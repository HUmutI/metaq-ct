"""fVLM preprocessing for the BCH pediatric cohort.

Replicates fvlm/data/resize.py + fvlm/data/preprocess.py for our data layout.
The repo scripts cannot be run as-is (they crash on `type='str'` in argparse, and
they assume the CT-RATE directory tree + metadata CSV). This is a faithful
re-implementation for our inputs; every numeric constant is taken from the repo.

Pipeline (per volume), matching the official order:
  1. Load image (already HU; header scl_slope is NaN -> slope 1.0 / intercept 0.0).
     [repo step fix_data.py: set spacing/origin/direction + rescale. Our NIfTIs are
      already in HU with correct pixdim, so only the spacing is needed.]
  2. Load TotalSegmentator mask, merge the 117 TS-v2 labels into fVLM's 4 organ ids
     exactly as resize.py's `merged_organ_id`:
         lung_upper_lobe_left(10), lung_lower_lobe_left(11), lung_upper_lobe_right(12),
         lung_middle_lobe_right(13), lung_lower_lobe_right(14)  -> 1 (lung)
         heart(51), atrial_appendage_left(61)                   -> 2 (heart)
         esophagus(15)                                          -> 3 (esophagus)
         aorta(52)                                              -> 4 (aorta)
  3. Resample image (trilinear) and mask (nearest) to 1.0 x 1.0 x 3.0 mm  [resize.py]
  4. Transpose (x,y,z) -> (z,y,x) and ScaleIntensityRange a_min=-1150 a_max=350
     -> [0,1] clip                                                      [preprocess.py]
  5. Crop to the nonzero-label bbox with margins (+5 z, +20 y, +20 x)    [preprocess.py]
  6. SpatialPad (symmetric) to at least (112, 256, 352)                  [preprocess.py]
  7. Save image float16 + mask uint8 to a compressed .npz

NOTE ON MASK GEOMETRY (verified, see check_align.py):
  Our TS masks are stored on a 1.5x1.5x3.0 mm grid with a reset (identity-diagonal)
  affine, while the images are on their native grid. The two grids cover the SAME
  field of view to <1 mm in every axis and share axis order/orientation -- verified
  empirically: with identity alignment, mean HU inside the lung labels is ~-830 and
  inside heart/aorta ~+50..+65 (physiologic), while all 4 axis-flip variants give
  nonsense. We therefore resample the mask directly from its own grid to the shared
  1x1x3 mm target grid (nearest), which is the composition of (mask grid -> image
  grid) and (image grid -> 1x1x3 grid). FOV agreement is re-asserted per volume.
"""
import os, sys, json, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

IMG_DIR  = '/temp_work/ch278233/BCH_DATASET/THIN_LUNG_NII'
MASK_DIR = '/temp_work/ch278233/TS_MASKS_PEDS'
OUT_DIR  = '/temp_work/ch278233/PEDS_BENCH/fvlm/processed_valid'

TS_MERGE = {10: 1, 11: 1, 12: 1, 13: 1, 14: 1,   # lung lobes -> lung
            51: 2, 61: 2,                        # heart + atrial appendage -> heart
            15: 3,                               # esophagus
            52: 4}                               # aorta
REF_SPACING = (1.0, 1.0, 3.0)
A_MIN, A_MAX = -1150.0, 350.0
PAD_TO = (112, 256, 352)
EXT_D, EXT_HW = 5, 20


def _resize(t, size, mode):
    """t: (X,Y,Z) tensor -> (size) via F.interpolate (what monai Resized uses)."""
    t = t[None, None]
    if mode == 'nearest':
        out = F.interpolate(t.float(), size=size, mode='nearest')
    else:
        out = F.interpolate(t.float(), size=size, mode='trilinear', align_corners=False)
    return out[0, 0]


def process_one(acc, vol_name, out_dir=OUT_DIR):
    ipath = os.path.join(IMG_DIR, f'{acc}.nii')
    mpath = os.path.join(MASK_DIR, f'{acc}.nii.gz')
    qc = {'volume': vol_name, 'accession': acc}

    im = nib.load(ipath)
    mk = nib.load(mpath)
    izo = np.asarray(im.header.get_zooms()[:3], dtype=float)
    mzo = np.asarray(mk.header.get_zooms()[:3], dtype=float)
    ish = np.asarray(im.shape[:3]); msh = np.asarray(mk.shape[:3])
    fov_i, fov_m = ish * izo, msh * mzo
    qc['fov_img'] = fov_i.round(2).tolist()
    qc['fov_mask'] = fov_m.round(2).tolist()
    qc['fov_max_abs_diff_mm'] = float(np.abs(fov_i - fov_m).max())
    if qc['fov_max_abs_diff_mm'] > 5.0:
        qc['status'] = 'FAIL_FOV_MISMATCH'
        return None, qc

    # --- image: raw HU (slope NaN in header => nibabel ignores scaling; assert range)
    img = np.asanyarray(im.dataobj).astype(np.float32)
    qc['hu_min'], qc['hu_max'] = float(img.min()), float(img.max())
    if not (-3100 <= img.min() <= 100 and 100 <= img.max() <= 32000):
        qc['status'] = 'FAIL_HU_RANGE'
        return None, qc

    # --- mask: merge TS labels -> 4 fVLM organ ids
    msk_raw = np.asanyarray(mk.dataobj)
    merged = np.zeros(msk_raw.shape, dtype=np.uint8)
    present = {}
    for ts_id, organ_id in TS_MERGE.items():
        sel = msk_raw == ts_id
        n = int(sel.sum())
        if n:
            merged[sel] = organ_id
            present[ts_id] = n
    qc['ts_labels_used'] = present
    if merged.max() == 0:
        qc['status'] = 'FAIL_EMPTY_MASK'
        return None, qc

    # --- step 3: resample both to 1.0 x 1.0 x 3.0 mm (target from the IMAGE grid)
    tgt = [max(1, int(round(ish[i] * izo[i] / REF_SPACING[i]))) for i in range(3)]
    qc['target_size_xyz'] = tgt
    img_t = _resize(torch.from_numpy(img), tgt, 'trilinear')
    del img
    msk_t = _resize(torch.from_numpy(merged.astype(np.float32)), tgt, 'nearest').to(torch.uint8)

    # --- step 4: transpose (x,y,z)->(z,y,x); scale intensity
    img_t = img_t.permute(2, 1, 0).contiguous()
    msk_t = msk_t.permute(2, 1, 0).contiguous()
    img_t = ((img_t - A_MIN) / (A_MAX - A_MIN)).clamp_(0.0, 1.0)

    # --- step 5: crop to label bbox + margins
    nz = torch.nonzero(msk_t)
    mn = nz.min(0).values; mx = nz.max(0).values
    ext = torch.tensor([EXT_D, EXT_HW, EXT_HW])
    lo = torch.maximum(mn - ext, torch.zeros(3, dtype=torch.long))
    hi = torch.minimum(mx + ext, torch.tensor(list(msk_t.shape)))
    img_t = img_t[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    msk_t = msk_t[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    qc['cropped_shape'] = list(img_t.shape)

    # --- step 6: symmetric pad to at least PAD_TO (monai SpatialPad default)
    pads = []
    for d in range(3):
        need = max(0, PAD_TO[d] - img_t.shape[d])
        pads.append((need // 2, need - need // 2))
    # F.pad wants last-dim-first ordering
    fp = [pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]]
    img_t = F.pad(img_t[None], fp, mode='constant', value=0.0)[0]
    msk_t = F.pad(msk_t[None].float(), fp, mode='constant', value=0.0)[0].to(torch.uint8)
    qc['final_shape'] = list(img_t.shape)

    organ_vox = {o: int((msk_t == i + 1).sum()) for i, o in
                 enumerate(['lung', 'heart', 'esophagus', 'aorta'])}
    qc['organ_voxels'] = organ_vox
    qc['status'] = 'OK'

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, vol_name.replace('.nii.gz', '.npz')),
                        image=img_t.numpy().astype(np.float16),
                        mask=msk_t.numpy())
    return True, qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vollist', default='/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt')
    ap.add_argument('--map', default='/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv')
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--qc-out', default='/temp_work/ch278233/PEDS_BENCH/fvlm/qc')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    vol2acc = {}
    with open(args.map) as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 2:
                vol2acc[p[1]] = p[0]

    vols = [l.strip() for l in open(args.vollist) if l.strip()]
    if args.limit:
        vols = vols[:args.limit]
    vols = vols[args.shard::args.nshards]

    os.makedirs(args.qc_out, exist_ok=True)
    qc_path = os.path.join(args.qc_out, f'qc_{args.shard:03d}.jsonl')
    done = set()
    if os.path.exists(qc_path):
        for line in open(qc_path):
            try: done.add(json.loads(line)['volume'])
            except Exception: pass

    with open(qc_path, 'a') as qf:
        for i, v in enumerate(vols):
            if v in done and os.path.exists(os.path.join(args.out, v.replace('.nii.gz', '.npz'))):
                continue
            acc = vol2acc.get(v)
            if acc is None:
                qf.write(json.dumps({'volume': v, 'status': 'FAIL_NO_ACCESSION'}) + '\n'); qf.flush(); continue
            if not os.path.exists(os.path.join(MASK_DIR, f'{acc}.nii.gz')):
                qf.write(json.dumps({'volume': v, 'accession': acc, 'status': 'FAIL_NO_TS_MASK'}) + '\n'); qf.flush(); continue
            if not os.path.exists(os.path.join(IMG_DIR, f'{acc}.nii')):
                qf.write(json.dumps({'volume': v, 'accession': acc, 'status': 'FAIL_NO_IMAGE'}) + '\n'); qf.flush(); continue
            try:
                _, qc = process_one(acc, v, args.out)
            except Exception as e:
                qc = {'volume': v, 'accession': acc, 'status': f'FAIL_EXC:{type(e).__name__}:{e}'}
            qf.write(json.dumps(qc) + '\n'); qf.flush()
            if i % 20 == 0:
                print(f'[{args.shard}] {i}/{len(vols)} {v} {qc.get("status")}', flush=True)

if __name__ == '__main__':
    main()

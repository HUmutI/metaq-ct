"""HARNESS VALIDATION CONTROL: run the exact same fVLM zero-shot code on CT-RATE
validation, where the paper reports numbers, to check that our model build,
checkpoint load, text prototypes and scoring logic are correct.

IMPORTANT CAVEAT: the raw CT-RATE validation NIfTIs are not on this filesystem
(they were deleted after npz conversion). The only local copies are preprocessed
npz at 1.5 x 1.5 x 3.0 mm (key 'arr_0', shape (z,y,x), values = HU/1000). The
official fVLM pipeline resamples from NATIVE (~0.7 mm) in-plane spacing to 1.0 mm;
we can only go 1.5 mm -> 1.0 mm, which upsamples already-downsampled data. This
control therefore gives a LOWER BOUND on what the harness would score under the
paper's exact preprocessing -- it validates correctness, not exact reproduction.

Mask alignment verified empirically (see report): CT-RATE TS masks read with
nibabel are (x,y,z) and must be transposed (2,0,1) to align elementwise with the
npz. With that transpose, mean HU inside the lung labels is ~-817 and heart ~+50;
every other transpose/flip gives non-physiologic values.
"""
import os, sys, json, glob, argparse, time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peds_data import TS_MERGE, ORGANS, A_MIN, A_MAX, PAD_TO, EXT_D, EXT_HW, _resize
from infer_peds import center_crop, ROI_SIZE, OFFICIAL_ITEMS

NPZ_ROOT = '/temp_work/ch278233/CTRATE/mps_ct_npz/valid'
MASK_DIR = '/temp_work/ch278233/CTRATE_TS_RAW/valid'
LABELS = '/temp_work/ch278233/CTRATE/multi_abnormality_labels/valid_predicted_labels.csv'
SRC_SPACING_ZYX = (3.0, 1.5, 1.5)
REF_SPACING_ZYX = (3.0, 1.0, 1.0)

def find_npz(name):
    pid = name.rsplit('_', 2)[0]
    c = glob.glob(f'{NPZ_ROOT}/{pid}/*/{name}.npz')
    return c[0] if c else None

def preprocess_ctrate(name):
    qc = {'volume': name + '.nii.gz'}
    p = find_npz(name); mp = os.path.join(MASK_DIR, f'{name}.nii.gz')
    if p is None: qc['status'] = 'FAIL_NO_NPZ'; return None, None, qc
    if not os.path.exists(mp): qc['status'] = 'FAIL_NO_MASK'; return None, None, qc

    a = np.load(p)['arr_0'].astype(np.float32) * 1000.0        # (z,y,x) HU
    m = np.asanyarray(nib.load(mp).dataobj)                     # (x,y,z)
    m = np.transpose(m, (2, 0, 1))                              # -> aligns with npz
    if m.shape != a.shape:
        qc['status'] = f'FAIL_SHAPE {m.shape} vs {a.shape}'; return None, None, qc

    merged = np.zeros(m.shape, dtype=np.uint8)
    for ts_id, oid in TS_MERGE.items():
        merged[m == ts_id] = oid
    if merged.max() == 0: qc['status'] = 'FAIL_EMPTY_MASK'; return None, None, qc

    lung = np.isin(m, [10, 11, 12, 13, 14])
    qc['lung_mean_hu'] = round(float(a[lung].mean()), 1) if lung.sum() else None
    if qc['lung_mean_hu'] is not None and qc['lung_mean_hu'] > -400:
        qc['status'] = 'FAIL_LUNG_HU_IMPLAUSIBLE'; return None, None, qc

    tgt = [max(1, int(round(a.shape[i] * SRC_SPACING_ZYX[i] / REF_SPACING_ZYX[i]))) for i in range(3)]
    qt = _resize(torch.from_numpy(a), tgt, 'trilinear'); del a
    mt = _resize(torch.from_numpy(merged.astype(np.float32)), tgt, 'nearest').to(torch.uint8)
    qt = ((qt - A_MIN) / (A_MAX - A_MIN)).clamp_(0.0, 1.0)

    nz = torch.nonzero(mt); mn = nz.min(0).values; mx = nz.max(0).values
    ext = torch.tensor([EXT_D, EXT_HW, EXT_HW])
    lo = torch.maximum(mn - ext, torch.zeros(3, dtype=torch.long))
    hi = torch.minimum(mx + ext, torch.tensor(list(mt.shape)))
    qt = qt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; mt = mt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    pads = [(max(0, PAD_TO[d] - qt.shape[d]) // 2,
             max(0, PAD_TO[d] - qt.shape[d]) - max(0, PAD_TO[d] - qt.shape[d]) // 2) for d in range(3)]
    fp = [pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]]
    qt = F.pad(qt[None], fp, value=0.0)[0]
    mt = F.pad(mt[None].float(), fp, value=0.0)[0].to(torch.uint8)
    qc['final_shape'] = list(qt.shape); qc['status'] = 'OK'
    return qt[None], mt[None], qc

class CTRateDS(Dataset):
    def __init__(self, names): self.names = names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        try: return preprocess_ctrate(self.names[i])
        except Exception as e:
            return None, None, {'volume': self.names[i] + '.nii.gz', 'status': f'FAIL_EXC:{type(e).__name__}:{e}'}

def collate(b): return b[0]

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--qc-out', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--n', type=int, default=1000); ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    from monai import transforms
    from fvlm_model import build_model
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, info = build_model(device=dev)
    print('CKPT matched', info['matched'], flush=True)

    # the 16 OFFICIAL fVLM test items, verbatim
    items = [tuple(v) for v in OFFICIAL_ITEMS.values()]
    tfd = model.prepare_text_feat(items)
    pad_func = transforms.DivisiblePadd(keys=["image", "label"], k=(16, 16, 32),
                                        mode='constant', constant_values=0, method="end")

    import pandas as pd
    names = [n[:-7] for n in pd.read_csv(LABELS)['VolumeName'].tolist()]
    rng = np.random.RandomState(0); rng.shuffle(names)
    names = names[:args.n][args.shard::args.nshards]

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try: done.add(json.loads(l)['volume'])
            except Exception: pass
    names = [n for n in names if n + '.nii.gz' not in done]
    print(f'shard {args.shard}: {len(names)} volumes', flush=True)

    dl = DataLoader(CTRateDS(names), batch_size=1, num_workers=args.workers, collate_fn=collate)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fo = open(args.out, 'a'); fq = open(args.qc_out, 'a'); t0 = time.time()
    for n, (image, mask, qc) in enumerate(dl):
        fq.write(json.dumps(qc) + '\n'); fq.flush()
        if image is None:
            fo.write(json.dumps({'volume': qc['volume'], 'status': qc['status']}) + '\n'); fo.flush(); continue
        try:
            image = image[None].to(dev).float(); mask = mask[None].to(dev).float()
            wos = {o: torch.eq(mask, ORGANS.index(o) + 1).sum().item() for o in ORGANS}
            tos = [o for o in ORGANS if wos[o] > 0]
            its = [it for it in items if it[0] in tos]
            ol = {it: [] for it in its}; ofeat = {}
            for k, v in ol.items():
                if not len(v):
                    oid = ORGANS.index(k[0])
                    wp, wm = center_crop(image, torch.eq(mask, oid + 1), crop_size=ROI_SIZE)
                    wm = wm.float(); wm[wm == 1] = oid + 1
                    pdd = pad_func({'image': wp[0], 'label': wm[0]})
                    ol = model.forward_test_win(pdd['image'][None], pdd['label'][None], ol, tos,
                                                tfd, ofeat, wos, skip_organ=oid)
            rec = {'volume': qc['volume'], 'status': 'OK', 'probs': {}}
            for it, pr in ol.items():
                if len(pr): rec['probs'][it[1]] = float(np.concatenate(pr).mean(0)[1])
            fo.write(json.dumps(rec) + '\n'); fo.flush()
            del image, mask
        except Exception as e:
            fo.write(json.dumps({'volume': qc['volume'], 'status': f'FAIL:{type(e).__name__}:{e}'}) + '\n'); fo.flush()
        if n % 25 == 0:
            print(f'[{args.shard}] {n}/{len(names)} {time.time()-t0:.0f}s', flush=True)
    print('SHARD_DONE', args.shard, flush=True)

if __name__ == '__main__':
    main()

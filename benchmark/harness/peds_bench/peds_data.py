"""fVLM preprocessing for the BCH pediatric cohort -- streaming (no intermediate files).

Replicates fvlm/data/resize.py + fvlm/data/preprocess.py for our data layout.
The repo scripts cannot be run as-is: both crash on `parser.add_argument(..., type='str')`
(a string where a callable is required), and both assume the CT-RATE directory tree +
metadata CSV. This is a faithful re-implementation for our inputs; every numeric
constant is taken verbatim from the repo.

Pipeline per volume (official order):
  1. Load image. READ PATH: np.asanyarray(img.dataobj), which APPLIES the NIfTI
     rescale carried on the ArrayProxy (slope=1.0, inter=-1024.0). The header fields
     header['scl_slope']/['scl_inter'] read back as NaN, but the scaling is real and
     is already applied by this path -- it must NOT be re-applied. Verified: arrays
     land in -1024..3071 HU, and mean HU inside the TS lung labels is ~-830 while
     heart/aorta are ~+50..+65 (physiologic). Enforced per volume by QC gates below.
  2. Merge the 117 TotalSegmentator-v2 labels into fVLM's 4 organ ids, exactly as
     resize.py's `merged_organ_id` dict:
       lung_upper_lobe_left(10), lung_lower_lobe_left(11), lung_upper_lobe_right(12),
       lung_middle_lobe_right(13), lung_lower_lobe_right(14)   -> 1 (lung)
       heart(51), atrial_appendage_left(61)                    -> 2 (heart)
       esophagus(15)                                           -> 3 (esophagus)
       aorta(52)                                               -> 4 (aorta)
  3. Resample image (trilinear) + mask (nearest) to 1.0 x 1.0 x 3.0 mm   [resize.py]
  4. Transpose (x,y,z)->(z,y,x); ScaleIntensityRange a_min=-1150 a_max=350
     -> [0,1] clip                                                      [preprocess.py]
  5. Crop to nonzero-label bbox, margins +5 z / +20 y / +20 x           [preprocess.py]
  6. SpatialPad (symmetric) to at least (112,256,352)                   [preprocess.py]

MASK GEOMETRY (verified): our TS masks sit on a 1.5x1.5x3.0 mm grid with a reset
identity-diagonal affine, images on their native grid. The grids cover the SAME FOV
to <1 mm on every axis and share axis order/orientation -- verified empirically:
identity alignment gives physiologic HU inside every organ label while all four
axis-flip variants give nonsense. We therefore resample the mask directly from its
own grid onto the shared 1x1x3 mm target grid, which is the composition of
(mask grid -> image grid) and (image grid -> 1x1x3 grid). Both FOV agreement and
lung-HU plausibility are re-asserted per volume and recorded in QC.

AXIS ORDER: nibabel is used for BOTH image and mask, so both arrays are (x,y,z);
we never mix a nibabel array with a SimpleITK (z,y,x) array.
"""
import os, json
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG_DIR  = '/temp_work/ch278233/BCH_DATASET/THIN_LUNG_NII'
MASK_DIR = '/temp_work/ch278233/TS_MASKS_PEDS'
VOL_MAP  = '/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv'

TS_MERGE = {10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 51: 2, 61: 2, 15: 3, 52: 4}
ORGANS = ['lung', 'heart', 'esophagus', 'aorta']
REF_SPACING = (1.0, 1.0, 3.0)
A_MIN, A_MAX = -1150.0, 350.0
PAD_TO = (112, 256, 352)
EXT_D, EXT_HW = 5, 20

# QC gates
FOV_TOL_MM = 5.0          # image vs mask field-of-view agreement
LUNG_HU_MAX = -400.0      # mean HU inside lung labels must be clearly air-like
Z_SPACING_RANGE = (0.1, 10.0)
# Physiologic extent gate. Some headers carry a corrupt z-spacing that is still
# inside Z_SPACING_RANGE but implies an impossible scan length (e.g. ped_02059_7:
# 1185 slices x 5.1 mm = 6043 mm of chest). The TS mask inherits the same bad
# spacing, so the FOV-agreement check cannot catch this -- it needs its own gate.
Z_EXTENT_MM_RANGE = (50.0, 1000.0)
XY_EXTENT_MM_RANGE = (50.0, 700.0)


def load_vol_map(path=VOL_MAP):
    m = {}
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 2:
                m[p[1]] = p[0]
    return m


def _resize(t, size, mode):
    t = t[None, None].float()
    if mode == 'nearest':
        out = F.interpolate(t, size=size, mode='nearest')
    else:
        out = F.interpolate(t, size=size, mode='trilinear', align_corners=False)
    return out[0, 0]


def preprocess(acc, vol_name):
    """Returns (image (1,112+,256+,352+) float32, mask same uint8, qc dict) or (None,None,qc)."""
    qc = {'volume': vol_name, 'accession': acc}
    ipath = os.path.join(IMG_DIR, f'{acc}.nii')
    mpath = os.path.join(MASK_DIR, f'{acc}.nii.gz')

    im = nib.load(ipath)
    mk = nib.load(mpath)

    # --- geometry / orientation QC (report, never silently transform) -----------
    axcodes = ''.join(nib.aff2axcodes(im.affine))
    qc['axcodes'] = axcodes
    qc['axial_stored'] = axcodes[2] in ('S', 'I')
    izo = np.asarray(im.header.get_zooms()[:3], dtype=float)
    mzo = np.asarray(mk.header.get_zooms()[:3], dtype=float)
    qc['spacing_img'] = izo.round(4).tolist()
    if not (Z_SPACING_RANGE[0] <= izo[2] <= Z_SPACING_RANGE[1]) or not np.all(np.isfinite(izo)):
        qc['status'] = 'FAIL_BAD_Z_SPACING'; return None, None, qc
    if not qc['axial_stored']:
        qc['status'] = 'FAIL_NON_AXIAL_STORAGE'; return None, None, qc
    ext_mm = np.asarray(im.shape[:3], dtype=float) * izo
    qc['extent_mm'] = ext_mm.round(1).tolist()
    if not (Z_EXTENT_MM_RANGE[0] <= ext_mm[2] <= Z_EXTENT_MM_RANGE[1]):
        qc['status'] = 'FAIL_IMPLAUSIBLE_Z_EXTENT'; return None, None, qc
    if not (XY_EXTENT_MM_RANGE[0] <= ext_mm[0] <= XY_EXTENT_MM_RANGE[1] and
            XY_EXTENT_MM_RANGE[0] <= ext_mm[1] <= XY_EXTENT_MM_RANGE[1]):
        qc['status'] = 'FAIL_IMPLAUSIBLE_XY_EXTENT'; return None, None, qc

    ish = np.asarray(im.shape[:3]); msh = np.asarray(mk.shape[:3])
    fov_i, fov_m = ish * izo, msh * mzo
    qc['fov_max_abs_diff_mm'] = float(np.abs(fov_i - fov_m).max())
    if qc['fov_max_abs_diff_mm'] > FOV_TOL_MM:
        qc['status'] = 'FAIL_FOV_MISMATCH'; return None, None, qc

    # --- image in HU (rescale already applied by the dataobj path) --------------
    img = np.asanyarray(im.dataobj).astype(np.float32)
    qc['hu_min'], qc['hu_max'] = float(img.min()), float(img.max())
    if not (-3200 <= img.min() <= 100 and 100 <= img.max() <= 32000):
        qc['status'] = 'FAIL_HU_RANGE'; return None, None, qc

    # --- mask merge -------------------------------------------------------------
    msk_raw = np.asanyarray(mk.dataobj)
    merged = np.zeros(msk_raw.shape, dtype=np.uint8)
    present = {}
    for ts_id, organ_id in TS_MERGE.items():
        sel = msk_raw == ts_id
        n = int(sel.sum())
        if n:
            merged[sel] = organ_id; present[ts_id] = n
    qc['ts_labels_used'] = present
    if merged.max() == 0:
        qc['status'] = 'FAIL_EMPTY_MASK'; return None, None, qc

    # --- HU plausibility inside the lung labels (catches any misalignment) -------
    lung_sel = np.isin(msk_raw, [10, 11, 12, 13, 14])
    if lung_sel.sum() > 0:
        idx = [np.clip((np.arange(o) + 0.5) * s / o - 0.5, 0, s - 1).round().astype(int)
               for s, o in zip(img.shape, msk_raw.shape)]
        lung_hu = float(img[np.ix_(*idx)][lung_sel].mean())
        qc['lung_mean_hu'] = round(lung_hu, 1)
        if lung_hu > LUNG_HU_MAX:
            qc['status'] = 'FAIL_LUNG_HU_IMPLAUSIBLE'; return None, None, qc
    else:
        qc['lung_mean_hu'] = None

    # --- step 3: resample to 1.0 x 1.0 x 3.0 mm ---------------------------------
    tgt = [max(1, int(round(ish[i] * izo[i] / REF_SPACING[i]))) for i in range(3)]
    qt = _resize(torch.from_numpy(img), tgt, 'trilinear')
    del img
    mt = _resize(torch.from_numpy(merged.astype(np.float32)), tgt, 'nearest').to(torch.uint8)

    # --- step 4: (x,y,z) -> (z,y,x); intensity scaling --------------------------
    qt = qt.permute(2, 1, 0).contiguous()
    mt = mt.permute(2, 1, 0).contiguous()
    qt = ((qt - A_MIN) / (A_MAX - A_MIN)).clamp_(0.0, 1.0)

    # --- step 5: crop to label bbox + margins -----------------------------------
    nz = torch.nonzero(mt)
    mn = nz.min(0).values; mx = nz.max(0).values
    ext = torch.tensor([EXT_D, EXT_HW, EXT_HW])
    lo = torch.maximum(mn - ext, torch.zeros(3, dtype=torch.long))
    hi = torch.minimum(mx + ext, torch.tensor(list(mt.shape)))
    qt = qt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    mt = mt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # --- step 6: symmetric pad to at least PAD_TO -------------------------------
    pads = [(max(0, PAD_TO[d] - qt.shape[d]) // 2,
             max(0, PAD_TO[d] - qt.shape[d]) - max(0, PAD_TO[d] - qt.shape[d]) // 2)
            for d in range(3)]
    fp = [pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]]
    qt = F.pad(qt[None], fp, mode='constant', value=0.0)[0]
    mt = F.pad(mt[None].float(), fp, mode='constant', value=0.0)[0].to(torch.uint8)

    qc['final_shape'] = list(qt.shape)
    qc['organ_voxels'] = {o: int((mt == i + 1).sum()) for i, o in enumerate(ORGANS)}
    qc['status'] = 'OK'
    return qt[None], mt[None], qc


class PedsVolumes(Dataset):
    """Streams preprocessed volumes; nothing is written to disk."""
    def __init__(self, vols, vol2acc):
        self.vols = vols; self.vol2acc = vol2acc

    def __len__(self):
        return len(self.vols)

    def __getitem__(self, i):
        v = self.vols[i]
        acc = self.vol2acc.get(v)
        if acc is None:
            return None, None, {'volume': v, 'status': 'FAIL_NO_ACCESSION'}
        if not os.path.exists(os.path.join(IMG_DIR, f'{acc}.nii')):
            return None, None, {'volume': v, 'accession': acc, 'status': 'FAIL_NO_IMAGE'}
        if not os.path.exists(os.path.join(MASK_DIR, f'{acc}.nii.gz')):
            return None, None, {'volume': v, 'accession': acc, 'status': 'FAIL_NO_TS_MASK'}
        try:
            return preprocess(acc, v)
        except Exception as e:
            return None, None, {'volume': v, 'accession': acc,
                                'status': f'FAIL_EXC:{type(e).__name__}:{e}'}


def collate(batch):
    return batch[0]

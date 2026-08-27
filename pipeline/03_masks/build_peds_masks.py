#!/usr/bin/env python3
"""TotalSegmentator output -> the 13-region pediatric mask arc-ct consumes.

Mirrors tools/prepare_data.py:preprocess_mask (pad/crop to 240x240x120, nearest
resample to 192x192x96) with two changes:

  * FINE_LABEL_MAP_PEDS instead of the adult map, adding region 12 (chest wall
    and thoracic skeleton) and 13 (upper abdomen).
  * Region 11, the pleural space, is DERIVED. TotalSegmentator has no pleura
    class, and the obvious workaround is wrong: routing effusion or pneumothorax
    to the lung lobes masks out exactly the voxels the query needs, because
    TotalSegmentator labels the collapsed lung and the fluid or air sits outside
    it. So dilate the lung union, subtract the lungs, and keep only voxels no
    real organ claimed.

The dilation happens AFTER the resample to 192x192x96, not on the native grid.
build_role_mask nearest-downsamples 192x192x96 to a 12x12x12 feature grid, so
one feature token is 16x16x8 mask voxels and a 2-3 voxel shell would vanish.
(8, 8, 4) guarantees at least one token of thickness.

Prints aggregates only.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, nibabel as nib
from scipy import ndimage

# The CT and the mask must reach 192x192x96 by the SAME geometric route, so
# import the loader's own function instead of reimplementing it. They diverged
# before: the mask padded end-only and then interpolated 240->192 (a 0.8 scale)
# while the loader pads symmetrically and CENTER-CROPS. Measured on real data,
# a 186-voxel-wide child ended up with mean HU under the lung label at -281
# while the background sat at -528 - the mask was more air-like outside the lung
# than inside it. Volumes at or above 240 looked fine, which is why an
# adult-sized check would never have caught it.
sys.path.insert(0, "/home/ch278233/bch-arc-ct")
from arcct.dataset import _pad_crop_hwd as _ct_pad_crop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ch278233/pipeline/lib")   # shared modules: ts_roi, qwen_extract, prepare_reports
from ts_roi import (FINE_LABEL_MAP_PEDS, FINE_LABEL_MAP_PEDS10, PEDS10_NAMES,
                     CHEST_WALL, PEDS)

# The bony ring, used to derive the pleural region from the RIBCAGE rather than
# from the lung. Sternum, ribs and vertebrae only -- not muscle, scapula or
# clavicle, which sit outside the cavity and would swallow it.
_BONE = [n for n in CHEST_WALL
         if n == "sternum" or n.startswith(("rib_", "vertebrae_"))]
RIBCAGE_IDS = [PEDS.index(n) + 1 for n in _BONE if n in PEDS]

NAMES = {1: "lung_upper_lobe_left", 2: "lung_lower_lobe_left",
         3: "lung_upper_lobe_right", 4: "lung_middle_lobe_right",
         5: "lung_lower_lobe_right", 6: "trachea", 7: "heart", 8: "aorta",
         9: "vessels", 10: "esophagus", 11: "pleural_space (derived)",
         12: "chest_wall_skeleton", 13: "upper_abdomen"}

PAD_SHAPE = (240, 240, 120)
TARGET_SHAPE = (192, 192, 96)
LUNG_IDS = (1, 2, 3, 4, 5)
PLEURA_ID = 11
DILATE = (8, 8, 4)
# Bridging the intercostal gaps before filling. At 1.5 mm in plane a gap between
# adjacent ribs is roughly 10-20 mm, so the closing element has to be wider than
# that or the fill leaks straight out through it.
CLOSE_XY = 9
PLEURA_MODE = "lung"

# Small tubular structures vanish at the feature grid. Measured on 20 pediatric
# volumes with the exact build_role_mask downsample (192x192x96 -> 12x12x12,
# nearest): trachea reached >=1 token in only 50% of volumes, esophagus 60%,
# aorta and vessels 80%. An empty region is not dropped - anatomy_qformer.py
# turns it into an unrestricted global query - so half the tracheal queries were
# silently not anatomy queries at all. One feature token is 24x16x16 mm and a
# child's trachea does not fill it.
#
# Dilating them into background only (never into another organ) fixes it:
#   radius     trachea  aorta  vessels  esophagus
#   none          50%    80%     80%       60%
#   (2,2,1)       70%   100%     95%       80%
#   (4,4,2)      100%   100%     95%       85%   <- chosen
#   (6,6,3)      100%   100%     95%       90%
# Past (4,4,2) the gain nearly stops, so more dilation would trade anatomical
# fidelity for almost nothing. 4 voxels x 1.5 mm = 6 mm in-plane, which is the
# mediastinal fat immediately around the structure - tissue a radiologist reads
# together with it anyway.
SMALL_IDS = (6, 8, 9, 10)          # trachea, aorta, vessels, esophagus
SMALL_DILATE = (4, 4, 2)


def _pad_crop(vol: np.ndarray, target) -> np.ndarray:
    for ax in range(3):
        if vol.shape[ax] > target[ax]:
            start = (vol.shape[ax] - target[ax]) // 2
            sl = [slice(None)] * 3
            sl[ax] = slice(start, start + target[ax])
            vol = vol[tuple(sl)]
    pads = [(0, max(0, t - s)) for s, t in zip(vol.shape, target)]
    return np.pad(vol, pads, mode="constant", constant_values=0)


def _ellipsoid(rx: int, ry: int, rz: int) -> np.ndarray:
    zz, yy, xx = np.ogrid[-rx:rx + 1, -ry:ry + 1, -rz:rz + 1]
    return ((zz / rx) ** 2 + (yy / ry) ** 2 + (xx / rz) ** 2) <= 1.0


def _ribcage_cavity(bone: np.ndarray) -> np.ndarray:
    """Everything enclosed by the bony thorax, slice by slice.

    Why this instead of dilating the lung: the shell built by dilating the lung
    union is not an anatomical definition, it is a FUNCTION OF THE PATHOLOGY.
    TotalSegmentator folds pneumothorax air into the lung field, so a positive
    volume has a larger 'lung', the shell lands further out in the chest wall,
    and the region ends up DENSER on positives than on negatives. Measured on 400
    volumes stratified for the class: +6 HU on positives against -98 HU on
    negatives, a 104 HU inversion, with the region twice the size. That is why
    the routed read-out scores pneumothorax at 0.317, below chance -- the pooled
    feature is anti-correlated with the finding.

    The ribcage does not move when the lung collapses, so a region defined from it
    cannot carry that confound.

    Filled per axial slice, not in 3D: the ribs are separate objects and a 3D fill
    escapes through the gaps between them, above the first rib and below the
    twelfth. Closing first bridges the intercostal spaces so the 2D fill has a
    closed contour to fill.
    """
    out = np.zeros_like(bone, dtype=bool)
    el = np.ones((CLOSE_XY, CLOSE_XY), dtype=bool)
    for z in range(bone.shape[2]):
        sl = bone[:, :, z]
        if not sl.any():
            continue
        closed = ndimage.binary_closing(sl, structure=el)
        out[:, :, z] = ndimage.binary_fill_holes(closed) & ~closed
    return out


def build(ts_path: str, out_path: str) -> dict:
    raw = np.asanyarray(nib.load(ts_path).dataobj).astype(np.int16)
    merged = np.zeros_like(raw, dtype=np.uint8)
    # Source ids 1-9 collide numerically with destination ids 1-9. Safe only
    # because we write into a separate array; an in-place remap would turn
    # spleen into a lung lobe.
    for src, dst in FINE_LABEL_MAP_PEDS.items():
        merged[raw == src] = dst

    # identical transform to the CT: pad/crop to 240 then centre-crop to 192.
    # Pure pad/crop, so integer labels survive unchanged - no interpolation.
    m = np.round(_ct_pad_crop(merged.astype(np.float32), TARGET_SHAPE)).astype(np.uint8)

    # Expand the small mediastinal structures first, into background only.
    for r in SMALL_IDS:
        b = (m == r)
        if b.any():
            grown = ndimage.binary_dilation(b, structure=_ellipsoid(*SMALL_DILATE))
            m[grown & (m == 0)] = r

    # Then the pleural region. Both modes claim BACKGROUND only, so a real organ
    # always wins; they differ in what they propose.
    if PLEURA_MODE == "cavity":
        # Carried through the SAME pad/crop as the labels, then filled at
        # 192x192x96 rather than on the native grid -- 96 slices of 192x192
        # instead of ~300 of 512x512, for an identical result, since the
        # transform is pure pad/crop with no interpolation.
        bone = np.round(_ct_pad_crop(
            np.isin(raw, RIBCAGE_IDS).astype(np.float32), TARGET_SHAPE)).astype(bool)
        cav = _ribcage_cavity(bone)
        m[cav & (m == 0)] = PLEURA_ID
    else:
        lung = np.isin(m, LUNG_IDS)
        if lung.any():
            shell = ndimage.binary_dilation(lung, structure=_ellipsoid(*DILATE)) & ~lung
            m[shell & (m == 0)] = PLEURA_ID
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(m, affine=np.eye(4)), out_path)

    # Survival at the feature grid, using the exact mechanism from
    # anatomy_qformer.build_role_mask: (D,H,W) nearest-downsampled to 12x12x12.
    # A voxel-count proxy is not good enough - nearest sampling can keep a
    # structure smaller than a token, or drop a larger one that misses the
    # sample points. This is the number that decides whether an anatomy query
    # is real or silently degenerates into a global query.
    lo = F.interpolate(torch.from_numpy(m.transpose(2, 0, 1)).float()[None, None],
                       size=(12, 12, 12), mode="nearest")[:, 0].long().flatten(1)[0]
    # max(...) is already the highest region id; the +1 that used to be here
    # made this return a count for region 11 in the 10-region schema, and the
    # caller's counters only go to 10 - a KeyError per volume, which the loop
    # caught and turned into a skipped mask.
    n = max(max(FINE_LABEL_MAP_PEDS.values()), PLEURA_ID)
    return ({r: int((m == r).sum()) for r in range(1, n + 1)},
            {r: int((lo == r).sum()) for r in range(1, n + 1)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/temp_work/ch278233/TS_MASKS_PEDS")
    ap.add_argument("--out-dir", default="/temp_work/ch278233/PEDS_MASKS_192")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=int(os.environ.get("SHARD_ID", "0")))
    ap.add_argument("--shard-n", type=int, default=int(os.environ.get("SHARD_N", "1")))
    ap.add_argument("--no-map", action="store_true",
                    help="output stem already equals the VolumeName (CT-RATE)")
    ap.add_argument("--pleura", choices=["lung", "cavity"], default="lung",
                    help="how region 11/8 is derived: dilate the lung union "
                         "(the original, confounded by pneumothorax) or fill the "
                         "bony ribcage (independent of the lung segmentation)")
    ap.add_argument("--schema", choices=["13", "10"], default="13",
                    help="10 = the measured pediatric schema with merged small regions")
    a = ap.parse_args()

    global FINE_LABEL_MAP_PEDS, PLEURA_ID, SMALL_IDS, NAMES, N_REGIONS, PLEURA_MODE
    PLEURA_MODE = a.pleura
    if a.schema == "10":
        FINE_LABEL_MAP_PEDS = FINE_LABEL_MAP_PEDS10
        PLEURA_ID = 8            # pleura moves from 11 to 8 in the merged schema
        SMALL_IDS = (6, 7)       # airway+esophagus, heart+great vessels
        NAMES = PEDS10_NAMES
        N_REGIONS = 10
    else:
        N_REGIONS = 13

    # Name the mask by the de-identified VolumeName, not the accession.
    # dataset._npz_to_mask_path derives the mask filename from the npz filename,
    # and the npz files are named ped_NNNNN_S. Keeping the accession here meant
    # the two name spaces never intersected - measured: 8816 npz stems, 8816
    # mask stems, intersection 0 - so every anatomy query would have run
    # unmasked with only "masked/batch=0.00" in the log to show for it.
    # The pediatric TS outputs are named by accession and must be renamed to the
    # de-identified VolumeName the npz files use; CT-RATE outputs already carry
    # the volume name, so the map is skipped rather than faked. Getting this
    # wrong once produced 8,816 masks and 8,816 npz whose stems never
    # intersected - every anatomy query ran unmasked with nothing in the log but
    # "masked/batch=0.00".
    import csv as _csv
    if a.no_map:
        a2v = None
    else:
        a2v = {r["accession"]: r["volume_name"] for r in
               _csv.DictReader(open("/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv"),
                               delimiter="\t")}
    files = sorted(f for f in os.listdir(a.in_dir) if f.endswith(".nii.gz"))
    if a.shard_n > 1:
        files = files[a.shard_id::a.shard_n]
    if a.limit:
        files = files[:a.limit]
    print("[mask] %d volumes" % len(files), flush=True)

    empty = {r: 0 for r in range(1, N_REGIONS + 1)}
    tok = {r: 0 for r in range(1, N_REGIONS + 1)}
    n = 0
    # `fail` was used below without ever being initialised - a latent NameError
    # that only fires when a TS mask has no volume_map entry, i.e. the first
    # time this is rerun on newly pulled data.
    fail = 0
    for i, fn in enumerate(files):
        vol = fn[:-7] if a2v is None else a2v.get(fn[:-7])
        if not vol:
            fail += 1
            continue
        dst = os.path.join(a.out_dir, vol.replace(".nii.gz", "").replace(".nii", "") + ".nii.gz")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        try:
            counts, grid = build(os.path.join(a.in_dir, fn), dst)
            n += 1
            for r, c in counts.items():
                if c == 0:
                    empty[r] += 1
            for r, g in grid.items():
                if g > 0:
                    tok[r] += 1
        except Exception as exc:
            print("[mask] FAIL %s: %s" % (fn[:16], type(exc).__name__), flush=True)
        if (i + 1) % 200 == 0:
            print("[mask] %d/%d" % (i + 1, len(files)), flush=True)

    print("\n[mask] built %d masks (%d skipped: no volume_map entry)" % (n, fail))
    print("[mask] %-4s %-24s %8s %14s" % ("id", "region", "empty", "reaches grid"))
    for r in range(1, N_REGIONS + 1):
        pct = 100.0 * tok[r] / max(n, 1)
        flag = "   <<< becomes a global query" if pct < 98 else ""
        print("[mask] %-4d %-24s %8d %13.1f%%%s" % (r, NAMES.get(r, "?"), empty[r], pct, flag))
    print("[mask] 'reaches grid' = has >=1 token after the 12x12x12 nearest downsample.")
    print("[mask] Anything below 98%% is an anatomy query that silently stops being one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cache CT-CLIP's own preprocessing for the pediatric cohort.

CT-CLIP's CTReportDataset.nii_img_to_tensor resamples to 0.75x0.75x1.5 mm, clips
HU to [-1000,1000], scales by /1000 and centre-crops/pads to (480,480,240),
returning (1,240,480,480) with pad value -1.  Its trainer runs that transform
inside __getitem__, i.e. it re-reads and re-resamples a ~500 MB volume on every
epoch.  For 5,946 pediatric volumes that is not affordable, so we run the
IDENTICAL transform once and cache the result.

Fidelity notes -- this is a cache, not a reimplementation:
  * `resize_array` is imported from CT-CLIP's own scripts/data.py, not copied.
  * The only deliberate deviation is float32 instead of float64 during resample
    (halves peak RAM; far below CT's ~1 HU quantisation).
  * Our pediatric NIfTIs are already in HU (dcm2niix applied rescale; verified
    min -1024 / max ~3070), and their headers carry scl_slope=nan, so we use
    slope=1.0 intercept=0.0 rather than propagating NaN.
  * Spacing is taken from the NIfTI header pixdim, which is what a metadata CSV
    would have recorded anyway.
Cached as float16: the values live in [-1,1] and float16 has ~3 decimal digits
there, well below CT noise.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, "/home/ch278233/BENCHMARK/CT-CLIP/scripts")
from data import resize_array          # CT-CLIP's own function

TARGET_SPACING = (1.5, 0.75, 0.75)     # (z, x, y) -- data.py:104-106
TARGET_SHAPE = (480, 480, 240)         # (h, w, d) -- data.py:131
HU_MIN, HU_MAX = -1000, 1000           # data.py:122


def one(nii_path: str, out_path: str) -> tuple[bool, str]:
    img = nib.load(nii_path)
    hdr = img.header
    zx = hdr["pixdim"][1:4].astype(np.float64)   # (x, y, z) mm
    xy_spacing, z_spacing = float(zx[0]), float(zx[2])
    if not np.isfinite([xy_spacing, z_spacing]).all() or min(xy_spacing, z_spacing) <= 0:
        return False, f"bad spacing {zx}"

    a = np.asanyarray(img.dataobj, dtype=np.float32)    # already HU
    a = a.transpose(2, 0, 1)                            # data.py:113 -> (z,x,y)
    t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    del a
    out = resize_array(t, (z_spacing, xy_spacing, xy_spacing), TARGET_SPACING)
    del t
    out = out[0][0]
    out = np.transpose(out, (1, 2, 0))                  # -> (x,y,z)

    out = np.clip(out, HU_MIN, HU_MAX) / 1000.0
    t = torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
    del out

    dh, dw, dd = TARGET_SHAPE
    h, w, d = t.shape
    hs, ws, ds = max((h - dh) // 2, 0), max((w - dw) // 2, 0), max((d - dd) // 2, 0)
    t = t[hs:min(hs + dh, h), ws:min(ws + dw, w), ds:min(ds + dd, d)]
    pads = []
    for size, target in ((t.size(2), dd), (t.size(1), dw), (t.size(0), dh)):
        before = (target - size) // 2
        pads += [before, target - size - before]
    t = torch.nn.functional.pad(t, pads, value=-1)       # data.py:156
    t = t.permute(2, 0, 1)                               # -> (240,480,480)

    if tuple(t.shape) != (dd, dh, dw):
        return False, f"shape {tuple(t.shape)}"
    tmp = out_path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, arr=t.numpy().astype(np.float16))
    os.replace(tmp, out_path)
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vollist", required=True)
    ap.add_argument("--map", default="/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    vol2nii = {}
    with open(a.map) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                vol2nii[p[1]] = p[2]

    vols = [l.strip() for l in open(a.vollist) if l.strip()]
    mine = vols[a.shard::a.nshards]
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[prep] shard {a.shard}/{a.nshards}: {len(mine)} of {len(vols)} volumes", flush=True)

    ok = skip = miss = fail = 0
    t0 = time.time()
    for i, v in enumerate(mine):
        stem = v[:-7] if v.endswith(".nii.gz") else v
        out = os.path.join(a.out_dir, stem + ".npz")
        if os.path.exists(out):
            skip += 1
            continue
        src = vol2nii.get(v)
        if not src or not os.path.exists(src):
            miss += 1
            print(f"[prep] MISSING {v} -> {src}", flush=True)
            continue
        try:
            good, why = one(src, out)
            if good:
                ok += 1
            else:
                fail += 1
                print(f"[prep] FAIL {v}: {why}", flush=True)
        except Exception as e:
            fail += 1
            print(f"[prep] ERROR {v}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[prep] {i+1}/{len(mine)} ok={ok} skip={skip} miss={miss} fail={fail} "
                  f"{el/max(ok,1):.1f}s/vol", flush=True)

    print(f"[prep] DONE shard {a.shard}: ok={ok} skip={skip} miss={miss} fail={fail} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    return 1 if (miss or fail) else 0


if __name__ == "__main__":
    sys.exit(main())

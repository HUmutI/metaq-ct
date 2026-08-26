#!/usr/bin/env python3
"""Resampled pediatric NIfTI -> the .npz the training pipeline reads.

No re-resampling from raw: prepare_data.py's preprocess_volume resamples to
1.5x1.5x3.0 mm, transposes to (D,H,W) and divides by 1000, and TS_RESAMPLED is
already that volume in HU on that exact grid. So this is a transpose and a
scale, not a second interpolation - which also avoids re-reading 3.5 TB of raw
volumes over NFS.

The pediatric NIfTIs are already in Hounsfield units (dcm2niix applied the
rescale at conversion), so unlike the CT-RATE path there is no slope/intercept
to apply here.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, nibabel as nib


def convert(src: str, dst: str) -> dict:
    vol = np.asanyarray(nib.load(src).dataobj).astype(np.float32)   # (H, W, D), HU
    dhw = np.transpose(vol, (2, 0, 1)).astype(np.float32) / 1000.0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, arr_0=dhw)
    return {"shape": dhw.shape, "min": float(dhw.min()), "max": float(dhw.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/temp_work/ch278233/TS_RESAMPLED")
    ap.add_argument("--out-dir", default="/temp_work/ch278233/PEDS_NPZ")
    ap.add_argument("--map", default="/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv")
    a = ap.parse_args()
    sid = int(os.environ.get("SHARD_ID", "0"))
    sn = int(os.environ.get("SHARD_N", "1"))

    # name the npz by the de-identified VolumeName, not the accession, so the
    # training set carries no identifier
    import csv
    a2v = {r["accession"]: r["volume_name"] for r in
           csv.DictReader(open(a.map), delimiter="\t")}

    files = sorted(f for f in os.listdir(a.in_dir) if f.endswith(".nii.gz"))
    files = files[sid::sn]
    print("[npz] shard %d/%d: %d volumes" % (sid, sn, len(files)), flush=True)
    ok = skip = fail = 0
    for i, fn in enumerate(files):
        acc = fn[:-7]
        vol_name = a2v.get(acc)
        if not vol_name:
            fail += 1
            continue
        stem = vol_name.replace(".nii.gz", "").replace(".nii", "")
        dst = os.path.join(a.out_dir, stem + ".npz")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1
            continue
        try:
            info = convert(os.path.join(a.in_dir, fn), dst)
            ok += 1
            if ok <= 2:
                print("[npz] %s -> %s  HU/1000 [%.2f, %.2f]"
                      % (stem, info["shape"], info["min"], info["max"]), flush=True)
        except Exception as exc:
            fail += 1
            print("[npz] FAIL %s: %s" % (stem, type(exc).__name__), flush=True)
        if (i + 1) % 300 == 0:
            print("[npz] %d/%d" % (i + 1, len(files)), flush=True)
    print("[npz] DONE ok=%d skip=%d fail=%d" % (ok, skip, fail))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Preprocess CT-RATE volumes and TotalSegmentator masks for ARC-CT.

Volumes: rescale to Hounsfield units with the per-scan slope/intercept from the
CT-RATE metadata CSV (the released NIfTI headers do not carry them), resample to
1.5x1.5x3.0 mm, store as (D, H, W) float32 in HU/1000 under key ``arr_0``.
Cropping to 192x192x96 happens in the data loader, not here.

Masks: merge the TotalSegmentator label map into the ten thoracic groups used by
the anatomy queries (five lung lobes, trachea, heart, aorta, mediastinal vessels,
esophagus), pad/crop to 240x240x120, then nearest-resample to 192x192x96.

Usage:
    python tools/prepare_data.py --split train \
        --nifti-dir  /data/ct_rate/train \
        --mask-dir   /data/ct_rate/train_masks_raw \
        --metadata   /data/ct_rate/metadata/train_metadata.csv \
        --out-npz    /data/ct_rate/npz/train \
        --out-mask   /data/ct_rate/masks_192/train
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

NPZ_TARGET_SPACING = (1.5, 1.5, 3.0)
MASK_PAD_SHAPE = (240, 240, 120)
MASK_TARGET_SHAPE = (192, 192, 96)

# TotalSegmentator label -> ARC-CT organ group.
# 1-5 lung lobes, 6 trachea, 7 heart, 8 aorta, 9 mediastinal vessels, 10 esophagus.
FINE_LABEL_MAP = {
    10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 16: 6,
    51: 7, 61: 7, 52: 8, 53: 9, 54: 9, 62: 9, 63: 9, 15: 10,
}


def _pad_crop(vol: np.ndarray, target_shape) -> np.ndarray:
    for ax in range(3):
        if vol.shape[ax] > target_shape[ax]:
            start = (vol.shape[ax] - target_shape[ax]) // 2
            sl = [slice(None)] * 3
            sl[ax] = slice(start, start + target_shape[ax])
            vol = vol[tuple(sl)]
    pads = [(0, max(0, t - s)) for s, t in zip(vol.shape, target_shape)]
    return np.pad(vol, pads, mode="constant", constant_values=0)


def preprocess_volume(nii_path: Path, meta_row: dict, out_path: Path) -> str:
    if out_path.exists():
        return "skip"
    nii = nib.load(str(nii_path))
    vol = nii.get_fdata().astype(np.float32)
    vol = vol * float(meta_row.get("RescaleSlope", 1.0)) + float(meta_row.get("RescaleIntercept", 0.0))
    try:
        xy_val = float(ast.literal_eval(str(meta_row.get("XYSpacing", "[1.0,1.0]")))[0])
    except Exception:
        xy_val = 1.0
    try:
        z_val = float(meta_row.get("ZSpacing", 1.0))
    except Exception:
        z_val = 1.0
    orig = (xy_val, xy_val, z_val)
    target_shape = tuple(max(1, round(s * so / st))
                         for s, so, st in zip(vol.shape, orig, NPZ_TARGET_SPACING))
    t = torch.from_numpy(vol)[None, None]
    t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    vol_dhw = np.transpose(t.squeeze().numpy(), (2, 0, 1)).astype(np.float32) / 1000.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, arr_0=vol_dhw)
    return "ok"


def preprocess_mask(ts_path: Path, out_path: Path) -> str:
    if out_path.exists():
        return "skip"
    raw = nib.load(str(ts_path)).get_fdata().astype(np.int16)
    merged = np.zeros_like(raw, dtype=np.uint8)
    for src, dst in FINE_LABEL_MAP.items():
        merged[raw == src] = dst
    padded = _pad_crop(merged, MASK_PAD_SHAPE)
    t = torch.from_numpy(padded).float()[None, None]
    t = F.interpolate(t, size=MASK_TARGET_SHAPE, mode="nearest")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(t.squeeze().numpy().astype(np.uint8), affine=np.eye(4)), str(out_path))
    return "ok"


def _job(kind, src, meta_row, dst):
    try:
        return preprocess_volume(src, meta_row, dst) if kind == "vol" else preprocess_mask(src, dst)
    except Exception as exc:                                   # noqa: BLE001
        return f"err:{exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nifti-dir", type=Path)
    ap.add_argument("--mask-dir", type=Path, help="raw TotalSegmentator output")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path)
    ap.add_argument("--out-mask", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata).set_index("VolumeName").to_dict("index")
    jobs = []
    if args.nifti_dir and args.out_npz:
        for f in sorted(args.nifti_dir.rglob("*.nii.gz")):
            jobs.append(("vol", f, meta.get(f.name, {}), args.out_npz / f.name.replace(".nii.gz", ".npz")))
    if args.mask_dir and args.out_mask:
        for f in sorted(args.mask_dir.rglob("*.nii.gz")):
            jobs.append(("mask", f, {}, args.out_mask / f.name))

    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_job, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            st = fut.result()
            counts[st.split(":")[0]] = counts.get(st.split(":")[0], 0) + 1
            if i % 500 == 0:
                print(f"{i}/{len(jobs)} {counts}", flush=True)
    print(f"done {counts}")


if __name__ == "__main__":
    main()

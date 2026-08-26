#!/usr/bin/env python3
"""Resample a BCH pediatric NIfTI to 1.5x1.5x3.0 mm and write .nii.gz for TotalSegmentator.

TotalSegmentator MUST run on this grid, not on the native volume. arc-ct's
tools/prepare_data.py:preprocess_mask pad/crops the mask to 240x240x120 and
interpolates to 192x192x96 *without ever resampling by voxel spacing*, while the
CT goes through a 1.5x1.5x3.0 mm resample first. Feed it a mask built at native
512x512x~1000 resolution and it will centre-crop a sliver out of the middle of
the chest, silently.

Unlike CT-RATE, these volumes are already in Hounsfield units - dcm2niix applied
the rescale slope/intercept at conversion - so no rescale is applied here.
Spacing comes from the NIfTI header rather than a metadata CSV.

Prints shapes and spacings only; never a path component that identifies a study.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, nibabel as nib

TARGET = (1.5, 1.5, 3.0)


class NoSpacing(ValueError):
    """Header carries no real voxel spacing, so a resample would be meaningless."""


def resample(src: str, dst: str) -> dict:
    nii = nib.load(src)
    vol = np.asanyarray(nii.dataobj)
    # dcm2niix stacks multiple reconstructions of one series into a 4th
    # dimension. Same family as the _Eq_1 duplicate we fixed at pull time.
    # Keep the first frame and say so - silently averaging or reshaping would
    # mix two different recons into one volume.
    extra = 0
    if vol.ndim > 3:
        extra = vol.shape[3]
        vol = vol[..., 0]
    vol = vol.astype(np.float32)                             # already HU
    zx, zy, zz = (float(v) for v in nii.header.get_zooms()[:3])
    # A header with exactly 1.0 mm on all three axes is dcm2niix's default when
    # spacing was never written, not a real isotropic acquisition: 512 px at
    # 1.0 mm is a 512 mm FOV and 87 slices at 1.0 mm is an 87 mm chest.
    # Resampling on that would put the volume at the wrong physical scale.
    if (zx, zy, zz) == (1.0, 1.0, 1.0):
        raise NoSpacing("header spacing is the 1.0/1.0/1.0 default")
    shape = tuple(max(1, round(s * o / t))
                  for s, o, t in zip(vol.shape, (zx, zy, zz), TARGET))
    t = torch.from_numpy(vol)[None, None]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)
    out = t.squeeze().numpy().astype(np.float32)
    # Affine carrying the new spacing, so TotalSegmentator resamples correctly
    aff = np.diag([TARGET[0], TARGET[1], TARGET[2], 1.0]).astype(np.float64)
    img = nib.Nifti1Image(out, affine=aff)
    img.header.set_zooms(TARGET)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    nib.save(img, dst)
    return {"in_shape": vol.shape, "in_spacing": (round(zx, 4), round(zy, 4), round(zz, 4)),
            "out_shape": shape, "hu_min": float(vol.min()), "hu_max": float(vol.max()),
            "extra_frames": extra}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="file of .nii paths, one per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    paths = [p.strip() for p in open(a.list) if p.strip()]
    if a.limit:
        paths = paths[:a.limit]
    print("[resample] %d volumes -> %s" % (len(paths), a.out_dir))
    for i, p in enumerate(paths):
        stem = os.path.basename(p)[:-4]
        dst = os.path.join(a.out_dir, stem + ".nii.gz")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        try:
            info = resample(p, dst)
            if info["extra_frames"]:
                print("[resample] #%d was 4D with %d frames, kept frame 0"
                      % (i, info["extra_frames"]))
            if i < 3:
                print("[resample] #%d %s @ %s -> %s   HU[%.0f, %.0f]"
                      % (i, info["in_shape"], info["in_spacing"], info["out_shape"],
                         info["hu_min"], info["hu_max"]))
                if info["hu_min"] > -500:
                    print("[resample] WARNING HU min > -500: not Hounsfield units?")
        except Exception as exc:
            print("[resample] FAIL #%d %s" % (i, type(exc).__name__))
    print("[resample] done, %d files in out-dir" % len(os.listdir(a.out_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

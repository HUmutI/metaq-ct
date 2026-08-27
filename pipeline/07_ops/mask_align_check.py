#!/usr/bin/env python3
"""Do these masks line up with the volumes the loader will pair them with?

Inside the lung labels the mean HU must sit well BELOW the mean outside the
mask: lung is air at roughly -600 HU, everything else is soft tissue and bone. A
mask that is misaligned, or that reached 192x192x96 by a different geometric
route than the loader uses, shows up as lung no more air-like than its
surroundings. That is not hypothetical -- build_peds_masks.py carries a comment
about exactly this failure, where a 186-voxel-wide child ended up with lung at
-281 HU against a background at -528, and where volumes at or above 240 looked
fine so an adult-sized check would never have caught it.

Two controls run beside the set under test: our own CT-RATE validation masks and
the pediatric masks, both produced by the pipeline in this repo. Whatever rate
this metric fails at on KNOWN-GOOD masks is the number to compare against, not
zero -- consolidated lung genuinely is not air, and a run with real pathology
will always have some.

Run on a compute node; each check reads a full npz off NFS.
"""
from __future__ import annotations

import glob
import os
import statistics
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, "/home/ch278233/bch-arc-ct")
from arcct.dataset import _pad_crop_hwd                      # noqa: E402

WANT = int(os.environ.get("MASKCHK_N", "60"))


def run(label: str, masks: list[str], npz_for, want: int = WANT) -> None:
    ok = n = 0
    diffs: list[float] = []
    for p in masks:
        name = os.path.basename(p)[:-7]
        npz = npz_for(name)
        if not npz or not os.path.exists(npz):
            continue
        try:
            m = np.asanyarray(nib.load(p).dataobj)
            hu = np.transpose(np.load(npz)["arr_0"], (1, 2, 0)).astype(np.float32) * 1000.0
        except Exception as exc:                              # noqa: BLE001
            print(f"    {name}: okunamadi ({exc})", flush=True)
            continue
        hu = _pad_crop_hwd(hu, (192, 192, 96))
        lung = (m >= 1) & (m <= 5)
        if lung.sum() < 200:
            continue
        n += 1
        ins = float(hu[lung].mean())
        out = float(hu[m == 0].mean())
        diffs.append(ins - out)
        if ins < out - 100:
            ok += 1
        if n >= want:
            break
    if not n:
        print(f"{label:<32s} eslesen hacim bulunamadi", flush=True)
        return
    d = sorted(diffs)
    print(f"{label:<32s} {ok:>3d}/{n:<3d} (%{100 * ok / n:>3.0f})  "
          f"ortalama {statistics.mean(d):>7.0f} HU  medyan {d[len(d) // 2]:>7.0f}  "
          f"en kotu {d[-1]:>7.0f}", flush=True)


def main() -> int:
    print(f"her set icin {WANT} hacim; olcut: akciger HU'su maske disindan "
          ">=100 dusuk\n", flush=True)
    run("Bilkent CT-RATE train",
        sorted(glob.glob("/home/ch278233/bch-arc-ct/ct-rate-train-masks/*/*/*.nii.gz"))[:400],
        lambda n: f"/temp_work/ch278233/COMBINED_NPZ/{n.rsplit('_', 2)[0]}/{n}/{n}.npz")
    run("bizim CT-RATE valid (kontrol)",
        sorted(glob.glob("/temp_work/ch278233/CTRATE_MASKS10_192/valid/*.nii.gz"))[:400],
        lambda n: f"/temp_work/ch278233/CTRATE/mps_ct_npz/valid/"
                  f"{n.rsplit('_', 2)[0]}/{n.rsplit('_', 1)[0]}/{n}.npz")
    run("peds (kontrol)",
        sorted(glob.glob("/temp_work/ch278233/PEDS_MASKS10_192/*.nii.gz"))[:400],
        lambda n: f"/temp_work/ch278233/COMBINED_NPZ/{n.rsplit('_', 1)[0]}/{n}/{n}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

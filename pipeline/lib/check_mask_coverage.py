#!/usr/bin/env python3
"""Refuse to start when the mask set is incomplete.

RAC_REQUIRE_MASK=1 -- which configs/stage2.env sets, because stage 2 is trained
with organ masks -- makes dataset.py DROP any training volume that has no mask:

    if self.require_mask and mask_path is None:
        skipped_nomask += 1
        continue

So an incomplete mask set does not fail. It trains on a smaller cohort and
reports it in one line among many. At 67% coverage that is 28,494 CT-RATE volumes
instead of 42,544, against a published model that had all of them -- and on the
dropped third the anatomy queries would have run unrestricted with the per-organ
alignment loss never firing at all.

Exit 78 (the value the sbatch chain treats as fatal) when coverage is below the
threshold. Passing --min explicitly is how you say "yes, I mean to run short".

    python pipeline/lib/check_mask_coverage.py \\
        --vollist .../VOLLIST_TRAIN.txt --masks .../COMBINED_MASKS10_192 --min 99
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vollist", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--min", type=float, default=99.0, help="percent")
    a = ap.parse_args()

    if not os.path.isdir(a.masks):
        print(f"FATAL: maske dizini yok: {a.masks}")
        return 78
    have = {f[:-7] for f in os.listdir(a.masks) if f.endswith(".nii.gz")}
    names = [l.strip().replace(".nii.gz", "")
             for l in open(a.vollist, encoding="utf-8") if l.strip()]
    if not names:
        print(f"FATAL: bos hacim listesi: {a.vollist}")
        return 78
    n = sum(x in have for x in names)
    cov = 100.0 * n / len(names)
    print(f"[mask-cov] {n:,}/{len(names):,} = %{cov:.1f}  (esik %{a.min:g})")
    if cov + 1e-9 < a.min:
        print(f"FATAL: kapsam %{cov:.1f}, esik %{a.min:g}. RAC_REQUIRE_MASK=1 ile bu "
              f"kosu {len(names) - n:,} hacmi sessizce dusururdu.")
        print("       Maskeler bitsin, ya da bilerek kisa kosmak icin --min ver.")
        return 78
    return 0


if __name__ == "__main__":
    sys.exit(main())

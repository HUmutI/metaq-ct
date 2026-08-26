#!/usr/bin/env python3
"""Region survival over the whole pediatric cohort.

Answers one question per region: does its mask reach the AnatomyQFormer feature
grid, or does the query silently stop being an anatomy query? Uses the exact
mechanism from anatomy_qformer.build_role_mask - (D,H,W) nearest-downsampled to
12x12x12 - because a voxel-count proxy neither guarantees a structure survives
nor that a large one is sampled.

A region below 98% is not an error: anatomy_qformer.py turns an empty region
into an unrestricted global query, so the model still runs. It just means that
query is not doing the job the architecture claims for it.
"""
import sys, os
sys.path.insert(0, "/home/ch278233/pipeline/lib")
from __future__ import annotations
import argparse, os, sys
import numpy as np, nibabel as nib, torch, torch.nn.functional as F

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from ts_roi import PEDS10_NAMES as NAMES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/temp_work/ch278233/PEDS_MASKS_192")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.dir) if f.endswith(".nii.gz"))
    sid = int(os.environ.get("SHARD_ID", "0")); sn = int(os.environ.get("SHARD_N", "1"))
    if sn > 1:
        files = files[sid::sn]
    n = len(files)
    print("region survival over %d masks" % n, flush=True)

    empty = {r: 0 for r in NAMES}
    grid = {r: 0 for r in NAMES}
    toks = {r: [] for r in NAMES}
    for i, fn in enumerate(files):
        m = np.asanyarray(nib.load(os.path.join(a.dir, fn)).dataobj)
        lo = F.interpolate(torch.from_numpy(m.transpose(2, 0, 1)).float()[None, None],
                           size=(12, 12, 12), mode="nearest")[:, 0].long().flatten(1)[0]
        for r in NAMES:
            if not (m == r).any():
                empty[r] += 1
            c = int((lo == r).sum())
            toks[r].append(c)
            if c > 0:
                grid[r] += 1
        if (i + 1) % 1000 == 0:
            print("  %d/%d" % (i + 1, n), flush=True)

    print("\n%-4s %-26s %9s %14s %12s" % ("id", "region", "empty", "reaches grid", "med tokens"))
    below = []
    for r in NAMES:
        pct = 100.0 * grid[r] / max(n, 1)
        if pct < 98:
            below.append((r, pct))
        print("%-4d %-26s %9d %13.1f%% %12d%s"
              % (r, NAMES[r], empty[r], pct, int(np.median(toks[r])),
                 "   <<<" if pct < 98 else ""))
    print("\ntotal tokens on the grid: 1728")
    if below:
        print("below 98%% - these queries degenerate to global on some volumes:")
        for r, p in below:
            print("   %-26s %.1f%%  (fails on %d volumes)"
                  % (NAMES[r], p, n - grid[r]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

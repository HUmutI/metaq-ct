#!/usr/bin/env python3
"""Lay the pediatric npz out the way the dataset scanner expects.

dataset.py walks data_folder/*/*/*.npz - three levels, mirroring CT-RATE's
mps_ct_npz/<split>/<patient>/<series>/<name>.npz. PEDS_NPZ was flat, so the
scanner found zero samples and said nothing: the loop simply never enters
(dataset.py:305-312 has no counter for files at the wrong depth).

Hardlinks, not copies - 125 GB stays 125 GB.
"""
from __future__ import annotations
import os, re, sys

SRC = "/temp_work/ch278233/PEDS_NPZ"
DST = "/temp_work/ch278233/PEDS_NPZ_NESTED"


def main() -> int:
    os.makedirs(DST, exist_ok=True)
    n = skip = bad = 0
    for f in sorted(os.listdir(SRC)):
        if not f.endswith(".npz"):
            continue
        stem = f[:-4]
        m = re.match(r"(ped_\d+)_(\d+)$", stem)
        if not m:
            bad += 1
            continue
        subj = m.group(1)
        d = os.path.join(DST, subj, stem)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, f)
        if os.path.exists(dst):
            skip += 1
            continue
        os.link(os.path.join(SRC, f), dst)
        n += 1
    print("[nest] linked=%d skip=%d unparsable=%d" % (n, skip, bad))
    import glob
    found = glob.glob(os.path.join(DST, "*", "*", "*.npz"))
    print("[nest] scanner pattern */*/*.npz finds: %d" % len(found))
    return 0 if len(found) >= 8800 else 1


if __name__ == "__main__":
    sys.exit(main())

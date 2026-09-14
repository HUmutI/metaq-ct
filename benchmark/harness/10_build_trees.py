#!/usr/bin/env python3
"""Build cohort-isolated npz trees for the MPS-CT / GreenRFM benchmark.

Both repos glob data_folder/*/*/*.npz and key on VolumeName ending '.nii.gz'.
COMBINED_NPZ holds BOTH cohorts and both CT-RATE splits in one tree, so
globbing it directly would silently mix CT-RATE train with its dev split and
with pediatrics.  Physical separation by hardlink is cheap (same filesystem,
no bytes copied) and auditable afterwards with a file count.

Layout produced:  <dst>/<subject>/<stem>/<stem>.npz
"""
from __future__ import annotations
import os, sys

SRC_COMBINED = "/temp_work/ch278233/COMBINED_NPZ"
SRC_OFFICIAL = "/temp_work/ch278233/CTRATE/mps_ct_npz/valid"
DST = "/temp_work/ch278233/BENCHMARK_DATA"


def subject_of(stem: str) -> str:
    """COMBINED_NPZ nests by subject: train_1_a_1 -> train_1 ; ped_00001_1 -> ped_00001."""
    if stem.startswith(("train_", "valid_")):
        return "_".join(stem.split("_")[:2])
    return stem.rsplit("_", 1)[0]


def build(name: str, vollist: str, src: str, official: bool = False) -> int:
    dst = os.path.join(DST, name)
    os.makedirs(dst, exist_ok=True)
    vols = [l.strip() for l in open(vollist) if l.strip()]
    n = skip = miss = 0
    missing = []
    for v in vols:
        stem = v[:-7] if v.endswith(".nii.gz") else v
        subj = subject_of(stem)
        if official:
            # mps_ct_npz/valid nests patient/study/volume: valid_1/valid_1_a/valid_1_a_1.npz
            study = "_".join(stem.split("_")[:3])
            s = os.path.join(src, subj, study, stem + ".npz")
        else:
            s = os.path.join(src, subj, stem, stem + ".npz")
        if not os.path.exists(s):
            miss += 1
            if len(missing) < 5:
                missing.append(s)
            continue
        d = os.path.join(dst, subj, stem)
        os.makedirs(d, exist_ok=True)
        t = os.path.join(d, stem + ".npz")
        if os.path.exists(t):
            skip += 1
            continue
        os.link(s, t)
        n += 1
    print(f"[{name:14s}] wanted={len(vols):6d} linked={n:6d} already={skip:6d} MISSING={miss:5d}")
    for m in missing:
        print(f"    missing e.g. {m}")
    return miss


if __name__ == "__main__":
    T = "/temp_work/ch278233"
    rc = 0
    rc += build("ctrate_train", f"{T}/CTRATE_ONLY/VOLLIST_TRAIN.txt", SRC_COMBINED)
    rc += build("ctrate_dev",   f"{T}/CTRATE_ONLY/VOLLIST_VALID.txt", SRC_COMBINED)
    sys.exit(0 if rc == 0 else 1)

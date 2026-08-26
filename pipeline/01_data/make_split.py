#!/usr/bin/env python3
"""Patient-level train/valid split for the pediatric cohort.

Study-level splitting would leak: 4004 subjects hold 8817 studies, and one child
has 48 of them. Split by study and that child's scans land on both sides, so the
model sees the same anatomy in training and is scored on it again.

Stratified by age band as well, because the age distribution is the thing we
most need to compare across splits - 0-11y is the group where the anatomy
routing actually breaks down, and it must not end up concentrated in one side.
"""
from __future__ import annotations
import argparse, csv, json, random, re, sys
from collections import Counter, defaultdict

BANDS = [(0, 2, "0-1y"), (2, 6, "2-5y"), (6, 12, "6-11y"), (12, 18, "12-17y"),
         (18, 30, "18-29y"), (30, 200, "30+")]


def band(a: float) -> str:
    for lo, hi, n in BANDS:
        if lo <= a < hi:
            return n
    return "30+"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv")
    ap.add_argument("--reports", default="/temp_work/ch278233/BCH_DATASET/reports_8k.csv")
    ap.add_argument("--npz-dir", default="/temp_work/ch278233/PEDS_NPZ")
    ap.add_argument("--out", default="/temp_work/ch278233/PEDS_SPLIT.json")
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    csv.field_size_limit(2 ** 31 - 1)
    a2v = {r["accession"]: r["volume_name"] for r in
           csv.DictReader(open(a.map), delimiter="\t")}
    age = {}
    with open(a.reports, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            acc = (r.get("Accession Number") or "").strip()
            try:
                age[acc] = float((r.get("Patient Age") or "").strip())
            except Exception:
                pass

    import os
    have = {f[:-4] for f in os.listdir(a.npz_dir) if f.endswith(".npz")}

    subj = defaultdict(list)
    subj_age = {}
    for acc, vol in a2v.items():
        stem = vol.replace(".nii.gz", "").replace(".nii", "")
        if stem not in have:
            continue
        m = re.match(r"(ped_\d+)_\d+", stem)
        if not m:
            continue
        s = m.group(1)
        subj[s].append(stem)
        if acc in age:
            subj_age.setdefault(s, []).append(age[acc])

    rng = random.Random(a.seed)
    by_band = defaultdict(list)
    for s in subj:
        ages = subj_age.get(s, [])
        by_band[band(sum(ages) / len(ages)) if ages else "unknown"].append(s)

    train, valid = [], []
    for b, subs in by_band.items():
        rng.shuffle(subs)
        k = int(round(len(subs) * a.valid_frac))
        valid += subs[:k]
        train += subs[k:]

    tr_v = [v for s in train for v in subj[s]]
    va_v = [v for s in valid for v in subj[s]]
    overlap = set(train) & set(valid)
    vol_overlap = set(tr_v) & set(va_v)

    print("subjects : train %d  valid %d" % (len(train), len(valid)))
    print("studies  : train %d  valid %d" % (len(tr_v), len(va_v)))
    print("subject overlap : %d   study overlap : %d" % (len(overlap), len(vol_overlap)))
    assert not overlap and not vol_overlap, "LEAK"

    print("\n%-10s %8s %8s %8s" % ("band", "train", "valid", "valid%"))
    for _lo, _hi, b in BANDS + [(0, 0, "unknown")]:
        t = sum(len(subj[s]) for s in train if band(sum(subj_age.get(s, [0])) / max(len(subj_age.get(s, [0])), 1)) == b)
        v = sum(len(subj[s]) for s in valid if band(sum(subj_age.get(s, [0])) / max(len(subj_age.get(s, [0])), 1)) == b)
        if t or v:
            print("%-10s %8d %8d %7.1f%%" % (b, t, v, 100.0 * v / max(t + v, 1)))

    json.dump({"train": sorted(tr_v), "valid": sorted(va_v),
               "train_subjects": sorted(train), "valid_subjects": sorted(valid),
               "seed": a.seed}, open(a.out, "w"), indent=1)
    print("\nyazildi: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

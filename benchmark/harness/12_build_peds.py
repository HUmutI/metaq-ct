#!/usr/bin/env python3
"""Pediatric artefacts for the benchmark: 23-class label CSVs + isolated trees.

The 23-class task drops four adult-specific targets from the 27-label schema.
Per the 2 Sep results doc this is a genuine 23-output task, not a 27-way model
scored on a subset, so the columns are removed from the CSV the model builds
its head from -- both repos size their head off len(sample['labels']).

Coverage is 5,946/5,948 train and 1,507/1,509 valid: two volumes on each side
have no harmonised label row.  1,507 is exactly the clean V2 validation cohort
the pediatric results are reported on, so the shortfall is the known cohort
definition, not a new loss.  Volumes without labels are dropped from the tree
too, so the scanned sample count matches the label count exactly.
"""
from __future__ import annotations
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
bt = import_module("10_build_trees")

T = "/temp_work/ch278233"
OUT = f"{T}/BENCHMARK_DATA"
DROP = ["Arterial wall calcification", "Coronary artery wall calcification",
        "Hiatal hernia", "Emphysema"]


def write_labels(vollist_path, out_csv):
    want = [l.strip() for l in open(vollist_path) if l.strip()]
    wantset = set(want)
    rows, cols = [], None
    with open(f"{T}/COMBINED_labels_harmonised.csv") as f:
        rd = csv.DictReader(f)
        allcols = rd.fieldnames
        cols = ["VolumeName"] + [c for c in allcols if c != "VolumeName" and c not in DROP]
        for r in rd:
            if r["VolumeName"] in wantset:
                rows.append({c: r[c] for c in cols})
    kept = {r["VolumeName"] for r in rows}
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[labels] {os.path.basename(out_csv):28s} rows={len(rows):5d} "
          f"classes={len(cols)-1} of wanted {len(want)} (no label row: {len(wantset-kept)})")
    return kept


def write_vollist(keep, path):
    with open(path, "w") as f:
        f.write("\n".join(sorted(keep)) + "\n")
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    missing = 0
    for split, src in (("train", f"{T}/PEDS23_GE5_VOLLIST_TRAIN.txt"),
                       ("valid", f"{T}/PEDS23_GE5_VOLLIST_VALID.txt")):
        lab = f"{OUT}/peds23_labels_{split}.csv"
        keep = write_labels(src, lab)
        vl = write_vollist(keep, f"{OUT}/peds23_vollist_{split}.txt")
        missing += bt.build(f"peds_{split}", vl, bt.SRC_COMBINED)
    sys.exit(0 if missing == 0 else 1)

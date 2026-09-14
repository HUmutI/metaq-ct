#!/usr/bin/env python3
"""Official CT-RATE validation tree (3,002 studies) -- the untouched test split.

Kept separate from ctrate_dev (4,589 train-side studies used for checkpoint
selection).  The pediatric results doc is explicit that the official split was
opened only after training against the dev split; mixing the two here would
quietly undo that.
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
bt = import_module("10_build_trees")

T = "/temp_work/ch278233"
OUT = f"{T}/BENCHMARK_DATA"
lab = f"{T}/CTRATE/multi_abnormality_labels/valid_predicted_labels.csv"

vols = [r["VolumeName"] for r in csv.DictReader(open(lab))]
vl = f"{OUT}/ctrate_test_vollist.txt"
open(vl, "w").write("\n".join(vols) + "\n")
print(f"[test] official validation volumes: {len(vols)}")
sys.exit(bt.build("ctrate_test", vl, bt.SRC_OFFICIAL, official=True))

#!/usr/bin/env python3
"""Tiny CT-RATE trees so a smoke test does not have to scan 42k volumes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
bt = import_module("10_build_trees")

T = "/temp_work/ch278233"
SM = "/temp_work/ch278233/BENCHMARK_DATA/smoke"
os.makedirs(SM, exist_ok=True)

def head(src, n, out):
    vols = [l.strip() for l in open(src) if l.strip()][:n]
    with open(out, "w") as f:
        f.write("\n".join(vols) + "\n")
    return out

tr = head(f"{T}/CTRATE_ONLY/VOLLIST_TRAIN.txt", 40, f"{SM}/vollist_train.txt")
dv = head(f"{T}/CTRATE_ONLY/VOLLIST_VALID.txt", 16, f"{SM}/vollist_dev.txt")
rc  = bt.build("smoke/ctrate_train", tr, bt.SRC_COMBINED)
rc += bt.build("smoke/ctrate_dev",   dv, bt.SRC_COMBINED)
sys.exit(0 if rc == 0 else 1)

#!/usr/bin/env python3
"""What is inside a stage-2 checkpoint -- specifically, the query bank shape.

The Q-Former is saved under its OWN top-level key, `qformer_module`, not inside
`model`. Looking for "queries" in the model state_dict finds only BERT's
attention query projections and reports the bank as absent.

Run on a compute node: torch.load of these 1.3 GB files was killed on the login
node, and put the script somewhere shared -- /tmp is node-local, so a script
written on the login node is not there when the job runs.
"""
from __future__ import annotations

import argparse
import os

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    for p in a.ckpts:
        if not os.path.isfile(p):
            print(f"{p}: yok"); continue
        pkg = torch.load(p, map_location="cpu", weights_only=False)
        print(f"\n===== {os.path.basename(p)}  ({os.path.getsize(p)/1e6:.0f} MB)")
        if not isinstance(pkg, dict):
            print("  dict degil:", type(pkg)); continue
        print(f"  rac_schema={pkg.get('rac_schema')}  best_auc={pkg.get('best_auc')}  "
              f"update_step={pkg.get('update_step')}")
        qm = pkg.get("qformer_module")
        if isinstance(qm, dict):
            print(f"  qformer_module: {len(qm)} tensor")
            for k, v in qm.items():
                if hasattr(v, "shape") and v.dim() <= 2 and v.shape[0] <= 128:
                    print(f"     {k:46s} {tuple(v.shape)}")
        else:
            print(f"  qformer_module: {type(qm)}")
        cfg = pkg.get("config")
        if isinstance(cfg, dict):
            for k in sorted(cfg):
                v = cfg[k]
                if isinstance(v, (list, tuple)):
                    print(f"  config.{k}: {len(v)} oge")
                elif isinstance(v, (str, int, float, bool)):
                    print(f"  config.{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Macro AUROC on present- versus missing-indication subsets of saved predictions."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auc(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> tuple[float, dict]:
    per = {}
    for j in range(pred.shape[1]):
        use = mask & np.isfinite(true[:, j])
        y = true[use, j]
        per[j] = (float(roc_auc_score(y, pred[use, j]))
                  if len(y) and len(np.unique(y)) > 1 else None)
    vals = [v for v in per.values() if v is not None]
    return (float(np.mean(vals)) if vals else float("nan")), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.pred, allow_pickle=True)
    pred, true = z["pred"], z["true"].astype(float)
    accessions = np.asarray(z["accessions"]).astype(str)
    names = [str(x) for x in z["pathologies"]]
    ctx = pd.read_csv(a.context, keep_default_na=False).set_index("VolumeName")
    rows = ctx.reindex(accessions)
    present = (rows["ind_status"].eq("present").to_numpy()
               & rows["Indication_EN"].astype(str).str.strip().ne("").to_numpy())

    out = {}
    for label, mask in (("all", np.ones(len(accessions), dtype=bool)),
                        ("indication_present", present),
                        ("indication_missing", ~present)):
        auc, per = macro_auc(pred, true, mask)
        out[label] = {
            "n": int(mask.sum()), "macro_auc": auc,
            "n_evaluable_classes": sum(v is not None for v in per.values()),
            "per_class_auc": {names[j]: per[j] for j in range(len(names))},
        }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    for label, result in out.items():
        print(f"[subset] {label}: n={result['n']:,} "
              f"auc={result['macro_auc']:.5f} "
              f"classes={result['n_evaluable_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

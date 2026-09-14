#!/home/ch278233/micromamba/envs/arcct/bin/python
"""Score a saved prediction file with ARC-CT's OWN compute_metric_bundle.

Every row of the pediatric benchmark table must be scored by identical metric
code, or the Prec/Acc/F1 columns are not comparable. Rather than reimplement the
CT-CLIP protocol here (support-weighted F1, positive-class precision) and risk
drifting from it, we import the function ARC-CT itself uses.

Usage: 77_metric_bundle.py --preds X_preds.npz --name "GreenRFM (classifier)" --out Y.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, "/home/ch278233/arc-ct")
sys.path.insert(0, "/home/ch278233/arc-ct/tools")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--readout", default="", help="which readout produced these scores")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from evaluate import compute_metric_bundle

    d = np.load(a.preds, allow_pickle=True)
    pred, true = d["pred"].astype(np.float64), d["true"].astype(np.int32)
    classes = [str(c) for c in d["classes"]]
    if pred.shape != true.shape:
        sys.exit(f"[bundle] FATAL: pred {pred.shape} != true {true.shape}")

    # Pediatric patients contribute ~2.33 volumes each (1507 volumes, 648 patients),
    # so bootstrap must resample PATIENTS, not volumes, or the CI is far too narrow.
    # compute_metric_bundle groups on the "ped_00001" prefix of each name, which is
    # exactly ARC-CT's own patient_prefix unit.
    acc = d["volumes"] if "volumes" in d.files else None
    if acc is None:
        print("[bundle] WARNING: no volume names stored -- CI will be volume-level, "
              "which UNDERSTATES the interval. Point estimates are unaffected.")
    b = compute_metric_bundle(pred, true, classes, accessions=acc,
                              threshold=a.threshold, n_bootstrap=a.bootstrap)
    b["bootstrap_unit"] = "patient_prefix" if acc is not None else "volume"
    b["model"] = a.name
    b["readout"] = a.readout
    b["preds_file"] = os.path.abspath(a.preds)   # mutlak: sonuclar sonradan denetlenebilsin
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(b, open(a.out, "w"), indent=2)
    m = b["macro"]
    ci = ""
    if m.get("auc_ci_lo") is not None:
        ci = f" [{m['auc_ci_lo']:.4f}-{m['auc_ci_hi']:.4f}]"
    print(f"[bundle] {a.name:32s} AUC={m['auc']:.4f}{ci} Prec={m['precision']:.4f} "
          f"Acc={m['acc']:.4f} F1={m['f1']:.4f}  n={b['n_samples']} c={b['n_classes']} "
          f"unit={b['bootstrap_unit']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

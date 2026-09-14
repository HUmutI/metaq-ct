#!/usr/bin/env python3
"""Compare U18 versus age>=5 training on exactly the same age>=5 test cases."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze_predictions_paper import patient_groups


WORK = Path("/temp_work/ch278233")
U18 = Path(os.environ.get("PEDS_U18_EVAL_ROOT", str(WORK / "eval_matrix/peds23_u18_r2plus1d_18_test")))
GE5 = Path(os.environ.get("PEDS_GE5_EVAL_ROOT", str(WORK / "eval_matrix/peds23_u18_ge5_r2plus1d_18_test")))
OUT = Path(os.environ.get("PEDS_AGE_FACTORIAL_OUT", str(WORK / "eval_matrix/peds23_u18_age_factorial")))
TEX = Path(os.environ.get("PEDS_AGE_FACTORIAL_TEX", "/home/ch278233/paper/generated/peds23_age_factorial.tex"))
SNAPSHOT = os.environ.get("PEDS_U18_SNAPSHOT", "0") == "1"
LABEL_SUFFIX = "-snapshot" if SNAPSHOT else ""


def load(root: Path, arm: str):
    z = np.load(root / f"{arm}_ind-true/predictions.npz", allow_pickle=True)
    return z["pred"], z["true"].astype(int), z["accessions"], [str(x) for x in z["pathologies"]]


def macro(y, p):
    auc, ap = [], []
    for j in range(y.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            auc.append(roc_auc_score(y[:, j], p[:, j]))
            ap.append(average_precision_score(y[:, j], p[:, j]))
    return float(np.mean(auc)), float(np.mean(ap))


def main() -> int:
    demographics = pd.read_csv(WORK / "CONTEXT/demographics.csv").set_index("VolumeName")
    rows, predictions, reference = [], {}, None
    for seed in range(3):
        for model, arm in (("Full", f"full_seed{seed}"), ("CT-only", f"ct_only_seed{seed}")):
            up, uy, uv, uc = load(U18, arm)
            age = pd.to_numeric(demographics.loc[list(map(str, uv)), "AgeYears"], errors="coerce").to_numpy()
            take = (age >= 5) & (age < 18)
            gp, gy, gv, gc = load(GE5, arm)
            assert uc == gc
            u_index = {str(v): i for i, v in enumerate(uv[take])}
            order = np.asarray([u_index[str(v)] for v in gv])
            up, uy, uv = up[take][order], uy[take][order], uv[take][order]
            assert np.array_equal(uy, gy) and np.array_equal(uv, gv)
            if reference is None:
                reference = (gy, gv)
            else:
                assert np.array_equal(reference[0], gy) and np.array_equal(reference[1], gv)
            for cohort, pred in (("0--<18", up), ("5--<18", gp)):
                auc, ap = macro(gy, pred)
                rows.append({"seed": seed, "model": model, "training_age": cohort,
                             "auroc": auc, "auprc": ap})
                predictions[(seed, model, cohort)] = pred

    y, volumes = reference
    groups = patient_groups(volumes)
    patients = np.unique(groups)
    rng = np.random.default_rng(51)
    bootstrap = {model: [] for model in ("Full", "CT-only")}
    for _ in range(2000):
        sampled = rng.choice(patients, len(patients), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == patient) for patient in sampled])
        for model in bootstrap:
            delta = []
            for seed in range(3):
                younger = macro(y[idx], predictions[(seed, model, "0--<18")][idx])[0]
                ge5 = macro(y[idx], predictions[(seed, model, "5--<18")][idx])[0]
                delta.append(ge5 - younger)
            bootstrap[model].append(float(np.mean(delta)))

    summary = {"studies": int(len(y)), "patients": int(len(patients)), "rows": rows, "comparison": {}}
    for model, values in bootstrap.items():
        point = np.mean([r["auroc"] for r in rows if r["model"] == model and r["training_age"] == "5--<18"]) - np.mean([r["auroc"] for r in rows if r["model"] == model and r["training_age"] == "0--<18"])
        summary["comparison"][model] = {"delta_auroc": float(point),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Auto-generated age-factorial result.",
             ("\\subsection{20:00 checkpoint snapshot: does excluding ages 0--5 help?}"
              if SNAPSHOT else "\\subsection{Does excluding ages 0--5 help?}"),
             "\\begin{table}[t]", "\\centering",
             ("\\caption{Interim 20:00 age-training checkpoint snapshot evaluated on the identical 5--$<18$ MRN-disjoint test subset.}"
              if SNAPSHOT else "\\caption{Age-training factorial evaluated on the identical 5--$<18$ MRN-disjoint test subset.}"),
             f"\\label{{tab:peds-age-factorial{LABEL_SUFFIX}}}", "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
             "Model & Training ages & AUROC & AUPRC \\\\", "\\midrule"]
    for model in ("CT-only", "Full"):
        for cohort in ("0--<18", "5--<18"):
            selected = [r for r in rows if r["model"] == model and r["training_age"] == cohort]
            display_cohort = cohort.replace("<18", "$<18$")
            lines.append(f'{model} & {display_cohort} & {np.mean([r["auroc"] for r in selected]):.4f} & {np.mean([r["auprc"] for r in selected]):.4f} \\\\')
    lines += ["\\bottomrule", "\\end{tabular}"]
    for model in ("CT-only", "Full"):
        item = summary["comparison"][model]
        lines.append(f'\\parbox{{\\columnwidth}}{{\\footnotesize {model}: excluding ages 0--5 changes AUROC by {item["delta_auroc"]:+.4f} (patient-bootstrap 95\\% CI [{item["ci95"][0]:.4f}, {item["ci95"][1]:.4f}]).}}')
    lines += ["\\end{table}"]
    TEX.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

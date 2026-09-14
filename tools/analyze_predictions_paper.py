#!/usr/bin/env python3
"""Paper metrics with true MRN-level bootstrap for pediatric predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def patient_groups(volume_names: np.ndarray) -> np.ndarray:
    work = Path("/temp_work/ch278233")
    with (work / "BCH_DATASET/LABELS/volume_map.tsv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        volume_to_accession = {
            row["volume_name"]: row["accession"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
        }
    with (work / "BCH_DATASET/reports_8k.csv").open(
        newline="", encoding="utf-8-sig", errors="replace"
    ) as handle:
        accession_to_mrn = {
            row["Accession Number"].strip(): row["Patient MRN"].strip()
            for row in csv.DictReader(handle)
        }
    groups = []
    for value in volume_names:
        volume = str(value)
        accession = volume_to_accession.get(volume)
        mrn = accession_to_mrn.get(accession or "")
        if not mrn:
            raise RuntimeError("Prediction row could not be mapped to an MRN")
        groups.append(mrn)
    return np.asarray(groups)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        take = (p >= left) & (p < right if right < 1.0 else p <= right)
        if take.any():
            total += take.mean() * abs(float(p[take].mean()) - float(y[take].mean()))
    return float(total)


def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    take = (1.0 - fpr) >= target
    return float(np.max(tpr[take])) if take.any() else 0.0


def per_class(y: np.ndarray, p: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for index, name in enumerate(names):
        yt, pt = y[:, index].astype(int), p[:, index]
        valid = len(np.unique(yt)) > 1
        rows.append({
            "class": name,
            "n": int(len(yt)),
            "positive_n": int(yt.sum()),
            "prevalence": float(yt.mean()),
            "auroc": float(roc_auc_score(yt, pt)) if valid else float("nan"),
            "auprc": float(average_precision_score(yt, pt)) if yt.sum() else float("nan"),
            "brier": float(brier_score_loss(yt, pt)),
            "ece_10bin": ece(yt, pt),
            "sensitivity_at_90_specificity": sensitivity_at_specificity(yt, pt, 0.90) if valid else float("nan"),
            "sensitivity_at_95_specificity": sensitivity_at_specificity(yt, pt, 0.95) if valid else float("nan"),
        })
    return rows


def macro(rows: list[dict], key: str) -> float:
    return float(np.nanmean([row[key] for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    package = np.load(args.pred, allow_pickle=True)
    pred = package["pred"].astype(np.float64)
    true = package["true"].astype(int)
    names = [str(value) for value in package["pathologies"]]
    volumes = package["accessions"]
    groups = patient_groups(volumes)
    rows = per_class(true, pred, names)

    rng = np.random.default_rng(args.seed)
    patients = np.unique(groups)
    samples = {key: [] for key in ("auroc", "auprc", "brier", "ece_10bin")}
    for _ in range(args.bootstrap):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == patient) for patient in chosen])
        boot_rows = per_class(true[indices], pred[indices], names)
        for key in samples:
            samples[key].append(macro(boot_rows, key))

    summary = {
        "prediction_file": str(Path(args.pred).resolve()),
        "n_studies": int(len(true)),
        "n_patients": int(len(patients)),
        "n_classes": int(len(names)),
        "bootstrap_unit": "MRN/patient",
        "bootstrap_replicates": args.bootstrap,
        "macro": {},
    }
    for key, values in samples.items():
        summary["macro"][key] = macro(rows, key)
        summary["macro"][f"{key}_ci95"] = [
            float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))
        ]
    summary["macro"]["sensitivity_at_90_specificity"] = macro(rows, "sensitivity_at_90_specificity")
    summary["macro"]["sensitivity_at_95_specificity"] = macro(rows, "sensitivity_at_95_specificity")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_metrics.json").write_text(json.dumps(summary, indent=2))
    with (output / "paper_metrics_per_class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

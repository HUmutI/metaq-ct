#!/usr/bin/env python3
"""Aggregate heuristic audit of explicit target terms in BCH indications."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from arcct.schema import PEDS23_PATHOLOGIES


WORK = Path("/temp_work/ch278233")
LEXICON = {
    "Medical material": r"\b(?:catheter|central line|picc|port|chest tube|device|hardware|stent|drain|pacemaker)\b",
    "Cardiomegaly": r"\b(?:cardiomegal\w*|enlarged heart)\b",
    "Pericardial effusion": r"\bpericardial effusion\b",
    "Lymphadenopathy": r"\b(?:lymphadenopathy|adenopathy|enlarged lymph node)\b",
    "Atelectasis": r"\batelecta\w*\b",
    "Lung nodule": r"\b(?:lung|pulmonary) nodul\w*\b",
    "Lung opacity": r"\b(?:lung|pulmonary|airspace) opacit\w*\b",
    "Pulmonary fibrotic sequela": r"\b(?:pulmonary fibrosis|fibrotic lung|lung scarring)\b",
    "Pleural effusion": r"\bpleural effusion\b",
    "Mosaic attenuation pattern": r"\bmosaic attenuation\b",
    "Peribronchial thickening": r"\bperibronchial thickening\b",
    "Consolidation": r"\bconsolidat\w*\b",
    "Bronchiectasis": r"\bbronchiecta\w*\b",
    "Interlobular septal thickening": r"\b(?:interlobular )?septal thickening\b",
    "Post-surgical or post-treatment change": r"\b(?:post[- ]?operative|post[- ]?surgical|post[- ]?treatment|status post|resection bed)\b",
    "Pulmonary metastases": r"\b(?:pulmonary|lung) metast\w*\b",
    "Tree-in-bud": r"\btree[- ]in[- ]bud\b",
    "Pulmonary cyst": r"\b(?:pulmonary|lung) cyst\w*\b",
    "Mass or neoplasm": r"\b(?:mass|neoplasm|tumou?r)\b",
    "Mucus plugging": r"\b(?:mucus|mucous) plug\w*\b",
    "Pleural thickening or nodule": r"\bpleural (?:thickening|nodul\w*)\b",
    "Bone lesion or fracture": r"\b(?:bone|osseous) lesion\b|\bfracture\b",
    "Pneumothorax": r"\bpneumothora\w*\b",
}


def main() -> int:
    if set(LEXICON) != set(PEDS23_PATHOLOGIES):
        raise RuntimeError("Mention lexicon does not exactly cover peds23")
    context = pd.read_csv(WORK / "CONTEXT/indication.csv", keep_default_na=False).set_index("VolumeName")
    labels = pd.read_csv(WORK / "BCH_DATASET/LABELS27/labels.csv").set_index("VolumeName")
    output_rows = []
    split_summary = {}
    for split in ("TRAIN", "DEV", "TEST"):
        names = [line.strip() for line in (WORK / f"PEDS23_U18_VOLLIST_{split}.txt").read_text().splitlines() if line.strip()]
        text = context.loc[names, "Indication_EN"].astype(str)
        present = context.loc[names, "ind_status"].eq("present").to_numpy()
        y = labels.loc[names, PEDS23_PATHOLOGIES].to_numpy(np.int32)
        any_positive_explicit = np.zeros(len(names), dtype=bool)
        for index, class_name in enumerate(PEDS23_PATHOLOGIES):
            mention = text.str.contains(LEXICON[class_name], case=False, regex=True).to_numpy() & present
            positive = y[:, index] == 1
            any_positive_explicit |= mention & positive
            output_rows.append({
                "split": split.lower(), "class": class_name, "studies": len(names),
                "positive_n": int(positive.sum()), "mention_n": int(mention.sum()),
                "positive_and_mentioned_n": int((positive & mention).sum()),
                "positive_mention_fraction": float((positive & mention).sum() / max(positive.sum(), 1)),
                "prevalence_if_mentioned": float(positive[mention].mean()) if mention.any() else float("nan"),
                "prevalence_if_not_mentioned": float(positive[~mention].mean()),
            })
        split_summary[split.lower()] = {
            "studies": len(names), "indication_present_n": int(present.sum()),
            "studies_with_at_least_one_positive_label_explicitly_mentioned": int(any_positive_explicit.sum()),
            "fraction_with_positive_label_explicitly_mentioned": float(any_positive_explicit.mean()),
        }
    output = WORK / "eval_matrix/peds23_u18_audits"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "indication_label_mentions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    package = {
        "method": "case-insensitive explicit-term regex audit; suspicion and confirmed diagnosis are not distinguished",
        "splits": split_summary,
        "lexicon": LEXICON,
    }
    (output / "indication_label_mentions.json").write_text(json.dumps(package, indent=2))
    print(json.dumps(split_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

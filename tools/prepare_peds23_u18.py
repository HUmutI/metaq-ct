#!/usr/bin/env python3
"""Create aggregate-audited, patient-disjoint BCH <18 paper splits.

The historical clean training pool is split into train/development by MRN.
The historical patient-disjoint validation pool is retained as the test pool.
Only aggregate, non-PHI audit information is written.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from arcct.schema import PEDS23_PATHOLOGIES


WORK = Path("/temp_work/ch278233")
DEMOGRAPHICS = WORK / "CONTEXT/demographics.csv"
LABELS = WORK / "BCH_DATASET/LABELS27/labels.csv"
VOLUME_MAP = WORK / "BCH_DATASET/LABELS/volume_map.tsv"
RAW_PHI = WORK / "BCH_DATASET/reports_8k.csv"
SOURCE_TRAIN = WORK / "PEDS_VOLLIST_TRAIN_clean.txt"
SOURCE_TEST = WORK / "PEDS_VOLLIST_VALID_clean.txt"
PREFIX = WORK / "PEDS23_U18"


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_list(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values))


def patient_map() -> dict[str, str]:
    with VOLUME_MAP.open(newline="", encoding="utf-8-sig") as handle:
        volume_to_accession = {
            row["volume_name"]: row["accession"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
        }
    with RAW_PHI.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        accession_to_mrn = {
            row["Accession Number"].strip(): row["Patient MRN"].strip()
            for row in csv.DictReader(handle)
        }
    mapping = {
        volume: accession_to_mrn.get(accession, "")
        for volume, accession in volume_to_accession.items()
    }
    if any(not value for value in mapping.values()):
        raise RuntimeError("At least one volume could not be mapped to an MRN")
    return mapping


def choose_group_split(names: list[str], groups: np.ndarray, labels: np.ndarray):
    """Choose the best of deterministic group splits by size/prevalence balance."""
    best = None
    overall = labels.mean(axis=0)
    scale = np.maximum(overall, 0.01)
    for seed in range(500):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_idx, dev_idx = next(splitter.split(names, groups=groups))
        dev_prev = labels[dev_idx].mean(axis=0)
        size_error = abs(len(dev_idx) / len(names) - 0.15)
        prevalence_error = float(np.mean(np.abs(dev_prev - overall) / scale))
        score = prevalence_error + 2.0 * size_error
        candidate = (score, seed, train_idx, dev_idx)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best


def aggregate(names: list[str], labels: pd.DataFrame, mrn: dict[str, str]) -> dict:
    frame = labels.loc[names, PEDS23_PATHOLOGIES]
    return {
        "studies": len(names),
        "patients": len({mrn[name] for name in names}),
        "positive_counts": {key: int(value) for key, value in frame.sum().items()},
        "prevalence": {key: float(value) for key, value in frame.mean().items()},
    }


def main() -> int:
    demographics = pd.read_csv(DEMOGRAPHICS).set_index("VolumeName")
    ages = pd.to_numeric(demographics["AgeYears"], errors="coerce")
    labels = pd.read_csv(LABELS).set_index("VolumeName")
    labels[PEDS23_PATHOLOGIES] = labels[PEDS23_PATHOLOGIES].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    mrn = patient_map()

    historical_train = [
        name for name in read_list(SOURCE_TRAIN)
        if name in ages.index and name in labels.index and 0 <= ages[name] < 18
    ]
    test = [
        name for name in read_list(SOURCE_TEST)
        if name in ages.index and name in labels.index and 0 <= ages[name] < 18
    ]
    groups = np.asarray([mrn[name] for name in historical_train])
    y = labels.loc[historical_train, PEDS23_PATHOLOGIES].to_numpy(np.float32)
    score, seed, train_idx, dev_idx = choose_group_split(historical_train, groups, y)
    train = [historical_train[index] for index in train_idx]
    dev = [historical_train[index] for index in dev_idx]

    train_patients = {mrn[name] for name in train}
    dev_patients = {mrn[name] for name in dev}
    test_patients = {mrn[name] for name in test}
    assert not train_patients & dev_patients
    assert not train_patients & test_patients
    assert not dev_patients & test_patients
    assert set(train).isdisjoint(dev)
    assert set(train).isdisjoint(test)
    assert set(dev).isdisjoint(test)

    for split, names in (("TRAIN", train), ("DEV", dev), ("TEST", test)):
        write_list(Path(f"{PREFIX}_VOLLIST_{split}.txt"), names)
        output_labels = labels.loc[names, PEDS23_PATHOLOGIES].reset_index()
        output_labels.to_csv(Path(f"{PREFIX}_{split}_labels.csv"), index=False)

    audit = {
        "cohort": "BCH studies with 0 <= age < 18 years",
        "classes": list(PEDS23_PATHOLOGIES),
        "split_unit": "MRN/patient",
        "historical_train_pool_studies": len(historical_train),
        "historical_test_pool_studies": len(test),
        "group_split_candidate_seed": int(seed),
        "group_split_balance_objective": float(score),
        "patient_overlap": {"train_dev": 0, "train_test": 0, "dev_test": 0},
        "splits": {
            "train": aggregate(train, labels, mrn),
            "development": aggregate(dev, labels, mrn),
            "test": aggregate(test, labels, mrn),
        },
        "caveat": (
            "The source validation pool was used in historical exploratory work; "
            "the exact under-18 recipe and split are frozen prospectively from this audit."
        ),
    }
    Path(f"{PREFIX}_AUDIT.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({
        "train_studies": len(train), "dev_studies": len(dev), "test_studies": len(test),
        "train_patients": len(train_patients), "dev_patients": len(dev_patients),
        "test_patients": len(test_patients), "patient_overlap": 0,
        "split_seed": int(seed),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

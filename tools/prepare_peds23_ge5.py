#!/usr/bin/env python3
"""Prepare the genuine 23-class, age >=5 pediatric experiment artifacts."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

from arcct.schema import PEDS_PATHOLOGIES, PEDS23_PATHOLOGIES


WORK = Path(os.environ.get("RAC_WORK", "/temp_work/ch278233"))
DEM = WORK / "CONTEXT/demographics.csv"
SOURCE_CKPT = WORK / "arcct_checkpoints/arcct_seed0.peds27.pt"
TARGET_CKPT = WORK / "arcct_checkpoints/arcct_seed0.peds23.pt"


def filtered_list(src: Path, ages: pd.Series) -> list[str]:
    names = [x.strip() for x in src.read_text().splitlines() if x.strip()]
    return [n for n in names if n in ages.index and float(ages[n]) >= 5.0]


def main() -> int:
    dem = pd.read_csv(DEM).set_index("VolumeName")
    ages = pd.to_numeric(dem["AgeYears"], errors="coerce")
    out_lists = {}
    for split in ("TRAIN", "VALID"):
        src = WORK / f"PEDS_VOLLIST_{split}_clean.txt"
        keep = filtered_list(src, ages)
        dst = WORK / f"PEDS23_GE5_VOLLIST_{split}.txt"
        dst.write_text("".join(f"{n}\n" for n in keep))
        out_lists[split] = keep
        print(f"[peds23-ge5] {split}: {len(keep):,}/{sum(1 for x in src.read_text().splitlines() if x.strip()):,}")

    v2 = pd.read_csv(WORK / "PEDS_VALID_labels_v2.csv")
    valid_set = set(out_lists["VALID"])
    v2 = v2[v2["VolumeName"].isin(valid_set)]
    v2.to_csv(WORK / "PEDS23_GE5_VALID_labels_v2.csv", index=False)
    print(f"[peds23-ge5] evaluator labels: {len(v2):,}")

    pkg = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    qsd = dict(pkg["qformer_module"])
    q = qsd["qformer.queries"]
    if q.shape[0] != 39:
        raise RuntimeError(f"expected 39 pediatric query rows, found {tuple(q.shape)}")
    keep_path = [PEDS_PATHOLOGIES.index(c) for c in PEDS23_PATHOLOGIES]
    keep_rows = list(range(10)) + [10 + i for i in keep_path] + [37, 38]
    if len(keep_rows) != 35:
        raise AssertionError(keep_rows)
    qsd["qformer.queries"] = q[keep_rows].clone()
    pkg["qformer_module"] = qsd
    pkg["rac_schema"] = "peds23"
    cfg = dict(pkg.get("config") or {})
    cfg["schema_remap"] = {
        "source": "peds27", "target": "peds23",
        "excluded": sorted(set(PEDS_PATHOLOGIES) - set(PEDS23_PATHOLOGIES)),
        "query_rows": keep_rows,
    }
    pkg["config"] = cfg
    torch.save(pkg, TARGET_CKPT)
    print(f"[peds23-ge5] checkpoint: {TARGET_CKPT} queries={tuple(qsd['qformer.queries'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

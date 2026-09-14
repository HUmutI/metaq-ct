#!/usr/bin/env python3
"""Filter the frozen U18 split without changing any patient assignment."""

from pathlib import Path

import pandas as pd


WORK = Path("/temp_work/ch278233")


def main() -> int:
    demographics = pd.read_csv(WORK / "CONTEXT/demographics.csv").set_index("VolumeName")
    ages = pd.to_numeric(demographics["AgeYears"], errors="coerce")
    audit = []
    for split in ("TRAIN", "DEV", "TEST"):
        source = WORK / f"PEDS23_U18_VOLLIST_{split}.txt"
        names = [line.strip() for line in source.read_text().splitlines() if line.strip()]
        selected = [name for name in names if name in ages.index and 5 <= ages[name] < 18]
        (WORK / f"PEDS23_U18_GE5_VOLLIST_{split}.txt").write_text(
            "".join(f"{name}\n" for name in selected)
        )
        labels = pd.read_csv(WORK / f"PEDS23_U18_{split}_labels.csv").set_index("VolumeName")
        labels.loc[selected].reset_index().to_csv(
            WORK / f"PEDS23_U18_GE5_{split}_labels.csv", index=False
        )
        audit.append((split, len(names), len(selected)))
    for split, before, after in audit:
        print(f"{split}: {after}/{before} studies with 5 <= age < 18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

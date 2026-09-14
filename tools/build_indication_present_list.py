#!/usr/bin/env python3
"""Intersect a volume list with audited rows having a usable indication."""
from __future__ import annotations

import argparse
import csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.context, newline="", encoding="utf-8") as fh:
        present = {
            row["VolumeName"] for row in csv.DictReader(fh)
            if row.get("ind_status") == "present" and
            str(row.get("Indication_EN", "")).strip()
        }
    with open(args.input, encoding="utf-8") as fh:
        source = [line.strip() for line in fh if line.strip()]
    kept = [volume for volume in source if volume in present]
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("".join(f"{volume}\n" for volume in kept))
    print(f"{len(kept)}/{len(source)} indication-present volumes -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

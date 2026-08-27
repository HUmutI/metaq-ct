#!/usr/bin/env python3
"""Build one age/sex table keyed by de-identified VolumeName, for both cohorts.

Neither cohort's demographics reach the model today. The pediatric age and sex
live in the PHI spreadsheet and were only ever read by a one-off analysis script;
CT-RATE's live in the DICOM metadata CSVs and are read by nothing. This produces
the single table the dataset consumes.

Reads
  peds    /temp_work/ch278233/BCH_DATASET/reports_8k.csv   cols D / L / M
          (accession, Patient Sex, Patient Age) -- PHI, never leaves the cluster
          /temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv  accession -> volume
  ctrate  /temp_work/ch278233/CTRATE/ct_meta_hf/dataset/metadata/{train,validation}_metadata.csv

Writes
  /temp_work/ch278233/CONTEXT/demographics.csv
      VolumeName, AgeYears, AgeBand, Sex, SexIdx, cohort, age_clamped

The model consumes AgeBand ONLY. AgeYears is kept because the design requires
every pediatric number to be reported three ways -- pooled, under 18, over 18 --
and because the "scalar age" ablation arm needs it. It is a quasi-identifier, so
it stays cluster-side; --no-age-years drops it from any exported copy.

Nothing patient-identifying is written: the key is the de-identified VolumeName,
and the only other fields are an age, a band and a sex.

Run: python pipeline/02_extract/16_build_demographics.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from demographics import (BAND_NAMES, BAND_UNKNOWN, N_BANDS,      # noqa: E402
                          age_band, clamp_age, parse_age, parse_sex)

PEDS_REPORTS = "/temp_work/ch278233/BCH_DATASET/reports_8k.csv"
PEDS_MAP = "/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv"
CTRATE_META = [
    ("train", "/temp_work/ch278233/CTRATE/ct_meta_hf/dataset/metadata/train_metadata.csv"),
    ("valid", "/temp_work/ch278233/CTRATE/ct_meta_hf/dataset/metadata/validation_metadata.csv"),
]
OUT_DIR = "/temp_work/ch278233/CONTEXT"
OUT = os.path.join(OUT_DIR, "demographics.csv")

# The pediatric band histogram published in the design doc. Reproducing it is a
# real test of the parser: if a pattern regresses, the infant bands collapse
# first and the count moves by hundreds.
EXPECTED_PEDS_BANDS = (301, 222, 807, 1291, 1097, 2701, 2045, 364, 39, 3)


def load_peds() -> list[dict]:
    csv.field_size_limit(10 ** 9)
    acc2vol: dict[str, str] = {}
    with open(PEDS_MAP, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            acc2vol[str(row["accession"]).strip()] = str(row["volume_name"]).strip()

    rows, unmapped, unparsed = [], 0, 0
    raw_bands = Counter()
    with open(PEDS_REPORTS, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            acc = str(row.get("Accession Number", "")).strip()
            age = parse_age(row.get("Patient Age"))
            raw_bands[age_band(age)] += 1          # over ALL rows, to match the doc
            vol = acc2vol.get(acc)
            if vol is None:
                unmapped += 1
                continue
            if age is None:
                unparsed += 1
            age_c, clamped = clamp_age(age)
            rows.append({
                "VolumeName": vol,
                "AgeYears": "" if age_c is None else f"{age_c:.6f}",
                "AgeBand": age_band(age_c),
                "Sex": ("F", "M", "U")[parse_sex(row.get("Patient Sex"))],
                "SexIdx": parse_sex(row.get("Patient Sex")),
                "cohort": "peds",
                "age_clamped": int(clamped),
            })
    print(f"[peds] {len(rows):,} volumes  unmapped_accessions={unmapped:,}  "
          f"unparsed_age={unparsed:,}")
    return rows, raw_bands


def load_ctrate() -> list[dict]:
    rows, unparsed = [], 0
    for split, path in CTRATE_META:
        if not os.path.isfile(path):
            print(f"[ctrate] MISSING {path} -- skipping {split}")
            continue
        n = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                vol = str(row.get("VolumeName", "")).strip()
                if not vol:
                    continue
                age = parse_age(row.get("PatientAge"))
                if age is None:
                    unparsed += 1
                age_c, clamped = clamp_age(age)
                rows.append({
                    "VolumeName": vol,
                    "AgeYears": "" if age_c is None else f"{age_c:.6f}",
                    "AgeBand": age_band(age_c),
                    "Sex": ("F", "M", "U")[parse_sex(row.get("PatientSex"))],
                    "SexIdx": parse_sex(row.get("PatientSex")),
                    "cohort": "ctrate",
                    "age_clamped": int(clamped),
                })
                n += 1
        print(f"[ctrate/{split}] {n:,} volumes")
    print(f"[ctrate] unparsed_age={unparsed:,}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--no-age-years", action="store_true",
                    help="drop the AgeYears column (for an exported copy)")
    a = ap.parse_args()

    peds, peds_raw_bands = load_peds()
    ctrate = load_ctrate()
    rows = peds + ctrate

    problems = []
    seen = Counter(r["VolumeName"] for r in rows)
    dupes = [v for v, c in seen.items() if c > 1]
    if dupes:
        problems.append(f"{len(dupes)} duplicate VolumeName keys, e.g. {dupes[:3]}")
    bad_band = [r for r in rows if not 0 <= int(r["AgeBand"]) <= BAND_UNKNOWN]
    if bad_band:
        problems.append(f"{len(bad_band)} rows with an out-of-range AgeBand")
    bad_sex = [r for r in rows if not 0 <= int(r["SexIdx"]) <= 2]
    if bad_sex:
        problems.append(f"{len(bad_sex)} rows with an out-of-range SexIdx")

    # The published pediatric histogram, over all spreadsheet rows.
    got = tuple(peds_raw_bands.get(i, 0) for i in range(N_BANDS))
    print("\n[peds] band histogram (all spreadsheet rows)")
    for i in range(N_BANDS):
        flag = "" if got[i] == EXPECTED_PEDS_BANDS[i] else \
            f"   <-- doc says {EXPECTED_PEDS_BANDS[i]}"
        print(f"   {i} {BAND_NAMES[i]:>8s} : {got[i]:6,d}{flag}")
    unknown = peds_raw_bands.get(BAND_UNKNOWN, 0)
    print(f"   {BAND_UNKNOWN} {'unknown':>8s} : {unknown:6,d}")
    if got != EXPECTED_PEDS_BANDS:
        problems.append(f"pediatric band histogram {got} != published "
                        f"{EXPECTED_PEDS_BANDS}; the parser has regressed or the "
                        "cohort changed -- do not write a table that silently "
                        "disagrees with the design doc")

    # Reported per split: the CT-RATE table in the design doc is TRAIN ONLY
    # (2,849 / 13,709 / 16,585 / 13,987 / 7 = 47,137), and a combined histogram
    # silently disagrees with it by the size of the validation split.
    ct_bands = Counter(int(r["AgeBand"]) for r in ctrate)
    valid_vols = set()
    vpath = dict(CTRATE_META).get("valid")
    if vpath and os.path.isfile(vpath):
        with open(vpath, newline="", encoding="utf-8", errors="replace") as fh:
            valid_vols = {str(r.get("VolumeName", "")).strip() for r in csv.DictReader(fh)}
    ct_train = Counter(int(r["AgeBand"]) for r in ctrate if r["VolumeName"] not in valid_vols)
    ct_valid = Counter(int(r["AgeBand"]) for r in ctrate if r["VolumeName"] in valid_vols)
    print("\n[ctrate] band histogram        train    valid    total")
    for i in range(N_BANDS + 1):
        if ct_bands.get(i):
            print(f"   {i} {BAND_NAMES[i]:>8s} : {ct_train.get(i,0):8,d} "
                  f"{ct_valid.get(i,0):8,d} {ct_bands[i]:8,d}")
    print(f"   {'':>10s}   {sum(ct_train.values()):8,d} {sum(ct_valid.values()):8,d} "
          f"{sum(ct_bands.values()):8,d}")
    doc_train = {1: 7, 6: 2849, 7: 13709, 8: 16585, 9: 13987}
    drift = {i: (ct_train.get(i, 0), v) for i, v in doc_train.items()
             if ct_train.get(i, 0) != v}
    if drift:
        print(f"   NOTE: train histogram differs from the design doc at {drift} "
              "(got, doc) -- not fatal, but the doc table should be refreshed")
    sexes = Counter(r["Sex"] for r in rows)
    print(f"\n[both] sex: {dict(sexes)}   rows: {len(rows):,}")
    empty_train_bands = [i for i in range(N_BANDS)
                         if got[i] == 0 and ct_bands.get(i, 0) == 0]
    if empty_train_bands:
        # An untrained increment propagates through the ordinal cumulative sum
        # into every band above it, so an empty band is not a local problem.
        problems.append(f"bands {empty_train_bands} are empty in BOTH cohorts")

    if problems:
        print("\nFATAL:")
        for p in problems:
            print("   ", p)
        return 1

    if a.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["VolumeName", "AgeYears", "AgeBand", "Sex", "SexIdx", "cohort", "age_clamped"]
    if a.no_age_years:
        cols.remove("AgeYears")
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["VolumeName"]):
            w.writerow(r)
    os.replace(tmp, OUT)
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    with open(OUT + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    print(f"\nwrote {OUT}  ({len(rows):,} rows)\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

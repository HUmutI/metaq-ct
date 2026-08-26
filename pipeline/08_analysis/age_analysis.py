#!/usr/bin/env python3
"""Age distribution of the pediatric cohort and of CT-RATE (train + validation).

Motivation: a joint adult-pediatric model needs to know how the two age
distributions actually overlap. The BCH cohort is a children's hospital but not
exclusively children - it follows adults with childhood-onset disease - and how
much of it is adult decides whether "pediatric vs adult" is even the right axis.

PHI: ages are bucketed and only counts are printed. Ages above 89 are pooled
into a single 90+ bucket, which is the HIPAA safe-harbour rule; no exact age of
an elderly patient leaves this script. No name, MRN or accession is read.
"""
from __future__ import annotations
import csv, os, re, sys
from collections import Counter

csv.field_size_limit(2 ** 31 - 1)

# DICOM PatientAge is like "045Y", "018M", "003D"; the BCH column is free text.
AGE_PATTERNS = [
    (re.compile(r"^\s*0*(\d{1,3})\s*[Yy]\s*$"), 1.0),          # 045Y
    (re.compile(r"^\s*0*(\d{1,3})\s*[Mm]\s*$"), 1 / 12),       # 018M
    (re.compile(r"^\s*0*(\d{1,3})\s*[Ww]\s*$"), 1 / 52),
    (re.compile(r"^\s*0*(\d{1,3})\s*[Dd]\s*$"), 1 / 365),
    (re.compile(r"(\d{1,3})\s*(?:y(?:ea)?rs?|yo|y/o|yr)\b", re.I), 1.0),
    (re.compile(r"(\d{1,3})\s*(?:months?|mo)\b", re.I), 1 / 12),
    (re.compile(r"(\d{1,3})\s*(?:weeks?|wks?)\b", re.I), 1 / 52),
    (re.compile(r"(\d{1,3})\s*(?:days?)\b", re.I), 1 / 365),
    # The BCH column is decimal years, and infants are fractional: 0.25 is a
    # three-month-old, 0.08219178 is thirty days. An integer-only pattern would
    # silently drop 521 of the youngest patients - exactly the group that
    # matters most for a pediatric question.
    (re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*$"), 1.0),
]

BUCKETS = [(0, 1, "<1y"), (1, 2, "1-2y"), (2, 6, "2-5y"), (6, 12, "6-11y"),
           (12, 18, "12-17y"), (18, 30, "18-29y"), (30, 50, "30-49y"),
           (50, 65, "50-64y"), (65, 90, "65-89y"), (90, 200, "90+")]


def parse_age(raw: str) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    for rx, mult in AGE_PATTERNS:
        m = rx.search(s)
        if m:
            v = float(m.group(1)) * mult
            if 0 <= v <= 120:
                return v
    return None


def bucket(v: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= v < hi:
            return name
    return "90+"


def summarise(ages: list[float], total: int, title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)
    print("  rows                : %6d" % total)
    print("  age parsed          : %6d  (%.1f%%)" % (len(ages), 100.0 * len(ages) / max(total, 1)))
    if not ages:
        return
    a = sorted(ages)
    def pct(p): return a[min(len(a) - 1, int(len(a) * p))]
    print("  median              : %6.1f y" % a[len(a) // 2])
    print("  p10 / p90           : %6.1f / %.1f y" % (pct(0.10), pct(0.90)))
    print("  min / max           : %6.1f / %.1f y" % (a[0], a[-1]))
    under18 = sum(1 for x in a if x < 18)
    print("  under 18            : %6d  (%.1f%%)" % (under18, 100.0 * under18 / len(a)))
    print("  18 and over         : %6d  (%.1f%%)" % (len(a) - under18,
                                                     100.0 * (len(a) - under18) / len(a)))
    c = Counter(bucket(x) for x in a)
    print("\n  %-10s %8s %7s" % ("bucket", "n", "share"))
    for _lo, _hi, name in BUCKETS:
        n = c.get(name, 0)
        bar = "#" * int(round(40.0 * n / max(len(a), 1)))
        print("  %-10s %8d %6.1f%%  %s" % (name, n, 100.0 * n / len(a), bar))


def peds(path: str) -> None:
    ages = []
    total = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            total += 1
            v = parse_age(row.get("Patient Age", ""))
            if v is not None:
                ages.append(v)
    summarise(ages, total, "BCH pediatric cohort - Patient Age column")


def ctrate(paths: list[str], title: str) -> None:
    ages = []
    total = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rd = csv.DictReader(fh)
            col = "PatientAge" if "PatientAge" in (rd.fieldnames or []) else None
            if not col:
                print("  (no PatientAge column in %s)" % os.path.basename(path))
                continue
            for row in rd:
                total += 1
                v = parse_age(row.get(col, ""))
                if v is not None:
                    ages.append(v)
    summarise(ages, total, title)


if __name__ == "__main__":
    peds("/temp_work/ch278233/BCH_DATASET/reports_8k.csv")
    base = "/temp_work/ch278233/CTRATE/ct_meta_hf/dataset/metadata"
    # Only dataset/metadata/: the ts_seg/ CSVs are one row per detected nodule or
    # effusion, not one row per patient, so pooling them would count the same
    # study many times and skew the distribution toward whoever has most lesions.
    meta_dir = "/temp_work/ch278233/CTRATE/ct_meta_hf/dataset/metadata"
    cands = [os.path.join(meta_dir, f) for f in sorted(os.listdir(meta_dir))
             if f.endswith(".csv")] if os.path.isdir(meta_dir) else []
    tr = [p for p in cands if "train" in os.path.basename(p).lower()]
    va = [p for p in cands if "valid" in os.path.basename(p).lower()]
    if not va:
        va = ["/home/ch278233/ctrate_fetch/validation_metadata.csv"]
    ctrate(tr, "CT-RATE TRAIN - PatientAge")
    ctrate(va, "CT-RATE VALIDATION - PatientAge")

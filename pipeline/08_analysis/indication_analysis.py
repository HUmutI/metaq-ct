#!/usr/bin/env python3
"""How often does a chest-CT report actually carry an indication?

Compares CT-RATE (adult, public) with the BCH pediatric cohort. Motivation: a
model that conditions on indication text can only learn from reports that have
one, so the first thing to know is the coverage, and how much of that coverage
is real rather than boilerplate.

CT-RATE keeps the indication in its own column (ClinicalInformation_EN).
Pediatric reports keep it inside the free-text report under a section header,
and not every report has one.

PHI DISCIPLINE
--------------
Only aggregates leave this script. The one exception is the vacuous-phrase
table, and it is gated three ways: a phrase is printed only if it occurs in at
least MIN_SHARED reports, is at most MAX_LEN characters, and contains no digit.
Boilerplate like "no prior knowledge" passes; anything patient-specific does not.
Report text, accessions, names and MRNs are never read into the output.
"""
from __future__ import annotations
import csv, re, sys, os
from collections import Counter

csv.field_size_limit(2 ** 31 - 1)

MIN_SHARED = 50      # a phrase must be shared by this many reports to be shown
MAX_LEN = 40         # ...and be this short
DIGIT = re.compile(r"\d")

# Section headers that introduce an indication in the pediatric reports.
IND_HEADERS = (r"INDICATION|INDICATIONS|CLINICAL INDICATION|CLINICAL INFORMATION|"
               r"CLINICAL HISTORY|HISTORY|REASON FOR (?:EXAM|EXAMINATION|STUDY)|"
               r"CLINICAL DATA|REASON FOR THE EXAM")
SECTION = re.compile(
    r"^[ \t]*\**[ \t]*(?P<h>[A-Z][A-Z /&()'\-]{2,44})[ \t]*\**[ \t]*:",
    re.M)

VACUOUS = re.compile(
    r"^(?:"
    r"n/?a|none|none\.|not provided|not given|not available|unknown|"
    r"no prior knowledge|no priors?|no history|no clinical history|"
    r"no clinical information|not specified|not stated|nil|-+|\.+|\?+"
    r")\.?$", re.I)


def classify(text: str) -> str:
    t = (text or "").strip().strip(".;:- \t\n")
    if not t:
        return "absent"
    if VACUOUS.match(t):
        return "vacuous"
    if len(t) < 4:
        return "vacuous"
    return "present"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip(".;:- ")


def report_bucket(counts: Counter, total: int, title: str, phrases: Counter) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)
    print("  reports total          : %6d" % total)
    for k, label in (("present", "indication present"),
                     ("vacuous", "present but vacuous"),
                     ("absent", "no indication at all")):
        c = counts[k]
        print("  %-22s : %6d  (%5.1f%%)" % (label, c, 100.0 * c / max(total, 1)))
    usable = counts["present"]
    print("  %-22s : %6d  (%5.1f%%)" % ("-> usable for training", usable,
                                        100.0 * usable / max(total, 1)))
    shown = [(p, c) for p, c in phrases.most_common(40)
             if c >= MIN_SHARED and len(p) <= MAX_LEN and not DIGIT.search(p)]
    if shown:
        print("\n  shared boilerplate (>=%d reports, <=%d chars, no digits):" % (MIN_SHARED, MAX_LEN))
        for p, c in shown[:10]:
            print("    %-42s %6d" % (p[:42], c))


def do_ctrate(paths: list[str]) -> None:
    counts = Counter(); phrases = Counter(); total = 0
    lens = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                col = row.get("ClinicalInformation_EN", "")
                total += 1
                k = classify(col)
                counts[k] += 1
                if k != "present":
                    phrases[norm(col)] += 1
                else:
                    lens.append(len(col.strip()))
    report_bucket(counts, total, "CT-RATE (adult) - ClinicalInformation_EN column", phrases)
    if lens:
        lens.sort()
        print("\n  length of a real indication (chars): median %d, p10 %d, p90 %d"
              % (lens[len(lens) // 2], lens[len(lens) // 10], lens[len(lens) * 9 // 10]))


def do_peds(path: str) -> None:
    counts = Counter(); phrases = Counter(); total = 0
    lens = []
    headers = Counter()
    ind_re = re.compile(r"^(?:%s)$" % IND_HEADERS)
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("Report Text") or "").replace("_x000D_", "\n")
            if not text.strip():
                continue
            total += 1
            marks = [(m.start(), m.end(), m.group("h").strip()) for m in SECTION.finditer(text)]
            for _s, _e, h in marks:
                headers[h] += 1
            body = ""
            for i, (s, e, h) in enumerate(marks):
                if ind_re.match(h):
                    end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
                    body = text[e:end]
                    break
            k = classify(body)
            counts[k] += 1
            if k != "present":
                phrases[norm(body)] += 1
            else:
                lens.append(len(body.strip()))
    report_bucket(counts, total, "BCH pediatric - INDICATION section inside Report Text", phrases)
    if lens:
        lens.sort()
        print("\n  length of a real indication (chars): median %d, p10 %d, p90 %d"
              % (lens[len(lens) // 2], lens[len(lens) // 10], lens[len(lens) * 9 // 10]))
    print("\n  section headers seen (top 14, header names only):")
    for h, c in headers.most_common(14):
        mark = " <- indication" if ind_re.match(h) else ""
        print("    %-34s %6d%s" % (h[:34], c, mark))


if __name__ == "__main__":
    base = "/temp_work/ch278233/CTRATE/ct_reports_hf/dataset/radiology_text_reports"
    do_ctrate(["%s/train_reports.csv" % base, "%s/validation_reports.csv" % base])
    do_peds("/temp_work/ch278233/BCH_DATASET/reports_8k.csv")

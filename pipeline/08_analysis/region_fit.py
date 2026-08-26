#!/usr/bin/env python3
"""Does the adult 10-region scheme actually fit pediatric chest CT reports?

The 10 groups (5 lung lobes, trachea, heart, aorta, mediastinal vessels,
esophagus) were chosen for CT-RATE, an adult cohort, and are driven by what
TotalSegmentator segments - not by what pediatric radiologists write about. This
measures the mismatch three ways:

  1. coverage      - how often each region gets any sentence at all
  2. unmapped      - sentences the model assigns to NO region
  3. missing anatomy - which thoracic structures those unmapped sentences talk
                       about, counted against a fixed vocabulary

Aggregates only: prints counts and percentages, never a sentence, never an
accession.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ch278233/pipeline/lib")   # shared modules: ts_roi, qwen_extract, prepare_reports
from qwen_extract import REGIONS

# Thoracic structures a chest CT report may discuss that have NO home among the
# ten anatomy queries. Presence here is the evidence for adding a region.
OUTSIDE = {
    "pleura":        ["pleura", "pleural"],
    "chest wall":    ["chest wall", "pectus", "costal", "intercostal"],
    "bone/spine":    ["rib", "ribs", "osseous", "bony", "vertebra", "vertebral",
                      "spine", "spinal", "scoliosis", "sternum", "sternal", "clavicle"],
    "thymus":        ["thymus", "thymic"],
    "diaphragm":     ["diaphragm", "hemidiaphragm", "diaphragmatic"],
    "airways(small)":["bronchiole", "bronchiolar", "small airway", "air trapping",
                      "bronchiectasis", "peribronchial"],
    "mediastinum":   ["mediastinum", "mediastinal"],
    "upper abdomen": ["liver", "hepatic", "spleen", "splenic", "abdomen",
                      "abdominal", "adrenal", "kidney"],
    "thyroid/neck":  ["thyroid", "neck", "supraclavicular"],
    "soft tissue":   ["soft tissue", "subcutaneous", "axilla", "axillary"],
    "devices":       ["catheter", "tube", "line", "stent", "port", "pacemaker",
                      "hardware", "clip", "drain", "shunt"],
    "breast":        ["breast"],
}


def read_jsonl_many(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok":
                    out[r["VolumeName"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/temp_work/ch278233/BCH_DATASET/LABELS")
    args = ap.parse_args()
    D = args.dir

    sents = {}
    with open(os.path.join(D, "sentences.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            sents[r["VolumeName"]] = r["sentences"]

    regions = read_jsonl_many(os.path.join(D, "regions*.jsonl"))
    keys = sorted(set(regions) & set(sents))
    if not keys:
        print("no region records yet"); return 1
    n = len(keys)
    print("=" * 74)
    print("DO THE 10 ADULT REGIONS FIT PEDIATRIC REPORTS?   (n=%d reports)" % n)
    print("=" * 74)

    cov = Counter(); sent_per_region = Counter()
    unmapped_total = 0; sent_total = 0
    unmapped_terms = Counter(); all_terms = Counter()
    reports_with_unmapped = 0

    for v in keys:
        ss = sents[v]; sent_total += len(ss)
        used = set()
        for k in REGIONS:
            idxs = [i for i in regions[v]["regions"].get(str(k), []) if 0 <= i < len(ss)]
            if idxs:
                cov[k] += 1
            sent_per_region[k] += len(idxs)
            used.update(idxs)
        unmapped = [i for i in range(len(ss)) if i not in used]
        unmapped_total += len(unmapped)
        if unmapped:
            reports_with_unmapped += 1
        for i, s in enumerate(ss):
            low = s.lower()
            for grp, kws in OUTSIDE.items():
                if any(w in low for w in kws):
                    all_terms[grp] += 1
                    if i in unmapped:
                        unmapped_terms[grp] += 1

    print("\n1. REGION COVERAGE  (share of reports where the region gets >=1 sentence)")
    print("   %-4s %-62s %8s %10s" % ("id", "region", "coverage", "sent/rep"))
    for k, name in REGIONS.items():
        print("   %-4d %-62s %7.1f%% %10.2f"
              % (k, name[:62], 100.0 * cov[k] / n, sent_per_region[k] / n))

    print("\n2. UNMAPPED SENTENCES  (assigned to none of the 10 regions)")
    print("   sentences total        : %d" % sent_total)
    print("   unmapped               : %d  (%.1f%%)" % (unmapped_total, 100.0 * unmapped_total / max(sent_total, 1)))
    print("   reports with >=1       : %d  (%.1f%%)" % (reports_with_unmapped, 100.0 * reports_with_unmapped / n))

    print("\n3. WHAT THE UNMAPPED SENTENCES TALK ABOUT")
    print("   (structures with no home among the 10 anatomy queries)")
    print("   %-16s %12s %12s %10s" % ("structure", "unmapped", "all sents", "share"))
    for grp, c in unmapped_terms.most_common():
        tot = all_terms[grp]
        print("   %-16s %12d %12d %9.1f%%" % (grp, c, tot, 100.0 * c / max(tot, 1)))

    print("\n4. READING")
    weak = [k for k in REGIONS if 100.0 * cov[k] / n < 20]
    if weak:
        print("   regions covered in <20%% of reports: %s" % ", ".join(str(k) for k in weak))
        print("   -> these anatomy queries would train on mostly-empty text")
    if unmapped_total / max(sent_total, 1) > 0.25:
        print("   >25%% of sentences map nowhere: the scheme misses a lot of what")
        print("   pediatric reports actually discuss - see section 3 for what.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())

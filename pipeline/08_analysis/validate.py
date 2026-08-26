#!/usr/bin/env python3
"""QC the local-LLM extraction. Aggregates only - never prints report text.

Four checks, each answering "would I notice if the model quietly degraded?":

1. **Coverage** - how many reports produced an ok record, how many had schema
   problems the validator had to repair.
2. **Prevalence vs a rule-based baseline** - every label is also scored by a
   keyword+negation matcher. Large disagreement in either direction is where to
   look; exact agreement is not expected (the whole point of the LLM is to beat
   regex on negation and context).
3. **Negation audit** - for each positive label, the evidence sentence is
   re-checked for a negation cue in front of the matched term. A positive whose
   own evidence sentence reads as negated is very likely a false positive.
4. **Region agreement** - regions are compared against arc-ct's own
   REGION_KEYWORDS fallback, which is what training would otherwise use.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ch278233/pipeline/lib")   # shared modules: ts_roi, qwen_extract, prepare_reports
sys.path.insert(0, os.path.expanduser("~/arc-ct"))
from qwen_extract import PATHOLOGIES, REGIONS  # noqa: E402

try:
    from arcct.dataset import REGION_KEYWORDS, LUNG_GENERAL_KEYWORDS
except Exception:                                    # arc-ct env not importable
    REGION_KEYWORDS, LUNG_GENERAL_KEYWORDS = {}, []

# Keyword baseline for the 18 labels. Deliberately simple: it exists to be
# disagreed with, not to be right.
LABEL_KEYWORDS = {
    "Medical material": ["catheter", "tube", "line", "stent", "clip", "hardware",
                         "pacemaker", "port", "drain"],
    "Arterial wall calcification": ["arterial calcification", "calcified aorta",
                                    "aortic calcification"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "Pericardial effusion": ["pericardial effusion", "pericardial fluid"],
    "Coronary artery wall calcification": ["coronary calcification",
                                           "coronary artery calcification"],
    "Hiatal hernia": ["hiatal hernia"],
    "Lymphadenopathy": ["lymphadenopathy", "enlarged lymph node", "adenopathy"],
    "Emphysema": ["emphysema", "bulla", "bullae", "pneumatocele",
                  "hyperinflation", "overinflation"],
    "Atelectasis": ["atelecta", "collapse", "volume loss"],
    "Lung nodule": ["nodule", "nodular", "granuloma"],
    "Lung opacity": ["opacity", "opacities", "ground-glass", "ground glass"],
    "Pulmonary fibrotic sequela": ["fibrosis", "fibrotic", "scarring", "scar",
                                   "reticulation", "architectural distortion"],
    "Pleural effusion": ["pleural effusion", "pleural fluid"],
    "Mosaic attenuation pattern": ["mosaic", "air trapping", "air-trapping"],
    "Peribronchial thickening": ["peribronchial thickening", "bronchial wall thickening",
                                 "peribronchial cuffing"],
    "Consolidation": ["consolidation", "consolidative"],
    "Bronchiectasis": ["bronchiecta"],
    "Interlobular septal thickening": ["septal thickening", "interlobular septal"],
}

NEG_CUES = ["no ", "no evidence", "without", "negative for", "absence of",
            "absent", "not seen", "not identified", "not present", "free of",
            "resolved", "unremarkable", "normal", "rule out", "r/o",
            "there is no", "denies"]
CLAUSE_BREAK = re.compile(r"[,;:]| but | however | although | except ")


def negated(sentence: str, term: str) -> bool:
    """Is `term` preceded by a negation cue within the same clause?"""
    s = sentence.lower()
    i = s.find(term.lower())
    if i < 0:
        return False
    window = s[max(0, i - 70):i]
    parts = CLAUSE_BREAK.split(window)
    window = parts[-1] if parts else window
    return any(c in window for c in NEG_CUES)


def baseline_label(sentences: list[str], label: str) -> bool:
    for kw in LABEL_KEYWORDS.get(label, []):
        for s in sentences:
            if kw in s.lower() and not negated(s, kw):
                return True
    return False


def baseline_region(sentences: list[str], k: int) -> list[int]:
    kws = REGION_KEYWORDS.get(k, [])
    out = []
    for i, s in enumerate(sentences):
        low = s.lower()
        if any(w in low for w in kws):
            out.append(i)
        elif 1 <= k <= 5 and any(w in low for w in LUNG_GENERAL_KEYWORDS):
            out.append(i)
    return out


def read_jsonl_many(pattern):
    out, bad = {}, Counter()
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad["unreadable_line"] += 1
                    continue
                if r.get("status") == "ok":
                    out[r["VolumeName"]] = r
                else:
                    bad[r.get("status", "unknown")] += 1
    return out, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/temp_work/ch278233/BCH_DATASET/LABELS")
    ap.add_argument("--sample", type=int, default=0,
                    help="limit the baseline comparison to N reports (0 = all)")
    args = ap.parse_args()
    D = args.dir

    sents = {}
    with open(os.path.join(D, "sentences.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            sents[r["VolumeName"]] = r["sentences"]
    total = len(sents)

    regions, rbad = read_jsonl_many(os.path.join(D, "regions*.jsonl"))
    labels, lbad = read_jsonl_many(os.path.join(D, "labels*.jsonl"))

    print("=" * 68)
    print("1. COVERAGE     (%d reports in sentences.jsonl)" % total)
    print("=" * 68)
    for name, ok, bad in (("regions", regions, rbad), ("labels", labels, lbad)):
        prob = sum(1 for r in ok.values() if r.get("problems"))
        print("  %-8s ok=%-6d (%.1f%%)  missing=%-6d  repaired_schema=%d"
              % (name, len(ok), 100.0 * len(ok) / max(total, 1),
                 total - len(ok), prob))
        for k, v in bad.items():
            print("      failed: %-14s %d" % (k, v))
        pc = Counter(p.rsplit("_", 1)[-1] for r in ok.values()
                     for p in r.get("problems", []))
        for k, v in pc.most_common(5):
            print("      problem: %-13s %d" % (k, v))

    keys = sorted(set(labels) & set(sents))
    if args.sample:
        keys = keys[:args.sample]

    if keys:
        print()
        print("=" * 68)
        print("2. LABEL PREVALENCE vs KEYWORD BASELINE   (n=%d)" % len(keys))
        print("=" * 68)
        print("  %-38s %7s %7s %7s %7s" % ("label", "llm%", "base%", "both", "llm_only"))
        for p in PATHOLOGIES:
            a = b = both = llm_only = base_only = 0
            for v in keys:
                L = labels[v]["labels"].get(p, {}).get("p", 0) == 1
                B = baseline_label(sents[v], p)
                a += L
                b += B
                both += L and B
                llm_only += L and not B
                base_only += B and not L
            n = len(keys)
            print("  %-38s %6.1f%% %6.1f%% %7d %7d"
                  % (p, 100.0 * a / n, 100.0 * b / n, both, llm_only))

        print()
        print("=" * 68)
        print("3. NEGATION AUDIT — positives whose own evidence reads negated")
        print("=" * 68)
        flagged = Counter()
        no_ev = Counter()
        for v in keys:
            ss = sents[v]
            for p in PATHOLOGIES:
                rec = labels[v]["labels"].get(p, {})
                if rec.get("p") != 1:
                    continue
                e = rec.get("e", -1)
                if not (0 <= e < len(ss)):
                    no_ev[p] += 1
                    continue
                if any(negated(ss[e], kw) for kw in LABEL_KEYWORDS.get(p, [])):
                    flagged[p] += 1
        tot_flag = sum(flagged.values())
        tot_noev = sum(no_ev.values())
        print("  positives with negated evidence : %d" % tot_flag)
        print("  positives with no valid evidence: %d" % tot_noev)
        for p, c in flagged.most_common(8):
            print("      %-38s %d" % (p, c))

    rkeys = sorted(set(regions) & set(sents))
    if args.sample:
        rkeys = rkeys[:args.sample]
    if rkeys and REGION_KEYWORDS:
        print()
        print("=" * 68)
        print("4. REGION AGREEMENT vs arc-ct's keyword fallback  (n=%d)" % len(rkeys))
        print("=" * 68)
        print("  %2s %-40s %8s %8s %8s" % ("", "region", "llm_cov", "base_cov", "jaccard"))
        for k, name in REGIONS.items():
            lc = bc = 0
            jac = []
            for v in rkeys:
                ss = sents[v]
                L = set(regions[v]["regions"].get(str(k), []))
                B = set(baseline_region(ss, k))
                lc += bool(L)
                bc += bool(B)
                if L or B:
                    jac.append(len(L & B) / len(L | B))
            n = len(rkeys)
            print("  %2d %-40s %7.1f%% %7.1f%% %8.2f"
                  % (k, name[:40], 100.0 * lc / n, 100.0 * bc / n,
                     sum(jac) / max(len(jac), 1)))
    elif rkeys:
        print("\n(region agreement skipped: arcct.dataset not importable here)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Turn the LLM's sentence indices into the two artifacts ARC-CT consumes.

  region_cache.json   {VolumeName: {"1": [sentence, ...], ..., "10": [...]}}
                      -> point RAC_REGION_CACHE at this.
                      dataset.py:_coerce_region_text joins a list with spaces,
                      so lists are consumed natively; an empty list falls back
                      to the rule-based localize_findings, which is the right
                      behaviour for a region the report never mentions.

  labels.csv          VolumeName + the PATHOLOGIES columns, 0/1
                      -> point RAC_LABELS_TRAIN / RAC_LABELS_VALID at this.
                      Written with explicit 0s and never blanks: dataset.py:299
                      does ``float(row.get(c, 0.0) or 0.0)``, and because
                      ``bool(nan) is True`` a blank cell survives as NaN and
                      poisons the loss.

  evidence.tsv        VolumeName, label, the sentence that justified a 1
                      -> for spot-checking the labeller. Stays on the cluster.

Reports coverage aggregates only; never prints report text.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ch278233/pipeline/lib")   # shared modules: ts_roi, qwen_extract, prepare_reports
from qwen_extract import PATHOLOGIES, REGIONS  # noqa: E402


def read_jsonl_many(pattern: str) -> dict:
    """Last ok record per VolumeName wins (resume appends, so later = newer)."""
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
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
    labels = read_jsonl_many(os.path.join(D, "labels*.jsonl"))
    print("sentences=%d  regions=%d  labels=%d" % (len(sents), len(regions), len(labels)))

    # ---- region cache ----------------------------------------------------- #
    cache, empty_regions, per_region_cov = {}, Counter(), Counter()
    for vol, rec in regions.items():
        ss = sents.get(vol, [])
        obj = {}
        for k in REGIONS:
            idxs = rec.get("regions", {}).get(str(k), [])
            picked = [ss[i] for i in idxs if 0 <= i < len(ss)]
            obj[str(k)] = picked
            if picked:
                per_region_cov[k] += 1
            else:
                empty_regions[k] += 1
        cache[vol] = obj
    cpath = os.path.join(D, "region_cache.json")
    with open(cpath, "w") as fh:
        json.dump(cache, fh, ensure_ascii=False)

    # ---- labels csv ------------------------------------------------------- #
    lpath = os.path.join(D, "labels.csv")
    epath = os.path.join(D, "evidence.tsv")
    pos = Counter()
    n_no_ev = 0
    with open(lpath, "w", newline="") as lf, open(epath, "w", newline="") as ef:
        w = csv.writer(lf)
        w.writerow(["VolumeName"] + PATHOLOGIES)
        ew = csv.writer(ef, delimiter="\t")
        ew.writerow(["VolumeName", "label", "evidence_sentence"])
        for vol in sorted(labels):
            rec = labels[vol]["labels"]
            ss = sents.get(vol, [])
            row = [vol]
            for p in PATHOLOGIES:
                v = rec.get(p, {"p": 0, "e": -1})
                row.append(int(v.get("p", 0)))
                if v.get("p") == 1:
                    pos[p] += 1
                    e = v.get("e", -1)
                    if 0 <= e < len(ss):
                        ew.writerow([vol, p, ss[e]])
                    else:
                        n_no_ev += 1
                        ew.writerow([vol, p, "<no evidence sentence returned>"])
            w.writerow(row)

    n = max(len(labels), 1)
    print("\n--- label prevalence (n=%d) ---" % len(labels))
    for p in PATHOLOGIES:
        print("  %-38s %5d  %5.1f%%" % (p, pos[p], 100.0 * pos[p] / n))
    print("  positives with no evidence sentence: %d" % n_no_ev)

    nr = max(len(regions), 1)
    print("\n--- region coverage (n=%d) ---" % len(regions))
    for k, v in REGIONS.items():
        print("  %2d %-62s %5.1f%%" % (k, v[:62], 100.0 * per_region_cov[k] / nr))

    print("\nwrote:\n  %s\n  %s\n  %s" % (cpath, lpath, epath))
    return 0


if __name__ == "__main__":
    sys.exit(main())

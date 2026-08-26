#!/usr/bin/env python3
"""CT-RATE reports -> sentences.jsonl, byte-identical in shape to the pediatric one.

Why this exists: the combined dataset labels all 16,400 adults 0 for the nine
pediatric-only classes. Measured, that is 0 positives out of 16,400 - a
fabricated negative, not an observed one. CT-RATE has no labels for those
findings; it does not follow that adults never have a pneumothorax, a rib
fracture or a post-surgical change. Supervising on it taught the combined model
that an adult-looking scan cannot have them, and 'Bone lesion or fracture'
landed at AUC 0.373 - below chance.

So label them from the reports instead of inventing zeros.

Two things this file is careful about:

  * It imports sentences_of/clean_text from prepare_reports rather than
    re-implementing them. Different sentence boundaries would make the adult and
    pediatric label sets incomparable, which is precisely the failure we are
    here to fix - 'Arterial wall calcification' already drifted from aortic
    atherosclerosis (adult, 28.4%) to calcified catheter tracts in veins
    (pediatric, 3.3%) and cost that class 0.30 AUC.

  * It deduplicates by report TEXT. 47,149 volumes carry only 23,013 distinct
    reports, because the same study appears under several reconstructions. The
    LLM sees each report once; the fan-out map puts the labels back on every
    volume that shares it.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ch278233/pipeline/lib")   # shared modules: ts_roi, qwen_extract, prepare_reports
from prepare_reports import sentences_of, clean_text

csv.field_size_limit(10 ** 7)

SRC = ("/temp_work/ch278233/CTRATE/ct_reports_hf/dataset/"
       "radiology_text_reports/train_reports.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=SRC)
    ap.add_argument("--out-dir", default="/temp_work/ch278233/CTRATE/LABELS27")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(a.reports, newline="", encoding="utf-8",
                                    errors="replace")))
    if a.limit:
        rows = rows[:a.limit]

    # text -> representative VolumeName, and the full fan-out list
    rep_of: dict[str, str] = {}
    fan: dict[str, list[str]] = {}
    empty = 0
    for r in rows:
        find = clean_text(r.get("Findings_EN") or "")
        imp = clean_text(r.get("Impressions_EN") or "")
        if not find and not imp:
            empty += 1
            continue
        key = find + "\x00" + imp
        vol = r["VolumeName"]
        if key not in rep_of:
            rep_of[key] = vol
            fan[vol] = []
        fan[rep_of[key]].append(vol)

    sent_path = os.path.join(a.out_dir, "sentences.jsonl")
    map_path = os.path.join(a.out_dir, "report_fanout.json")
    nsent: Counter = Counter()
    with open(sent_path, "w") as sf:
        for key, vol in sorted(rep_of.items(), key=lambda kv: kv[1]):
            find, imp = key.split("\x00", 1)
            fs = sentences_of(find)
            sents = fs + sentences_of(imp)
            if not sents:
                continue
            nsent[min(len(sents), 60)] += 1
            sf.write(json.dumps({
                "VolumeName": vol,
                "n_findings_sentences": len(fs),
                "sentences": sents,
            }) + "\n")
    json.dump(fan, open(map_path, "w"))

    vals = sorted(nsent.elements())
    print("  hacim satiri        : %d" % len(rows))
    print("  bos rapor           : %d" % empty)
    print("  benzersiz rapor     : %d   <- LLM bu kadarini isleyecek" % len(rep_of))
    print("  fan-out toplami     : %d" % sum(len(v) for v in fan.values()))
    print("  cumle/rapor         : medyan %d  p90 %d  max(60'ta kirpik) %d"
          % (vals[len(vals) // 2], vals[int(len(vals) * 0.9)], max(vals)))
    print("\n  yazildi:\n    %s\n    %s" % (sent_path, map_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

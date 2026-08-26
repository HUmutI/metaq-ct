#!/usr/bin/env python3
"""Does Qwen actually beat the keyword fallback at splitting a report into the
10 anatomical regions? Measure it on CT-RATE validation, no training involved.

Why this is worth running even though ablation T11 said "no gain": T11 measured
the END of the chain - retrain stage 2 with a Qwen cache, look at macro AUC.
That answers "does it move the metric", not "is the extraction better". A better
extraction can still fail to move AUC because the per-organ alignment loss is
one term among four. This script measures the extraction itself.

The keyword baseline is arcct's own localize_findings(), i.e. exactly what the
augmd checkpoint trained on.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, statistics
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/arc-ct"))
from arcct.dataset import localize_findings, REGION_KEYWORDS, FINE_LABEL_NAMES  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ch278233/pipeline/lib")   # shared modules: ts_roi, qwen_extract, prepare_reports
from qwen_extract import REGIONS, region_schema, region_prompt, make_sampling, chat_texts, SYSTEM  # noqa

import re
SENT = re.compile(r"(?<=[.!?])\s+")


def sentences_of(text):
    return [s.strip() for s in SENT.split((text or "").strip()) if len(s.strip()) > 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="/temp_work/ch278233/CTRATE/ct_reports/valid_reports.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="/temp_work/ch278233/CTRATE/region_compare.json")
    a = ap.parse_args()

    rows = []
    with open(a.reports, newline="") as fh:
        for r in csv.DictReader(fh):
            f = (r.get("Findings_EN") or "").strip()
            i = (r.get("Impressions_EN") or "").strip()
            txt = (f + " " + i).strip()
            if txt:
                rows.append((r["VolumeName"], txt))
            if len(rows) >= a.n:
                break
    print("[cmp] %d reports" % len(rows), flush=True)

    per = [(v, t, sentences_of(t)) for v, t in rows]
    from vllm import LLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.90, trust_remote_code=True)
    sp = make_sampling(region_schema(), 1024)
    texts = chat_texts(tok, [region_prompt(s) for _, _, s in per])
    outs = llm.generate(texts, sp)

    recs, bad = [], 0
    for (vol, txt, sents), o in zip(per, outs):
        try:
            obj = json.loads(o.outputs[0].text)
        except json.JSONDecodeError:
            bad += 1
            continue
        q, k, both = {}, {}, {}
        for r in REGIONS:
            idxs = [i for i in (obj.get(str(r)) or []) if isinstance(i, int) and 0 <= i < len(sents)]
            qtext = " ".join(sents[i] for i in idxs)
            ktext = localize_findings(txt, r)
            q[str(r)], k[str(r)] = qtext, ktext
            # does the keyword baseline just return the WHOLE report? that is its
            # documented fallback when nothing matches, and it is the failure we
            # care about: no localisation at all.
            both[str(r)] = {
                "qwen_chars": len(qtext), "kw_chars": len(ktext),
                "qwen_empty": not qtext,
                "kw_is_whole_report": ktext.strip()[:380] == txt.strip()[:380],
                "kw_truncated_400": len(ktext) >= 400,
            }
        recs.append({"VolumeName": vol, "n_sent": len(sents), "stats": both})

    print("\n[cmp] parse failures: %d" % bad)
    print("\n%-4s %-46s %10s %10s %10s %10s" %
          ("id", "region", "qwen_empty", "kw=whole", "qwen_chr", "kw_chr"))
    for r in REGIONS:
        s = str(r)
        qe = sum(1 for x in recs if x["stats"][s]["qwen_empty"])
        kw = sum(1 for x in recs if x["stats"][s]["kw_is_whole_report"])
        qc = statistics.mean([x["stats"][s]["qwen_chars"] for x in recs] or [0])
        kc = statistics.mean([x["stats"][s]["kw_chars"] for x in recs] or [0])
        print("%-4d %-46s %9.1f%% %9.1f%% %10.0f %10.0f"
              % (r, REGIONS[r][:46], 100.0*qe/max(len(recs),1),
                 100.0*kw/max(len(recs),1), qc, kc))
    n = max(len(recs), 1)
    tot_kw = sum(1 for x in recs for s in x["stats"] if x["stats"][s]["kw_is_whole_report"])
    print("\nkeyword returned the WHOLE report (no localisation) in %.1f%% of region slots"
          % (100.0*tot_kw/(n*10)))
    trunc = sum(1 for x in recs for s in x["stats"] if x["stats"][s]["kw_truncated_400"])
    print("keyword hit its 400-char cap in %.1f%% of region slots" % (100.0*trunc/(n*10)))
    with open(a.out, "w") as fh:
        json.dump(recs, fh)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())

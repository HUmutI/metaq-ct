#!/usr/bin/env python3
"""Measure, for models whose extractor did not record it, how many of the 1,507
canonical reports exceed the model's trained text context.

GreenRFM and MPS-CT both encode text through 70_eval.py:encode_text, which is
BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",
do_lower_case=True) at max_length=128 (70_eval.py:88, :118, :197). This script
replays that exact tokenizer over the exact same strings and writes a sidecar
JSON that the table generator reads, so the paper's truncation note is measured
rather than asserted."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peds_report_text import valid_volumes, texts_for
from transformers import BertTokenizer

R = "/temp_work/ch278233/PEDS_BENCH/retrieval"
BERT = "microsoft/BiomedVLP-CXR-BERT-specialized"
SPEC = {"greenrfm": 128, "mpsct": 128}

stems = valid_volumes(); texts = texts_for(stems)
tok = BertTokenizer.from_pretrained(BERT, do_lower_case=True)
n = np.array([len(tok(t)["input_ids"]) for t in texts])
for stem, ctx in SPEC.items():
    out = {"n": len(n), "context_length": ctx, "n_truncated": int((n > ctx).sum()),
           "ntok_median": int(np.median(n)), "ntok_max": int(n.max()),
           "tokenizer": BERT, "tokenizer_class": "BertTokenizer(do_lower_case=True)",
           "source": "replay of 70_eval.py:encode_text (max_length=%d) over peds_report_text.texts_for" % ctx}
    json.dump(out, open(f"{R}/{stem}_trunc.json", "w"), indent=1)
    print(stem, out["n_truncated"], "/", out["n"], "median", out["ntok_median"], "max", out["ntok_max"])

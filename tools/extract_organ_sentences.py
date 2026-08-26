"""Build organ-text cache from CT-RATE reports using Qwen3-8B (thinking mode).

For each CT report, run a single Qwen3 call that returns JSON with sentences for
each of the 10 fine-grained anatomical regions (lung lobes 1-5, trachea, heart,
aorta, vessels, esophagus). The output JSON file is consumed by
``dataset.py:region_text_from_cache`` at training time when
``RAC_REGION_CACHE`` env var points at it.

Env:
  RAC_REPORTS_CSV       path to a reports CSV (default: train_reports.csv)
  RAC_REGION_OUT        output JSON path
  RAC_SAMPLE_N          how many random reports to process (0 = all)
  RAC_SEED              numpy/torch seed (default 42)
  RAC_QWEN_MODEL        HF id (default Qwen/Qwen3-8B)
  RAC_MAX_NEW_TOKENS    generation cap (default 1024)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FINE_LABEL_NAMES = {
    1: "lung upper lobe left",
    2: "lung lower lobe left",
    3: "lung upper lobe right",
    4: "lung middle lobe right",
    5: "lung lower lobe right",
    6: "trachea or main bronchi",
    7: "heart, pericardium, coronary arteries",
    8: "aorta",
    9: "mediastinal vessels, vena cava, lymph nodes",
    10: "esophagus",
}

SYSTEM_PROMPT = (
    "You are a radiology report anatomical-region extractor. "
    "Given a chest CT report, return sentences that describe findings in each anatomical region. "
    "Quote verbatim from the report where possible. "
    "If a region has no specific finding mentioned, output \"No specific finding described.\""
)


def build_user_prompt(report_text: str) -> str:
    region_lines = "\n".join(
        f'  "{k}": "<sentences about {v}>",' for k, v in FINE_LABEL_NAMES.items()
    )
    return f"""Report:
\"\"\"
{report_text}
\"\"\"

Extract for each anatomical region. Return strict JSON only, no commentary:
{{
{region_lines}
}}

Rules:
- Use exact wording from the report where possible.
- If a sentence spans multiple regions, include it in each relevant key.
- Empty findings = "No specific finding described."
- Output JSON only, no other text."""


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
JSON_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(generated: str) -> dict | None:
    cleaned = THINK_RE.sub("", generated).strip()
    match = JSON_RE.search(cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_keys(obj: dict) -> dict:
    out = {}
    for k, v in obj.items():
        try:
            kk = str(int(str(k).strip()))
        except ValueError:
            continue
        if kk in {str(i) for i in range(1, 11)} and isinstance(v, str):
            out[kk] = v.strip()
    for i in range(1, 11):
        out.setdefault(str(i), "No specific finding described.")
    return out


def main() -> None:
    reports_csv = os.environ.get(
        "RAC_REPORTS_CSV",
        "/mnt/amax5_drive/alp_ozaydin_0/data/ct_reports/train_reports.csv",
    )
    out_path = os.environ.get(
        "RAC_REGION_OUT",
        "/mnt/amax5_drive/alp_ozaydin_0/data/region_cache_qwen3_100sample.json",
    )
    sample_n = int(os.environ.get("RAC_SAMPLE_N", "100"))
    seed = int(os.environ.get("RAC_SEED", "42"))
    model_id = os.environ.get("RAC_QWEN_MODEL", "Qwen/Qwen3-8B")
    max_new = int(os.environ.get("RAC_MAX_NEW_TOKENS", "1024"))

    print(f"[qwen3] reports={reports_csv}")
    print(f"[qwen3] out={out_path}")
    print(f"[qwen3] sample_n={sample_n} seed={seed} model={model_id} max_new={max_new}")

    df = pd.read_csv(reports_csv)
    rng = np.random.default_rng(seed)
    if sample_n > 0 and sample_n < len(df):
        idxs = rng.choice(len(df), size=sample_n, replace=False)
        df = df.iloc[sorted(idxs.tolist())].reset_index(drop=True)

    shard_id = int(os.environ.get("RAC_SHARD_ID", "0"))
    shard_n = int(os.environ.get("RAC_SHARD_N", "1"))
    if shard_n > 1:
        df = df.iloc[shard_id::shard_n].reset_index(drop=True)
        print(f"[qwen3] shard={shard_id}/{shard_n} keeps {len(df)} reports")
    print(f"[qwen3] processing {len(df)} reports")

    print(f"[qwen3] loading {model_id} bf16 on cuda...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    print(f"[qwen3] model loaded; param count = {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    cache: dict[str, dict[str, str]] = {}
    parse_fail = 0
    t0 = time.time()
    flush_every = int(os.environ.get("RAC_FLUSH_EVERY", "50"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        try:
            with open(out_path) as fp:
                cache = json.load(fp)
            print(f"[qwen3] resume: loaded {len(cache)} existing entries from {out_path}")
        except Exception as e:
            print(f"[qwen3] resume failed ({e}); starting fresh")
            cache = {}

    def _flush():
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fp:
            json.dump(cache, fp, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)

    for i, row in df.iterrows():
        acc = row["VolumeName"]
        if acc in cache:
            continue
        report_text = ((str(row.get("Findings_EN", "")) + " " + str(row.get("Impressions_EN", "")))).strip()
        if not report_text:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(report_text)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        out_ids = gen[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(out_ids, skip_special_tokens=True)

        parsed = extract_json(generated)
        if parsed is None:
            parse_fail += 1
            preview = generated[:200].replace("\n", " ")
            print(f"[qwen3] {i+1}/{len(df)} {acc}  PARSE_FAIL  raw='{preview}...'")
            cache[acc] = normalize_keys({})
        else:
            cache[acc] = normalize_keys(parsed)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(df) - i - 1) / max(rate, 1e-6)
            print(
                f"[qwen3] {i+1}/{len(df)} rate={rate:.2f} r/s eta={eta/60:.1f}min parse_fail={parse_fail}",
                flush=True,
            )

        if (i + 1) % flush_every == 0:
            _flush()

    _flush()
    print(f"[qwen3] DONE wrote {len(cache)} entries to {out_path} parse_fail={parse_fail}")


if __name__ == "__main__":
    main()

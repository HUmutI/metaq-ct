#!/usr/bin/env python3
"""
extract_labels.py
=================
Recreate ctrate_labeled_by_doubao.csv by calling an LLM to classify
CT-RATE radiology reports into 18 binary pathology labels.

Supports any OpenAI-compatible API (Doubao, GPT, etc.)
configured via env vars:

    DOUBAO_API_KEY    — API key (required)
    DOUBAO_BASE_URL   — API base URL (default: Doubao Ark endpoint)
    DOUBAO_MODEL      — model id (default: ep-20250628-xxx, set yours)
    RAC_REPORTS_CSV   — path to train_reports.csv
    RAC_OUT_CSV       — output path for the labels CSV
    RAC_SHARD_ID      — 0-indexed shard to process (default: 0)
    RAC_SHARD_N       — total shards (default: 1)
    RAC_SAVE_EVERY    — flush to disk every N rows (default: 100)

Resume: re-running with the same RAC_OUT_CSV will skip already-labeled rows.

Usage:
    # single-threaded, all rows
    python scripts/build_doubao_labels.py

    # 4 parallel shards (run in 4 separate terminals or sbatch array):
    RAC_SHARD_ID=0 RAC_SHARD_N=4 RAC_OUT_CSV=.../shard0.csv python ...
    RAC_SHARD_ID=1 RAC_SHARD_N=4 RAC_OUT_CSV=.../shard1.csv python ...
    ...
    # then merge shards:
    python scripts/merge_doubao_shards.py shard0.csv shard1.csv ... out.csv
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# 18 pathology labels — must match dataset.PATHOLOGIES exactly
# ---------------------------------------------------------------------------
PATHOLOGIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_LABELS_BLOCK = "\n".join(f'  "{p}": <0 or 1>,' for p in PATHOLOGIES)

SYSTEM_PROMPT = (
    "You are a chest CT radiology report classifier. "
    "Given a radiology report (Findings and Impressions sections), "
    "output a JSON object with exactly 18 keys — one per pathology. "
    "For each key output 1 if the pathology is present or strongly suspected, "
    "0 if absent, not mentioned, or normal. "
    "Do not output any text outside the JSON object."
)


def build_user_prompt(findings: str, impressions: str) -> str:
    report = f"Findings: {findings}\nImpressions: {impressions}".strip()
    return f"""Report:
\"\"\"
{report}
\"\"\"

Classify each pathology as 0 (absent/normal) or 1 (present/suspected).
Return strict JSON only:
{{
{_LABELS_BLOCK}
}}"""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{[\s\S]*?\}", re.DOTALL)


def parse_response(text: str) -> dict[str, int] | None:
    text = _THINK_RE.sub("", text).strip()
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    # normalise keys: lowercase + strip
    obj_lower = {k.strip().lower(): v for k, v in obj.items()}
    result = {}
    for p in PATHOLOGIES:
        v = obj_lower.get(p.lower())
        if v is None:
            return None  # missing key → full parse fail
        try:
            result[p] = int(bool(v))
        except (TypeError, ValueError):
            return None
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    api_key = os.environ.get("DOUBAO_API_KEY", "")
    if not api_key:
        print("ERROR: set DOUBAO_API_KEY", file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get(
        "DOUBAO_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/v3",
    )
    model = os.environ.get("DOUBAO_MODEL", "")
    if not model:
        print("ERROR: set DOUBAO_MODEL (e.g. ep-20250628-abcdef)", file=sys.stderr)
        sys.exit(1)

    data_root = os.environ.get(
        "RAC_DATA_ROOT",
        "/mnt/amax5_drive/alp_ozaydin_0/data",
    )
    reports_csv = os.environ.get(
        "RAC_REPORTS_CSV",
        f"{data_root}/ct_reports/train_reports.csv",
    )
    out_csv = os.environ.get(
        "RAC_OUT_CSV",
        f"{data_root}/llm_labels/ctrate_labeled_by_doubao.csv",
    )
    shard_id = int(os.environ.get("RAC_SHARD_ID", "0"))
    shard_n = int(os.environ.get("RAC_SHARD_N", "1"))
    save_every = int(os.environ.get("RAC_SAVE_EVERY", "100"))

    client = OpenAI(api_key=api_key, base_url=base_url)

    # load reports
    if not Path(reports_csv).exists():
        print(f"ERROR: reports CSV not found: {reports_csv}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(reports_csv)
    df["VolumeName"] = df["VolumeName"].apply(lambda x: str(x).strip())
    print(f"[doubao] reports_total={len(df)} model={model}")

    # shard
    if shard_n > 1:
        indices = list(range(shard_id, len(df), shard_n))
        df = df.iloc[indices].reset_index(drop=True)
        print(f"[doubao] shard={shard_id}/{shard_n} keeping {len(df)} rows")

    # resume
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if Path(out_csv).exists():
        df_done = pd.read_csv(out_csv)
        for _, row in df_done.iterrows():
            done[str(row["VolumeName"])] = row.to_dict()
        print(f"[doubao] resume: {len(done)} already done")

    remaining = [row for _, row in df.iterrows() if row["VolumeName"] not in done]
    print(f"[doubao] remaining={len(remaining)}")

    results = list(done.values())
    parse_fail = 0
    t0 = time.time()

    for i, row in enumerate(remaining):
        acc = str(row["VolumeName"])
        findings = str(row.get("Findings_EN", "") or "").strip()
        impressions = str(row.get("Impressions_EN", "") or "").strip()

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(findings, impressions)},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw = resp.choices[0].message.content or ""
                labels = parse_response(raw)
                if labels is not None:
                    break
                parse_fail += 1
                if attempt < 2:
                    time.sleep(1.0)
            except Exception as e:
                print(f"  [WARN] {acc} attempt={attempt} err={e}", flush=True)
                time.sleep(2.0 * (attempt + 1))
                labels = None

        if labels is None:
            # fallback: all zeros
            labels = {p: 0 for p in PATHOLOGIES}

        entry = {"VolumeName": acc, **labels}
        results.append(entry)

        if (i + 1) % save_every == 0 or (i + 1) == len(remaining):
            pd.DataFrame(results).to_csv(out_csv, index=False)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            print(
                f"[doubao] {i+1}/{len(remaining)} rate={rate:.1f} r/s "
                f"eta={eta/60:.0f}min parse_fail={parse_fail}",
                flush=True,
            )

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[doubao] DONE wrote {len(results)} rows to {out_csv} parse_fail={parse_fail}")


if __name__ == "__main__":
    main()

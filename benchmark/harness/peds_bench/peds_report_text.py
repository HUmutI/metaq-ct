"""Canonical pediatric report string + standard retrieval-npz writer.

EVERY retrieval producer (ours, CT-CLIP, GreenRFM, MPS-CT, HLIP, BiomedCLIP,
Merlin) must encode the byte-identical string, or report->image R@K is not
comparable. The rule is copied from bch-arc-ct/arcct/dataset.py:365-374 as it
runs under configs/stage2_peds.env, which sets RAC_IMPRESSION_FIRST=1:

    findings    = str(row.get("Findings_EN", "") or "")     # NaN -> "nan" (kept!)
    impressions = str(row.get("Impressions_EN", "") or "")
    concat = (impressions + " " + findings).strip()          # impression FIRST
    text   = re.sub(r"[\"'()]", "", concat)

The `nan` quirk affects 16 volumes with empty findings; it is what our own
model was evaluated with, so it is reproduced, not corrected.
"""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd

REPORTS = "/temp_work/ch278233/COMBINED_reports.csv"
LABELS  = "/temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv"
VOLLIST = "/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt"
_STRIP = re.compile(r"[\"'()]")


def _field(row, col):
    return str(row.get(col, "") or "")           # identical semantics to dataset.py


def canonical_text(findings, impressions):
    return _STRIP.sub("", (impressions + " " + findings).strip())


def stem(v):
    return v[:-7] if v.endswith(".nii.gz") else v


def valid_volumes():
    return [stem(l.strip()) for l in open(VOLLIST) if l.strip()]


def texts_for(stems):
    df = pd.read_csv(REPORTS)
    m = {}
    for _, row in df.iterrows():
        m[stem(str(row["VolumeName"]))] = canonical_text(_field(row, "Findings_EN"),
                                                          _field(row, "Impressions_EN"))
    return [m[s] for s in stems]


def labels_for(stems):
    df = pd.read_csv(LABELS).set_index("VolumeName")
    cols = list(df.columns)
    L = (df.loc[[s + ".nii.gz" for s in stems], cols].to_numpy(dtype=float) > 0.5).astype(np.int8)
    return L, cols


def write_retrieval_npz(path, img_lat, txt_lat, stems, meta=None):
    stems = [stem(str(s)) for s in stems]
    L, cols = labels_for(stems)
    texts = texts_for(stems)
    ok = np.array([t.strip() != "" for t in texts])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, img_lat=np.asarray(img_lat, np.float32), txt_lat=np.asarray(txt_lat, np.float32),
                        accessions=np.array(stems), labels=L, text_ok=ok,
                        classes=np.array(cols), meta=np.array(str(meta or {})))
    return path


if __name__ == "__main__":
    st = valid_volumes(); tx = texts_for(st)
    print(f"[canon] valid volumes={len(st)}  empty text={sum(1 for t in tx if not t.strip())}")
    for s, t in list(zip(st, tx))[:2]:
        print(f"  {s}: len={len(t)} | {t[:90]!r}")
    print(f"  ornek: 'nan' token iceren={sum(1 for t in tx if ' nan' in t or t.startswith('nan'))} (beklenen ~16)")

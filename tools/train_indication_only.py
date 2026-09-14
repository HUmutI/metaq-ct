#!/usr/bin/env python3
"""Frozen-CXR-BERT indication-only control for the pediatric 27-class task.

This deliberately has no CT, age, sex, report text, or anatomy input.  It uses
the same indication tower and 64-token budget as Context Q-Former, then trains
only a LayerNorm and linear 27-label head.  Three seeds quantify how much of the
refined model's result can be obtained from indication text alone.
"""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

from arcct.context import NO_INDICATION, TEXT_MODEL
from arcct.schema import active


ROOT = Path(os.environ.get("ARCCT_ROOT", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("IND_ONLY_RESULTS", "/temp_work/ch278233/runs/indication_only_peds27"))
CONTEXT_CSV = Path(os.environ.get("RAC_CONTEXT_CSV", "/temp_work/ch278233/CONTEXT/indication.csv"))
LABELS_TRAIN = Path(os.environ.get("RAC_LABELS_TRAIN", "/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv"))
LABELS_VALID = Path(os.environ.get("RAC_LABELS_VALID", "/temp_work/ch278233/PEDS_VALID_labels_v2.csv"))
LIST_TRAIN = Path(os.environ.get("RAC_VOLUME_LIST_TRAIN", "/temp_work/ch278233/PEDS_VOLLIST_TRAIN_clean.txt"))
LIST_VALID = Path(os.environ.get("RAC_VOLUME_LIST_VALID", "/temp_work/ch278233/PEDS_VOLLIST_VALID_clean.txt"))
MAX_LEN = int(os.environ.get("RAC_CTX_MAX_IND_LEN", "64"))
ENCODE_BATCH = int(os.environ.get("IND_ONLY_ENCODE_BATCH", "64"))
HEAD_BATCH = int(os.environ.get("IND_ONLY_HEAD_BATCH", "512"))
EPOCHS = int(os.environ.get("IND_ONLY_EPOCHS", "200"))
PATIENCE = int(os.environ.get("IND_ONLY_PATIENCE", "20"))
LR = float(os.environ.get("IND_ONLY_LR", "1e-3"))
IND_DROPOUT = float(os.environ.get("RAC_IND_DROPOUT", "0.10"))
SEEDS = tuple(int(x) for x in os.environ.get("IND_ONLY_SEEDS", "0,1,2").split(","))


def read_list(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def load_split(names: list[str], labels_path: Path, indications: dict[str, str], classes: list[str]):
    df = pd.read_csv(labels_path).set_index("VolumeName")
    keep = [n for n in names if n in df.index and n in indications]
    y = df.loc[keep, classes].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    # The production loss masks NaN targets. This dataset is complete, but keep
    # the mask explicit so a future label version cannot silently become zero.
    mask = np.isfinite(y)
    y = np.nan_to_num(y, nan=0.0)
    texts = [indications[n] for n in keep]
    return keep, texts, y, mask


@torch.inference_mode()
def encode(texts: list[str], tokenizer, bert, device) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), ENCODE_BATCH):
        tok = tokenizer(
            texts[start:start + ENCODE_BATCH], return_tensors="pt",
            padding="max_length", truncation=True, max_length=MAX_LEN)
        ids = tok["input_ids"].to(device)
        am = tok["attention_mask"].to(device)
        h = bert(input_ids=ids, attention_mask=am)[0]
        m = am.to(h.dtype).unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        chunks.append(pooled.float().cpu())
    return torch.cat(chunks)


def macro_auc(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> tuple[float, list[float]]:
    vals = []
    for j in range(y.shape[1]):
        take = mask[:, j]
        yt = y[take, j]
        vals.append(float(roc_auc_score(yt, p[take, j])) if len(np.unique(yt)) > 1 else float("nan"))
    return float(np.nanmean(vals)), vals


class IndicationHead(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.classifier(self.norm(x))


def fit_seed(seed: int, xtr, ytr, mtr, xva, yva, mva, x_empty, classes, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = IndicationHead(xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(xtr, ytr, mtr)
    best_auc, best_epoch, best_state, stale = -1.0, -1, None, 0
    rng = torch.Generator(device=device).manual_seed(seed + 1000)

    for epoch in range(EPOCHS):
        model.train()
        loader = DataLoader(ds, batch_size=HEAD_BATCH, shuffle=True, generator=gen)
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            if IND_DROPOUT > 0:
                drop = torch.rand(len(xb), generator=rng, device=device) < IND_DROPOUT
                xb = torch.where(drop[:, None], x_empty.to(device), xb)
            logits = model(xb)
            loss_raw = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (loss_raw * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(xva.to(device))).cpu().numpy()
        score, _ = macro_auc(yva.numpy(), pred, mva.numpy().astype(bool))
        if score > best_auc + 1e-5:
            best_auc, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best_state); model.to(device).eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(xva.to(device))).cpu().numpy()
    score, per = macro_auc(yva.numpy(), pred, mva.numpy().astype(bool))
    torch.save({"model": best_state, "seed": seed, "best_epoch": best_epoch,
                "macro_auc": score, "classes": classes}, OUT / f"seed{seed}.pt")
    np.savez_compressed(OUT / f"predictions_seed{seed}.npz", pred=pred, true=yva.numpy(),
                        mask=mva.numpy(), pathologies=np.asarray(classes))
    return {"seed": seed, "best_epoch": best_epoch, "macro_auc": score,
            "per_class": dict(zip(classes, per))}


def main() -> int:
    os.environ["RAC_SCHEMA"] = "peds"
    classes = list(active()["PATHOLOGIES"])
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = pd.read_csv(CONTEXT_CSV, keep_default_na=False)
    indications = {}
    for row in ctx.to_dict("records"):
        text = str(row.get("Indication_EN", "")).strip()
        if row.get("ind_status") != "present" or not text:
            text = NO_INDICATION
        indications[str(row["VolumeName"])] = text
    train_names, train_text, ytr, mtr = load_split(read_list(LIST_TRAIN), LABELS_TRAIN, indications, classes)
    valid_names, valid_text, yva, mva = load_split(read_list(LIST_VALID), LABELS_VALID, indications, classes)
    print(f"[ind-only] train={len(train_names):,} valid={len(valid_names):,} classes={len(classes)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL, do_lower_case=True)
    bert = BertModel.from_pretrained(TEXT_MODEL).to(device).eval()
    for p in bert.parameters():
        p.requires_grad = False
    xtr = encode(train_text, tokenizer, bert, device)
    xva = encode(valid_text, tokenizer, bert, device)
    x_empty = encode([NO_INDICATION], tokenizer, bert, device)
    del bert
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ytr_t, mtr_t = torch.from_numpy(ytr), torch.from_numpy(mtr.astype(np.float32))
    yva_t, mva_t = torch.from_numpy(yva), torch.from_numpy(mva.astype(np.float32))
    results = [fit_seed(s, xtr, ytr_t, mtr_t, xva, yva_t, mva_t,
                        x_empty, classes, device) for s in SEEDS]
    summary = {
        "control": "indication_only_frozen_cxrbert_linear",
        "train_n": len(train_names), "valid_n": len(valid_names),
        "max_len": MAX_LEN, "indication_dropout": IND_DROPOUT,
        "seeds": results,
        "mean_macro_auc": float(np.mean([r["macro_auc"] for r in results])),
        "sd_macro_auc": float(np.std([r["macro_auc"] for r in results], ddof=1)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT / "per_class_auc.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["class"] + [f"seed{s}" for s in SEEDS] + ["mean"])
        for c in classes:
            vals = [r["per_class"][c] for r in results]
            w.writerow([c] + vals + [float(np.mean(vals))])
    print(json.dumps({k: v for k, v in summary.items() if k != "seeds"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

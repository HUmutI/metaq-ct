#!/usr/bin/env python3
"""No-CT pediatric controls using indication and/or demographics only."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

from arcct.context import NO_INDICATION, TEXT_MODEL
from arcct.schema import active


WORK = Path("/temp_work/ch278233")
MODE = os.environ.get("META_ONLY_MODE", "all")
OUT = Path(os.environ.get("META_ONLY_RESULTS", f"{WORK}/runs/peds23_u18_metadata_only/{MODE}"))
SEEDS = [int(value) for value in os.environ.get("META_ONLY_SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("META_ONLY_EPOCHS", "200"))
PATIENCE = int(os.environ.get("META_ONLY_PATIENCE", "20"))
LR = float(os.environ.get("META_ONLY_LR", "1e-3"))


def names(split: str) -> list[str]:
    path = WORK / f"PEDS23_U18_VOLLIST_{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def safe_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, list[float], list[float]]:
    aucs, aps = [], []
    for index in range(y.shape[1]):
        aucs.append(float(roc_auc_score(y[:, index], p[:, index]))
                    if len(np.unique(y[:, index])) > 1 else float("nan"))
        aps.append(float(average_precision_score(y[:, index], p[:, index]))
                   if y[:, index].sum() > 0 else float("nan"))
    return float(np.nanmean(aucs)), float(np.nanmean(aps)), aucs, aps


@torch.inference_mode()
def encode_text(texts: list[str], device: torch.device) -> torch.Tensor:
    tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL, do_lower_case=True)
    bert = BertModel.from_pretrained(TEXT_MODEL).to(device).eval()
    chunks = []
    for start in range(0, len(texts), 64):
        tokenized = tokenizer(texts[start:start + 64], return_tensors="pt",
                              padding="max_length", truncation=True, max_length=64)
        ids = tokenized["input_ids"].to(device)
        mask = tokenized["attention_mask"].to(device)
        hidden = bert(input_ids=ids, attention_mask=mask)[0]
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        chunks.append(((hidden * weights).sum(1) /
                       weights.sum(1).clamp(min=1)).float().cpu())
    del bert
    return torch.cat(chunks)


def build_features(split_names: list[str], device: torch.device) -> torch.Tensor:
    parts = []
    if MODE in ("indication", "all"):
        ctx = pd.read_csv(WORK / "CONTEXT/indication.csv", keep_default_na=False).set_index("VolumeName")
        texts = []
        for name in split_names:
            row = ctx.loc[name]
            text = str(row["Indication_EN"]).strip()
            texts.append(text if row["ind_status"] == "present" and text else NO_INDICATION)
        parts.append(encode_text(texts, device))
    if MODE in ("demographics", "all"):
        dem = pd.read_csv(WORK / "CONTEXT/demographics.csv").set_index("VolumeName")
        age = pd.to_numeric(dem.loc[split_names, "AgeYears"], errors="coerce").fillna(-1).to_numpy()
        age = np.clip(age / 18.0, -1.0, 1.0)[:, None]
        sex = pd.to_numeric(dem.loc[split_names, "SexIdx"], errors="coerce").fillna(-1).astype(int).to_numpy()
        one_hot = np.stack([(sex == 0), (sex == 1), (sex < 0)], axis=1).astype(np.float32)
        parts.append(torch.from_numpy(np.concatenate([age.astype(np.float32), one_hot], axis=1)))
    if not parts:
        raise ValueError("META_ONLY_MODE must be indication, demographics, or all")
    return torch.cat(parts, dim=1)


class Head(nn.Module):
    def __init__(self, width: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, classes))

    def forward(self, features):
        return self.net(features)


def fit(seed, x_train, y_train, x_dev, y_dev, x_test, y_test, classes, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = Head(x_train.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=512,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_auc, best_epoch, best_state, stale = -1.0, -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for features, labels in loader:
            logits = model(features.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_pred = torch.sigmoid(model(x_dev.to(device))).cpu().numpy()
        dev_auc, _, _, _ = safe_metrics(y_dev.numpy(), dev_pred)
        if dev_auc > best_auc + 1e-5:
            best_auc, best_epoch = dev_auc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE: break
    model.load_state_dict(best_state); model.to(device).eval()
    with torch.no_grad():
        dev_pred = torch.sigmoid(model(x_dev.to(device))).cpu().numpy()
        test_pred = torch.sigmoid(model(x_test.to(device))).cpu().numpy()
    dev_auc, dev_ap, _, _ = safe_metrics(y_dev.numpy(), dev_pred)
    test_auc, test_ap, test_aucs, test_aps = safe_metrics(y_test.numpy(), test_pred)
    torch.save({"model": best_state, "seed": seed, "mode": MODE, "classes": classes}, OUT / f"seed{seed}.pt")
    np.savez_compressed(OUT / f"test_predictions_seed{seed}.npz", pred=test_pred,
                        true=y_test.numpy(), pathologies=np.asarray(classes))
    return {"seed": seed, "best_epoch": best_epoch, "dev_auc": dev_auc,
            "dev_auprc": dev_ap, "test_auc": test_auc, "test_auprc": test_ap,
            "test_per_class_auc": dict(zip(classes, test_aucs)),
            "test_per_class_auprc": dict(zip(classes, test_aps))}


def main() -> int:
    if MODE not in ("indication", "demographics", "all"):
        raise ValueError(MODE)
    classes = list(active()["PATHOLOGIES"])
    labels = pd.read_csv(WORK / "BCH_DATASET/LABELS27/labels.csv").set_index("VolumeName")
    split_names = {split: names(split) for split in ("TRAIN", "DEV", "TEST")}
    y = {split: torch.from_numpy(labels.loc[values, classes].to_numpy(np.float32))
         for split, values in split_names.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = {split: build_features(values, device) for split, values in split_names.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    results = [fit(seed, features["TRAIN"], y["TRAIN"], features["DEV"], y["DEV"],
                   features["TEST"], y["TEST"], classes, device) for seed in SEEDS]
    summary = {"control": "no_ct_metadata_only", "mode": MODE,
               "train_n": len(split_names["TRAIN"]), "dev_n": len(split_names["DEV"]),
               "test_n": len(split_names["TEST"]), "seeds": results,
               "mean_test_auc": float(np.mean([result["test_auc"] for result in results])),
               "sd_test_auc": float(np.std([result["test_auc"] for result in results], ddof=1)),
               "mean_test_auprc": float(np.mean([result["test_auprc"] for result in results]))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "seeds"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

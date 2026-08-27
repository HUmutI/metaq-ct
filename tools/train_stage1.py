"""Stage-1 supervised 3-channel vision pretraining for RAC-CLIP.

This trains the R3D-18 image encoder on the 18 CT-RATE abnormality labels and
saves weights that ``train.py`` can load through ``RAC_STAGE1_CKPT``.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_HOME", f"{HERE}/.cache/torch")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tools"))

from arcct.dataset import PATHOLOGIES, N_PATH, RACDatasetV4, rac_collate  # noqa: E402
from arcct.image_encoder import RACImageEncoder  # noqa: E402


DATA_ROOT = os.environ.get("RAC_DATA_ROOT", "/mnt/amax5_drive/alp_ozaydin_0/data")
DATA_TRAIN = f"{DATA_ROOT}/mps_ct_npz/train"
DATA_VALID = f"{DATA_ROOT}/mps_ct_npz/valid"
MASK_TRAIN = os.environ.get("RAC_MASK_TRAIN", "/mnt/amax2_drive/alp_ozaydin_0/data/fine_train_masks_192")
MASK_VALID = os.environ.get("RAC_MASK_VALID", "/mnt/amax2_drive/alp_ozaydin_0/data/fine_valid_masks_192")
REPORTS_TRAIN = f"{DATA_ROOT}/ct_reports/train_reports.csv"
REPORTS_VALID = f"{DATA_ROOT}/ct_reports/valid_reports.csv"
LABELS_TRAIN = os.environ.get("RAC_LABELS_TRAIN", f"{DATA_ROOT}/multi_abnormality_labels/train_predicted_labels.csv")
LABELS_VALID = os.environ.get("RAC_LABELS_VALID", f"{DATA_ROOT}/multi_abnormality_labels/valid_predicted_labels.csv")
RESULTS_DIR = os.environ.get("RAC_STAGE1_RESULTS_DIR", f"{HERE}/runs/stage1_seed{os.environ.get('RAC_SEED', '0')}")

EPOCHS = int(os.environ.get("RAC_STAGE1_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("RAC_STAGE1_BATCH_SIZE", "8"))
LR = float(os.environ.get("RAC_STAGE1_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("RAC_STAGE1_WEIGHT_DECAY", "1e-4"))
NUM_WORKERS = int(os.environ.get("RAC_NUM_WORKERS", "6"))
TRAIN_LIMIT = int(os.environ.get("RAC_TRAIN_LIMIT", "0"))
VAL_LIMIT = int(os.environ.get("RAC_VAL_LIMIT", "0"))
FAIL_FAST = os.environ.get("RAC_FAIL_FAST", "1") == "1"
USE_AMP = os.environ.get("RAC_AMP", "1") == "1"
KINETICS_PRETRAINED = os.environ.get("RAC_KINETICS_PRETRAINED", "1") == "1"
MAX_GRAD_NORM = float(os.environ.get("RAC_MAX_GRAD_NORM", "1.0"))
LOSS_NAME = os.environ.get("RAC_STAGE1_LOSS", "asl").lower()

_seed_env = os.environ.get("RAC_SEED", "").strip()
if _seed_env:
    SEED = int(_seed_env)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    _GEN = torch.Generator()
    _GEN.manual_seed(SEED)

    def _worker(wid):
        ws = (SEED + wid) & 0xFFFFFFFF
        np.random.seed(ws)
        random.seed(ws)
else:
    _GEN = None
    _worker = None


def _loader_kwargs():
    return {"generator": _GEN, "worker_init_fn": _worker} if _GEN else {}


def _amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return torch.bfloat16 if bf16_ok else torch.float16


def amp_context(device):
    if USE_AMP and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=_amp_dtype())
    return nullcontext()


class Stage1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = RACImageEncoder(pretrained_kinetics=KINETICS_PRETRAINED)
        self.classifier = nn.Linear(RACImageEncoder.FEAT_DIM, N_PATH)

    def forward(self, x):
        return self.classifier(self.image_encoder(x))


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pos = probs
        neg = 1.0 - probs
        if self.clip and self.clip > 0:
            neg = (neg + self.clip).clamp(max=1.0)
        loss_pos = targets * torch.log(pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(neg.clamp(min=self.eps))
        pt = pos * targets + neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        loss = (loss_pos + loss_neg) * torch.pow(1.0 - pt, gamma)
        return -loss.mean()


def _aucs(pred, true):
    vals = []
    per = {}
    for j, name in enumerate(PATHOLOGIES):
        if len(np.unique(true[:, j])) > 1:
            v = float(roc_auc_score(true[:, j], pred[:, j]))
            vals.append(v)
            per[name] = v
        else:
            per[name] = float("nan")
    return float(np.nanmean(vals)) if vals else 0.0, per


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for ct, _, _, labels, _, _, _, _ in tqdm.tqdm(loader, desc="Stage1-val", leave=False):
        ct = ct.to(device, non_blocking=True)
        with amp_context(device):
            logits = model(ct)
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(labels.numpy())
    model.train()
    return _aucs(np.concatenate(preds), np.concatenate(trues))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage1] device={device} results_dir={RESULTS_DIR}")
    print(f"[stage1] epochs={EPOCHS} batch={BATCH_SIZE} lr={LR} loss={LOSS_NAME}")

    train_ds = RACDatasetV4(
        DATA_TRAIN,
        REPORTS_TRAIN,
        LABELS_TRAIN,
        mask_root=MASK_TRAIN if os.path.isdir(MASK_TRAIN) else None,
        is_train=True,
        limit=TRAIN_LIMIT,
        fail_fast=FAIL_FAST,
    )
    val_ds = RACDatasetV4(
        DATA_VALID,
        REPORTS_VALID,
        LABELS_VALID,
        mask_root=MASK_VALID if os.path.isdir(MASK_VALID) else None,
        is_train=False,
        limit=VAL_LIMIT,
        fail_fast=FAIL_FAST,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=rac_collate, drop_last=True, pin_memory=True, persistent_workers=NUM_WORKERS > 0, **_loader_kwargs())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=rac_collate, pin_memory=True, **_loader_kwargs())

    model = Stage1Model().to(device)
    criterion = AsymmetricLoss() if LOSS_NAME == "asl" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS * len(train_loader), 1), eta_min=LR * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda" and _amp_dtype() == torch.float16)

    # Resume from latest saved epoch checkpoint if available
    start_epoch = 1
    best_auc = 0.0
    for ep in range(EPOCHS, 0, -1):
        ckpt_path = f"{RESULTS_DIR}/stage1_epoch{ep}.pt"
        if os.path.isfile(ckpt_path):
            pkg = torch.load(ckpt_path, map_location=device)
            model.image_encoder.load_state_dict(pkg["image_encoder"])
            model.classifier.load_state_dict(pkg["classifier"])
            best_auc = pkg.get("best_auc", 0.0)
            start_epoch = ep + 1
            print(f"[stage1] resumed from epoch={ep} best_auc={best_auc:.4f}")
            break

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"Stage1 epoch {epoch}/{EPOCHS}")
        for ct, _, _, labels, _, _, _, _ in pbar:
            ct = ct.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(ct)
                loss = criterion(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        mean_auc, per_auc = validate(model, val_loader, device)
        print(f"[stage1] epoch={epoch} loss={np.mean(losses):.4f} val_auc={mean_auc:.4f} best={best_auc:.4f} t={(time.time()-t0)/60:.1f}m")
        pkg = {
            "image_encoder": model.image_encoder.state_dict(),
            "classifier": model.classifier.state_dict(),
            "epoch": epoch,
            "best_auc": max(best_auc, mean_auc),
            "per_auc": per_auc,
            "config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "loss": LOSS_NAME},
        }
        torch.save(pkg, f"{RESULTS_DIR}/stage1_epoch{epoch}.pt")
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(pkg, f"{RESULTS_DIR}/stage1_best.pt")
            with open(f"{RESULTS_DIR}/stage1_best_auc.txt", "w") as f:
                for name in PATHOLOGIES:
                    f.write(f"{name:45s}: {per_auc[name]:.4f}\n")
                f.write(f"{'Mean AUC':45s}: {mean_auc:.4f}\n")

    print(f"[stage1] Done. best_val_auc={best_auc:.4f} ckpt={RESULTS_DIR}/stage1_best.pt")


if __name__ == "__main__":
    main()

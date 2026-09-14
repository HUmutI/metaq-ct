#!/usr/bin/env python3
"""Drive GreenRFM's two-stage recipe on our cohorts.

Stage 1 (`image`, `text`): independent supervised pre-training of each encoder
against the label set.  Stage 2 (`align`): contrastive alignment with the
shared-classifier constraint.  Upstream ships these as __main__ blocks with
hardcoded ./data paths and a CONFIG dict; we import the run functions and
supply the config, leaving the clone untouched.

Faithfulness notes, deliberate:
  * upstream builds the stage-2 val_dataset with section="train", so validation
    inherits RandSpatialCrop rather than a centre crop.  Kept as-is -- changing
    it would make our numbers incomparable to the authors' recipe -- but it
    means stage-2 val loss is noisier than a normal eval.
  * num_classes is detected from the first sample, so a 23-column pediatric
    labels CSV yields a 23-way head with no code change.
  * stage "text" never reads the volume: supervise_pretrain.py:117-126 uses only
    batch['text_input'] and batch['labels'].  Left alone, the loader would still
    decompress 42,544 npz per epoch for nothing, and EXPERIMENTS.md records a
    1000x slowdown when a concurrent job saturated NFS the same way.  We feed
    that stage text and labels straight from the scanned sample list, which is
    the identical arithmetic with none of the I/O.

inference/run_zero_shot.py:207 falls back to RANDOM weights when no checkpoint
is found, printing only "No checkpoints found. Testing with initialized
weights."  Nothing here can produce that, but any eval wrapper must gate on it.
"""
from __future__ import annotations
import os, sys

REPO = "/home/ch278233/BENCHMARK/GreenRFM"


def env(k, default=None, required=False):
    v = os.environ.get(k, default)
    if required and not v:
        sys.exit(f"[greenrfm] FATAL: {k} is unset")
    return v


def _encoder_state_dict(ck, attr):
    """Pull one encoder's tensors out of whichever checkpoint shape we were given.

    supervise_pretrain.py:150 saves stage 1 as {'encoder','classifier','optimizer','epoch'}
    where 'encoder' holds BARE module keys (stem.0.weight / embeddings.*).
    alignment.py:131 saves stage 2 as {'model',...} holding the whole CTCLIP,
    so the same tensors appear PREFIXED as image_encoder.* / text_encoder.*.

    Warm-starting a pediatric fine-tune from the CT-RATE aligned model uses the
    second shape; chaining stage 2 onto stage 1 uses the first.  Guessing wrong
    silently loads nothing, which is why the caller aborts on a zero match.
    """
    if isinstance(ck, dict) and "encoder" in ck and isinstance(ck["encoder"], dict):
        return ck["encoder"], "stage1:encoder"
    full = None
    for k in ("model", "model_state_dict", "state_dict"):
        if isinstance(ck, dict) and k in ck and isinstance(ck[k], dict):
            full = ck[k]
            break
    if full is None:
        full = ck
    pref = attr + "."
    sub = {k[len(pref):]: v for k, v in full.items() if k.startswith(pref)}
    if sub:
        return sub, f"full-model:{pref}*"
    return full, "raw"


class _TextOnly:
    """Text + labels straight from the scan, with no volume read.

    GreenRFM's stage-1 text arm never touches batch['image'], so reading the
    npz for it is pure NFS traffic.  Wraps the same sample list the real
    dataset built, so text, labels and ordering are unchanged.
    """

    def __init__(self, samples):
        import numpy as np
        self._np = np
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import torch
        s = self.samples[i]
        lab = self._np.asarray(s["labels"], dtype=self._np.float32)
        return {"text": s["text"], "labels": torch.from_numpy(lab)}


def main() -> int:
    stage = env("BM_STAGE", required=True)
    if stage not in ("image", "text", "align"):
        sys.exit(f"[greenrfm] FATAL: BM_STAGE must be image|text|align, got {stage!r}")
    if os.environ.get("WANDB_MODE") not in ("disabled", "offline"):
        sys.exit("[greenrfm] FATAL: set WANDB_MODE=disabled (upstream calls wandb.init unconditionally)")

    sys.path.insert(0, REPO)
    os.chdir(REPO)  # upstream resolves a couple of relative paths

    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer
    from data.ct_rate import CTRATEDataset

    train_folder = env("BM_DATA_TRAIN", required=True)
    val_folder   = env("BM_DATA_VALID", required=True)
    train_reports = env("BM_REPORTS_TRAIN", required=True)
    val_reports   = env("BM_REPORTS_VALID", required=True)
    train_labels  = env("BM_LABELS_TRAIN", required=True)
    val_labels    = env("BM_LABELS_VALID", required=True)
    save_dir      = env("BM_SAVE_DIR", required=True)

    epochs  = int(env("BM_EPOCHS", "5"))
    batch   = int(env("BM_BATCH", "10"))
    workers = int(env("BM_WORKERS", "4"))
    lr      = float(env("BM_LR", "1e-4" if stage != "align" else "1e-5"))
    limit   = int(env("BM_LIMIT", "0"))
    bert    = env("BM_BERT", "microsoft/BiomedVLP-CXR-BERT-specialized")

    for p in (train_folder, val_folder, train_reports, val_reports, train_labels, val_labels):
        if not os.path.exists(p):
            sys.exit(f"[greenrfm] FATAL: missing input {p}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[greenrfm] stage={stage} torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    print("[greenrfm] scanning train tree ...", flush=True)
    train_ds = CTRATEDataset(data_folder=train_folder, reports_csv=train_reports,
                             labels_csv=train_labels, section="train")
    print("[greenrfm] scanning val tree ...", flush=True)
    val_ds = CTRATEDataset(data_folder=val_folder, reports_csv=val_reports,
                           labels_csv=val_labels, section="train")
    print(f"[greenrfm] train={len(train_ds)} val={len(val_ds)}", flush=True)
    if len(train_ds) == 0:
        sys.exit("[greenrfm] FATAL: train dataset scanned to ZERO samples")

    if limit:
        train_ds = torch.utils.data.Subset(train_ds, range(min(limit, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(max(limit // 4, 2), len(val_ds))))
        print(f"[greenrfm] SMOKE limit -> train={len(train_ds)} val={len(val_ds)}", flush=True)

    if stage == "text":
        base_tr = train_ds.dataset.samples if hasattr(train_ds, "dataset") else train_ds.samples
        base_va = val_ds.dataset.samples if hasattr(val_ds, "dataset") else val_ds.samples
        if limit:
            base_tr = base_tr[: len(train_ds)]
            base_va = base_va[: len(val_ds)]
        train_ds, val_ds = _TextOnly(base_tr), _TextOnly(base_va)
        print(f"[greenrfm] text stage: volume reads skipped "
              f"(train={len(train_ds)} val={len(val_ds)})", flush=True)

    sample = train_ds[0]
    num_classes = len(sample["labels"])
    print(f"[greenrfm] detected num_classes={num_classes}", flush=True)

    config = {
        "mode": stage if stage != "align" else "image",
        "save_dir": save_dir,
        "num_classes": num_classes,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch,
        "wd": float(env("BM_WD", "0.0")),
        "lite_version": env("BM_LITE", "0") == "1",
        "bert_name": bert,
        "dataset": "ct_rate",
        "project_name": f"benchmark_greenrfm_{stage}",
    }

    tokenizer = BertTokenizer.from_pretrained(bert, do_lower_case=True)

    def collate_fn(batch):
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        if stage == "text":
            texts = [b["text"] for b in batch]
            return {"labels": labels,
                    "text_input": tokenizer(texts, padding="max_length", truncation=True,
                                            max_length=int(env("BM_MAXLEN", "128")),
                                            return_tensors="pt")}
        images = torch.stack([b["image"] for b in batch], dim=0)
        texts = [b["text"] for b in batch]
        text_inputs = tokenizer(texts, padding="max_length", truncation=True,
                                max_length=int(env("BM_MAXLEN", "128")), return_tensors="pt")
        return {"image": images, "labels": labels, "text_input": text_inputs}

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=workers, collate_fn=collate_fn)

    if stage in ("image", "text"):
        from training.supervise_pretrain import run_supervised_training
        run_supervised_training(mode=stage, train_loader=train_loader,
                                val_loader=val_loader, config=config)
    else:
        from models.clip import CTCLIP
        from training.alignment import run_alignment_training
        model = CTCLIP(num_classes=num_classes, text_encoder_name=bert,
                       dim_image=512, dim_text=768, dim_latent=768,
                       lite_version=config["lite_version"])
        vis = env("BM_PRETRAINED_VISION", "")
        txt = env("BM_PRETRAINED_TEXT", "")
        for tag, path, attr in (("vision", vis, "image_encoder"), ("text", txt, "text_encoder")):
            if not path:
                continue
            if not os.path.exists(path):
                sys.exit(f"[greenrfm] FATAL: BM_PRETRAINED_{tag.upper()} set but missing: {path}")
            ck = torch.load(path, map_location="cpu")
            sd, origin = _encoder_state_dict(ck, attr)
            tgt = getattr(model, attr)
            missing, unexpected = tgt.load_state_dict(sd, strict=False)
            loaded = len(sd) - len(unexpected)
            print(f"[greenrfm] loaded {tag} weights from {path} [{origin}]: "
                  f"{loaded}/{len(sd)} tensors (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
            if loaded == 0:
                sys.exit(f"[greenrfm] FATAL: {tag} checkpoint matched ZERO tensors -- key layout mismatch")
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                 "weight_decay": config["wd"]},
                {"params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                 "weight_decay": 0.0},
            ], lr=lr)
        run_alignment_training(model=model, train_loader=train_loader,
                               val_loader=val_loader, optimizer=optimizer, config=config)
    print("[greenrfm] stage complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

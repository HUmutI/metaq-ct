"""Train Qwen3-8B LoRA and a CT-token resampler for report generation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arcct.report_generator import CTTokenResampler  # noqa: E402

SYSTEM = ("You are an expert thoracic radiologist. Generate an accurate, concise chest CT "
          "report from the encoded scan. Do not invent findings. State uncertainty when needed.")
USER = ("The following learned tokens encode one chest CT examination. Produce FINDINGS and "
        "IMPRESSION sections. Prioritize clinically important abnormalities and relevant negatives.")


class CacheDataset(Dataset):
    def __init__(self, root: str, limit: int = 0, token_mode: str = "all"):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        parts = [torch.load(p, map_location="cpu") for p in sorted(self.root.glob("shard_*.pt"))]
        self.tokens = torch.cat([p["tokens"] for p in parts], 0)
        if token_mode == "general":
            self.tokens = self.tokens[:, :int(self.manifest["n_general_tokens"])]
        self.findings = sum((p["findings"] for p in parts), [])
        self.impressions = sum((p["impression"] for p in parts), [])
        self.labels = torch.cat([p["labels"] for p in parts], 0)
        if limit:
            self.tokens = self.tokens[:limit]
            self.findings = self.findings[:limit]
            self.impressions = self.impressions[:limit]
            self.labels = self.labels[:limit]

    def __len__(self): return len(self.findings)

    def __getitem__(self, i):
        target = f"FINDINGS:\n{self.findings[i].strip()}\n\nIMPRESSION:\n{self.impressions[i].strip()}"
        return self.tokens[i], target, self.labels[i]


def collate(batch):
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch],
            torch.stack([x[2] for x in batch]))


def prefix_ids(tokenizer, device):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--dev-cache", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--resampler-lr", type=float, default=5e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-target-tokens", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--token-mode", choices=("general", "all"), default="all",
                   help="general is the matched CT-only arm; all includes C1 tokens")
    p.add_argument("--eval-batches", type=int, default=100)
    p.add_argument("--aux-weight", type=float, default=0.1,
                   help="weight of pathology BCE factuality anchor")
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               local_files_only=True, attn_implementation="sdpa")
    lm.gradient_checkpointing_enable()
    lm.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    lm = get_peft_model(lm, lora).cuda()
    lm.print_trainable_parameters()
    hidden = lm.get_input_embeddings().embedding_dim
    resampler = CTTokenResampler(output_dim=hidden).cuda().to(torch.bfloat16)

    train_ds = CacheDataset(args.train_cache, args.train_limit, args.token_mode)
    dev_ds = CacheDataset(args.dev_cache, args.dev_limit, args.token_mode)
    label_valid = torch.isfinite(train_ds.labels)
    label_pos = (torch.nan_to_num(train_ds.labels, nan=0.0) * label_valid).sum(0)
    label_neg = label_valid.sum(0) - label_pos
    aux_pos_weight = (label_neg / label_pos.clamp_min(1)).clamp(1.0, 10.0).cuda().to(torch.bfloat16)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=True)
    aux_classifier = torch.nn.Sequential(torch.nn.LayerNorm(hidden),
                                         torch.nn.Linear(hidden, 18)).cuda().to(torch.bfloat16)
    groups = [{"params": [p for p in lm.parameters() if p.requires_grad], "lr": args.lr},
              {"params": list(resampler.parameters()) + list(aux_classifier.parameters()),
               "lr": args.resampler_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total_updates * args.warmup_ratio), total_updates)
    fixed_prefix = prefix_ids(tokenizer, "cuda")

    def make_batch(ct_tokens, targets):
        encoded = tokenizer(targets, padding=False, truncation=True,
                            max_length=args.max_target_tokens - 1,
                            add_special_tokens=False)["input_ids"]
        encoded = [torch.tensor(ids + [tokenizer.eos_token_id], dtype=torch.long)
                   for ids in encoded]
        target_ids = torch.nn.utils.rnn.pad_sequence(
            encoded, batch_first=True, padding_value=tokenizer.pad_token_id).to("cuda")
        target_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(ids) for ids in encoded], batch_first=True,
            padding_value=0).to("cuda")
        pref = fixed_prefix.expand(target_ids.shape[0], -1)
        pref_emb = lm.get_input_embeddings()(pref)
        visual = resampler(ct_tokens.cuda(non_blocking=True).to(torch.bfloat16))
        target_emb = lm.get_input_embeddings()(target_ids)
        embeds = torch.cat([pref_emb, visual, target_emb], 1)
        prefix_mask = torch.ones(embeds.shape[0], pref_emb.shape[1] + visual.shape[1],
                                 dtype=torch.long, device="cuda")
        attention = torch.cat([prefix_mask, target_mask], 1)
        ignore = torch.full_like(prefix_mask, -100)
        labels = torch.cat([ignore, target_ids.masked_fill(target_mask == 0, -100)], 1)
        return embeds, attention, labels, visual

    @torch.no_grad()
    def validate():
        lm.eval(); resampler.eval(); losses = []
        for bi, (tokens, targets, path_labels) in enumerate(dev_loader):
            if bi >= args.eval_batches: break
            emb, att, lab, visual = make_batch(tokens, targets)
            lm_loss = lm(inputs_embeds=emb, attention_mask=att, labels=lab).loss
            valid = torch.isfinite(path_labels).cuda()
            y = torch.nan_to_num(path_labels, nan=0.0).cuda().to(torch.bfloat16)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                aux_classifier(visual.mean(1)), y, reduction="none",
                pos_weight=aux_pos_weight)
            aux_loss = (raw * valid).sum() / valid.sum().clamp_min(1)
            losses.append((lm_loss + args.aux_weight * aux_loss).float().item())
        lm.train(); resampler.train()
        return float(np.mean(losses))

    best = float("inf"); update = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        lm.train(); resampler.train()
        for step, (tokens, targets, path_labels) in enumerate(train_loader):
            emb, att, lab, visual = make_batch(tokens, targets)
            lm_loss = lm(inputs_embeds=emb, attention_mask=att, labels=lab).loss
            valid = torch.isfinite(path_labels).cuda()
            y = torch.nan_to_num(path_labels, nan=0.0).cuda().to(torch.bfloat16)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                aux_classifier(visual.mean(1)), y, reduction="none",
                pos_weight=aux_pos_weight)
            aux_loss = (raw * valid).sum() / valid.sum().clamp_min(1)
            total_loss = lm_loss + args.aux_weight * aux_loss
            loss = total_loss / args.grad_accum
            loss.backward()
            boundary = (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader)
            if boundary:
                torch.nn.utils.clip_grad_norm_(resampler.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); update += 1
                if update % 100 == 0:
                    print(f"epoch={epoch} update={update}/{total_updates} "
                          f"loss={total_loss.item():.4f} lm={lm_loss.item():.4f} "
                          f"aux={aux_loss.item():.4f}", flush=True)
        dev_loss = validate()
        print(f"epoch={epoch} dev_loss={dev_loss:.6f}", flush=True)
        state = {"resampler": resampler.state_dict(),
                 "aux_classifier": aux_classifier.state_dict(), "epoch": epoch,
                 "dev_loss": dev_loss, "args": vars(args),
                 "train_manifest": train_ds.manifest, "dev_manifest": dev_ds.manifest}
        torch.save(state, out / "resampler_last.pt")
        lm.save_pretrained(out / "adapter_last")
        if dev_loss < best:
            best = dev_loss
            torch.save(state, out / "resampler_best.pt")
            lm.save_pretrained(out / "adapter_best")
            tokenizer.save_pretrained(out / "adapter_best")
        (out / "metrics.json").write_text(json.dumps({"best_dev_loss": best, "last_dev_loss": dev_loss,
                                                       "epoch": epoch, "update": update}, indent=2) + "\n")


if __name__ == "__main__":
    main()

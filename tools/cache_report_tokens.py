"""Cache frozen CT/Q-Former tokens for report-generation experiments.

The cache is deliberately produced by the original ARC-CT environment.  The
large language model can then be trained in a modern Transformers environment
without repeatedly running the expensive 3-D encoder or changing its numerics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

from arcct.dataset import RACDatasetV4, rac_collate  # noqa: E402
from evaluate import build_model_for_eval  # noqa: E402


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean_report_field(value: str) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in ("nan", "none", "null") else text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--volume-list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--masks", default="")
    p.add_argument("--context", default="")
    p.add_argument("--demographics", default="")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--indication-mode", choices=("true", "none", "shuffled"), default="true")
    p.add_argument("--use-masks", action="store_true")
    args = p.parse_args()

    os.environ["RAC_EVAL_IND_MODE"] = args.indication_mode
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("shard_*.pt")):
        raise RuntimeError(f"output already contains shards: {out}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, tokenizer = build_model_for_eval(args.ckpt)
    clip = clip.to(device).eval()
    for param in clip.parameters():
        param.requires_grad_(False)

    ds = RACDatasetV4(
        args.data, args.reports, args.labels,
        mask_root=args.masks if args.masks and os.path.isdir(args.masks) else None,
        is_train=False, limit=args.limit, fail_fast=True,
        volume_list_path=args.volume_list, context_csv=args.context,
        demographics_csv=args.demographics,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=rac_collate,
                        pin_memory=True)
    qformer = getattr(clip, "qformer_module", None)
    if qformer is None:
        raise RuntimeError("checkpoint has no Q-Former; report generation needs query tokens")
    use_context = bool(getattr(clip, "use_context_qformer", False))

    pending: dict[str, list] = {k: [] for k in
                                ("tokens", "labels", "accession", "findings", "impression")}
    shard_idx = total = 0

    def flush() -> None:
        nonlocal shard_idx
        if not pending["tokens"]:
            return
        obj = {
            "tokens": torch.cat(pending["tokens"], 0).to(torch.float16),
            "labels": torch.cat(pending["labels"], 0).to(torch.float32),
            "accession": list(pending["accession"]),
            "findings": list(pending["findings"]),
            "impression": list(pending["impression"]),
        }
        torch.save(obj, out / f"shard_{shard_idx:05d}.pt")
        shard_idx += 1
        for value in pending.values():
            value.clear()

    with torch.inference_mode():
        for ct, _, findings, labels, masks, has_masks, accessions, ctx in tqdm.tqdm(loader, desc="cache"):
            ct = ct.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            has_masks = has_masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                feat = clip.visual_transformer.forward_spatial(ct)
                if use_context:
                    cfg = qformer.cfg
                    tok = qformer.context.tokenize(tokenizer, ctx["indication"],
                                                   cfg.max_ind_len, device)
                    bundle = qformer.context(
                        tok["input_ids"], tok["attention_mask"],
                        ctx["age_band"].to(device), ctx["sex"].to(device),
                        age_years=ctx["age_years"].to(device), age_mode=cfg.age_mode)
                    qout = qformer(feat, masks, has_masks, context=bundle,
                                   return_parts=True, suppress_mask=not args.use_masks)
                    qtoks = qout.tokens
                else:
                    result = qformer(feat, masks, has_masks, return_tokens=True,
                                     suppress_mask=not args.use_masks)
                    qtoks = result[1]
            qtoks = qtoks.float().cpu()
            labels = labels.cpu()
            for i, acc in enumerate(accessions):
                pending["tokens"].append(qtoks[i:i + 1])
                pending["labels"].append(labels[i:i + 1])
                pending["accession"].append(acc)
                pending["findings"].append(_clean_report_field(findings[i]))
                pending["impression"].append(_clean_report_field(ds.acc2impressions.get(acc, "")))
                total += 1
                if len(pending["tokens"]) >= args.shard_size:
                    flush()
    flush()
    manifest = {
        "format_version": 1, "n_samples": total, "n_shards": shard_idx,
        "token_shape": list(torch.load(out / "shard_00000.pt", map_location="cpu")["tokens"].shape[1:]),
        "checkpoint": str(Path(args.ckpt).resolve()), "checkpoint_sha256": _sha256(args.ckpt),
        "data": str(Path(args.data).resolve()), "reports": str(Path(args.reports).resolve()),
        "labels": str(Path(args.labels).resolve()), "volume_list": str(Path(args.volume_list).resolve()),
        "context_qformer": use_context, "indication_mode": args.indication_mode,
        "n_general_tokens": int(qformer.layout.n_gen) if use_context else int(
            torch.load(out / "shard_00000.pt", map_location="cpu")["tokens"].shape[1]),
        "mask_free": not args.use_masks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

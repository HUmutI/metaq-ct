"""Generate reports from cached CT tokens and compute text metrics.

F1-RadGraph is intentionally optional: the official RadGraph model is run as a
second pass over the saved predictions so generation and metric failures cannot
lose one another's output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from arcct.report_generator import CTTokenResampler  # noqa: E402
from train_report_generator import CacheDataset, SYSTEM, USER, collate, prefix_ids  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--token-mode", choices=("general", "all"), default="all")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--num-beams", type=int, default=4)
    args = p.parse_args()

    from peft import PeftModel
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    state = torch.load(Path(args.run) / "resampler_best.pt", map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(Path(args.run) / "adapter_best", local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                local_files_only=True, attn_implementation="sdpa")
    lm = PeftModel.from_pretrained(base, Path(args.run) / "adapter_best").cuda().eval()
    hidden = lm.get_input_embeddings().embedding_dim
    resampler = CTTokenResampler(output_dim=hidden).cuda().to(torch.bfloat16).eval()
    resampler.load_state_dict(state["resampler"])
    ds = CacheDataset(args.cache, args.limit, args.token_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate, pin_memory=True)
    fixed = prefix_ids(tokenizer, "cuda")
    shards = sorted(Path(args.cache).glob("shard_*.pt"))
    meta = [torch.load(x, map_location="cpu") for x in shards]
    accessions = sum((x["accession"] for x in meta), [])[:len(ds)]
    labels = torch.cat([x["labels"] for x in meta], 0)[:len(ds)].numpy()

    predictions, references = [], []
    with torch.inference_mode():
        for tokens, targets, _ in loader:
            b = tokens.shape[0]
            pref = fixed.expand(b, -1)
            p_emb = lm.get_input_embeddings()(pref)
            visual = resampler(tokens.cuda(non_blocking=True).to(torch.bfloat16))
            embeds = torch.cat([p_emb, visual], 1)
            attention = torch.ones(embeds.shape[:2], dtype=torch.long, device="cuda")
            seq = lm.generate(inputs_embeds=embeds, attention_mask=attention,
                              max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                              do_sample=False, early_stopping=True,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id)
            # Current Transformers returns only newly generated ids for inputs_embeds;
            # older releases prepend dummy prefix ids. Handle both explicitly.
            if seq.shape[1] > args.max_new_tokens:
                seq = seq[:, embeds.shape[1]:]
            predictions.extend(tokenizer.batch_decode(seq, skip_special_tokens=True))
            references.extend(targets)

    smooth = SmoothingFunction().method1
    bleu = {f"bleu_{n}": [] for n in range(1, 5)}
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = []
    for hyp, ref in zip(predictions, references):
        ht, rt = hyp.split(), ref.split()
        for n in range(1, 5):
            weights = tuple([1.0 / n] * n)
            bleu[f"bleu_{n}"].append(sentence_bleu([rt], ht, weights=weights,
                                                     smoothing_function=smooth))
        rouge_l.append(rouge.score(ref, hyp)["rougeL"].fmeasure)
    metrics = {k: float(np.mean(v)) for k, v in bleu.items()}
    metrics.update({"rouge_l": float(np.mean(rouge_l)), "n_samples": len(predictions),
                    "token_mode": args.token_mode, "cache": args.cache})
    records = [{"accession": a, "prediction": h, "reference": r,
                "labels": y.tolist()} for a, h, r, y in zip(accessions, predictions, references, labels)]
    (out / "predictions.json").write_text(json.dumps(records, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

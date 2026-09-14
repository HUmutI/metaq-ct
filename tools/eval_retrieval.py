"""Cross-modal report-to-volume retrieval on CT-RATE valid (Table T3).

Builds image and text latents for every CT-RATE valid scan, then computes:
- Image->Text and Text->Image R@1/5/10/50/100 with scan-level bootstrap CIs.
- Image->Image MeanJaccard@1/5/10/50 mirroring mps-ct volume_to_volume.py
  (relevance = label-set Jaccard between query and each top-K gallery scan;
  gallery = scans with >=1 positive label; query = all valid scans).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

from arcct.dataset import PATHOLOGIES, RACDatasetV4, rac_collate  # noqa: E402
from evaluate import (  # noqa: E402
    DATA_VALID, LABELS_VALID, MASK_VALID, REPORTS_VALID,
    PROMPT_LEN, build_model_for_eval,
)


KS = (1, 5, 10, 50, 100)
KS_I2I = (1, 5, 10, 50)


def _encode_text_latents(clip, tokenizer, texts, device, max_len: int):
    enc = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                    max_length=max_len).to(device)
    out = clip.text_transformer(enc["input_ids"], enc["attention_mask"])
    lat = clip.to_text_latent(out[0][:, 0, :])
    return F.normalize(lat, dim=-1)


def _recall_at_k(sim: torch.Tensor, ks=KS) -> dict[str, float]:
    n = sim.shape[0]
    ranks = (-sim).argsort(dim=1)
    diag = torch.arange(n, device=sim.device).unsqueeze(1)
    gt_pos = (ranks == diag).nonzero(as_tuple=True)[1]
    return {f"R@{k}": float((gt_pos < k).float().mean().item() * 100) for k in ks}


def _dsl_rerank(sim: torch.Tensor, temp: float) -> torch.Tensor:
    """Dual-softmax (CAMoE) transductive test-time reranking: row-softmax over the
    gallery times column-softmax over queries, damping hub items. Report as a
    separate '+DSL' row; it uses the full query-gallery matrix at inference."""
    return torch.softmax(sim / temp, dim=1) * torch.softmax(sim / temp, dim=0)


def _load_label_vectors(accessions: list[str], labels_csv: str) -> np.ndarray:
    df = pd.read_csv(labels_csv)
    key_col = "VolumeName" if "VolumeName" in df.columns else df.columns[0]
    df = df.set_index(key_col)
    cols = [c for c in PATHOLOGIES if c in df.columns]
    if len(cols) != len(PATHOLOGIES):
        missing = sorted(set(PATHOLOGIES) - set(cols))
        raise RuntimeError(f"[retrieval] valid labels CSV missing columns: {missing}")
    out = np.zeros((len(accessions), len(PATHOLOGIES)), dtype=np.int32)
    for i, acc in enumerate(accessions):
        if acc in df.index:
            row = df.loc[acc, cols]
            out[i] = (np.asarray(row, dtype=np.float32) > 0.5).astype(np.int32)
    return out


def _jaccard_labels(q: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Per mps-ct calc_similarity: |both 1| / (|both 1| + |xor|).

    q: [L] int, gs: [K, L] int. Returns [K] float in [0, 1] (0 if no shared positives).
    """
    inter = (gs * q[None, :]).sum(axis=1)
    union_or = ((gs + q[None, :]) > 0).sum(axis=1)
    denom = union_or
    return np.where(denom > 0, inter / np.maximum(denom, 1), 0.0)


def _mean_jaccard_at_k(img_lat: np.ndarray, labels: np.ndarray,
                       ks: tuple[int, ...]) -> dict[str, float]:
    """For each query scan, find top-K gallery (gallery=scans with >=1 pos label),
    compute Jaccard(query_labels, gallery_labels) per retrieved, mean over K.
    Then mean over queries. Matches mps-ct/volume_to_volume.py exactly.
    """
    n = img_lat.shape[0]
    img_t = torch.from_numpy(img_lat).float()
    img_t = F.normalize(img_t, dim=-1)
    gallery_mask = labels.sum(axis=1) > 0
    gallery_idx = np.where(gallery_mask)[0]
    gal_lat = img_t[gallery_idx]
    gal_lab = labels[gallery_idx]
    sim = img_t @ gal_lat.T
    self_mask = (np.arange(n)[:, None] == gallery_idx[None, :])
    sim_np = sim.numpy()
    sim_np[self_mask] = -1.0
    order = np.argsort(-sim_np, axis=1)
    out = {}
    max_k = max(ks)
    per_query = np.zeros((n, max_k), dtype=np.float32)
    for i in range(n):
        top = order[i, :max_k]
        per_query[i] = _jaccard_labels(labels[i], gal_lab[top])
    for k in ks:
        out[f"Jaccard@{k}"] = float(per_query[:, :k].mean(axis=1).mean() * 100.0)
    return out


def _bootstrap_jaccard(img_lat: np.ndarray, labels: np.ndarray,
                       n_bootstrap: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = img_lat.shape[0]
    vals = {f"Jaccard@{k}": [] for k in KS_I2I}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        res = _mean_jaccard_at_k(img_lat[idx], labels[idx], KS_I2I)
        for k in KS_I2I:
            vals[f"Jaccard@{k}"].append(res[f"Jaccard@{k}"])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in vals.items()}


def _bootstrap_recall(img_lat: np.ndarray, txt_lat: np.ndarray,
                      n_bootstrap: int, seed: int) -> dict[str, dict[str, tuple[float, float]]]:
    """Resample N scans (with replacement) per draw, recompute R@K for both directions."""
    rng = np.random.default_rng(seed)
    n = img_lat.shape[0]
    i2t = {f"R@{k}": [] for k in KS}
    t2i = {f"R@{k}": [] for k in KS}
    img = torch.from_numpy(img_lat)
    txt = torch.from_numpy(txt_lat)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        i = img[idx]
        t = txt[idx]
        sim_it = i @ t.T  # image -> text
        sim_ti = t @ i.T  # text -> image
        r_it = _recall_at_k(sim_it)
        r_ti = _recall_at_k(sim_ti)
        for k in KS:
            i2t[f"R@{k}"].append(r_it[f"R@{k}"])
            t2i[f"R@{k}"].append(r_ti[f"R@{k}"])
    def _ci(vals):
        a = np.asarray(vals)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    return {
        "image_to_text": {k: _ci(i2t[k]) for k in i2t},
        "text_to_image": {k: _ci(t2i[k]) for k in t2i},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("EVAL_CKPT", ""))
    ap.add_argument("--out_dir", default=os.environ.get("EVAL_RESULTS_DIR", ""))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "4")))
    ap.add_argument("--num_workers", type=int, default=int(os.environ.get("EVAL_NUM_WORKERS", "4")))
    ap.add_argument("--max_txt_len", type=int, default=int(os.environ.get("RAC_RETRIEVAL_TXT_LEN", "256")))
    ap.add_argument("--n_bootstrap", type=int, default=int(os.environ.get("EVAL_BOOTSTRAP", "1000")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("EVAL_VAL_LIMIT", "0")))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default=os.environ.get("EVAL_RUN_NAME", "seed0"))
    ap.add_argument("--labels_csv", default=os.environ.get("EVAL_LABELS_VALID", LABELS_VALID))
    ap.add_argument(
        "--embedding_mode",
        choices=("general", "conditioned", "mixed", "final", "backbone"),
        default=os.environ.get("RAC_RETRIEVAL_EMBEDDING", "general"),
        help=("Image representation used for retrieval. For a context Q-Former, "
              "general=z_gen (canonical shared embedding), conditioned=z_ind, "
              "mixed=normalize(z_gen+z_ind), and final=z_final. backbone is the "
              "legacy pre-Q-Former global-pool path."),
    )
    ap.add_argument(
        "--use_masks",
        action="store_true",
        help="Enable anatomy routing masks at inference (default: mask-free).",
    )
    args = ap.parse_args()
    assert args.ckpt and args.out_dir, "EVAL_CKPT and EVAL_RESULTS_DIR (or --ckpt/--out_dir) are required"
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, tokenizer = build_model_for_eval(args.ckpt)
    clip = clip.to(device).eval()

    ds = RACDatasetV4(
        DATA_VALID, REPORTS_VALID, LABELS_VALID,
        mask_root=MASK_VALID if os.path.isdir(MASK_VALID) else None,
        is_train=False, limit=args.limit, fail_fast=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=rac_collate, pin_memory=True)

    qformer = getattr(clip, "qformer_module", None)
    use_context = bool(getattr(clip, "use_context_qformer", False))
    use_anatomy = bool(getattr(clip, "use_anatomy_qformer", False))
    if args.embedding_mode in ("conditioned", "mixed") and not use_context:
        raise RuntimeError(f"embedding_mode={args.embedding_mode} requires a context Q-Former checkpoint")

    print(f"[retrieval] embedding={args.embedding_mode} mask_free={not args.use_masks} "
          f"context_qformer={use_context} anatomy_qformer={use_anatomy}")

    img_latents, txt_latents, accs_all = [], [], []
    with torch.no_grad():
        for ct, texts, _, _, masks_fine, has_masks, accessions, ctx in tqdm.tqdm(loader, desc="Encode img+txt"):
            ct = ct.to(device, non_blocking=True)
            masks_fine = masks_fine.to(device, non_blocking=True)
            has_masks = has_masks.to(device, non_blocking=True)
            feat_map = clip.visual_transformer.forward_spatial(ct)
            if args.embedding_mode == "backbone" or qformer is None:
                raw = clip.visual_transformer.global_pool(feat_map)
                image_repr = clip.to_visual_latent(raw)
            elif use_context:
                cfg = qformer.cfg
                tok = qformer.context.tokenize(
                    tokenizer, ctx["indication"], cfg.max_ind_len, device)
                bundle = qformer.context(
                    tok["input_ids"], tok["attention_mask"],
                    ctx["age_band"].to(device), ctx["sex"].to(device),
                    age_years=ctx["age_years"].to(device),
                    age_mode=cfg.age_mode,
                )
                out = qformer(
                    feat_map, masks_fine, has_masks, context=bundle,
                    return_parts=True, suppress_mask=not args.use_masks,
                )
                if args.embedding_mode == "general":
                    image_repr = out.z_gen
                elif args.embedding_mode == "conditioned":
                    image_repr = out.z_ind
                elif args.embedding_mode == "mixed":
                    image_repr = out.z_gen + out.z_ind
                else:
                    image_repr = out.z_final
            elif use_anatomy:
                image_repr = qformer(
                    feat_map, masks_fine, has_masks,
                    suppress_mask=not args.use_masks,
                )
            else:
                image_repr = qformer(feat_map)
            img_lat = F.normalize(image_repr, dim=-1)
            txt_lat = _encode_text_latents(clip, tokenizer, list(texts), device, args.max_txt_len)
            img_latents.append(img_lat.float().cpu())
            txt_latents.append(txt_lat.float().cpu())
            accs_all.extend(accessions)

    img_lat = torch.cat(img_latents, 0)
    txt_lat = torch.cat(txt_latents, 0)
    n = img_lat.shape[0]
    print(f"[retrieval] N={n}  img_lat={tuple(img_lat.shape)} txt_lat={tuple(txt_lat.shape)}")

    sim_it = img_lat @ txt_lat.T
    sim_ti = txt_lat @ img_lat.T
    r_it = _recall_at_k(sim_it)
    r_ti = _recall_at_k(sim_ti)

    diag_rank = (-sim_it).argsort(dim=1)
    gt_pos = (diag_rank == torch.arange(n).unsqueeze(1)).nonzero(as_tuple=True)[1]
    mean_rank_it = float((gt_pos.float() + 1).mean().item())
    diag_rank = (-sim_ti).argsort(dim=1)
    gt_pos = (diag_rank == torch.arange(n).unsqueeze(1)).nonzero(as_tuple=True)[1]
    mean_rank_ti = float((gt_pos.float() + 1).mean().item())

    dsl_temp = float(os.environ.get("RAC_DSL_TEMP", "0.02"))
    r_it_dsl = _recall_at_k(_dsl_rerank(sim_it, dsl_temp))
    r_ti_dsl = _recall_at_k(_dsl_rerank(sim_ti, dsl_temp))

    print(f"\n{'Direction':<22}" + "  ".join(f"R@{k:>3}" for k in KS) + "  mean_rank")
    print(f"{'image -> text':<22}" + "  ".join(f"{r_it[f'R@{k}']:5.2f}" for k in KS) + f"  {mean_rank_it:6.1f}")
    print(f"{'  + DSL':<22}" + "  ".join(f"{r_it_dsl[f'R@{k}']:5.2f}" for k in KS))
    print(f"{'text -> image':<22}" + "  ".join(f"{r_ti[f'R@{k}']:5.2f}" for k in KS) + f"  {mean_rank_ti:6.1f}")
    print(f"{'  + DSL':<22}" + "  ".join(f"{r_ti_dsl[f'R@{k}']:5.2f}" for k in KS))

    if args.n_bootstrap > 0:
        print(f"[retrieval] computing {args.n_bootstrap} bootstrap resamples ...")
        ci = _bootstrap_recall(img_lat.numpy(), txt_lat.numpy(), args.n_bootstrap, args.seed)
    else:
        print("[retrieval] bootstrap disabled")
        ci = {"image_to_text": {}, "text_to_image": {}}

    print("[retrieval] computing I->I MeanJaccard@K (mps-ct convention) ...")
    labels = _load_label_vectors(accs_all, args.labels_csv)
    j_ii = _mean_jaccard_at_k(img_lat.numpy(), labels, KS_I2I)
    j_ii_ci = (_bootstrap_jaccard(img_lat.numpy(), labels, args.n_bootstrap, args.seed)
               if args.n_bootstrap > 0 else {})
    print(f"{'image -> image':<22}" + "  ".join(f"{j_ii[f'Jaccard@{k}']:5.2f}" for k in KS_I2I))

    report = {
        "ckpt": args.ckpt,
        "n_samples": int(n),
        "n_bootstrap": int(args.n_bootstrap),
        "embedding_mode": args.embedding_mode,
        "mask_free": not args.use_masks,
        "bootstrap_unit": "scan",
        "image_to_text": {**r_it, "mean_rank": mean_rank_it,
                          "ci": {k: {"lo": ci["image_to_text"][k][0], "hi": ci["image_to_text"][k][1]}
                                 for k in r_it if k in ci["image_to_text"]}},
        "text_to_image": {**r_ti, "mean_rank": mean_rank_ti,
                          "ci": {k: {"lo": ci["text_to_image"][k][0], "hi": ci["text_to_image"][k][1]}
                                 for k in r_ti if k in ci["text_to_image"]}},
        "dsl_temp": dsl_temp,
        "image_to_text_dsl": {**r_it_dsl, "note": "dual-softmax transductive test-time rerank"},
        "text_to_image_dsl": {**r_ti_dsl, "note": "dual-softmax transductive test-time rerank"},
        "image_to_image": {**j_ii,
                           "ci": {k: {"lo": j_ii_ci[k][0], "hi": j_ii_ci[k][1]} for k in j_ii_ci},
                           "metric_definition": ("mean Jaccard@K of multilabel positive sets; "
                                                 "mirrors mps-ct/volume_to_volume.py calc_similarity. "
                                                 "gallery = scans with >=1 positive label; query excludes self.")},
    }
    out_json = Path(args.out_dir) / f"retrieval_{args.run_name}.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[retrieval] saved {out_json}")

    np.savez_compressed(Path(args.out_dir) / f"retrieval_latents_{args.run_name}.npz",
                        img_lat=img_lat.numpy().astype(np.float32),
                        txt_lat=txt_lat.numpy().astype(np.float32),
                        accessions=np.asarray(accs_all),
                        labels=labels.astype(np.int8))

    txt_path = Path(args.out_dir) / f"retrieval_{args.run_name}_table.txt"
    with txt_path.open("w") as f:
        f.write(f"ckpt={args.ckpt}\nN={n}\n\n")
        f.write(f"{'Direction':<22}" + "  ".join(f"R@{k:>3}" for k in KS) + "  mean_rank\n")
        f.write(f"{'image -> text':<22}" + "  ".join(f"{r_it[f'R@{k}']:5.2f}" for k in KS) + f"  {mean_rank_it:6.1f}\n")
        if args.n_bootstrap > 0:
            for k in KS:
                lo, hi = ci["image_to_text"][f"R@{k}"]
                f.write(f"  CI R@{k:<3}: [{lo:.2f}, {hi:.2f}]\n")
        f.write(f"{'text -> image':<22}" + "  ".join(f"{r_ti[f'R@{k}']:5.2f}" for k in KS) + f"  {mean_rank_ti:6.1f}\n")
        if args.n_bootstrap > 0:
            for k in KS:
                lo, hi = ci["text_to_image"][f"R@{k}"]
                f.write(f"  CI R@{k:<3}: [{lo:.2f}, {hi:.2f}]\n")
        f.write(f"\n{'image -> image':<22}" + "  ".join(f"Jacc@{k:<3}" for k in KS_I2I) + "  (mps-ct convention)\n")
        f.write(f"{'mean Jaccard@K':<22}" + "  ".join(f"{j_ii[f'Jaccard@{k}']:7.2f}" for k in KS_I2I) + "\n")
        if args.n_bootstrap > 0:
            for k in KS_I2I:
                lo, hi = j_ii_ci[f"Jaccard@{k}"]
                f.write(f"  CI Jaccard@{k:<3}: [{lo:.2f}, {hi:.2f}]\n")
    print(f"[retrieval] saved {txt_path}")


if __name__ == "__main__":
    main()

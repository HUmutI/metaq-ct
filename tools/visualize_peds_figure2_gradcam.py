#!/usr/bin/env python3
"""Paper-style pediatric Grad-CAM comparison (novel vs matched CT-only).

The layout deliberately follows ARC-CT Figure 2: one study, four known-positive
findings, one row per model, smooth fixed-alpha ``jet`` Grad-CAM overlays, and
no probabilities, routing contours, or debugging panels in the figure.  The
models still run with their saved evaluation recipes and normal anatomy routing;
that distinction is recorded in the private manifest rather than hidden by a
misleading "mask-free" caption.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import string
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arcct.dataset import PATHOLOGIES, RACDatasetV4, rac_collate  # noqa: E402
from tools.evaluate import TEMPERATURE, build_model_for_eval, encode_prompts  # noqa: E402

REFERENCE_COMMIT = "cde75a956aa719c89cbbf3624df5dd95a497f402"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--novel-checkpoint", required=True)
    p.add_argument("--control-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--case", action="append", required=True,
                   metavar="ACCESSION|FINDING|FINDING|FINDING|FINDING")
    p.add_argument("--data", default=os.environ.get("RAC_DATA_VALID", ""))
    p.add_argument("--reports", default=os.environ.get("RAC_REPORTS_VALID", ""))
    p.add_argument("--labels", default=os.environ.get("RAC_LABELS_VALID", ""))
    p.add_argument("--masks", default=os.environ.get("RAC_MASK_VALID", ""))
    p.add_argument("--context", default=os.environ.get("RAC_CONTEXT_CSV", ""))
    p.add_argument("--demographics", default=os.environ.get("RAC_DEMOGRAPHICS_CSV", ""))
    p.add_argument("--volume-list", default=os.environ.get("RAC_VOLUME_LIST_VALID", ""))
    p.add_argument("--exclude", default=os.environ.get("RAC_VOLUME_EXCLUDE", ""))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def parse_cases(raw_cases):
    result = []
    for raw in raw_cases:
        fields = [x.strip() for x in raw.split("|")]
        if len(fields) != 5:
            raise ValueError(f"case needs accession plus exactly four findings: {raw!r}")
        accession, findings = fields[0], fields[1:]
        if not accession.endswith(".nii.gz"):
            accession += ".nii.gz"
        unknown = [x for x in findings if x not in PATHOLOGIES]
        if unknown:
            raise ValueError(f"unknown findings {unknown}; schema={PATHOLOGIES}")
        result.append((accession, findings))
    return result


def context_bundle(qf, tokenizer, ctx, device):
    tok = qf.context.tokenize(tokenizer, list(ctx["indication"]),
                              qf.cfg.max_ind_len, device)
    return qf.context(
        tok["input_ids"], tok["attention_mask"],
        ctx["age_band"].to(device), ctx["sex"].to(device),
        age_years=ctx["age_years"].to(device), age_mode=qf.cfg.age_mode,
    )


def gradcam(clip, tokenizer, batch, class_index, pos, neg):
    """Grad-CAM of the exact prompt decision margin with normal inference."""
    ct, _, _, _, masks, has_masks, _, ctx = batch
    device = next(clip.parameters()).device
    ct = ct.to(device)
    masks = masks.to(device)
    has_masks = has_masks.to(device)
    suppress = bool(clip.eval_recipe.get("suppress_routing_mask", False))
    no_l2 = bool(clip.eval_recipe.get("no_l2_prompt", False))

    # We need d(score)/d(feature-map), not parameter gradients. Detaching here
    # preserves the exact derivative with respect to the feature map and avoids
    # storing the full 3-D encoder backward graph.
    with torch.no_grad():
        fmap = clip.visual_transformer.forward_spatial(ct)
    fmap = fmap.detach().requires_grad_(True)
    qf = clip.qformer_module
    if getattr(clip, "use_context_qformer", False):
        bundle = context_bundle(qf, tokenizer, ctx, device)
        out = qf(fmap, masks, has_masks, context=bundle, return_parts=True,
                 suppress_mask=suppress)
        if qf.cfg.fusion == "class_logit":
            probability = qf.class_prompt_probs(
                out, pos, neg, TEMPERATURE, normalize=not no_l2)[0, class_index]
            # logit(p) gives a better-scaled derivative of the same decision.
            score = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        else:
            z = out.z_final if no_l2 else F.normalize(out.z_final, dim=-1)
            score = ((z[0] * pos[class_index]).sum()
                     - (z[0] * neg[class_index]).sum()) / TEMPERATURE
            probability = torch.sigmoid(score)
    else:
        z = qf(fmap, masks, has_masks, suppress_mask=suppress)
        z = z if no_l2 else F.normalize(z, dim=-1)
        score = ((z[0] * pos[class_index]).sum()
                 - (z[0] * neg[class_index]).sum()) / TEMPERATURE
        probability = torch.sigmoid(score)

    grad = torch.autograd.grad(score, fmap, retain_graph=False)[0]
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * fmap.detach()).sum(dim=1).relu()
    cam = F.interpolate(cam[:, None], size=ct.shape[2:], mode="trilinear",
                        align_corners=False)[0, 0].float().cpu().numpy()
    # Match the public Figure-2 implementation's smoothness exactly.
    cam = gaussian_filter(cam, sigma=(1.2, 3.0, 3.0))
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    return cam, float(probability.detach().cpu())


def upsample(x, scale=4, *, clamp=False):
    t = torch.from_numpy(np.ascontiguousarray(x))[None, None].float()
    y = F.interpolate(t, scale_factor=scale, mode="bicubic",
                      align_corners=False)[0, 0].numpy()
    return np.clip(y, 0, 1) if clamp else y


def compute_model_maps(checkpoint, cases, tokenizer_batches, role):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, tokenizer = build_model_for_eval(checkpoint)
    clip = clip.to(device).eval()
    is_context = bool(getattr(clip, "use_context_qformer", False))
    if role == "novel" and not is_context:
        raise RuntimeError("novel checkpoint did not reconstruct ContextQFormer")
    if role == "control" and is_context:
        raise RuntimeError("control checkpoint unexpectedly reconstructed ContextQFormer")
    pos, neg = encode_prompts(clip, tokenizer, device)
    pos, neg = pos.detach(), neg.detach()
    result = {}
    for accession, findings in cases:
        result.setdefault(accession, {})
        batch = tokenizer_batches[accession]
        for finding in findings:
            # The same volume may be proposed with multiple four-finding sets.
            # Reuse maps already computed for an earlier proposal.
            if finding in result[accession]:
                continue
            j = PATHOLOGIES.index(finding)
            cam, probability = gradcam(clip, tokenizer, batch, j, pos, neg)
            result[accession][finding] = {"cam": cam, "probability": probability}
            print(f"[figure2] {role} {accession} {finding}: p={probability:.4f}", flush=True)
    recipe = dict(clip.eval_recipe)
    del clip, tokenizer, pos, neg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, recipe


def render_case(out_path, accession, findings, batch, novel, control, dpi):
    ct = batch[0][0, 0].float().cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.65))
    rows = [("Ours", novel), ("CT-only", control)]
    for row, (row_name, maps) in enumerate(rows):
        for col, finding in enumerate(findings):
            ax = axes[row, col]
            cam = maps[accession][finding]["cam"]
            # Same deterministic peak-axial rule as the reference repository.
            z = int(cam.mean(axis=(1, 2)).argmax())
            ct_slice = upsample(ct[z], clamp=True)
            heat_slice = upsample(cam[z], clamp=True)
            ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1,
                      interpolation="bicubic")
            ax.imshow(heat_slice, cmap="jet", alpha=0.45, vmin=0, vmax=1,
                      interpolation="bilinear")
            if row == 0:
                ax.set_title(finding, fontsize=11.5, pad=8)
            panel = string.ascii_lowercase[row * 4 + col]
            ax.text(0.035, 0.94, f"({panel})", transform=ax.transAxes,
                    color="white", fontsize=11, fontweight="bold",
                    va="top", ha="left",
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "black",
                          "edgecolor": "none", "alpha": 0.88})
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.text(0.022, 0.705, rows[0][0], rotation=90, va="center", ha="center",
             fontsize=12.5, fontweight="bold")
    fig.text(0.022, 0.285, rows[1][0], rotation=90, va="center", ha="center",
             fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(left=0.052, right=0.995, top=0.925, bottom=0.015,
                        wspace=0.025, hspace=0.045)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03,
                facecolor="white")
    plt.close(fig)


def main():
    a = parse_args()
    cases = parse_cases(a.case)
    required = [a.novel_checkpoint, a.control_checkpoint, a.data, a.reports,
                a.labels, a.masks, a.context, a.demographics, a.volume_list]
    missing = [x for x in required if not x or not Path(x).exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = RACDatasetV4(
        a.data, a.reports, a.labels, mask_root=a.masks, is_train=False,
        fail_fast=True, volume_list_path=a.volume_list, context_csv=a.context,
        demographics_csv=a.demographics, volume_exclude_path=a.exclude,
    )
    by_accession = {sample[1]: i for i, sample in enumerate(ds.samples)}
    batches = {}
    for accession, findings in cases:
        if accession not in by_accession:
            raise KeyError(f"case absent from validation dataset: {accession}")
        subset = torch.utils.data.Subset(ds, [by_accession[accession]])
        batch = next(iter(DataLoader(subset, batch_size=1, num_workers=a.workers,
                                     collate_fn=rac_collate)))
        labels = batch[3][0]
        not_positive = [x for x in findings if int(labels[PATHOLOGIES.index(x)]) != 1]
        if not_positive:
            raise RuntimeError(f"{accession}: requested findings are not GT positive: {not_positive}")
        if not bool(batch[5][0]):
            raise RuntimeError(f"{accession}: registered anatomy mask is unavailable")
        batches[accession] = batch

    novel, novel_recipe = compute_model_maps(
        a.novel_checkpoint, cases, batches, "novel")
    control, control_recipe = compute_model_maps(
        a.control_checkpoint, cases, batches, "control")

    manifest = {
        "reference_style": "ARC-CT Figure 2",
        "reference_code_commit": REFERENCE_COMMIT,
        "novel_checkpoint": str(Path(a.novel_checkpoint).resolve()),
        "control_checkpoint": str(Path(a.control_checkpoint).resolve()),
        "novel_recipe": novel_recipe,
        "control_recipe": control_recipe,
        "routing_masks_visible": False,
        "raw_indication_saved": False,
        "cases": [],
    }
    for number, (accession, findings) in enumerate(cases, 1):
        out_path = out_dir / f"candidate_{number:02d}_figure2_style.png"
        render_case(out_path, accession, findings, batches[accession], novel,
                    control, a.dpi)
        manifest["cases"].append({
            "candidate": number,
            "accession": accession,
            "ground_truth": {x: 1 for x in findings},
            "findings": findings,
            "novel_probabilities": {x: novel[accession][x]["probability"] for x in findings},
            "control_probabilities": {x: control[accession][x]["probability"] for x in findings},
            "figure": out_path.name,
        })
        print(f"[figure2] wrote {out_path}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

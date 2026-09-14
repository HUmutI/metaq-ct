#!/usr/bin/env python3
"""Qualitative heatmaps for the metadata-conditioned pediatric Q-Former.

This adapts the visualization strategy in HUmutI/heatmap_codes (commit
cde75a956aa719c89cbbf3624df5dd95a497f402) to the *actual* ContextQFormer.
The portable repository only knows the legacy 18-class query layout; using it
directly would mis-index the 23-class pediatric conditioned slots.

For each selected (study, pathology) pair the script renders a common axial
slice with:

  1. CT only;
  2. metadata-isolated general pathology-query cross-attention;
  3. conditioned pathology-query attention with correct indication/age/sex;
  4. conditioned attention after blanking indication but retaining age/sex;
  5. the signed correct-minus-blank attention change; and
  6. Grad-CAM for the final conditioned prompt score.

The anatomy mask is used exactly as it is during primary-model inference and
is outlined in white.  Consequently, these panels describe evidence *within*
the registered routing region; they are not a free-localization benchmark.
No report or indication text is written into figures or the output manifest.
Generated images are hospital-governed derived data and must remain off Git.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arcct.context import AGE_BAND_NAMES, NO_INDICATION  # noqa: E402
from arcct.dataset import (  # noqa: E402
    FINE_LABEL_NAMES,
    PATHOLOGIES,
    PATHOLOGY_FINE_ORGANS,
    RACDatasetV4,
    rac_collate,
)
from tools.evaluate import (  # noqa: E402
    TEMPERATURE,
    build_model_for_eval,
    encode_prompts,
    prompt_probs,
)

HEATMAP_SOURCE_COMMIT = "cde75a956aa719c89cbbf3624df5dd95a497f402"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data", default=os.environ.get("RAC_DATA_VALID", ""))
    p.add_argument("--reports", default=os.environ.get("RAC_REPORTS_VALID", ""))
    p.add_argument("--labels", default=os.environ.get("RAC_LABELS_VALID", ""))
    p.add_argument("--masks", default=os.environ.get("RAC_MASK_VALID", ""))
    p.add_argument("--context", default=os.environ.get("RAC_CONTEXT_CSV", ""))
    p.add_argument("--demographics", default=os.environ.get("RAC_DEMOGRAPHICS_CSV", ""))
    p.add_argument("--volume-list", default=os.environ.get("RAC_VOLUME_LIST_VALID", ""))
    p.add_argument("--exclude", default=os.environ.get("RAC_VOLUME_EXCLUDE", ""))
    p.add_argument("--predictions", default="",
                   help="NPZ used only to choose high-confidence true-positive candidates")
    p.add_argument("--pathologies", default=(
        "Pneumothorax,Pleural effusion,Bronchiectasis,"
        "Pulmonary metastases,Bone lesion or fracture"))
    p.add_argument("--case", action="append", default=[], metavar="ACCESSION:PATHOLOGY",
                   help="Explicit case; repeat for multiple panels")
    p.add_argument("--max-cases", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--show-accessions", action="store_true",
                   help="Put de-identified volume IDs in PNG titles (off by default)")
    return p.parse_args()


def _validate_args(a: argparse.Namespace) -> None:
    required = {
        "checkpoint": a.checkpoint, "data": a.data, "reports": a.reports,
        "labels": a.labels, "masks": a.masks, "context": a.context,
        "demographics": a.demographics, "volume-list": a.volume_list,
    }
    missing = [f"{k}={v!r}" for k, v in required.items() if not v or not Path(v).exists()]
    if missing:
        raise FileNotFoundError("missing heatmap input(s): " + ", ".join(missing))


def _context_bundle(qf, tokenizer, ctx, device, *, blank_indication: bool):
    indications = ([NO_INDICATION] * len(ctx["indication"])
                   if blank_indication else list(ctx["indication"]))
    tok = qf.context.tokenize(tokenizer, indications, qf.cfg.max_ind_len, device)
    return qf.context(
        tok["input_ids"], tok["attention_mask"],
        ctx["age_band"].to(device), ctx["sex"].to(device),
        age_years=ctx["age_years"].to(device), age_mode=qf.cfg.age_mode,
    )


def _run_with_attention(clip, tokenizer, ct, masks, has_masks, ctx, *, blank=False):
    qf = clip.qformer_module
    bundle = _context_bundle(qf, tokenizer, ctx, ct.device, blank_indication=blank)
    feat = clip.visual_transformer.forward_spatial(ct)
    attn_out, attn = qf(
        feat, masks, has_masks, context=bundle, return_parts=True,
        return_attn=True,
        suppress_mask=bool(clip.eval_recipe.get("suppress_routing_mask", False)),
    )
    # Requesting weights can select a different PyTorch attention kernel.  Use
    # the ordinary evaluator path for all displayed probabilities and assert
    # that the analysis path remains numerically equivalent.
    standard_out = qf(
        feat, masks, has_masks, context=bundle, return_parts=True,
        suppress_mask=bool(clip.eval_recipe.get("suppress_routing_mask", False)),
    )
    max_delta = float((standard_out.z_final - attn_out.z_final).abs().max())
    if max_delta > 1e-4:
        raise RuntimeError(f"return-attention path changed z_final by {max_delta:.3e}")
    return standard_out, attn, feat.shape[-3:], max_delta


def _probabilities(qf, out, pos, neg, no_l2: bool):
    if qf.cfg.fusion == "class_logit":
        conditioned = qf.class_prompt_probs(
            out, pos, neg, TEMPERATURE, normalize=not no_l2)
    else:
        conditioned = prompt_probs(out.z_final, pos, neg, no_l2=no_l2)
    general = prompt_probs(out.z_gen, pos, neg, no_l2=no_l2)
    return general, conditioned


def _attention_volume(attn, slot: int, spatial, target_shape) -> np.ndarray:
    x = attn[0, :, slot].mean(dim=0).reshape(*spatial)
    x = F.interpolate(x[None, None], size=target_shape, mode="trilinear",
                      align_corners=False)[0, 0]
    x = x.clamp_min(0)
    x = x / x.sum().clamp_min(1e-12)
    return x.detach().float().cpu().numpy()


def _display_scale(x: np.ndarray) -> np.ndarray:
    positive = x[x > 0]
    if positive.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(positive, [2, 99.5])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _gradcam(clip, tokenizer, ct, masks, has_masks, ctx, class_idx, pos, neg):
    """Grad-CAM of the exact final conditioned prompt margin."""
    qf = clip.qformer_module
    no_l2 = bool(clip.eval_recipe.get("no_l2_prompt", False))
    bundle = _context_bundle(qf, tokenizer, ctx, ct.device, blank_indication=False)
    clip.zero_grad(set_to_none=True)
    qf.zero_grad(set_to_none=True)
    with torch.enable_grad():
        feat = clip.visual_transformer.forward_spatial(ct)
        feat.retain_grad()
        out = qf(
            feat, masks, has_masks, context=bundle, return_parts=True,
            suppress_mask=bool(clip.eval_recipe.get("suppress_routing_mask", False)),
        )
        if qf.cfg.fusion == "class_logit":
            score = qf.class_prompt_probs(
                out, pos, neg, TEMPERATURE, normalize=not no_l2)[0, class_idx]
        else:
            z = out.z_final if no_l2 else F.normalize(out.z_final, dim=-1)
            score = ((z[0] * pos[class_idx]).sum()
                     - (z[0] * neg[class_idx]).sum()) / TEMPERATURE
        score.backward()
        grad = feat.grad
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * feat.detach()).sum(dim=1).relu()
        cam = F.interpolate(cam[:, None], size=ct.shape[2:], mode="trilinear",
                            align_corners=False)[0, 0]
    clip.zero_grad(set_to_none=True)
    qf.zero_grad(set_to_none=True)
    return cam.float().cpu().numpy()


def _allowed_region(mask: np.ndarray, pathology: str) -> np.ndarray | None:
    labels = PATHOLOGY_FINE_ORGANS.get(pathology, [])
    if not labels:
        return None
    return np.isin(mask, labels)


def _draw_overlay(ax, ct_slice, heat_slice, title, *, cmap="turbo", outline=None,
                  signed=False):
    ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
    if signed:
        vmax = float(np.max(np.abs(heat_slice)))
        if vmax > 0:
            ax.imshow(heat_slice, cmap=cmap, alpha=0.58, vmin=-vmax, vmax=vmax)
    else:
        disp = _display_scale(heat_slice)
        ax.imshow(disp, cmap=cmap, alpha=0.58 * disp, vmin=0, vmax=1)
    if outline is not None and np.any(outline):
        ax.contour(outline.astype(float), levels=[0.5], colors="white", linewidths=0.55)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _render(out_path: Path, case_label: str, pathology: str, ct, mask,
            general, correct, blank, cam, probs, dpi: int) -> None:
    # One shared slice prevents each map from choosing a flattering plane.
    z = int(np.argmax(correct.sum(axis=(1, 2))))
    region = _allowed_region(mask, pathology)
    outline = region[z] if region is not None else None
    delta = correct - blank

    fig, axes = plt.subplots(1, 6, figsize=(17.2, 3.15), constrained_layout=True)
    axes[0].imshow(ct[z], cmap="gray", vmin=0, vmax=1)
    if outline is not None and np.any(outline):
        axes[0].contour(outline.astype(float), levels=[0.5], colors="white", linewidths=0.55)
    axes[0].set_title(f"CT (axial {z})", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    _draw_overlay(axes[1], ct[z], general[z], "General query", outline=outline)
    _draw_overlay(axes[2], ct[z], correct[z], "Conditioned: correct", outline=outline)
    _draw_overlay(axes[3], ct[z], blank[z], "Conditioned: blank indication", outline=outline)
    _draw_overlay(axes[4], ct[z], delta[z], "Correct minus blank", cmap="coolwarm",
                  outline=outline, signed=True)
    _draw_overlay(axes[5], ct[z], cam[z], "Final-score Grad-CAM", outline=outline)
    fig.suptitle(
        f"{case_label} — {pathology} (GT positive)  |  "
        f"Pgeneral={probs['general']:.3f}, Pcorrect={probs['correct']:.3f}, "
        f"Pblank={probs['blank']:.3f}", fontsize=10,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _parse_explicit(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"--case must be ACCESSION:PATHOLOGY, got {raw!r}")
        accession, pathology = raw.split(":", 1)
        accession = accession.strip()
        if not accession.endswith(".nii.gz"):
            accession += ".nii.gz"
        pathology = pathology.strip()
        if pathology not in PATHOLOGIES:
            raise ValueError(f"unknown pathology in --case: {pathology!r}")
        out.append((accession, pathology))
    return out


def _auto_cases(ds, pred_path: str, targets: list[str], max_cases: int):
    """Pick one high-confidence true positive per target, with unique patients."""
    samples = {s[1] for s in ds.samples}
    selected, used_patients = [], set()
    if pred_path:
        z = np.load(pred_path, allow_pickle=True)
        pred_names = [str(x) for x in z["pathologies"]]
        pred_index = {name: i for i, name in enumerate(pred_names)}
        for pathology in targets:
            if pathology not in pred_index:
                continue
            j = pred_index[pathology]
            order = np.argsort(-z["pred"][:, j])
            for i in order:
                acc = str(z["accessions"][i])
                patient = "_".join(acc.split("_")[:2])
                if (z["true"][i, j] == 1 and acc in samples
                        and patient not in used_patients):
                    selected.append((acc, pathology))
                    used_patients.add(patient)
                    break
            if len(selected) >= max_cases:
                break
    else:
        for pathology in targets:
            j = PATHOLOGIES.index(pathology)
            for sample in ds.samples:
                if sample[4][j] == 1:
                    patient = "_".join(sample[1].split("_")[:2])
                    if patient not in used_patients:
                        selected.append((sample[1], pathology))
                        used_patients.add(patient)
                        break
            if len(selected) >= max_cases:
                break
    return selected


def main() -> int:
    a = parse_args()
    _validate_args(a)
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[heatmap] WARNING: CUDA unavailable; inference will be slow", flush=True)

    clip, tokenizer = build_model_for_eval(a.checkpoint)
    clip = clip.to(device).eval()
    qf = getattr(clip, "qformer_module", None)
    if qf is None or not getattr(clip, "use_context_qformer", False):
        raise RuntimeError("checkpoint did not reconstruct a ContextQFormer")
    qf = qf.to(device).eval()
    if qf.layout.n_path != len(PATHOLOGIES):
        raise RuntimeError(f"layout has {qf.layout.n_path} pathologies, schema has {len(PATHOLOGIES)}")
    pos, neg = encode_prompts(clip, tokenizer, device)
    no_l2 = bool(clip.eval_recipe.get("no_l2_prompt", False))

    ds = RACDatasetV4(
        a.data, a.reports, a.labels, mask_root=a.masks, is_train=False,
        fail_fast=True, volume_list_path=a.volume_list, context_csv=a.context,
        demographics_csv=a.demographics, volume_exclude_path=a.exclude,
    )
    explicit = _parse_explicit(a.case)
    targets = [x.strip() for x in a.pathologies.split(",") if x.strip()]
    cases = explicit or _auto_cases(ds, a.predictions, targets, a.max_cases)
    if not cases:
        raise RuntimeError("no qualifying heatmap cases were found")
    wanted = {x[0] for x in cases}
    ds.samples = [s for s in ds.samples if s[1] in wanted]
    by_accession = {s[1]: i for i, s in enumerate(ds.samples)}

    manifest = {
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "heatmap_source_commit": HEATMAP_SOURCE_COMMIT,
        "selection": "explicit" if explicit else "top-confidence true-positive per target",
        "routing_mask_used": not bool(clip.eval_recipe.get("suppress_routing_mask", False)),
        "raw_indication_saved": False,
        "cases": [],
    }
    for case_no, (accession, pathology) in enumerate(cases, 1):
        if accession not in by_accession:
            print(f"[heatmap] skip missing dataset accession {accession}", flush=True)
            continue
        one = torch.utils.data.Subset(ds, [by_accession[accession]])
        batch = next(iter(DataLoader(one, batch_size=1, num_workers=a.workers,
                                     collate_fn=rac_collate)))
        ct, _, _, labels, masks, has_masks, accessions, ctx = batch
        ct = ct.to(device); masks = masks.to(device); has_masks = has_masks.to(device)
        j = PATHOLOGIES.index(pathology)
        if int(labels[0, j].item()) != 1:
            raise RuntimeError(f"selected pair is not GT-positive: {accession} / {pathology}")
        if not bool(has_masks[0]):
            raise RuntimeError(f"selected case has no anatomy mask: {accession}")

        with torch.no_grad():
            true_out, true_attn, spatial, true_attn_delta = _run_with_attention(
                clip, tokenizer, ct, masks, has_masks, ctx, blank=False)
            blank_out, blank_attn, blank_spatial, blank_attn_delta = _run_with_attention(
                clip, tokenizer, ct, masks, has_masks, ctx, blank=True)
            if spatial != blank_spatial:
                raise RuntimeError("true/blank feature grids differ")
            gen_prob, true_prob = _probabilities(qf, true_out, pos, neg, no_l2)
            _, blank_prob = _probabilities(qf, blank_out, pos, neg, no_l2)

        target_shape = tuple(ct.shape[2:])
        gen_slot = int(qf.path_gen_index[j])
        cond_slot = int(qf.path_cond_index[j])
        general_hm = _attention_volume(true_attn, gen_slot, spatial, target_shape)
        true_hm = _attention_volume(true_attn, cond_slot, spatial, target_shape)
        blank_hm = _attention_volume(blank_attn, cond_slot, spatial, target_shape)
        cam = _gradcam(clip, tokenizer, ct, masks, has_masks, ctx, j, pos, neg)

        case_label = accession if a.show_accessions else f"Case {case_no}"
        safe_path = pathology.lower().replace(" ", "_").replace("/", "_")
        png = out_dir / f"case_{case_no:02d}_{safe_path}.png"
        probabilities = {
            "general": float(gen_prob[0, j]),
            "correct": float(true_prob[0, j]),
            "blank": float(blank_prob[0, j]),
        }
        _render(
            png, case_label, pathology, ct[0, 0].float().cpu().numpy(),
            masks[0, 0].cpu().numpy(), general_hm, true_hm, blank_hm, cam,
            probabilities, a.dpi,
        )
        band_idx = int(ctx["age_band"][0])
        manifest["cases"].append({
            "case_label": f"Case {case_no}", "accession": accession,
            "pathology": pathology, "ground_truth": 1,
            "age_band": AGE_BAND_NAMES[band_idx] if 0 <= band_idx < len(AGE_BAND_NAMES) else "unknown",
            "sex_index": int(ctx["sex"][0]), "probabilities": probabilities,
            "attention_path_max_abs_delta": {
                "correct": true_attn_delta, "blank": blank_attn_delta},
            "figure": png.name,
        })
        print(f"[heatmap] {case_label} {pathology} -> {png}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[heatmap] wrote {len(manifest['cases'])} figures to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

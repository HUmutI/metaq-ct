"""Visualize what AnatomyQFormer + R3D-18 attends to per pathology.

Outputs three views per (volume, pathology):
    A) QFormer cross-attention heatmap for the pathology query token.
    B) Grad-CAM on R3D-18 layer4 wrt the class prompt cosine.
    C) Per-slice confidence chart (12 axial slabs, 18 pathologies).

Each (volume, pathology) writes a triptych PNG: axial / coronal / sagittal
mid-plane MIP of CT (lung window) with the heatmap overlaid (jet, alpha=0.45).
Also writes one bar chart per volume showing per-slice prob for the top
pathologies.

Env:
    VIZ_CKPT             path to ckpt (required)
    VIZ_RESULTS_DIR      output dir (required)
    VIZ_ACCESSIONS       comma list of accessions (default: first 3 val)
    VIZ_PATHOLOGIES      comma list (default: Lung nodule,Cardiomegaly,
                                              Pleural effusion,Emphysema)
"""
from __future__ import annotations

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
import tqdm
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tools"))

from arcct.dataset import PATHOLOGIES, RACDatasetV4, rac_collate  # noqa: E402
from evaluate import build_model_for_eval, encode_prompts, TEMPERATURE  # noqa: E402
from arcct.anatomy_qformer import build_role_mask  # noqa: E402

DATA_ROOT = os.environ.get("RAC_DATA_ROOT", "/mnt/amax5_drive/alp_ozaydin_0/data")
DATA_VALID = f"{DATA_ROOT}/mps_ct_npz/valid"
MASK_VALID = os.environ.get("RAC_MASK_VALID", "/mnt/amax2_drive/alp_ozaydin_0/data/fine_valid_masks_192")
REPORTS_VALID = f"{DATA_ROOT}/ct_reports/valid_reports.csv"
LABELS_VALID = os.environ.get("RAC_LABELS_VALID", f"{DATA_ROOT}/multi_abnormality_labels/valid_predicted_labels.csv")

CKPT = os.environ["VIZ_CKPT"]
RESULTS_DIR = Path(os.environ["VIZ_RESULTS_DIR"])
ACCESSIONS = [s.strip() for s in os.environ.get("VIZ_ACCESSIONS", "").split(",") if s.strip()]
TARGET_PATHS = [s.strip() for s in os.environ.get(
    "VIZ_PATHOLOGIES",
    "Lung nodule,Cardiomegaly,Pleural effusion,Emphysema",
).split(",") if s.strip()]
LUNG_WIN_LO = -1500.0
LUNG_WIN_HI = 500.0


def _ct_to_window(ct_np: np.ndarray) -> np.ndarray:
    """ct_np: [D,H,W] in HU. Returns [0,1] float lung-windowed."""
    x = (ct_np - LUNG_WIN_LO) / (LUNG_WIN_HI - LUNG_WIN_LO)
    return np.clip(x, 0.0, 1.0)


def _save_triptych(ct_win: np.ndarray, hm: np.ndarray, out_path: Path, title: str) -> None:
    D, H, W = ct_win.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    z, y, x = D // 2, H // 2, W // 2
    hm_axial = hm[z]
    hm_coronal = np.flipud(hm[:, y, :])
    hm_sagittal = np.flipud(hm[:, :, x])
    # Axial
    axes[0].imshow(ct_win[z], cmap="gray", vmin=0, vmax=1)
    axes[0].imshow(hm_axial, cmap="YlOrRd",
                   alpha=np.clip(hm_axial, 0, 1).astype(np.float32),
                   vmin=0, vmax=1)
    axes[0].set_title(f"axial z={z}")
    # Coronal
    axes[1].imshow(np.flipud(ct_win[:, y, :]), cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(hm_coronal, cmap="YlOrRd",
                   alpha=np.clip(hm_coronal, 0, 1).astype(np.float32),
                   vmin=0, vmax=1)
    axes[1].set_title(f"coronal y={y}")
    # Sagittal
    axes[2].imshow(np.flipud(ct_win[:, :, x]), cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(hm_sagittal, cmap="YlOrRd",
                   alpha=np.clip(hm_sagittal, 0, 1).astype(np.float32),
                   vmin=0, vmax=1)
    axes[2].set_title(f"sagittal x={x}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _normalize(hm: np.ndarray) -> np.ndarray:
    lo, hi = float(hm.min()), float(hm.max())
    if hi - lo < 1e-9:
        return np.zeros_like(hm)
    return (hm - lo) / (hi - lo)


def _qformer_with_attention(qf, feat_map, ts_mask_4d, has_mask):
    """Replay AnatomyQFormer forward; return last block cross-attn weights.

    Returns:
        attn: [B, num_heads, Q, N] cross-attention from the LAST block.
        spatial: (Dp, Hp, Wp)
    """
    img_tokens, (Dp, Hp, Wp) = qf.qformer._flatten_feat_map(feat_map)
    img_tokens = qf.qformer.image_proj(img_tokens)
    img_tokens = qf.qformer.image_norm(img_tokens)
    B = img_tokens.shape[0]

    bool_mask = build_role_mask(
        ts_mask_4d, (Dp, Hp, Wp),
        qf.anatomy_labels, qf.pathology_labels, num_global=qf.num_global,
    )
    if has_mask is not None:
        bool_mask = bool_mask | (~has_mask).view(B, 1, 1)
    ignore_mask = ~bool_mask
    Q = bool_mask.shape[1]
    N = bool_mask.shape[2]
    ignore_mask = ignore_mask.unsqueeze(1).expand(
        B, qf.num_heads, Q, N
    ).reshape(B * qf.num_heads, Q, N)

    q = qf.qformer.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
    last_attn = None
    for i, layer in enumerate(qf.qformer.layers):
        h = layer.norm1(q)
        attn_out, _ = layer.self_attn(h, h, h, need_weights=False)
        q = q + attn_out
        h = layer.norm2(q)
        is_last = (i == len(qf.qformer.layers) - 1)
        cross_out, attn_w = layer.cross_attn(
            h, img_tokens, img_tokens,
            attn_mask=ignore_mask,
            need_weights=is_last,
            average_attn_weights=False,
        )
        if is_last:
            last_attn = attn_w
        q = q + cross_out
        q = q + layer.ffn(layer.norm3(q))
    return last_attn, (Dp, Hp, Wp)


def _qformer_attn_to_heatmap(attn, query_idx, spatial_lo, target_hw):
    """attn: [1, H, Q, N] -> [D, H, W] upsampled.
    Average over heads, take query_idx row, reshape, upsample.
    """
    Dp, Hp, Wp = spatial_lo
    D, H, W = target_hw
    hm = attn[0, :, query_idx, :].mean(0)
    hm = hm.reshape(Dp, Hp, Wp).float()
    hm_up = F.interpolate(
        hm.unsqueeze(0).unsqueeze(0), size=(D, H, W),
        mode="trilinear", align_corners=False,
    )[0, 0]
    return _normalize(hm_up.cpu().numpy())


def _gradcam(clip, qf, pos_embs, neg_embs, ct, masks_fine, has_masks,
             class_idx, device):
    """Grad-CAM on layer4 wrt class-pos cosine. Returns [D, H, W] heatmap."""
    # Save / enable grad on visual + qformer params so feat_map has a grad_fn.
    saved_states = []
    for p in clip.parameters():
        saved_states.append((p, p.requires_grad))
        p.requires_grad_(True)
    clip.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            feat_map = clip.visual_transformer.forward_spatial(ct)
            feat_map.retain_grad()
            if qf is not None:
                pooled = qf(feat_map, masks_fine, has_masks)
                img_lat = pooled
            else:
                raw = clip.visual_transformer.global_pool(feat_map)
                img_lat = clip.to_visual_latent(raw)
            img_lat = F.normalize(img_lat, dim=-1)
            score = (img_lat[0] * pos_embs[class_idx]).sum()
            score.backward()
        grad = feat_map.grad
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * feat_map.detach()).sum(dim=1).relu()
        cam_up = F.interpolate(
            cam.unsqueeze(0), size=ct.shape[2:],
            mode="trilinear", align_corners=False,
        )[0, 0]
    finally:
        for p, rg in saved_states:
            p.requires_grad_(rg)
        clip.zero_grad(set_to_none=True)
    return _normalize(cam_up.cpu().numpy())


@torch.no_grad()
def _slice_scores(clip, qf, pos_embs, neg_embs, ct, masks_fine, has_masks):
    """Per-slice (T) per-class (P) prob via anatomy-routed QFormer."""
    feat_map = clip.visual_transformer.forward_spatial(ct)
    B, Cc, T, Hp, Wp = feat_map.shape
    A = len(qf.anatomy_labels)
    P = len(qf.pathology_labels)
    ts_mask_4d = masks_fine[:, 0] if masks_fine.ndim == 5 else masks_fine
    stride = ts_mask_4d.shape[1] // T
    out = torch.zeros(B, T, P, device=ct.device)
    for t in range(T):
        slab = feat_map[:, :, t:t + 1]
        lo = t * stride
        hi = (t + 1) * stride if t < T - 1 else ts_mask_4d.shape[1]
        slab_mask = torch.zeros_like(ts_mask_4d)
        slab_mask[:, lo:hi] = ts_mask_4d[:, lo:hi]
        _, tok = qf(slab, slab_mask, has_masks, return_tokens=True)
        ptok = tok[:, A:A + P, :]
        p_lat = F.normalize(ptok, dim=-1)
        sp = (p_lat * pos_embs.unsqueeze(0)).sum(dim=-1)
        sn = (p_lat * neg_embs.unsqueeze(0)).sum(dim=-1)
        out[:, t, :] = F.softmax(
            torch.stack([sp, sn], dim=-1) / TEMPERATURE, dim=-1
        )[..., 0]
    return out.cpu().numpy()


def _save_slice_chart(slice_probs, accession, out_path):
    """slice_probs: [T, P]. Plot lines for the top-confidence pathologies."""
    T, P = slice_probs.shape
    max_per_class = slice_probs.max(axis=0)
    order = np.argsort(-max_per_class)[:8]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(T)
    for idx in order:
        ax.plot(xs, slice_probs[:, idx], marker="o", label=f"{PATHOLOGIES[idx]} (peak={max_per_class[idx]:.2f})")
    ax.set_xlabel("axial slab index (0=foot, T-1=head)")
    ax.set_ylabel("p(class | slab)")
    ax.set_title(f"per-slice confidence — {accession}")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, tokenizer = build_model_for_eval(CKPT)
    clip = clip.to(device).eval()
    qf = clip.qformer_module
    if qf is None or not getattr(clip, "use_anatomy_qformer", False):
        print("[viz] ckpt has no AnatomyQFormer; abort")
        sys.exit(2)
    qf.to(device).eval()
    pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)

    ds = RACDatasetV4(
        DATA_VALID, REPORTS_VALID, LABELS_VALID,
        mask_root=MASK_VALID if os.path.isdir(MASK_VALID) else None,
        is_train=False, limit=0, fail_fast=True,
    )
    if ACCESSIONS:
        acc_set = {a if a.endswith(".nii.gz") else a + ".nii.gz" for a in ACCESSIONS}
        keep = [i for i, s in enumerate(ds.samples) if s[1] in acc_set]
        if not keep:
            print(f"[viz] no match for accessions {ACCESSIONS}; using first 3")
            keep = list(range(3))
        ds.samples = [ds.samples[i] for i in keep]
    else:
        ds.samples = ds.samples[:3]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=rac_collate)

    A = len(qf.anatomy_labels)
    target_path_indices = []
    for name in TARGET_PATHS:
        try:
            target_path_indices.append(PATHOLOGIES.index(name))
        except ValueError:
            print(f"[viz] unknown pathology {name}; skip")
    print(f"[viz] targets = {TARGET_PATHS}")

    summary = []
    for ct, _, _, labels, masks_fine, has_masks, accessions in tqdm.tqdm(loader, desc="viz"):
        ct = ct.to(device)
        masks_fine = masks_fine.to(device)
        has_masks = has_masks.to(device)
        acc = accessions[0]
        out_acc = RESULTS_DIR / acc
        out_acc.mkdir(exist_ok=True)

        ct_np = ct[0, 0].cpu().numpy() * (LUNG_WIN_HI - LUNG_WIN_LO) + LUNG_WIN_LO
        ct_win = _ct_to_window(ct_np)

        # A) QFormer attention per pathology
        ts_4d = masks_fine[:, 0] if masks_fine.ndim == 5 else masks_fine
        with torch.no_grad():
            feat_map = clip.visual_transformer.forward_spatial(ct)
            attn, spatial_lo = _qformer_with_attention(qf, feat_map, ts_4d, has_masks)
        D, H, W = ct.shape[2:]
        for class_idx, name in zip(target_path_indices, TARGET_PATHS):
            qidx = A + class_idx
            hm = _qformer_attn_to_heatmap(attn, qidx, spatial_lo, (D, H, W))
            _save_triptych(
                ct_win, hm,
                out_acc / f"A_attn_{name.replace(' ', '_')}.png",
                f"[A] QFormer attn — {acc} — {name}",
            )

        # B) Grad-CAM per pathology
        for class_idx, name in zip(target_path_indices, TARGET_PATHS):
            cam = _gradcam(clip, qf, pos_embs, neg_embs,
                           ct.clone(), masks_fine, has_masks,
                           class_idx, device)
            _save_triptych(
                ct_win, cam,
                out_acc / f"B_gradcam_{name.replace(' ', '_')}.png",
                f"[B] Grad-CAM — {acc} — {name}",
            )

        # C) Per-slice confidence chart
        with torch.no_grad():
            sp = _slice_scores(clip, qf, pos_embs, neg_embs, ct, masks_fine, has_masks)[0]
        _save_slice_chart(sp, acc, out_acc / "C_slice_confidence.png")
        np.save(out_acc / "C_slice_probs.npy", sp)

        peak_slab = {PATHOLOGIES[i]: {"slab": int(sp[:, i].argmax()), "prob": float(sp[:, i].max())}
                     for i in target_path_indices}
        summary.append({"accession": acc, "peak": peak_slab,
                        "labels": {PATHOLOGIES[i]: int(labels[0, i].item()) for i in target_path_indices}})
        print(f"[viz] {acc} done -> {out_acc}")

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[viz] summary -> {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()

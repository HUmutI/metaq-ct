"""Opt-in losses for imbalanced multilabel Stage-2 experiments.

The default Stage-2 recipe remains ordinary masked BCE.  These helpers make
ASL and Batch Nuclear-norm Maximization (BNM) explicit ablations rather than
silently changing historical runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_asymmetric_loss_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """ASL over finite multilabel cells, optionally relevance weighted.

    With ``class_weights``, normalization matches the existing conditioned
    indication loss: normalize each study by its sum of weights, then average
    studies.  Without weights, average all labeled cells.
    """
    with torch.cuda.amp.autocast(enabled=False):
        p = probs.float().clamp(1e-6, 1.0 - 1e-6)
        target = labels.float().to(p.device)
        mask = torch.isfinite(target)
        target = torch.nan_to_num(target, nan=0.0)
        neg = 1.0 - p
        if clip > 0:
            neg = (neg + float(clip)).clamp(max=1.0)
        loss_pos = target * torch.log(p)
        loss_neg = (1.0 - target) * torch.log(neg.clamp_min(1e-6))
        pt = p * target + neg * (1.0 - target)
        gamma = float(gamma_pos) * target + float(gamma_neg) * (1.0 - target)
        per_cell = -(loss_pos + loss_neg) * (1.0 - pt).pow(gamma)
        if class_weights is None:
            return (per_cell * mask).sum() / mask.sum().clamp(min=1)
        weight = class_weights.float().to(p.device) * mask
        return ((per_cell * weight).sum(dim=-1)
                / weight.sum(dim=-1).clamp(min=1e-6)).mean()


def masked_asymmetric_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return masked_asymmetric_loss_from_probs(
        torch.sigmoid(logits.float()), labels, **kwargs)


def batch_nuclear_norm_loss(probs: torch.Tensor) -> torch.Tensor:
    """Standard BNM objective ``-||P||_* / B`` for a BxC probability matrix."""
    if probs.ndim != 2:
        raise ValueError(f"BNM expects BxC probabilities, got {tuple(probs.shape)}")
    if probs.shape[0] < 2:
        # Rank diversity is undefined for a one-study batch. Returning an exact
        # graph-connected zero keeps callers simple and gradients well formed.
        return probs.sum() * 0.0
    with torch.cuda.amp.autocast(enabled=False):
        matrix = probs.float().clamp(0.0, 1.0)
        return -torch.linalg.matrix_norm(matrix, ord="nuc") / matrix.shape[0]

#!/usr/bin/env python3
"""Fast unit checks for the opt-in Stage-2 imbalance losses."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arcct.imbalance_losses import (
    batch_nuclear_norm_loss,
    masked_asymmetric_loss_from_logits,
    masked_asymmetric_loss_from_probs,
)


def test_asl_reduces_to_bce():
    p = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    got = masked_asymmetric_loss_from_probs(
        p, y, gamma_neg=0, gamma_pos=0, clip=0)
    expected = F.binary_cross_entropy(p, y)
    assert torch.allclose(got, expected, atol=1e-7), (got, expected)


def test_nan_mask_and_logits_gradient():
    logits = torch.tensor([[1.0, -1.0], [0.0, 2.0]], requires_grad=True)
    y = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    loss = masked_asymmetric_loss_from_logits(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0


def test_bnm_prefers_diverse_rows_and_has_gradient():
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    diverse = torch.eye(2, requires_grad=True)
    loss_collapsed = batch_nuclear_norm_loss(collapsed)
    loss_diverse = batch_nuclear_norm_loss(diverse)
    assert loss_diverse < loss_collapsed, (loss_diverse, loss_collapsed)
    loss_diverse.backward()
    assert torch.isfinite(diverse.grad).all()


if __name__ == "__main__":
    test_asl_reduces_to_bce()
    test_nan_mask_and_logits_gradient()
    test_bnm_prefers_diverse_rows_and_has_gradient()
    print("imbalance-loss checks passed")

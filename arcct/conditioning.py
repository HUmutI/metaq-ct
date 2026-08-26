"""C1 -- indication conditioning of the pathology queries.

Two levels, following the advisor's note: cross-attention decides *where* to
look, FiLM decides *what kind of feature* matters.

    q~ = q + g_x * CrossAttn(q, H_C, H_C)          # spatial question
    q~ = gamma(cbar) * q~ + beta(cbar)             # channel question

Applied once, to the query vectors, before block 1 -- not inside every block.
The conditioning lives in the queries and nowhere else, so the image tower is
never conditioned and stays reusable.

Everything that could move the query starts at exactly zero: ``g_x`` is a zero
scalar, and both FiLM projections have zero weight *and* zero bias, so
``gamma = 1 + eps*tanh(0) = 1`` and ``beta = 0``. The module is therefore the
identity at initialisation -- bit-exactly, not approximately, because
``1.0 * q + 0.0`` is exact in IEEE754. That is half of the step-0 claim.

``gamma`` is bounded to ``1 +- eps`` rather than the ``1 + tanh(.)`` of the first
draft. Unbounded, ``tanh -> -1`` drives ``gamma -> 0`` and erases the query
entirely, which would make the headline safety property -- "this is a residual,
the query is never erased" -- untrue for exactly the classes the indication
considers irrelevant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QueryConditioner(nn.Module):
    """Condition a block of query vectors on the clinical context."""

    def __init__(self, dim: int = 768, num_heads: int = 8, film_eps: float = 0.2,
                 dropout: float = 0.0):
        super().__init__()
        if not 0.0 < film_eps <= 1.0:
            raise ValueError(f"film_eps must be in (0, 1], got {film_eps}")
        self.dim = int(dim)
        self.film_eps = float(film_eps)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                           batch_first=True)
        self.g_x = nn.Parameter(torch.zeros(()))       # scalar gate, starts closed

        self.to_gamma = nn.Linear(dim, dim)
        self.to_beta = nn.Linear(dim, dim)
        for lin in (self.to_gamma, self.to_beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)                   # bias too: a nonzero bias
            #                                            would make beta != 0 and
            #                                            break the identity even
            #                                            with zero weights.

    @staticmethod
    def _masked_mean(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Mean over sequence positions, ignoring padding. ``(B, L, D) -> (B, D)``."""
        if key_padding_mask is None:
            return x.mean(dim=1)
        keep = key_padding_mask.logical_not().to(x.dtype).unsqueeze(-1)
        # The context always carries three real demographic tokens, so the
        # denominator is >= 3 by construction; clamp is belt-and-braces against
        # a caller that builds H_C some other way.
        return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

    def forward(self, q: torch.Tensor, H_C: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``q`` is ``(B, S_cond, D)``; returns the same shape."""
        if q.shape[-1] != self.dim or H_C.shape[-1] != self.dim:
            raise ValueError(f"expected width {self.dim}, got q={tuple(q.shape)} "
                             f"H_C={tuple(H_C.shape)}")
        if key_padding_mask is not None and bool(key_padding_mask.all(dim=1).any()):
            # A fully-padded row makes MultiheadAttention emit NaN, and the zero
            # gate does NOT stop it: 0 * NaN is NaN, so it would reach Z_final
            # and be blamed on the image tower.
            raise ValueError("a row of H_C is entirely padding")

        c = self.norm_c(H_C)
        attn, _ = self.cross(self.norm_q(q), c, c,
                             key_padding_mask=key_padding_mask, need_weights=False)
        q = q + self.g_x * attn

        cbar = self._masked_mean(c, key_padding_mask)
        gamma = 1.0 + self.film_eps * torch.tanh(self.to_gamma(cbar))
        beta = self.to_beta(cbar)
        return gamma.unsqueeze(1) * q + beta.unsqueeze(1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, film_eps={self.film_eps}"


def _selftest() -> int:
    """python -m arcct.conditioning"""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    torch.manual_seed(0)
    dim, B, S, L = 32, 3, 28, 9
    m = QueryConditioner(dim=dim, num_heads=4, film_eps=0.2)

    q = torch.randn(B, S, dim)
    H_C = torch.randn(B, L, dim)
    kpm = torch.zeros(B, L, dtype=torch.bool)
    kpm[0, 5:6] = True                       # one padded position

    # -- the identity at init, EXACTLY ----------------------------------------
    out = m(q, H_C, kpm)
    check(torch.equal(out, q),
          f"C1 must be the exact identity at init; max|d|={(out - q).abs().max():.3e}")

    # and it must stay the identity whatever the context says
    out2 = m(q, torch.randn(B, L, dim) * 10.0, kpm)
    check(torch.equal(out2, q), "C1 output moved with the context while the gates were zero")

    # -- structural zeros ------------------------------------------------------
    check(float(m.g_x) == 0.0, "g_x must start at zero")
    for name in ("to_gamma", "to_beta"):
        lin = getattr(m, name)
        check(float(lin.weight.abs().max()) == 0.0, f"{name}.weight must be zero")
        check(float(lin.bias.abs().max()) == 0.0, f"{name}.bias must be zero")

    # -- gamma is bounded away from zero --------------------------------------
    with torch.no_grad():
        m.to_gamma.weight.normal_(std=5.0)   # drive tanh hard into saturation
        m.to_gamma.bias.normal_(std=5.0)
        c = m.norm_c(H_C)
        cbar = m._masked_mean(c, kpm)
        gamma = 1.0 + m.film_eps * torch.tanh(m.to_gamma(cbar))
    check(float(gamma.min()) >= 1.0 - m.film_eps - 1e-6,
          f"gamma fell to {float(gamma.min()):.3f}; with eps=0.2 it must stay >= 0.8")
    check(float(gamma.max()) <= 1.0 + m.film_eps + 1e-6, "gamma exceeded 1+eps")
    check(float(gamma.min()) > 0.0, "gamma must never reach zero -- the query would be erased")

    # -- once open, it actually conditions ------------------------------------
    m2 = QueryConditioner(dim=dim, num_heads=4, film_eps=0.2)
    with torch.no_grad():
        m2.g_x.fill_(1.0)
        m2.to_gamma.weight.normal_(std=0.1)
        m2.to_beta.weight.normal_(std=0.1)
    a = m2(q, H_C, kpm)
    b = m2(q, torch.randn(B, L, dim), kpm)
    check(not torch.equal(a, q), "with an open gate the output must move")
    check(not torch.equal(a, b), "with an open gate a different context must give a different query")
    check(bool(torch.isfinite(a).all()), "conditioned output must be finite")

    # -- a fully padded row is refused, not silently NaN ----------------------
    bad = torch.ones(B, L, dtype=torch.bool)
    try:
        m(q, H_C, bad)
        fails.append("an entirely padded H_C row must raise, not return NaN")
    except ValueError:
        pass

    # -- gradients exist even though the output is the identity ---------------
    # This is why the grad-clip has to be split into two groups: the new modules
    # contribute zero to the output but a nonzero norm to the global clip, so the
    # inherited parameters would take smaller steps than the reference run.
    m3 = QueryConditioner(dim=dim, num_heads=4, film_eps=0.2)
    m3(q, H_C, kpm).pow(2).sum().backward()
    check(m3.g_x.grad is not None and float(m3.g_x.grad.abs()) > 0,
          "g_x must receive gradient at init -- zero output does not mean zero gradient")
    check(float(m3.to_gamma.weight.grad.abs().max()) > 0, "to_gamma must receive gradient")

    if fails:
        print("S5 FAIL (%d):" % len(fails))
        for f in fails:
            print("   ", f)
        return 1
    print("S5 OK · exact identity at init · gamma in [%.2f, %.2f] under saturation "
          "· padded context refused · gradients flow" % (float(gamma.min()), float(gamma.max())))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())

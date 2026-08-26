"""C2 -- how relevant is each pathology to this clinical question.

    r_c = sigmoid( MLP[ E(I) ; E(P_c) ; E(I) * E(P_c) ] )
    w_c = 1 + beta * r_c ,   beta = softplus(b)

Two properties matter more than the exact form.

**Never multiply the query by r.** The advisor's note is explicit: scaling by a
relevance in [0, 1] suppresses incidental findings, and suppressing incidental
findings is the one thing a chest CT model must not do. ``w = 1 + beta*r`` leaves
an irrelevant class at weight 1, not 0. The indication says "pay particular
attention to these"; it does not say "ignore everything else".

**beta must be non-negative structurally, not hopefully.** An earlier draft made
beta a free scalar and paired it with an un-normalised ``L_ind = sum_c (1+beta
r_c) L_c``. Since every ``L_c >= 0``, ``dL/dbeta = sum_c r_c L_c > 0`` always, so
gradient descent drives beta negative from the first step: ``w < 1`` violates the
headline safety property within the first hundred updates, and then ``sum w -> 0``
is a pole. ``beta = softplus(b)`` makes ``w >= 1`` a fact about the parameterisation
rather than something to check afterwards. Both the pool and the loss are then
normalised by ``sum_c w_c``, which turns ``dL/dbeta`` into a covariance between
relevance and per-class loss -- so beta learns to emphasise classes that are
relevant *and* badly learned, which is the intended behaviour.

The MLP, rather than a scaled dot product: ``sigmoid(E(I)^T E(P_c) / tau)`` is
elegant but has no trainable parameter between two embeddings that both come from
a frozen tower, so the relevance is pinned to CXR-BERT's pre-training similarity
structure and cannot be corrected by the data. The ``E(I) * E(P_c)`` term gives
the MLP a cheap interaction feature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelevanceHead(nn.Module):
    """Per-class indication relevance and the pooling weights it induces."""

    def __init__(self, dim: int = 768, n_classes: int = 27, hidden: int = 128,
                 beta_init: float = -6.0):
        super().__init__()
        self.dim = int(dim)
        self.n_classes = int(n_classes)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.b = nn.Parameter(torch.tensor(float(beta_init)))
        # Persistent: the class embeddings come from the FROZEN indication tower,
        # so they are a constant of the run and belong in the checkpoint. Saving
        # them also makes a schema mismatch on resume raise on shape rather than
        # silently score 27 classes against 18 embeddings.
        self.register_buffer("class_emb", torch.zeros(self.n_classes, self.dim))
        self.register_buffer("class_emb_set", torch.zeros((), dtype=torch.bool))

    @property
    def beta(self) -> torch.Tensor:
        """``softplus(b) >= 0`` -- structural, not enforced by a clamp."""
        return F.softplus(self.b)

    def set_class_embeddings(self, emb: torch.Tensor) -> None:
        """Install ``(C, dim)`` class-prompt embeddings from the frozen tower."""
        if tuple(emb.shape) != (self.n_classes, self.dim):
            raise ValueError(f"class embeddings must be {(self.n_classes, self.dim)}, "
                             f"got {tuple(emb.shape)}")
        with torch.no_grad():
            self.class_emb.copy_(F.normalize(emb.detach().to(self.class_emb.dtype), dim=-1))
            self.class_emb_set.fill_(True)

    def forward(self, e_ind: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``e_ind`` is ``(B, dim)``; returns ``(r, w)``, both ``(B, C)``."""
        if not bool(self.class_emb_set):
            raise RuntimeError("set_class_embeddings() has not been called; the "
                               "relevance head would score against zeros")
        if e_ind.shape[-1] != self.dim:
            raise ValueError(f"e_ind must be (B, {self.dim}), got {tuple(e_ind.shape)}")

        B, C = e_ind.shape[0], self.n_classes
        cls = self.class_emb.to(e_ind.dtype).unsqueeze(0).expand(B, C, self.dim)
        ind = e_ind.unsqueeze(1).expand(B, C, self.dim)
        feats = torch.cat([ind, cls, ind * cls], dim=-1)          # (B, C, 3*dim)
        r = torch.sigmoid(self.mlp(feats).squeeze(-1))            # (B, C)
        w = 1.0 + self.beta * r
        return r, w

    @staticmethod
    def weighted_pool(tokens: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``sum_c w_c z_c / sum_c w_c`` over ``(B, C, D)``.

        Normalised, and that is not cosmetic: with a raw ``sum w_c z_c`` the norm
        of the pool would scale with how many classes the indication happens to
        be relevant to, so a broad indication ("chest pain") would systematically
        produce a larger vector than a narrow one. Dividing by ``sum w`` keeps the
        relative emphasis and drops the magnitude side effect.
        """
        if tokens.shape[:2] != w.shape[:2]:
            raise ValueError(f"tokens {tuple(tokens.shape)} and weights "
                             f"{tuple(w.shape)} disagree on (B, C)")
        num = (w.unsqueeze(-1) * tokens).sum(dim=1)
        den = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return num / den

    @staticmethod
    def weighted_loss(per_class: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``sum_c w_c L_c / sum_c w_c``, normalised for the same reason."""
        num = (w * per_class).sum(dim=-1)
        den = w.sum(dim=-1).clamp(min=1e-6)
        return (num / den).mean()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_classes={self.n_classes}, beta={float(self.beta):.5f}"


def _selftest() -> int:
    """python -m arcct.relevance"""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    torch.manual_seed(0)
    dim, C, B, D = 32, 27, 5, 16
    h = RelevanceHead(dim=dim, n_classes=C, beta_init=-6.0)

    # must refuse to score before the class embeddings are installed
    try:
        h(F.normalize(torch.randn(B, dim), dim=-1))
        fails.append("scoring before set_class_embeddings() must raise")
    except RuntimeError:
        pass

    h.set_class_embeddings(torch.randn(C, dim))
    check(bool(h.class_emb_set), "class_emb_set must flip")
    check(abs(float(h.class_emb.norm(dim=-1).mean()) - 1.0) < 1e-5,
          "class embeddings must be stored normalised")
    try:
        h.set_class_embeddings(torch.randn(C - 1, dim))
        fails.append("a class-count mismatch must raise")
    except ValueError:
        pass

    e = F.normalize(torch.randn(B, dim), dim=-1)
    r, w = h(e)
    check(tuple(r.shape) == (B, C) and tuple(w.shape) == (B, C),
          f"shapes r={tuple(r.shape)} w={tuple(w.shape)}")
    check(bool(((r >= 0) & (r <= 1)).all()), "r must lie in [0, 1]")

    # -- w >= 1 for ANY b, which is the whole point of softplus ---------------
    worst = 1.0
    for b in (-50.0, -20.0, -6.0, 0.0, 3.0, 20.0):
        with torch.no_grad():
            h.b.fill_(b)
        _, wb = h(e)
        worst = min(worst, float(wb.min()))
        check(float(wb.min()) >= 1.0,
              f"b={b}: w.min()={float(wb.min()):.6f} < 1 -- an irrelevant class was suppressed")
        check(float(h.beta) >= 0.0, f"b={b}: beta must be non-negative")
    check(worst >= 1.0, "w dropped below 1 for some b")

    with torch.no_grad():
        h.b.fill_(-6.0)
    check(abs(float(h.beta) - 0.0024757) < 1e-6,
          f"softplus(-6) should be ~0.00248, got {float(h.beta):.7f}")

    # -- the normalised pool --------------------------------------------------
    tok = torch.randn(B, C, D)
    w1 = torch.ones(B, C)
    check(torch.allclose(RelevanceHead.weighted_pool(tok, w1), tok.mean(dim=1), atol=1e-6),
          "uniform weights must reduce to the plain mean")

    # Magnitude must not depend on how many classes are relevant. This is the
    # failure the normalisation exists to prevent: a broad indication would
    # otherwise produce a systematically longer vector than a narrow one.
    narrow = torch.ones(B, C); narrow[:, :2] = 4.0
    broad = torch.ones(B, C); broad[:, :20] = 4.0
    n_pool = RelevanceHead.weighted_pool(tok, narrow)
    b_pool = RelevanceHead.weighted_pool(tok, broad)
    ratio = float(b_pool.norm(dim=-1).mean() / n_pool.norm(dim=-1).mean())
    check(0.5 < ratio < 2.0, f"pool norm scaled with the number of relevant classes: {ratio:.2f}")
    raw_ratio = float((broad.unsqueeze(-1) * tok).sum(1).norm(dim=-1).mean()
                      / (narrow.unsqueeze(-1) * tok).sum(1).norm(dim=-1).mean())
    check(raw_ratio > 2.0,
          f"the un-normalised pool should show the pathology this guards against, got {raw_ratio:.2f}")

    # -- dL/dbeta is a covariance once the loss is normalised -----------------
    # Un-normalised, dL/db > 0 always and b is driven to its floor from step 1.
    per = torch.rand(B, C)                        # per-class losses, all >= 0
    h2 = RelevanceHead(dim=dim, n_classes=C, beta_init=0.0)
    h2.set_class_embeddings(h.class_emb.clone())
    r2, w2 = h2(e)
    RelevanceHead.weighted_loss(per, w2).backward()
    g_norm = float(h2.b.grad)

    h3 = RelevanceHead(dim=dim, n_classes=C, beta_init=0.0)
    h3.set_class_embeddings(h.class_emb.clone())
    h3.load_state_dict(h2.state_dict())
    _, w3 = h3(e)
    (w3 * per).sum(dim=-1).mean().backward()
    g_raw = float(h3.b.grad)
    check(g_raw > 0, f"the un-normalised gradient should be strictly positive, got {g_raw:.4f}")
    check(abs(g_norm) < abs(g_raw),
          f"normalising must break the one-way push on beta: |{g_norm:.4f}| vs |{g_raw:.4f}|")

    if fails:
        print("S6 FAIL (%d):" % len(fails))
        for f in fails:
            print("   ", f)
        return 1
    print("S6 OK · w >= 1 for b in [-50, 20] · beta=softplus(b) >= 0 · pool norm "
          "flat vs breadth (%.2f, raw %.2f) · dL/db normalised %.4f vs raw %.4f"
          % (ratio, raw_ratio, g_norm, g_raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())

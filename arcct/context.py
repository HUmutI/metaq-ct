"""The clinical-context pathway: indication text, age band, sex.

Produces ``H_C``, the token sequence the conditioned queries cross-attend to:

    H_ind = CXR-BERT_frozen(indication)      # a SEQUENCE, deliberately not pooled
    H_dem = [P_age e_age(b) ; P_sex e_sex(s) ; P_int e_int(b, s)]
    H_C   = [H_ind ; H_dem]

Three decisions in here are load-bearing.

**The indication tower is a second, frozen copy of CXR-BERT.** The report tower
is not frozen -- it carries LoRA adapters and ``to_text_latent`` is trained -- so
reusing it would make the indication representation drift underneath the
conditioning modules while they are trying to learn from it. A separate frozen
copy costs ~440 MB of RAM and nothing else; it is excluded from ``state_dict``
because it is exactly reconstructible from ``from_pretrained``.

**The indication is kept as a token sequence.** Pooling it first was the earlier
design, justified by "the indication collapses to one vector anyway". That is
false for this corpus: the pediatric indication has a measured median of 24
words, and the difference between "history of metastatic osteosarcoma, evaluate
pulmonary nodules" and "incidental 5 mm nodule follow up" does not survive a mean
over tokens.

**Age is a band, never a scalar.** CT-RATE contributes 7 volumes under 18 out of
47,137, so a scalar age forces the model to extrapolate linearly across a range
where it has no data. Bands represent that gap explicitly. The band embedding is
built as a cumulative sum of non-negative increments, so ordering is structural:
band 3 is between band 2 and band 4 by construction, while the *spacing* stays
learned. Age-unknown is an eleventh free row outside the chain -- folding it into
band 0 would label every missing age a neonate.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_MODEL = "microsoft/BiomedVLP-CXR-BERT-specialized"

# One canonical empty input, used by every path that has no indication: a
# pediatric report whose section is missing, a CT-RATE row whose
# ClinicalInformation_EN is "Not given.", a training-time dropout, and
# evaluation condition 3. If these used different strings they would become
# separate dropout channels with separate embeddings, and the effective dropout
# rate on the adult half -- where 75.9% of rows are vacuous -- would silently be
# far higher than the configured one.
NO_INDICATION = "no clinical indication provided."

# (lower, upper) in years, half-open [lo, hi).
AGE_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 13.0),
    (13.0, 18.0), (18.0, 25.0), (25.0, 40.0), (40.0, 60.0), (60.0, float("inf")),
)
AGE_BAND_NAMES = ("infant", "toddler", "preschool", "school age", "pre-adolescent",
                  "adolescent", "young adult", "adult", "middle age", "older adult")
N_BANDS = len(AGE_BANDS)            # 10 ordered bands
BAND_UNKNOWN = N_BANDS              # index 10, a free row outside the chain
N_BANDS_TOTAL = N_BANDS + 1
N_SEX = 3                           # female / male / unknown
SEX_UNKNOWN = 2


def age_to_band(age_years: float | None) -> int:
    """Map decimal years to a band index, or :data:`BAND_UNKNOWN`.

    ``None`` and NaN both mean unknown. Negative ages are treated as unknown
    rather than clamped to band 0, because a negative age is a parse failure and
    calling it "infant" would put a fabricated value into training.
    """
    if age_years is None:
        return BAND_UNKNOWN
    try:
        a = float(age_years)
    except (TypeError, ValueError):
        return BAND_UNKNOWN
    if a != a or a < 0.0:            # NaN or negative
        return BAND_UNKNOWN
    for i, (lo, hi) in enumerate(AGE_BANDS):
        if lo <= a < hi:
            return i
    return N_BANDS - 1               # unreachable: the last band is unbounded


class ContextBundle(NamedTuple):
    """What the conditioned queries and the relevance head consume."""

    H_C: torch.Tensor                 # (B, L_ind + 3, dim)
    key_padding_mask: torch.Tensor    # (B, L_ind + 3) bool, True = ignore
    e_ind: torch.Tensor               # (B, dim) pooled indication, detached


class ContextEncoder(nn.Module):
    """Indication text + demographics -> a context token sequence.

    Args
    ----
    dim : model width, 768 to match the Q-Former queries.
    text_model : HuggingFace id of the frozen indication tower.
    bert : an already-constructed encoder to freeze and adopt. Passing one in
        keeps unit tests off the network; production passes None.
    age_dim, sex_dim, int_dim : widths of the three demographic embeddings
        before they are projected to ``dim``. They are deliberately small: these
        are 3 tokens against a ~24-token indication, and giving them full width
        would let demographics dominate the context by sheer parameter count.
    """

    def __init__(
        self,
        dim: int = 768,
        text_model: str = TEXT_MODEL,
        bert: nn.Module | None = None,
        age_dim: int = 64,
        sex_dim: int = 16,
        int_dim: int = 32,
        delta_init: float = -3.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.text_model = text_model

        if bert is None:
            from transformers import BertModel          # imported late: heavy
            bert = BertModel.from_pretrained(text_model)
        for p in bert.parameters():
            p.requires_grad = False
        bert.eval()
        # Held in a list so nn.Module does not register it: the tower then stays
        # out of state_dict, out of parameters(), and out of the optimizer,
        # without having to special-case any of the three.
        self._bert = [bert]

        # -- ordinal-cumulative age ------------------------------------------
        # e_age(b) = e_0 + sum_{j <= b} softplus(delta_j)
        # softplus(0) = 0.693, so a zero init would put a ramp of magnitude 6.2
        # across the ten bands before any training. "Small init" means a small
        # INCREMENT: delta = -3 gives softplus = 0.049 per band.
        self.e_0 = nn.Parameter(torch.zeros(age_dim))
        self.delta = nn.Parameter(torch.full((N_BANDS - 1, age_dim), float(delta_init)))
        self.e_age_unknown = nn.Parameter(torch.zeros(age_dim))
        nn.init.normal_(self.e_0, std=0.02)
        nn.init.normal_(self.e_age_unknown, std=0.02)

        # The unordered table for the "flat" arm. Always constructed so the
        # checkpoint shape does not depend on the ablation flag.
        self.e_flat = nn.Embedding(N_BANDS_TOTAL, age_dim)
        nn.init.normal_(self.e_flat.weight, std=0.02)

        self.e_sex = nn.Embedding(N_SEX, sex_dim)
        nn.init.normal_(self.e_sex.weight, std=0.02)

        # Interaction starts at exactly zero, so the demographic signal is purely
        # additive until the data says otherwise. Weight decay x10 on this table
        # keeps it that way unless an interaction genuinely pays for itself.
        self.e_int = nn.Parameter(torch.zeros(N_BANDS_TOTAL, N_SEX, int_dim))

        self.P_age = nn.Linear(age_dim, dim, bias=False)
        self.P_sex = nn.Linear(sex_dim, dim, bias=False)
        self.P_int = nn.Linear(int_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.P_age.weight)
        nn.init.xavier_uniform_(self.P_sex.weight)
        nn.init.zeros_(self.P_int.weight)          # interaction token starts dead

        self.norm_dem = nn.LayerNorm(dim)
        self.norm_ind = nn.LayerNorm(dim)

    # -- the frozen tower ------------------------------------------------------

    @property
    def bert(self) -> nn.Module:
        return self._bert[0]

    def _apply(self, fn):                                   # noqa: D401
        """Follow .to()/.cuda()/.float() for the unregistered tower too."""
        super()._apply(fn)
        self._bert[0]._apply(fn)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self._bert[0].eval()        # the tower is frozen in both modes
        return self

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Frozen CXR-BERT token sequence, ``(B, L, dim)``."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return out.detach()

    # -- demographics ----------------------------------------------------------

    def age_table(self) -> torch.Tensor:
        """``(N_BANDS_TOTAL, age_dim)``, recomputed every call.

        delta is trainable, so caching this would silently freeze the ordinal
        structure at its initial values. It is a 9 x 64 softplus and a cumsum.
        """
        steps = F.softplus(self.delta).cumsum(dim=0)                    # (9, age_dim)
        zero = torch.zeros(1, steps.shape[1], dtype=steps.dtype, device=steps.device)
        ordered = self.e_0.unsqueeze(0) + torch.cat([zero, steps], dim=0)   # (10, age_dim)
        return torch.cat([ordered, self.e_age_unknown.unsqueeze(0)], dim=0)

    def demographic_tokens(self, age_band: torch.Tensor, sex: torch.Tensor,
                           age_years: torch.Tensor | None = None,
                           mode: str = "band") -> torch.Tensor:
        """``(B, 3, dim)`` -- age, sex, interaction.

        ``mode`` selects the R8 ablation arm and exists so the ordinal-cumulative
        parameterisation has to earn its place:

          band      the design: ordinal cumulative sum over 10 bands
          scalar    a single normalised number -- what the bands replaced, and
                    what forces linear extrapolation across the 18-25 bridge
                    where CT-RATE contributes 7 volumes under 18 out of 47,137
          flat      an unordered embedding table: bands, but with the ordering
                    thrown away, isolating what the cumulative sum buys
          shuffled  the band index permuted per sample -- the negative control;
                    if this scores like `band`, age was never being used
        """
        if age_band.dtype != torch.long:
            age_band = age_band.long()
        if sex.dtype != torch.long:
            sex = sex.long()
        # A dataset that has not been wired yet sends -1; map it to the unknown
        # row rather than letting negative indexing wrap to the last band.
        age_band = torch.where(age_band < 0, torch.full_like(age_band, BAND_UNKNOWN), age_band)
        sex = torch.where(sex < 0, torch.full_like(sex, SEX_UNKNOWN), sex)
        if int(age_band.max()) > BAND_UNKNOWN or int(sex.max()) >= N_SEX:
            raise ValueError(f"age_band max={int(age_band.max())} (allowed {BAND_UNKNOWN}), "
                             f"sex max={int(sex.max())} (allowed {N_SEX - 1})")

        if mode == "scalar":
            if age_years is None:
                raise ValueError("age_mode=scalar needs age_years")
            # /100 keeps it in roughly [0, 1]; unknown ages (-1) become 0 and are
            # indistinguishable from a newborn, which is precisely the failure
            # the bands were introduced to avoid.
            a = (age_years.to(self.e_0.dtype).clamp(min=0.0) / 100.0).unsqueeze(-1)
            e_age = self.e_0.unsqueeze(0) * a
        elif mode == "flat":
            e_age = self.e_flat(age_band)
        elif mode == "shuffled":
            perm = torch.randperm(N_BANDS_TOTAL, device=age_band.device)
            e_age = self.age_table()[perm[age_band]]
        else:
            e_age = self.age_table()[age_band]
        t_age = self.P_age(e_age)                                       # (B, dim)
        t_sex = self.P_sex(self.e_sex(sex))                             # (B, dim)
        t_int = self.P_int(self.e_int[age_band, sex])                   # (B, dim)
        return self.norm_dem(torch.stack([t_age, t_sex, t_int], dim=1))

    # -- the whole context -----------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        age_band: torch.Tensor,
        sex: torch.Tensor,
        to_latent: nn.Module | None = None,
        age_years: torch.Tensor | None = None,
        age_mode: str = "band",
    ) -> ContextBundle:
        """Build ``H_C`` and the pooled indication embedding.

        ``to_latent`` is an optional projection applied to the pooled indication
        before it reaches the relevance head. Leave it None: the relevance head
        compares indication text against class-prompt text, and both sides come
        from this same frozen tower, so they already share one stable basis.
        Routing that comparison through the trained visual-alignment projection
        would make the relevance geometry drift with the image tower for no
        benefit -- and would make the cached class embeddings stale on resume.
        """
        H_ind = self.norm_ind(self.encode_text(input_ids, attention_mask))   # (B, L, dim)
        H_dem = self.demographic_tokens(age_band, sex, age_years, age_mode)  # (B, 3, dim)
        H_C = torch.cat([H_ind, H_dem], dim=1)

        pad_ind = attention_mask.to(torch.bool).logical_not()                # True = ignore
        # The three demographic tokens are ALWAYS real. This is what makes a
        # fully-dropped indication safe: H_C can never be entirely padding, so
        # the masked mean below never divides by zero and the cross-attention
        # never sees a fully-masked row. A NaN here would pass straight through
        # the zero-initialised gates -- 0 * NaN is NaN -- and be blamed on the
        # image tower.
        pad_dem = torch.zeros(H_dem.shape[0], H_dem.shape[1],
                              dtype=torch.bool, device=H_dem.device)
        kpm = torch.cat([pad_ind, pad_dem], dim=1)

        # Masked mean over the indication tokens only: the pooled vector feeds
        # the relevance head, which is asking "what is this scan for", and the
        # demographic tokens would dilute that with information the head already
        # gets elsewhere.
        m = attention_mask.to(H_ind.dtype).unsqueeze(-1)
        e_ind = (H_ind * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if to_latent is not None:
            e_ind = to_latent(e_ind)
        e_ind = F.normalize(e_ind, dim=-1).detach()

        return ContextBundle(H_C=H_C, key_padding_mask=kpm, e_ind=e_ind)

    @torch.no_grad()
    def encode_class_prompts(self, tokenizer, class_names: list[str],
                             max_len: int = 32) -> torch.Tensor:
        """``(C, dim)`` class embeddings in the frozen indication basis.

        Computed once and cached by the relevance head. Because the tower never
        moves, this is a constant for the whole run -- unlike the training-side
        ``pos_embs``, which are re-encoded every 50 updates as the text LoRA
        drifts.
        """
        device = self.e_0.device
        tok = tokenizer([f"{c}." for c in class_names], return_tensors="pt",
                        padding="max_length", truncation=True, max_length=max_len)
        ids = tok["input_ids"].to(device)
        am = tok["attention_mask"].to(device)
        H = self.encode_text(ids, am)
        m = am.to(H.dtype).unsqueeze(-1)
        pooled = (H * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.norm_ind(pooled), dim=-1)

    # -- housekeeping ----------------------------------------------------------

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Two groups: the interaction table gets 10x weight decay."""
        interaction = [self.e_int]
        ids = {id(p) for p in interaction}
        rest = [p for p in self.parameters() if id(p) not in ids and p.requires_grad]
        return [
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
            {"params": interaction, "lr": lr, "weight_decay": weight_decay * 10.0},
        ]

    def tokenize(self, tokenizer, texts: list[str], max_len: int,
                 device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        tok = tokenizer(list(texts), return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_len)
        return {"input_ids": tok["input_ids"].to(device),
                "attention_mask": tok["attention_mask"].to(device)}


def _selftest() -> int:
    """python -m arcct.context -- runs without downloading CXR-BERT."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # bands
    cases = [(0.0, 0), (0.99, 0), (1.0, 1), (1.99, 1), (2.0, 2), (4.99, 2),
             (5.0, 3), (12.9, 4), (13.0, 5), (17.99, 5), (18.0, 6), (24.99, 6),
             (25.0, 7), (39.9, 7), (40.0, 8), (59.9, 8), (60.0, 9), (120.0, 9)]
    for a, want in cases:
        got = age_to_band(a)
        check(got == want, f"age_to_band({a}) = {got}, expected {want}")
    for bad in (None, float("nan"), -1.0, "x"):
        check(age_to_band(bad) == BAND_UNKNOWN, f"age_to_band({bad!r}) must be unknown")

    torch.manual_seed(0)
    dim, B, L = 32, 4, 7

    class _StubBert(nn.Module):
        """Stands in for CXR-BERT: same call shape, trainable params to prove
        they stay frozen and stay out of state_dict."""

        def __init__(self, d):
            super().__init__()
            self.emb = nn.Embedding(64, d)
            self.lin = nn.Linear(d, d)

        def forward(self, input_ids=None, attention_mask=None):
            return (self.lin(self.emb(input_ids)),)

    enc = ContextEncoder(dim=dim, bert=_StubBert(dim), age_dim=8, sex_dim=4, int_dim=4)

    # the frozen tower is invisible to the module
    check(not any("bert" in k or "emb" in k or "lin" in k for k in enc.state_dict()),
          f"the frozen tower leaked into state_dict: {sorted(enc.state_dict())[:5]}")
    check(all(not p.requires_grad for p in enc.bert.parameters()),
          "the indication tower must be frozen")
    enc.train()
    check(not enc.bert.training, "the tower must stay in eval mode even in train()")

    # ordinal structure
    tbl = enc.age_table()
    check(tuple(tbl.shape) == (N_BANDS_TOTAL, 8), f"age_table shape {tuple(tbl.shape)}")
    steps = (tbl[1:N_BANDS] - tbl[:N_BANDS - 1])
    check(bool((steps > 0).all()), "the age walk must be strictly monotone by construction")
    check(float(steps.max()) < 0.2,
          f"delta init too large: step={float(steps.max()):.3f}; softplus(0)=0.693 "
          "would put a ramp of 6.2 across the bands before training")
    # the unknown row is NOT part of the chain
    d_unknown = (tbl[BAND_UNKNOWN] - tbl[N_BANDS - 1]).abs().max()
    check(float(d_unknown) > 0, "the unknown row must be free, not the top of the walk")

    # demographic tokens
    ab = torch.tensor([0, 5, 9, -1])
    sx = torch.tensor([0, 1, 2, -1])
    H_dem = enc.demographic_tokens(ab, sx)
    check(tuple(H_dem.shape) == (B, 3, dim), f"H_dem shape {tuple(H_dem.shape)}")
    check(bool(torch.isfinite(H_dem).all()), "H_dem must be finite for age_band=-1")
    check(float(enc.e_int.abs().max()) == 0.0, "the interaction table must start at zero")
    check(float(enc.P_int.weight.abs().max()) == 0.0, "P_int must be zero-initialised")

    # full context
    ids = torch.randint(1, 64, (B, L))
    am = torch.ones(B, L, dtype=torch.long)
    am[0, 4:] = 0                                  # a short indication
    ctx = enc(ids, am, ab, sx)
    check(tuple(ctx.H_C.shape) == (B, L + 3, dim), f"H_C shape {tuple(ctx.H_C.shape)}")
    check(tuple(ctx.key_padding_mask.shape) == (B, L + 3), "key_padding_mask shape")
    check(not bool(ctx.key_padding_mask[:, -3:].any()),
          "the three demographic tokens must never be padded -- that is what "
          "keeps H_C from being all-padding when the indication is dropped")
    check(bool(ctx.key_padding_mask[0, 4:L].all()), "indication padding must be honoured")
    check(not ctx.e_ind.requires_grad, "e_ind must be detached")
    check(abs(float(ctx.e_ind.norm(dim=-1).mean()) - 1.0) < 1e-5, "e_ind must be unit norm")

    # an ENTIRELY empty indication must still produce a finite, usable context
    am0 = torch.zeros(B, L, dtype=torch.long)
    ctx0 = enc(ids, am0, ab, sx)
    check(bool(torch.isfinite(ctx0.H_C).all()), "H_C must be finite with an empty indication")
    check(bool(torch.isfinite(ctx0.e_ind).all()), "e_ind must be finite with an empty indication")
    check(not bool(ctx0.key_padding_mask.all(dim=1).any()),
          "no row of H_C may be entirely padding")

    # gradients reach the age walk but never the tower
    ctx.H_C.sum().backward()
    check(enc.delta.grad is not None and float(enc.delta.grad.abs().max()) > 0,
          "delta must receive gradient")
    check(all(p.grad is None for p in enc.bert.parameters()),
          "the frozen tower must receive no gradient")

    # param groups
    gs = enc.param_groups(lr=1e-4, weight_decay=1e-4)
    check(len(gs) == 2 and gs[1]["weight_decay"] == 1e-3,
          "the interaction table needs its own group at 10x weight decay")
    n_grouped = sum(len(g["params"]) for g in gs)
    check(n_grouped == len([p for p in enc.parameters() if p.requires_grad]),
          "every trainable parameter must land in exactly one group")

    if fails:
        print("S4 FAIL (%d):" % len(fails))
        for f in fails:
            print("   ", f)
        return 1
    print("S4 OK · bands exact · age walk monotone (step=%.3f) · tower frozen and "
          "out of state_dict · demographics never padded" % float(steps.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())

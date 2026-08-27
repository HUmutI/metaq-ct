"""The indication-conditioned Q-Former: 67 slots over 40 parameter rows.

Assembles the pieces built in :mod:`arcct.slots`, :mod:`arcct.context`,
:mod:`arcct.conditioning` and :mod:`arcct.relevance` into a drop-in replacement
for :class:`~arcct.anatomy_qformer.AnatomyQFormer`.

    Z_gen   = mean over the 39 UNCONDITIONED slots        # today's ARC-CT latent
    Z_ind   = combine( sum_c w_c z_c / sum_c w_c , z_clinical )
    Z_final = W [ Z_gen ; Z_ind ]        with  W initialised [ I | 0 ]

Four independent things keep the indication out of ``Z_gen``, and all four are
needed -- any one of them alone is decorative:

1.  **self-attention group mask.** Unconditioned slots may not attend to
    conditioned ones. Without it the "unconditioned" bank sees the indication
    through query-to-query attention after a single block.
2.  **group-restricted pooling.** ``Z_gen`` averages 39 slots, not 67. The
    original ``q.mean(dim=1)`` is both the CLIP latent and the checkpoint
    selection metric, so pooling the whole bank would leak into both while the
    other three fixes looked correct.
3.  **fusion initialised ``[I | 0]``.** At step 0 the conditioned half is
    multiplied by exact zeros, so ``Z_final == Z_gen`` bit for bit.
4.  **zero-initialised conditioning.** ``g_x = 0`` and both FiLM projections
    zero, so even the conditioned tokens start as their unconditioned selves.

The default return shape is deliberately a single ``(B, dim)`` tensor, matching
the module this replaces, so ``evaluate.py`` and the training loop work
unchanged. Ask for ``return_parts=True`` to see inside.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from arcct.anatomy_qformer import AnatomyQFormer, build_role_mask
from arcct.conditioning import QueryConditioner
from arcct.context import ContextBundle, ContextEncoder
from arcct.relevance import RelevanceHead
from arcct.slots import CtxConfig, SlotLayout


class ContextOutput(NamedTuple):
    z_final: torch.Tensor
    z_gen: torch.Tensor
    z_ind: torch.Tensor
    tokens: torch.Tensor
    r: torch.Tensor | None
    w: torch.Tensor | None
    rho: torch.Tensor | None
    empty: torch.Tensor | None


class ContextQFormer(AnatomyQFormer):
    """Anatomy-routed Q-Former with an indication-conditioned second bank."""

    def __init__(
        self,
        anatomy_labels: list[int],
        pathology_labels: list[list[int]],
        *,
        layout: SlotLayout | None = None,
        cfg: CtxConfig | None = None,
        dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        image_dim: int = 512,
        dropout: float = 0.0,
        bert: nn.Module | None = None,
        context: ContextEncoder | None = None,
    ):
        L = layout or SlotLayout.phase1(n_path=len(pathology_labels),
                                        n_anatomy=len(anatomy_labels))
        L.validate()
        if L.n_anatomy != len(anatomy_labels) or L.n_path != len(pathology_labels):
            raise ValueError(f"layout {L.describe()} does not match "
                             f"{len(anatomy_labels)} anatomy / "
                             f"{len(pathology_labels)} pathology labels")
        self.cfg = cfg or CtxConfig.from_env()

        # The inner QFormer holds n_rows query rows, NOT n_slots. Passing
        # n_glob_gen + n_glob_clin as num_global is what makes that come out
        # right, but the ROLE MASK is built with n_glob_gen only -- the extra
        # rows are parameters, not mask rows. Conflating the two is the easiest
        # mistake available here, so they get different names.
        super().__init__(
            anatomy_labels=anatomy_labels,
            pathology_labels=pathology_labels,
            num_global=L.n_glob_gen + L.n_glob_clin,
            dim=dim, depth=depth, num_heads=num_heads,
            image_dim=image_dim, dropout=dropout, pool="mean",
        )
        self.layout = L
        self.dim = int(dim)
        self.base_num_global = L.n_glob_gen
        if self.qformer.queries.shape[0] != L.n_rows:
            raise RuntimeError(f"inner bank has {self.qformer.queries.shape[0]} rows, "
                               f"layout wants {L.n_rows}")

        for name, vals in (("slot_to_row", L.slot_to_row), ("role_row", L.role_row),
                           ("gen_index", L.gen_index),
                           ("path_gen_index", L.path_gen_index),
                           ("path_cond_index", L.path_cond_index),
                           ("cond_index", L.cond_index),
                           ("glob_clin_index", L.glob_clin_index)):
            self.register_buffer(name, torch.tensor(vals, dtype=torch.long),
                                 persistent=False)

        # (S, S) bool, broadcast over batch and heads. The partition depends only
        # on the slot index, never on the sample, so materialising the
        # (B*heads, S, S) form would allocate ~700k constant elements per block
        # per step.
        m = torch.zeros(L.n_slots, L.n_slots, dtype=torch.bool)
        m[:L.n_gen, L.n_gen:] = True
        if not bool((~m).any(dim=-1).all()):
            raise RuntimeError("a slot would have no attendable key -- softmax NaN")
        self.register_buffer("group_self_attn_mask", m, persistent=False)

        self.context = context or ContextEncoder(dim=dim, bert=bert)
        self.conditioner = QueryConditioner(dim=dim, num_heads=num_heads,
                                            film_eps=self.cfg.film_eps, dropout=dropout)
        self.relevance = RelevanceHead(dim=dim, n_classes=L.n_path,
                                       beta_init=self.cfg.beta_init)

        self.fusion = nn.Linear(2 * dim, dim, bias=False)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :dim].copy_(torch.eye(dim))
        # The gated-sum alternative the advisor's note proposed first:
        #   Z_final = LN(Z_gen + g * Z_ind),  g = sigmoid(MLP(mean H_C))
        # The scalar g0 starts at zero so the gate is CLOSED at step 0 and the
        # identity holds for this arm too -- otherwise the two fusion arms would
        # not start from the same model and could not be compared.
        self.gate_mlp = nn.Linear(dim, dim)
        self.gate_scale = nn.Parameter(torch.zeros(()))
        self.gate_norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.gate_mlp.weight)
        nn.init.zeros_(self.gate_mlp.bias)

        self.last_role_stats: dict[str, torch.Tensor] | None = None

    # -- checkpoint ------------------------------------------------------------

    def load_arcct_state_dict(self, sd: dict, verbose: bool = True) -> tuple[list, list]:
        """Load a 39-row ARC-CT Q-Former into the 40-row bank.

        Only ``qformer.queries`` changes shape, and only by one appended row --
        every other tensor is a function of ``dim``. The new clinical row is
        seeded from a trained global rather than from noise: it feeds conditioned
        slots only, so any value preserves the step-0 identity, but a trained
        start is better conditioned than N(0, 0.02).
        """
        sd = dict(sd)
        key = "qformer.queries"
        if key in sd:
            src = sd[key]
            want = self.qformer.queries.shape
            if tuple(src.shape) != tuple(want):
                n_src, n_dst = src.shape[0], want[0]
                if src.shape[1] != want[1] or n_dst < n_src:
                    raise ValueError(f"cannot map {tuple(src.shape)} onto {tuple(want)}")
                pad = src[-1:].repeat(n_dst - n_src, 1).clone()
                sd[key] = torch.cat([src, pad], dim=0)
                if verbose:
                    print(f"[ctx] query bank {n_src} -> {n_dst} rows "
                          f"(+{n_dst - n_src} seeded from row {n_src - 1})")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        return list(missing), list(unexpected)

    def query_layout(self) -> dict:
        """Written into the checkpoint so evaluate.py never has to guess."""
        L = self.layout
        return {"n_anatomy": L.n_anatomy, "n_path": L.n_path,
                "n_glob_gen": L.n_glob_gen, "n_glob_clin": L.n_glob_clin,
                "n_path_cond": L.n_path_cond, "n_rows": L.n_rows, "n_slots": L.n_slots,
                "n_gen": L.n_gen}

    # -- forward ---------------------------------------------------------------

    def _base_role_mask(self, feat_map, ts_mask, has_mask):
        """``(B, n_gen, N)`` allowed-mask plus per-region statistics."""
        B = feat_map.shape[0]
        Dp, Hp, Wp = feat_map.shape[-3:]
        N = Dp * Hp * Wp
        if ts_mask is None:
            base = torch.ones(B, self.layout.n_gen, N, dtype=torch.bool,
                              device=feat_map.device)
            stats = {"rho": torch.ones(B, self.layout.n_gen, device=feat_map.device),
                     "empty": torch.zeros(B, self.layout.n_gen, dtype=torch.bool,
                                          device=feat_map.device)}
            return base, stats
        if ts_mask.ndim == 5:
            ts_mask = ts_mask[:, 0]
        base, stats = build_role_mask(
            ts_mask, (Dp, Hp, Wp), self.anatomy_labels, self.pathology_labels,
            num_global=self.base_num_global, return_stats=True)
        if has_mask is not None:
            base = base | (~has_mask).view(B, 1, 1)
        return base, stats

    def forward(
        self,
        feat_map: torch.Tensor,
        ts_mask: torch.Tensor | None = None,
        has_mask: torch.Tensor | None = None,
        return_tokens: bool = False,
        return_attn: bool = False,
        suppress_mask: bool = False,
        *,
        context: ContextBundle | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        age_band: torch.Tensor | None = None,
        sex: torch.Tensor | None = None,
        age_years: torch.Tensor | None = None,
        return_parts: bool = False,
    ):
        if return_attn:
            raise NotImplementedError("return_attn is not wired for the context bank")
        L = self.layout
        B = feat_map.shape[0]

        if context is None:
            if input_ids is None:
                raise ValueError("give either a prebuilt context= or input_ids/"
                                 "attention_mask/age_band/sex")
            if age_band is None:
                age_band = torch.full((B,), -1, dtype=torch.long, device=feat_map.device)
            if sex is None:
                sex = torch.full((B,), -1, dtype=torch.long, device=feat_map.device)
            context = self.context(input_ids, attention_mask, age_band, sex,
                                   age_years=age_years, age_mode=self.cfg.age_mode)

        base, stats = self._base_role_mask(feat_map, ts_mask, has_mask)
        self.last_role_stats = stats
        bool_mask = base.index_select(1, self.role_row)                  # (B, S, N)

        attn_mask = None
        if not suppress_mask:
            S, N = bool_mask.shape[1], bool_mask.shape[2]
            attn_mask = (~bool_mask).unsqueeze(1).expand(
                B, self.num_heads, S, N).reshape(B * self.num_heads, S, N)

        # Weight tying is this one gather. index_select's backward is index_add,
        # so the gradients of a conditioned slot and its general partner sum into
        # the same row automatically -- no parameter aliasing, no syncing, and
        # the checkpoint still holds n_rows rows.
        slots = self.qformer.queries.index_select(0, self.slot_to_row)
        q = slots.unsqueeze(0).expand(B, -1, -1)
        if self.cfg.c1:
            q_cond = self.conditioner(q[:, L.n_gen:], context.H_C,
                                      context.key_padding_mask)
            q = torch.cat([q[:, :L.n_gen], q_cond], dim=1)
        else:
            q = q.contiguous()

        # The group constraint is one-directional: an unconditioned slot attends
        # over n_gen keys, its conditioned twin over all n_slots. So a tied pair
        # shares a query vector but not an output token, by design -- the tying
        # is a statement about the input, not the answer.
        # "none" is the isolation ablation: the group constraint is REMOVED, so
        # the unconditioned rows can read the conditioned ones and Z_gen stops
        # being invariant. It exists to measure the leak rather than to argue
        # about it; a run in this mode must never be reported as a safe model.
        if self.cfg.selfattn == "none":
            split, gmask = None, None
        elif self.cfg.selfattn == "split":
            split, gmask = L.n_gen, None
        else:
            split, gmask = None, self.group_self_attn_mask
        z_gen, tokens = self.qformer(
            feat_map, cross_attn_mask=attn_mask, queries=q,
            self_attn_split=split, self_attn_mask=gmask,
            pool_index=self.gen_index, return_tokens=True)

        # -- the conditioned pool -------------------------------------------
        z_c = tokens.index_select(1, self.path_cond_index)               # (B, C, D)
        if self.cfg.c2:
            r, w = self.relevance(context.e_ind)
        else:
            r, w = None, torch.ones(B, L.n_path_cond, device=tokens.device,
                                    dtype=tokens.dtype)
        z_pool = RelevanceHead.weighted_pool(z_c, w.to(z_c.dtype))
        z_clin = tokens.index_select(1, self.glob_clin_index).mean(dim=1)
        # mean, not sum: it keeps ||Z_ind|| on the same scale as ||Z_gen||, and
        # the ||W_ind|| / ||W_gen|| training diagnostic is only readable if the
        # two halves of the fusion input are comparably scaled.
        z_ind = 0.5 * (z_pool + z_clin) if self.cfg.zind_combine == "mean" else (z_pool + z_clin)

        if self.cfg.fusion == "gated":
            g = self.gate_scale * torch.sigmoid(self.gate_mlp(context.e_ind.to(z_ind.dtype)))
            z_final = self.gate_norm(z_gen + g * z_ind)
        else:
            z_final = self.fusion(torch.cat([z_gen, z_ind], dim=-1))

        if self.cfg.debug and not torch.isfinite(z_ind).all():
            raise RuntimeError("Z_ind is not finite -- a NaN here passes straight "
                               "through the zero fusion block (0 * NaN = NaN)")

        if return_parts:
            return ContextOutput(z_final=z_final, z_gen=z_gen, z_ind=z_ind,
                                 tokens=tokens, r=r, w=w,
                                 rho=stats["rho"], empty=stats["empty"])
        if return_tokens:
            return z_final, tokens
        return z_final

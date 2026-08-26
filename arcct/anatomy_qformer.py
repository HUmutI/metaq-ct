"""Anatomy-bound Q-Former wrapper (Path 3).

Restricts each query token's cross-attention to a specific spatial region
derived from the per-sample TotalSegmentator mask. Query roles are:

    queries 0-9 (10) -> one per fine anatomy (lung lobes, trachea, heart,
        aorta, vessels, esophagus). Each queries-i can attend ONLY to voxels
        with TS label i+1 (since label 0 is background).

    queries 10-27 (18) -> one per pathology. Each pathology query attends
        to voxels inside ANY anatomy in PATHOLOGY_FINE_ORGANS[p].

    queries 28-31 (4) -> "global" queries with unrestricted attention.

When ``has_mask[i]`` is False for sample i, all 32 queries get unrestricted
attention for that row (no role enforcement). This guarantees the loss
remains well-defined on the small fraction of volumes that lack TS masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcct.qformer import QFormer


def build_role_mask(
    ts_mask: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    anatomy_labels: list[int],
    pathology_labels: list[list[int]],
    num_global: int = 4,
    *,
    row_index: torch.Tensor | None = None,
    return_stats: bool = False,
):
    """Construct ``(B, Q, N)`` boolean mask.

    Args
    ----
    ts_mask : ``(B, D, H, W)`` integer mask, label 0 = background.
    spatial_shape : feature map spatial shape ``(Dp, Hp, Wp)``.
    anatomy_labels : list of TS label ids for the anatomy queries (length A).
    pathology_labels : list of lists; each inner list is the anatomies a
        single pathology query may attend to (length P).
    num_global : number of unrestricted "global" queries to append.
    row_index : optional LongTensor of rows to gather after the mask is built,
        so a wider bank can reuse a row (the conditioned pathology slots take
        the same organ union as their weight-tied partners). Keyword-only, and
        applied last, so the statistics below are still reported once per
        distinct region rather than once per slot.
    return_stats : when True, return ``(mask, stats)`` with ``stats["rho"]`` the
        fraction of the feature grid each query may attend to and
        ``stats["empty"]`` a per-sample flag for regions that never reached the
        grid at all.

    Returns
    -------
    mask : ``(B, Q, N)`` boolean where True = "this query may attend to this
        voxel". Q = A + P + num_global, N = product(spatial_shape).
    """
    B = ts_mask.shape[0]
    Dp, Hp, Wp = spatial_shape
    N = Dp * Hp * Wp
    A = len(anatomy_labels)
    P = len(pathology_labels)
    Q = A + P + num_global

    # Downsample TS mask to feature grid using nearest-neighbour. Cast to
    # float for interpolate, then back to long.
    ts_lo = F.interpolate(
        ts_mask.float().unsqueeze(1),
        size=spatial_shape,
        mode="nearest",
    )[:, 0].long()                                         # (B, Dp, Hp, Wp)
    ts_flat = ts_lo.flatten(1)                             # (B, N)

    out = torch.zeros(B, Q, N, dtype=torch.bool, device=ts_mask.device)

    # Anatomy queries (one per organ).
    for ai, organ_label in enumerate(anatomy_labels):
        out[:, ai] = ts_flat.eq(int(organ_label))

    # Pathology queries (union of constituent organs).
    for pi, organs in enumerate(pathology_labels):
        if not organs:
            # Pathology without an anatomy mapping -> unrestricted attention.
            out[:, A + pi] = True
            continue
        any_organ = torch.zeros_like(ts_flat, dtype=torch.bool)
        for organ_label in organs:
            any_organ = any_organ | ts_flat.eq(int(organ_label))
        out[:, A + pi] = any_organ

    # Global queries: unrestricted.
    out[:, A + P:] = True

    # Routing density and the empty-region flag are computed BEFORE the override
    # below, and that ordering is the whole point. After the override an empty
    # region has every key allowed, so it reports rho = 1.0 -- the most tightly
    # routed case would read as the least routed one, exactly inverted.
    n_allowed = out.sum(dim=-1)                            # (B, Q)
    rho = n_allowed.to(torch.float32) / float(N)           # (B, Q)
    empty = n_allowed.eq(0)                                # (B, Q)

    # Safety: a query whose region is entirely empty in this volume would
    # have NO attendable keys, which makes nn.MultiheadAttention return NaN.
    # In that case, unrestrict the query so attention falls back to the
    # whole volume.
    #
    # A region with no voxels on the feature grid does not fail here - it is
    # turned into an unrestricted query, so the model keeps running and that
    # query quietly stops being an anatomy query. Measured over 8816 pediatric
    # volumes: right middle lobe reaches the grid in 89.9% of cases, central
    # airway 96.8%, upper abdomen 96.9%. Record it so the training log can say
    # how often anatomy routing was actually in force.
    #
    # last_empty is a function attribute overwritten by every call, so what a
    # caller reads is the most recent batch and nothing else. It is kept for
    # backward compatibility with the existing training log; anything that needs
    # a real rate should accumulate the returned stats instead.
    build_role_mask.last_empty = empty.float().mean(dim=0)  # (Q,)
    build_role_mask.last_rho = rho.mean(dim=0)              # (Q,)
    out = torch.where(empty.unsqueeze(-1), torch.ones_like(out), out)

    if row_index is not None:
        out = out.index_select(1, row_index)
        rho = rho.index_select(1, row_index)
        empty = empty.index_select(1, row_index)
    if return_stats:
        return out, {"rho": rho, "empty": empty}
    return out


class AnatomyQFormer(nn.Module):
    """Q-Former with per-query cross-attention masking.

    Wraps a base :class:`QFormer` and supplies an additive cross-attention
    mask built from the per-sample TS mask each forward pass.
    """

    def __init__(
        self,
        anatomy_labels: list[int],
        pathology_labels: list[list[int]],
        num_global: int = 4,
        dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        image_dim: int = 512,
        dropout: float = 0.0,
        pool: str = "mean",
    ):
        super().__init__()
        self.anatomy_labels = list(anatomy_labels)
        self.pathology_labels = [list(x) for x in pathology_labels]
        self.num_global = int(num_global)
        self.num_heads = int(num_heads)
        total_q = len(self.anatomy_labels) + len(self.pathology_labels) + self.num_global
        self.qformer = QFormer(
            num_queries=total_q,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            image_dim=image_dim,
            dropout=dropout,
            pool=pool,
        )

    def forward(
        self,
        feat_map: torch.Tensor,
        ts_mask: torch.Tensor | None = None,
        has_mask: torch.Tensor | None = None,
        return_tokens: bool = False,
        return_attn: bool = False,
        suppress_mask: bool = False,
    ):
        """Run Anatomy-Bound Q-Former.

        Args
        ----
        feat_map : ``(B, C, D, H, W)`` feature map.
        ts_mask : ``(B, D, H, W)`` or ``(B, 1, D, H, W)`` TS mask. May be
            None — in that case the role-mask is skipped and we reduce to a
            plain Q-Former.
        has_mask : optional ``(B,)`` boolean. Rows where this is False get
            unrestricted attention (no role enforcement).
        return_tokens : when True, return tokens too.
        return_attn : when True, also return last-block cross-attention
            (B, num_heads, Q, N) and the (B, Q, N) bool role-mask so callers
            can compute attention-supervision losses.
        suppress_mask : when True, disable the hard cross-attn routing mask.
            Used for attention-supervision passes where we want attention
            to be free + a soft penalty applied via the loss.
        """
        if ts_mask is None:
            return self.qformer(
                feat_map, cross_attn_mask=None,
                return_tokens=return_tokens, return_attn=return_attn,
            )
        if ts_mask.ndim == 5:
            ts_mask = ts_mask[:, 0]
        Dp, Hp, Wp = feat_map.shape[-3:]
        B = feat_map.shape[0]

        bool_mask = build_role_mask(
            ts_mask,
            (Dp, Hp, Wp),
            self.anatomy_labels,
            self.pathology_labels,
            num_global=self.num_global,
        )                                                  # (B, Q, N) True=allowed

        if has_mask is not None:
            # Rows without a mask get unrestricted attention.
            no_mask = (~has_mask).view(B, 1, 1)
            bool_mask = bool_mask | no_mask

        Q = bool_mask.shape[1]
        N = bool_mask.shape[2]

        if suppress_mask:
            attn_mask_arg = None
        else:
            ignore_mask = ~bool_mask
            attn_mask_arg = ignore_mask.unsqueeze(1).expand(
                B, self.num_heads, Q, N
            ).reshape(B * self.num_heads, Q, N)

        out = self.qformer(
            feat_map, cross_attn_mask=attn_mask_arg,
            return_tokens=return_tokens, return_attn=return_attn,
        )
        if return_attn:
            return (*out, bool_mask)
        return out

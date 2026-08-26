#!/usr/bin/env python3
"""S7 hard gate: at step 0 the context bank IS ARC-CT.

The whole experimental design rests on one claim -- that every number measured
during Phase 1 reads as "how far have I moved from ARC-CT", because at
initialisation the widened model reproduces the narrow one. This script turns
that claim into a test, and nothing downstream should run until it is green.

Five tiers, from cheapest and strictest to most informative:

  1  structural   exact ==, no forward pass: the four zero-initialisations and
                  the index vectors.
  2  numeric      a 39-slot AnatomyQFormer and the 67-slot bank, same weights,
                  same input: the unconditioned tokens and Z_gen agree, and
                  Z_final == Z_gen EXACTLY (W = [I|0] means adding exact zeros,
                  which is exact in IEEE754).
  3  invariance   same volume, two different indications -> Z_gen bit-identical.
                  Unlike tier 2 this stays true at every step of training, so it
                  is the assertion that protects the claim for the whole run.
  3b gradient     d Z_gen / d (context parameters) is exactly zero.
  4  tying        the bank has 40 rows, and a conditioned slot's gradient lands
                  on its general partner's row.

Tier 2 is a tolerance, not an equality, and the reason is worth stating: the
cross-attention GEMM has M=67 instead of M=39, so BLAS may tile it differently
and the last bits can move. The paper claim is therefore the one the design doc
already operationalises -- ||Z_final - img_lat_ARC|| < 1e-5 -- not "bit for bit".

Run: python tools/check_ctx_identity.py [--dim 64] [--full]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from arcct.anatomy_qformer import AnatomyQFormer          # noqa: E402
from arcct.context import ContextEncoder                   # noqa: E402
from arcct.context_qformer import ContextQFormer           # noqa: E402
from arcct.slots import CtxConfig, SlotLayout              # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


class _StubBert(nn.Module):
    """A stand-in for CXR-BERT so the gate runs offline and in seconds."""

    def __init__(self, d: int):
        super().__init__()
        self.emb = nn.Embedding(512, d)
        self.lin = nn.Linear(d, d)

    def forward(self, input_ids=None, attention_mask=None):
        return (self.lin(self.emb(input_ids)),)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64,
                    help="model width; 768 with --full to match production")
    ap.add_argument("--full", action="store_true",
                    help="use the real 27-class peds schema and the real CXR-BERT")
    a = ap.parse_args()

    # AMP is meaningless here: _amp_dtype() picks bf16 on modern GPUs, which has
    # 8 mantissa bits, and an identity claim measured at that precision claims
    # nothing. Everything below is fp32 on CPU, deterministic, TF32 off.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)

    if a.full:
        from arcct.schema import active
        sch = active()
        pathologies = sch["PATHOLOGIES"]
        pathology_labels = [sch["PATHOLOGY_FINE_ORGANS"][n] for n in pathologies]
        anatomy_labels = list(range(1, len(sch["FINE_LABEL_NAMES"])))
        dim, depth, heads, image_dim, grid = a.dim, 4, 8, 512, 12
    else:
        anatomy_labels = list(range(1, 11))
        pathology_labels = [[1], [1, 2], [], [4], [9], [2, 3, 5]]
        dim, depth, heads, image_dim, grid = a.dim, 2, 4, 32, 4

    n_path = len(pathology_labels)
    L = SlotLayout.phase1(n_path=n_path, n_anatomy=len(anatomy_labels))
    cfg = CtxConfig(enabled=True, c1=True, c2=True, selfattn="split",
                    zind_combine="mean", debug=True)

    # -- build both models with identical weights ---------------------------
    legacy = AnatomyQFormer(
        anatomy_labels=anatomy_labels, pathology_labels=pathology_labels,
        num_global=L.n_glob_gen, dim=dim, depth=depth, num_heads=heads,
        image_dim=image_dim, pool="mean").eval().double()

    bert = None if a.full else _StubBert(dim)
    ctx_enc = ContextEncoder(dim=dim, bert=bert, age_dim=16, sex_dim=8, int_dim=8)
    model = ContextQFormer(
        anatomy_labels=anatomy_labels, pathology_labels=pathology_labels,
        layout=L, cfg=cfg, dim=dim, depth=depth, num_heads=heads,
        image_dim=image_dim, context=ctx_enc).eval().double()
    missing, unexpected = model.load_arcct_state_dict(legacy.state_dict(), verbose=False)
    check(not unexpected, f"unexpected keys when loading the legacy bank: {unexpected[:4]}")
    model.relevance.set_class_embeddings(torch.randn(n_path, dim).double())

    B, N = 2, grid ** 3
    feat = torch.randn(B, image_dim, grid, grid, grid).double()
    ts = torch.randint(0, len(anatomy_labels) + 1, (B, grid * 2, grid * 2, grid * 2))
    ts[ts == 4] = 5                                  # keep one region absent
    has = torch.tensor([True, False])                # and one row mask-free
    ids_a = torch.randint(1, 400, (B, 9))
    ids_b = torch.randint(1, 400, (B, 9))
    am = torch.ones(B, 9, dtype=torch.long)
    band = torch.tensor([0, 7])
    sex = torch.tensor([0, 1])

    # ============================ TIER 1 =================================
    W = model.fusion.weight
    check(float(W[:, dim:].abs().max()) == 0.0, "the right fusion block must be exactly zero")
    check(torch.equal(W[:, :dim], torch.eye(dim, dtype=W.dtype)),
          "the left fusion block must be exactly the identity")
    check(float(model.conditioner.g_x) == 0.0, "g_x must start at zero")
    for nm in ("to_gamma", "to_beta"):
        lin = getattr(model.conditioner, nm)
        check(float(lin.weight.abs().max()) == 0.0, f"conditioner.{nm}.weight must be zero")
        check(float(lin.bias.abs().max()) == 0.0, f"conditioner.{nm}.bias must be zero")
    check(float(model.relevance.b) == cfg.beta_init, "b must start at RAC_CTX_BETA_INIT")
    check(float(model.relevance.beta) > 0.0, "beta = softplus(b) must be positive")
    check(float(model.context.P_int.weight.abs().max()) == 0.0, "P_int must be zero")
    check(float(model.context.e_int.abs().max()) == 0.0, "e_int must be zero")
    check(torch.equal(model.qformer.queries[:L.n_gen], legacy.qformer.queries),
          "the first n_gen rows must be the legacy bank verbatim")
    check(tuple(model.qformer.queries.shape) == (L.n_rows, dim),
          f"bank is {tuple(model.qformer.queries.shape)}, expected {(L.n_rows, dim)}")
    s2r, rr = model.slot_to_row.tolist(), model.role_row.tolist()
    check(s2r[:L.n_gen] == list(range(L.n_gen)), "slot_to_row prefix")
    check(s2r[L.n_gen:L.n_gen + n_path] == L.path_gen_index, "conditioned block must be tied")
    check([i for i, (x, y) in enumerate(zip(s2r, rr)) if x != y] == L.glob_clin_index,
          "slot_to_row and role_row must differ only at the clinical slot")

    # ============================ TIER 2 =================================
    with torch.no_grad():
        z_legacy, t_legacy = legacy(feat, ts, has, return_tokens=True)
        out = model(feat, ts, has, input_ids=ids_a, attention_mask=am,
                    age_band=band, sex=sex, return_parts=True)
    d_tok = float((out.tokens[:, :L.n_gen] - t_legacy).abs().max())
    d_gen = float((out.z_gen - z_legacy).abs().max())
    check(d_tok < 1e-10, f"unconditioned tokens drifted: max|d|={d_tok:.3e}")
    check(d_gen < 1e-10, f"Z_gen drifted from the legacy pooled latent: max|d|={d_gen:.3e}")
    check(torch.equal(out.z_final, out.z_gen),
          f"Z_final != Z_gen at step 0: max|d|={(out.z_final - out.z_gen).abs().max():.3e}")
    check(bool(torch.isfinite(out.z_ind).all()), "Z_ind must be finite even while ignored")
    check(float(out.w.min()) >= 1.0, "w >= 1 must hold at initialisation too")

    # Weight tying is a property of the QUERY INPUT, not of the output token. At
    # step 0 C1 is the identity, so a conditioned slot enters the stack with
    # exactly its general partner's vector -- that is what must hold, and it is
    # checked here.
    #
    # Their OUTPUT tokens legitimately differ, and it would be a bug if they did
    # not: the group constraint is one-directional, so a general slot attends
    # over n_gen keys while its conditioned twin attends over all n_slots. Same
    # query, wider key set, different answer. An earlier version of this gate
    # asserted the outputs matched and failed at 8.6e-01, which is the size of
    # that difference rather than a defect.
    with torch.no_grad():
        slots0 = model.qformer.queries.index_select(0, model.slot_to_row)
        cond_in = slots0.index_select(0, model.path_cond_index)
        gen_in = slots0.index_select(0, model.path_gen_index)
        d_pair = float((cond_in - gen_in).abs().max())
    check(torch.equal(cond_in, gen_in),
          f"tied query rows differ at the input: max|d|={d_pair:.3e}")

    # ============================ TIER 3 =================================
    with torch.no_grad():
        out_b = model(feat, ts, has, input_ids=ids_b, attention_mask=am,
                      age_band=torch.tensor([9, 2]), sex=torch.tensor([1, 2]),
                      return_parts=True)
    check(torch.equal(out.z_gen, out_b.z_gen),
          "Z_gen moved with the indication -- the isolation leaks")
    check(torch.equal(out.tokens[:, :L.n_gen], out_b.tokens[:, :L.n_gen]),
          "an unconditioned token moved with the indication")
    check(not torch.equal(out.tokens[:, L.n_gen:], out_b.tokens[:, L.n_gen:])
          or float(model.conditioner.g_x) == 0.0,
          "with open gates the conditioned tokens should move")

    # ... and it must still hold once the gates are OPEN, which is the state the
    # model spends the entire run in. Tier 3 at step 0 alone proves nothing.
    open_model = model
    with torch.no_grad():
        open_model.conditioner.g_x.fill_(1.0)
        open_model.conditioner.to_gamma.weight.normal_(std=0.05)
        open_model.conditioner.to_beta.weight.normal_(std=0.05)
        open_model.relevance.b.fill_(2.0)
        open_model.fusion.weight[:, dim:].normal_(std=0.05)
        o1 = open_model(feat, ts, has, input_ids=ids_a, attention_mask=am,
                        age_band=band, sex=sex, return_parts=True)
        o2 = open_model(feat, ts, has, input_ids=ids_b, attention_mask=am,
                        age_band=torch.tensor([9, 2]), sex=torch.tensor([1, 2]),
                        return_parts=True)
    check(torch.equal(o1.z_gen, o2.z_gen),
          "Z_gen moved with the indication once the gates were open")
    check(not torch.equal(o1.z_ind, o2.z_ind),
          "Z_ind did NOT move with the indication -- conditioning is not reaching it")
    check(not torch.equal(o1.z_final, o2.z_final),
          "Z_final did not move -- the fusion right block is not being used")
    check(not torch.equal(o1.z_final, o1.z_gen),
          "Z_final collapsed onto Z_gen with a nonzero right block")

    # ============================ TIER 3b ================================
    model.zero_grad(set_to_none=True)
    out_g = model(feat, ts, has, input_ids=ids_a, attention_mask=am,
                  age_band=band, sex=sex, return_parts=True)
    out_g.z_gen.pow(2).sum().backward()
    ctx_params = [("conditioner." + n, p) for n, p in model.conditioner.named_parameters()]
    ctx_params += [("relevance." + n, p) for n, p in model.relevance.named_parameters()]
    ctx_params += [("context." + n, p) for n, p in model.context.named_parameters()]
    for name, p in ctx_params:
        if p.grad is not None and float(p.grad.abs().max()) > 0:
            FAILS.append(f"Z_gen has a gradient path into {name} -- it is not isolated")
            break
    qg = model.qformer.queries.grad
    check(qg is not None, "the query bank must receive gradient from Z_gen")
    if qg is not None:
        clin_row = L.n_gen
        check(float(qg[clin_row].abs().max()) == 0.0,
              "the clinical query row must not receive gradient from Z_gen")
    check(model.fusion.weight.grad is None or
          float(model.fusion.weight.grad.abs().max()) == 0.0,
          "Z_gen is taken before the fusion and must not touch it")

    # ============================ TIER 4 =================================
    model.zero_grad(set_to_none=True)
    out_t = model(feat, ts, has, input_ids=ids_a, attention_mask=am,
                  age_band=band, sex=sex, return_parts=True)
    out_t.tokens.index_select(1, model.path_cond_index).pow(2).sum().backward()
    qg = model.qformer.queries.grad
    check(qg is not None and float(qg[L.path_gen_index].abs().max()) > 0,
          "a conditioned slot's gradient must land on its general partner's row "
          "-- that is what weight tying means")
    n_params = sum(1 for n, _ in model.qformer.named_parameters() if n == "queries")
    check(n_params == 1, "there must be exactly one query parameter, not an alias pair")

    if FAILS:
        print("S7 FAIL (%d):" % len(FAILS))
        for f in FAILS:
            print("   ", f)
        return 1
    print("S7 OK · Z_final == Z_gen exactly at step 0 · unconditioned tokens match "
          f"the 39-slot model to {d_tok:.1e} · Z_gen invariant to the indication "
          "with gates OPEN · no gradient path from Z_gen into any context module "
          "· one query parameter, tied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

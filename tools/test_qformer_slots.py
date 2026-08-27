#!/usr/bin/env python3
"""S2 regression gate for the Q-Former slot/self-attention plumbing.

Four things have to hold before anything is built on top of the widened bank.

1.  A plain 39-query forward is *bit-identical* to the pinned pre-change baseline.
    The new keyword arguments all default to None and take the same branch, but
    "should be identical" is exactly the claim that has to be measured rather
    than asserted, so the reference module is loaded straight out of git and run
    side by side.
2.  ``pool_index`` over every slot equals ``mean(dim=1)`` exactly.
3.  On a 67-slot bank the ``split`` and ``mask`` self-attention paths agree. They
    are two implementations of one constraint; if they ever disagree, one of the
    index vectors is wrong.
4.  A bank wider than ``num_queries`` without ``pool_index`` raises, because
    silently pooling the conditioned slots into the latent is the fourth and
    least visible indication-leak path.

Run: python tools/test_qformer_slots.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from arcct.qformer import QFormer                      # noqa: E402
from arcct.slots import SlotLayout                     # noqa: E402

FAILS: list[str] = []
# The reference is PINNED, not "HEAD". Once the change is committed, HEAD
# contains it, and a no-regression test that compares HEAD against HEAD passes
# while testing nothing at all. 8c5fc0d is the last commit before the Phase 1
# model changes; override with ARCCT_BASELINE_REF if the history is ever
# rewritten.
BASELINE_REF = os.environ.get("ARCCT_BASELINE_REF", "8c5fc0d")



def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def load_reference(ref: str = "") -> object | None:
    """Import arcct/qformer.py as it stands at ``ref`` (default: the pinned baseline) under a throwaway name."""
    ref = ref or BASELINE_REF
    try:
        src = subprocess.check_output(["git", "-C", HERE, "show", f"{ref}:arcct/qformer.py"],
                                      text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    fd, path = tempfile.mkstemp(suffix="_qformer_ref.py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("qformer_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.unlink(path)
    return mod


def build(cls, n_queries: int, seed: int = 0):
    torch.manual_seed(seed)
    return cls(num_queries=n_queries, dim=64, depth=2, num_heads=4,
               image_dim=32, pool="mean").eval()


def main() -> int:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    L = SlotLayout.phase1(27)
    Q, S, G = L.n_rows, L.n_slots, L.n_gen
    B, C, D = 2, 32, 4
    torch.manual_seed(1234)
    feat = torch.randn(B, C, D, D, D)

    # -- 1. the plain path is unchanged, measured against git ------------------
    ref_mod = load_reference()
    if ref_mod is None:
        FAILS.append("could not load arcct/qformer.py from the pinned baseline -- "
                     "the no-regression check did not run")
    else:
        new = build(QFormer, 39)
        old = build(ref_mod.QFormer, 39)
        old.load_state_dict(new.state_dict())          # identical weights, not just seeds
        with torch.no_grad():
            p_new, t_new = new(feat, return_tokens=True)
            p_old, t_old = old(feat, return_tokens=True)
        check(torch.equal(p_new, p_old),
              f"plain pooled output drifted from the baseline: max|d|={(p_new - p_old).abs().max():.3e}")
        check(torch.equal(t_new, t_old),
              f"plain tokens drifted from the baseline: max|d|={(t_new - t_old).abs().max():.3e}")

        # An explicitly supplied bank of exactly num_queries rows must also be
        # bit-identical: this is the path the context module will take.
        with torch.no_grad():
            qexp = new.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
            p_exp = new(feat, queries=qexp, pool_index=torch.arange(39))
        check(torch.equal(p_exp, p_old), "explicit queries + full pool_index must match the plain path")

    # -- 2. pool_index over everything == mean(dim=1) --------------------------
    qf = build(QFormer, 39)
    with torch.no_grad():
        a = qf(feat)
        b = qf(feat, pool_index=torch.arange(39))
    check(torch.equal(a, b), "pool_index=arange(Q) must equal mean(dim=1) exactly")

    # -- 3. split and mask agree on a 67-slot bank -----------------------------
    qf40 = build(QFormer, Q)
    slot_to_row = torch.tensor(L.slot_to_row)
    gen_index = torch.tensor(L.gen_index)
    with torch.no_grad():
        bank = qf40.queries.index_select(0, slot_to_row).unsqueeze(0).expand(B, -1, -1).contiguous()
        # deliberately perturb the conditioned half so a leak would show up
        bank = bank.clone()
        bank[:, G:] += 0.5 * torch.randn(B, S - G, qf40.dim)

        m = torch.zeros(S, S, dtype=torch.bool)
        m[:G, G:] = True                                # gen may NOT read cond
        check(bool((~m).any(-1).all()), "no row may be fully masked -- softmax would be NaN")

        p_split, t_split = qf40(feat, queries=bank, self_attn_split=G,
                                pool_index=gen_index, return_tokens=True)
        p_mask, t_mask = qf40(feat, queries=bank, self_attn_mask=m,
                              pool_index=gen_index, return_tokens=True)
    dt = (t_split - t_mask).abs().max().item()
    check(dt < 1e-5, f"split and mask self-attention disagree: max|d tokens|={dt:.3e}")
    dp = (p_split - p_mask).abs().max().item()
    check(dp < 1e-5, f"split and mask pooled outputs disagree: max|d|={dp:.3e}")

    # -- 3b. the isolation actually isolates ----------------------------------
    # Same image, same unconditioned rows, a completely different conditioned
    # half: Z_gen must not move. This is the claim, tested directly.
    with torch.no_grad():
        other = bank.clone()
        other[:, G:] = torch.randn(B, S - G, qf40.dim)
        p_other, t_other = qf40(feat, queries=other, self_attn_split=G,
                                pool_index=gen_index, return_tokens=True)
    check(torch.equal(p_split, p_other),
          "Z_gen moved when only the conditioned slots changed -- the split leaks")
    check(torch.equal(t_split[:, :G], t_other[:, :G]),
          "unconditioned tokens moved when only the conditioned slots changed")
    # ... and the mask path must isolate just as hard.
    with torch.no_grad():
        p_other_m = qf40(feat, queries=other, self_attn_mask=m, pool_index=gen_index)
    check(torch.equal(p_mask, p_other_m), "the mask path leaks the conditioned half into Z_gen")

    # -- 4. a wide bank without pool_index must raise --------------------------
    try:
        with torch.no_grad():
            qf40(feat, queries=bank, self_attn_split=G)
        FAILS.append("a 67-slot bank over 40 rows without pool_index must raise")
    except ValueError:
        pass

    # a wrong-width bank must also raise rather than broadcast
    try:
        with torch.no_grad():
            qf40(feat, queries=bank[:, :, :8], pool_index=gen_index)
        FAILS.append("a bank with the wrong feature dim must raise")
    except ValueError:
        pass

    if FAILS:
        print("S2 FAIL (%d):" % len(FAILS))
        for f in FAILS:
            print("   ", f)
        return 1
    print(f"S2 OK · plain path bit-identical to the baseline · split==mask (max|d|={dt:.1e}) "
          f"· Z_gen isolated exactly · wide-bank guard raises")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

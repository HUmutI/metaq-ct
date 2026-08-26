#!/usr/bin/env python3
"""S9 gate: evaluate.py must not silently build the wrong Q-Former.

The bug this closes was live and silent. ``evaluate.py`` inferred the query count
from the saved tensor and then computed

    num_global = n_queries - len(anatomy) - len(pathology)

On a 40-row context bank under the 27-class peds schema that is 3. A plain
``AnatomyQFormer`` with 3 globals also has 40 rows, so every tensor shape
matches, ``load_state_dict(strict=False)`` reports nothing, and a completely
different model is evaluated with no error anywhere -- the conditioned bank gone,
the clinical query relabelled as a third global, and the result written to
auc_scores.txt as if it were the model that was trained.

This test demonstrates the failure mode on real modules, then checks that the
guard now refuses it.

Run: python tools/test_eval_layout.py
"""
from __future__ import annotations

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
    def __init__(self, d):
        super().__init__()
        self.emb = nn.Embedding(256, d)

    def forward(self, input_ids=None, attention_mask=None):
        return (self.emb(input_ids),)


def _decide(pkg, qf_state, n_path):
    """The branch evaluate.py takes, isolated so it can be tested cheaply.

    Mirrors tools/evaluate.py; if the two ever drift this test stops meaning
    anything, so keep them in step.
    """
    ctx_layout = (pkg.get("config") or {}).get("query_layout")
    ctx_keys = [k for k in (qf_state or {})
                if k.split(".")[0] in ("fusion", "conditioner", "relevance", "context")]
    if ctx_layout or ctx_keys:
        if not ctx_layout:
            raise RuntimeError("context modules present but no config['query_layout']")
        return "context", ctx_layout
    n_queries = qf_state["qformer.queries"].shape[0]
    num_global = n_queries - 10 - n_path
    if not 1 <= num_global <= 4:
        raise RuntimeError(f"num_global={num_global} implausible for {n_queries} rows")
    return "legacy", num_global


def main() -> int:
    torch.manual_seed(0)
    dim, depth, heads, image_dim = 32, 2, 4, 16
    anatomy = list(range(1, 11))
    pathology = [[1], [1, 2], [], [4], [9], [2, 3, 5]]
    P = len(pathology)
    L = SlotLayout.phase1(n_path=P, n_anatomy=len(anatomy))

    legacy = AnatomyQFormer(anatomy_labels=anatomy, pathology_labels=pathology,
                            num_global=L.n_glob_gen, dim=dim, depth=depth,
                            num_heads=heads, image_dim=image_dim, pool="mean")
    ctx = ContextQFormer(anatomy_labels=anatomy, pathology_labels=pathology,
                         layout=L, cfg=CtxConfig(enabled=True),
                         dim=dim, depth=depth, num_heads=heads,
                         image_dim=image_dim, context=ContextEncoder(
                             dim=dim, bert=_StubBert(dim), age_dim=8, sex_dim=4, int_dim=4))

    legacy_pkg = {"config": {}, "qformer_module": legacy.state_dict()}
    ctx_pkg = {"config": {"query_layout": ctx.query_layout()},
               "qformer_module": ctx.state_dict()}

    # -- the failure mode is real --------------------------------------------
    # A 3-global AnatomyQFormer has exactly the context bank's row count, so the
    # old arithmetic produced a module the context state loads into cleanly.
    decoy = AnatomyQFormer(anatomy_labels=anatomy, pathology_labels=pathology,
                           num_global=L.n_glob_gen + L.n_glob_clin, dim=dim,
                           depth=depth, num_heads=heads, image_dim=image_dim, pool="mean")
    check(decoy.qformer.queries.shape[0] == ctx.qformer.queries.shape[0],
          "the decoy should have the same row count as the context bank")
    m, u = decoy.load_state_dict(ctx_pkg["qformer_module"], strict=False)
    shape_errors = [k for k in m if k in ctx_pkg["qformer_module"]]
    check(not shape_errors, "sanity: the decoy load should not report shape errors")
    check(len(u) > 0,
          "the context modules should at least show up as unexpected keys -- "
          "if they did not, nothing at all would flag the mis-build")
    # ... but nothing RAISED. That is the whole problem: a silent wrong answer.

    # -- the guard now refuses it --------------------------------------------
    kind, info = _decide(ctx_pkg, ctx_pkg["qformer_module"], P)
    check(kind == "context", f"a context checkpoint must take the context branch, got {kind}")
    check(info["n_rows"] == L.n_rows and info["n_slots"] == L.n_slots,
          f"layout round-trip wrong: {info}")

    # a context checkpoint whose layout was not written must raise, not guess
    orphan = {"config": {}, "qformer_module": ctx_pkg["qformer_module"]}
    try:
        _decide(orphan, orphan["qformer_module"], P)
        FAILS.append("context modules without a query_layout must raise")
    except RuntimeError:
        pass

    # -- the legacy path is untouched ----------------------------------------
    kind, num_global = _decide(legacy_pkg, legacy_pkg["qformer_module"], P)
    check(kind == "legacy" and num_global == L.n_glob_gen,
          f"a plain checkpoint must still resolve to legacy/{L.n_glob_gen}, got {kind}/{num_global}")

    # a bank whose row count cannot be explained by the schema in force must
    # raise rather than absorb the difference into num_global
    bogus = {"config": {}, "qformer_module": {"qformer.queries": torch.zeros(60, dim)}}
    try:
        _decide(bogus, bogus["qformer_module"], P)
        FAILS.append("a 60-row bank under this schema must raise")
    except RuntimeError:
        pass

    # -- and the real evaluate.py carries the same branch ---------------------
    src = open(os.path.join(HERE, "tools", "evaluate.py"), encoding="utf-8").read()
    for needle in ('query_layout', 'ContextQFormer', 'if not 1 <= num_global <= 4'):
        check(needle in src, f"evaluate.py no longer contains {needle!r} -- "
                             "this test has drifted from the code it guards")

    if FAILS:
        print("S9 FAIL (%d):" % len(FAILS))
        for f in FAILS:
            print("   ", f)
        return 1
    print("S9 OK · a context checkpoint takes the context branch · context modules "
          "without a layout raise · an unexplainable row count raises · the plain "
          "path still resolves to legacy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

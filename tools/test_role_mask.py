#!/usr/bin/env python3
"""S3 regression gate for build_role_mask: rho before the override, and row_index.

The bug being fixed: an anatomy region that never reaches the 12^3 feature grid
has no attendable keys, so the mask is unrestricted to stop MultiheadAttention
returning NaN. Any routing statistic read *after* that override sees a fully
open mask and reports rho = 1.0 -- the most tightly routed case reads as the
least routed one. Measured rates make this concrete rather than theoretical:
right middle lobe misses the grid in ~10% of pediatric volumes.

Checks:
  1. an absent region reports rho = 0 and empty = True, while its mask row is
     still all-True (the NaN guard must survive the fix);
  2. with row_index=None and return_stats=False the returned mask is byte-equal
     to the version at git HEAD, so nothing about training changed;
  3. row_index gathers rows correctly, and the tied conditioned slots inherit
     exactly the mask of their general partners;
  4. statistics are reported per distinct region, not per slot -- gathering must
     not make one absent region look like two.

Run: python tools/test_role_mask.py
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

from arcct.anatomy_qformer import build_role_mask        # noqa: E402
from arcct.slots import SlotLayout                       # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def load_reference(ref: str = "HEAD"):
    """build_role_mask as it stands at ``ref``, imported under a throwaway name."""
    try:
        src = subprocess.check_output(
            ["git", "-C", HERE, "show", f"{ref}:arcct/anatomy_qformer.py"],
            text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    # The reference file imports arcct.qformer, which is importable from HERE.
    fd, path = tempfile.mkstemp(suffix="_anat_ref.py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("anat_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.unlink(path)
    return mod.build_role_mask


def main() -> int:
    torch.manual_seed(0)
    B, Dp = 3, 4                       # 4^3 = 64 grid cells, enough to be exact
    N = Dp ** 3
    anatomy = list(range(1, 11))       # TS labels 1..10
    # A tiny pathology table with the three shapes that matter: one organ, a
    # union, and the deliberately unrouted [] case.
    pathology = [[1], [1, 2, 3], [], [4], [9]]
    A, P, G = len(anatomy), len(pathology), 2

    # A volume where label 4 is ABSENT everywhere: that is the empty region.
    ts = torch.randint(0, 11, (B, Dp, Dp, Dp))
    ts[ts == 4] = 5
    check(int((ts == 4).sum()) == 0, "test fixture: label 4 must be absent")
    # ... and label 9 present but rare, so rho is small but non-zero.
    ts[:, 0, 0, 0] = 9

    mask, st = build_role_mask(ts, (Dp, Dp, Dp), anatomy, pathology,
                               num_global=G, return_stats=True)
    rho, empty = st["rho"], st["empty"]
    Q = A + P + G
    check(tuple(mask.shape) == (B, Q, N), f"mask shape {tuple(mask.shape)}")
    check(tuple(rho.shape) == (B, Q), f"rho shape {tuple(rho.shape)}")

    # -- 1. the absent region ------------------------------------------------
    ai = 3                                            # anatomy query for label 4
    check(bool((rho[:, ai] == 0).all()), f"absent region must report rho=0, got {rho[:, ai]}")
    check(bool(empty[:, ai].all()), "absent region must be flagged empty")
    check(bool(mask[:, ai].all()), "absent region must still be fully unrestricted (NaN guard)")
    # the pathology query routed to [4] is the same story
    pi = A + 3
    check(bool((rho[:, pi] == 0).all()), "pathology routed to an absent organ must report rho=0")
    check(bool(mask[:, pi].all()), "pathology routed to an absent organ must be unrestricted")

    # -- and a present region must NOT look empty ----------------------------
    present = [i for i in range(A) if i != ai]
    check(bool((rho[:, present] > 0).all()), "a present region reported rho=0")
    check(not bool(empty[:, present].any()), "a present region was flagged empty")
    # unrouted pathology ([]) and globals are unrestricted by construction
    check(bool((rho[:, A + 2] == 1.0).all()), "an unrouted pathology must have rho=1")
    check(bool((rho[:, A + P:] == 1.0).all()), "global queries must have rho=1")
    # the rare label 9 must be small but non-zero -- this is the case the old
    # code could not distinguish from "absent", because both ended up all-True
    r9 = rho[:, A + 4]
    check(bool(((r9 > 0) & (r9 < 0.2)).all()), f"rare region rho should be small, got {r9}")

    # -- 2. no behavioural change for existing callers ------------------------
    ref = load_reference()
    if ref is None:
        FAILS.append("could not load anatomy_qformer.py from git HEAD -- "
                     "the no-regression check did not run")
    else:
        old = ref(ts, (Dp, Dp, Dp), anatomy, pathology, num_global=G)
        new = build_role_mask(ts, (Dp, Dp, Dp), anatomy, pathology, num_global=G)
        check(torch.equal(old, new), "the returned mask changed for existing callers")
        check(isinstance(new, torch.Tensor),
              "without return_stats the return type must stay a bare tensor")

    # -- 3. row_index and the tied conditioned block --------------------------
    L = SlotLayout.phase1(n_path=P, n_anatomy=A, n_glob_gen=G)
    ridx = torch.tensor(L.role_row)
    wide, wst = build_role_mask(ts, (Dp, Dp, Dp), anatomy, pathology, num_global=G,
                                row_index=ridx, return_stats=True)
    check(tuple(wide.shape) == (B, L.n_slots, N), f"wide mask shape {tuple(wide.shape)}")
    check(torch.equal(wide[:, :L.n_gen], mask), "the unconditioned block must be untouched")
    for k, (cond_slot, gen_slot) in enumerate(zip(L.path_cond_index, L.path_gen_index)):
        if not torch.equal(wide[:, cond_slot], mask[:, gen_slot]):
            FAILS.append(f"conditioned slot {cond_slot} does not inherit general slot {gen_slot}")
            break
    clin = L.glob_clin_index[0]
    check(bool(wide[:, clin].all()), "the clinical global must inherit an unrestricted row")

    # -- 4. statistics stay per-region ---------------------------------------
    # Gathering repeats rows, so an absent region appears twice in the wide
    # tensor. The reported rate must come from the 39-row base, not the 67-slot
    # bank, or every conditioned run would inflate its own empty-region rate.
    base_empty_rate = float(empty.float().mean(dim=0)[ai])
    check(abs(base_empty_rate - 1.0) < 1e-6, "fixture: label 4 empty in every sample")
    check(float(build_role_mask.last_empty[ai]) == 1.0,
          "last_empty must still report the absent region")
    check(build_role_mask.last_empty.shape[0] == Q,
          f"last_empty must be sized by the base bank ({Q}), "
          f"got {build_role_mask.last_empty.shape[0]}")
    check(build_role_mask.last_rho.shape[0] == Q, "last_rho must be sized by the base bank")

    # -- 5. a volume with NO mask at all -------------------------------------
    # ts all background: every region is empty, everything unrestricted, and rho
    # is 0 everywhere except the queries that are unrestricted by construction.
    zero = torch.zeros(1, Dp, Dp, Dp, dtype=torch.long)
    zmask, zst = build_role_mask(zero, (Dp, Dp, Dp), anatomy, pathology,
                                 num_global=G, return_stats=True)
    check(bool(zmask.all()), "an empty volume must leave every query unrestricted")
    check(bool(zst["empty"][0, :A].all()), "an empty volume must flag every anatomy region")
    check(float(zst["rho"][0, :A].max()) == 0.0,
          "an empty volume must report rho=0, not the post-override 1.0 -- "
          "this is the inversion the fix exists to prevent")

    if FAILS:
        print("S3 FAIL (%d):" % len(FAILS))
        for f in FAILS:
            print("   ", f)
        return 1
    print("S3 OK · rho computed before the override · mask byte-equal to HEAD "
          "· row_index inherits the tied rows · stats stay per-region")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

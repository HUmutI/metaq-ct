#!/usr/bin/env python3
"""S11 gate: the dataset's clinical-context contract.

Checks the parts that are easy to get subtly wrong and impossible to notice
afterwards:

  * the draw is a pure function of (seed, epoch, index), not random.random().
    This trainer auto-resumes on every SLURM requeue and Python's global RNG is
    not in the checkpoint, so a naive draw would silently change schedule
    mid-run and differ with the worker count.
  * rows that are already vacuous or absent are NOT counted as dropped. Counting
    them would put the effective rate on the adult half -- where 75.9% of
    CT-RATE indications are vacuous -- far above the configured one.
  * a counterfactual is never paired with a dropped row, never equals the row's
    own indication, and stays inside the cohort.
  * without set_epoch() the schedule is frozen, which is the persistent_workers
    trap: a fixed 30% of volumes that never see their indication is a different
    and worse experiment than 30% per epoch.

The dataset needs npz volumes and CSVs, so the context logic is exercised
through a light stand-in that reuses the real methods.

Run: python tools/test_dataset_context.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from arcct.context import NO_INDICATION                    # noqa: E402
from arcct.dataset import RACDatasetV4                     # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


class Fake(RACDatasetV4):
    """RACDatasetV4's context machinery without the filesystem."""

    def __init__(self, n=400, dropout=0.3, cf=0.25, seed=0, same_cohort=True):
        self.NO_INDICATION = NO_INDICATION
        self.acc2ind, self.acc2ind_status = {}, {}
        self.acc2band, self.acc2sex = {}, {}
        self.ind_pool, self.ind_cohort = [], {}
        for i in range(n):
            acc = f"vol_{i:04d}.nii.gz"
            cohort = "peds" if i % 2 == 0 else "ctrate"
            # a quarter of the corpus is vacuous, like CT-RATE
            if i % 4 == 3:
                self.acc2ind[acc] = NO_INDICATION
                self.acc2ind_status[acc] = "vacuous"
            else:
                self.acc2ind[acc] = f"indication text {i}"
                self.acc2ind_status[acc] = "present"
                self.ind_pool.append(acc)
                self.ind_cohort[acc] = cohort
            self.acc2band[acc] = i % 10
            self.acc2sex[acc] = i % 3
        self.ctx_seed = seed
        self.epoch = 0
        self.ind_dropout = dropout
        self.cf_prob = cf
        self.cf_same_cohort = same_cohort
        self.n = n

    def accs(self):
        return [f"vol_{i:04d}.nii.gz" for i in range(self.n)]


def main() -> int:
    N = 2000
    d = Fake(n=N)
    accs = d.accs()
    ctxs = [d._context_for(i, a) for i, a in enumerate(accs)]

    # -- reproducibility -------------------------------------------------------
    again = Fake(n=N)
    ctxs2 = [again._context_for(i, a) for i, a in enumerate(accs)]
    check(all(a["indication"] == b["indication"] and a["indication_cf"] == b["indication_cf"]
              for a, b in zip(ctxs, ctxs2)),
          "the same (seed, epoch, index) must give the same draw")

    # index order must not matter -- a worker sees a scattered subset
    shuffled = Fake(n=N)
    order = list(range(N))[::7] + list(range(N))[1::7]
    for i in order:
        got = shuffled._context_for(i, accs[i])
        if got["indication"] != ctxs[i]["indication"]:
            FAILS.append("the draw depends on visit order -- it must depend only on idx")
            break

    # a different seed must give a different schedule
    other = Fake(n=N, seed=1)
    diff = sum(1 for i, a in enumerate(accs)
               if other._context_for(i, a)["ind_dropped"] != ctxs[i]["ind_dropped"])
    check(diff > N * 0.05, f"a different RAC_CTX_SEED barely changed the schedule ({diff}/{N})")

    # a different epoch must give a different schedule -- and set_epoch is the
    # only way to move it, which is exactly what persistent_workers defeats
    ep = Fake(n=N)
    ep.set_epoch(1)
    diff_ep = sum(1 for i, a in enumerate(accs)
                  if ep._context_for(i, a)["ind_dropped"] != ctxs[i]["ind_dropped"])
    check(diff_ep > N * 0.05,
          f"epoch 1 gave the same dropout mask as epoch 0 ({diff_ep}/{N}) -- "
          "without set_epoch the mask is frozen for the whole run")

    # -- the dropout rate is measured over ELIGIBLE rows -----------------------
    present = [c for c in ctxs if c["ind_status"] == "present"]
    dropped = [c for c in present if c["ind_dropped"]]
    rate = len(dropped) / len(present)
    check(abs(rate - 0.3) < 0.04, f"dropout rate over present rows = {rate:.3f}, expected ~0.30")
    vacuous_dropped = [c for c in ctxs if c["ind_status"] != "present" and c["ind_dropped"]]
    check(not vacuous_dropped,
          f"{len(vacuous_dropped)} vacuous rows were counted as dropped -- the "
          "effective rate on the adult half would then be far above 0.30")
    check(all(c["indication"] == NO_INDICATION for c in dropped),
          "a dropped row must carry exactly NO_INDICATION")
    check(all(c["indication"] == NO_INDICATION
              for c in ctxs if c["ind_status"] == "vacuous"),
          "a vacuous row must carry exactly NO_INDICATION too -- one canonical "
          "empty input, or it becomes a second dropout channel")

    # -- the counterfactual ----------------------------------------------------
    cfs = [(i, c) for i, c in enumerate(ctxs) if c["indication_cf"]]
    check(len(cfs) > 0, "no counterfactual was ever drawn")
    eligible = [c for c in present if not c["ind_dropped"]]
    cf_rate = len(cfs) / len(eligible)
    check(abs(cf_rate - 0.25) < 0.05, f"cf rate over eligible rows = {cf_rate:.3f}, expected ~0.25")
    check(all(not ctxs[i]["ind_dropped"] for i, _ in cfs),
          "a counterfactual against a dropped row is not the experiment")
    check(all(c["indication_cf"] != c["indication"] for _, c in cfs),
          "a counterfactual must differ from the row's own indication")
    check(all(c["indication_cf"] != NO_INDICATION for _, c in cfs),
          "a counterfactual must never be the empty input")
    # cohort match: the partner text carries its index, so it can be resolved back
    bad_cohort = 0
    for i, c in cfs:
        j = int(c["indication_cf"].rsplit(" ", 1)[1])
        if (j % 2 == 0) != (i % 2 == 0):
            bad_cohort += 1
    check(bad_cohort == 0,
          f"{bad_cohort} counterfactuals crossed cohorts -- pairing a pediatric CT "
          "with an adult indication teaches a cohort cue")

    # off by default when not training
    quiet = Fake(n=200, dropout=0.0, cf=0.0)
    q = [quiet._context_for(i, a) for i, a in enumerate(quiet.accs())]
    check(not any(c["ind_dropped"] for c in q), "dropout must be off when the rate is 0")
    check(not any(c["indication_cf"] for c in q), "no counterfactual when the rate is 0")

    # -- demographics pass through --------------------------------------------
    check(all(c["age_band"] == i % 10 for i, c in enumerate(ctxs)), "age band lookup")
    check(all(c["sex"] == i % 3 for i, c in enumerate(ctxs)), "sex lookup")
    missing = d._context_for(0, "not_a_volume.nii.gz")
    check(missing["age_band"] == -1 and missing["sex"] == -1,
          "an unknown accession must report -1, not band 0 (which means infant)")
    check(missing["indication"] == NO_INDICATION, "an unknown accession must get the empty input")

    if FAILS:
        print("S11 FAIL (%d):" % len(FAILS))
        for f in FAILS:
            print("   ", f)
        return 1
    print(f"S11 OK · draw is a pure function of (seed, epoch, idx) · dropout "
          f"{rate:.3f} over eligible rows only · cf {cf_rate:.3f}, never on a "
          f"dropped row, never cross-cohort · unknown accession -> -1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

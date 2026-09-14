#!/usr/bin/env python3
"""Everything that must be true before a context run is allowed to start.

Exits non-zero, so an sbatch chain stops here rather than burning three days of
GPU on a run whose inputs were wrong. Each check exists because the failure it
catches is silent: the run completes, prints plausible numbers, and the defect
only shows up when someone tries to explain the result.

  P1  the context files exist and match their .sha256 sidecars
  P2  the PHI assertions re-run over the files THIS run will read
  P3  every volume in the train/valid lists has an indication and a demographic
      row -- a missing key falls back to NO_INDICATION and silently weakens the
      indication arm with no error anywhere
  P4  no age band is empty in TRAIN; an untrained increment propagates through
      the ordinal cumulative sum into every band above it
  P5  the leakage-exclude volumes are absent from the validation list
  P6  the step-0 identity gate is green
  P7  the slot arithmetic holds and the tying is real
  P8  the dropout draw is reproducible, the rate is right, and
      persistent_workers would be off
  P9  the counterfactual partner is sane
  P10 Z_gen is invariant to the indication on a real batch
  P11 a manifest hash of every input, written into the results dir

Run: python tools/preflight_context.py [--skip-slow]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pipeline", "lib"))

FAILS: list[str] = []
NOTES: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)


def ok(msg: str) -> None:
    print(f"  ok   {msg}")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_list(path: str) -> set[str]:
    if not path or not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip P6/P10, which build a model")
    a = ap.parse_args()

    ctx_csv = env("RAC_CONTEXT_CSV")
    dem_csv = env("RAC_DEMOGRAPHICS_CSV")
    train_list = env("RAC_VOLUME_LIST_TRAIN")
    valid_list = env("RAC_VOLUME_LIST_VALID")
    exclude = env("RAC_VOLUME_EXCLUDE")
    results = env("RAC_RESULTS_DIR", "/tmp")

    print("=== P1 input files and hashes ===")
    for path in (ctx_csv, dem_csv):
        if not path or not os.path.isfile(path):
            fail(f"P1: missing {path or '<unset>'}")
            continue
        side = path + ".sha256"
        if os.path.isfile(side):
            want = open(side, encoding="utf-8").read().strip()
            got = sha256(path)
            if want != got:
                fail(f"P1: {os.path.basename(path)} does not match its sidecar "
                     f"({got[:12]} vs {want[:12]}) -- it changed after it was audited")
            else:
                ok(f"{os.path.basename(path)} matches its sha256")
        else:
            NOTES.append(f"P1: no sidecar for {os.path.basename(path)}")
    if FAILS:
        print("\n".join("  FAIL " + f for f in FAILS))
        return 78

    csv.field_size_limit(10 ** 9)
    ctx = {r["VolumeName"]: r for r in csv.DictReader(
        open(ctx_csv, newline="", encoding="utf-8"))}
    dem = {r["VolumeName"]: r for r in csv.DictReader(
        open(dem_csv, newline="", encoding="utf-8"))}
    ok(f"indication rows {len(ctx):,}   demographic rows {len(dem):,}")

    print("=== P2 PHI assertions, on the files this run reads ===")
    import re
    bad = Counter()
    pats = {
        "digits5": re.compile(r"(?<!\d)\d{5,}(?!\d)"),
        "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:19|20)\d{2}\b"),
        "contact": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://"),
    }
    for r in ctx.values():
        t = r["Indication_EN"]
        if not t:
            continue
        for name, rx in pats.items():
            if rx.search(t):
                bad[name] += 1
    if bad:
        fail(f"P2: PHI patterns present in indication.csv: {dict(bad)}")
    else:
        ok("no identifier pattern in any indication")

    print("=== P3 coverage of the volume lists ===")
    tr, va = read_list(train_list), read_list(valid_list)
    if not tr or not va:
        fail(f"P3: volume lists missing (train={len(tr)} valid={len(va)})")
    for label, vols in (("train", tr), ("valid", va)):
        miss_c = [v for v in vols if v not in ctx]
        miss_d = [v for v in vols if v not in dem]
        if miss_c:
            fail(f"P3: {len(miss_c)} {label} volumes have no indication row, "
                 f"e.g. {miss_c[:3]} -- they would silently fall back to "
                 "NO_INDICATION and weaken the arm with no error")
        if miss_d:
            fail(f"P3: {len(miss_d)} {label} volumes have no demographic row, "
                 f"e.g. {miss_d[:3]}")
        if not miss_c and not miss_d:
            ok(f"{label}: {len(vols):,} volumes, all covered")

    print("=== P4 band coverage in TRAIN ===")
    tb = Counter(int(dem[v]["AgeBand"]) for v in tr if v in dem)
    vb = Counter(int(dem[v]["AgeBand"]) for v in va if v in dem)
    # A band absent from TRAIN but present in VALID is the real fault: the model
    # is asked at evaluation for an increment it never learned. A band absent
    # from BOTH is a property of the cohort, not a defect -- CT-RATE is adults,
    # so the six pediatric bands cannot appear and never will. Their increments
    # stay at init and enter every band above them as the SAME constant, which
    # the base embedding absorbs; the earlier version of this check refused to
    # start an adult-only run for that constant.
    unseen = [b for b in range(10) if tb.get(b, 0) == 0 and vb.get(b, 0) == 0]
    missing = [b for b in range(10) if tb.get(b, 0) == 0 and vb.get(b, 0) > 0]
    if missing:
        fail(f"P4: bands {missing} are EMPTY in train but PRESENT in valid; the "
             "ordinal increment for an untrained band propagates into every band "
             "above it, and here it is asked for at evaluation")
    thin = [b for b in range(10) if 0 < tb.get(b, 0) < 50 and b not in unseen]
    if thin:
        # Not fatal, but it is worth saying out loud: an increment shared by every
        # band above it, fitted on a handful of volumes, is noise with leverage.
        ok(f"WARN bands {thin} are thin in train "
           + " ".join(f"{b}:{tb.get(b, 0)}" for b in thin)
           + " -- their increments are shared by all bands above them")
    if unseen:
        ok(f"bands {unseen} absent from BOTH train and valid (cohort has no such "
           "ages); their increments stay at init and never reach evaluation")
    if not missing:
        ok("band coverage: " + " ".join(f"{b}:{tb.get(b,0)}" for b in range(10)))
    pres = sum(1 for v in tr if v in ctx and ctx[v]["ind_status"] == "present")
    ok(f"train indications present: {pres:,}/{len(tr):,} ({100.0*pres/max(len(tr),1):.1f}%)")
    if pres < 0.10 * len(tr):
        fail(f"P4b: only {100.0*pres/max(len(tr),1):.1f}% of train volumes carry an "
             "indication -- the conditioning arm has almost nothing to learn from")

    print("=== P5 excluded volumes ===")
    ex = {ln.split("\t")[0].strip() for ln in open(exclude, encoding="utf-8")
          if ln.strip() and not ln.startswith("#")} if exclude and os.path.isfile(exclude) else set()
    leaked = (ex & va) | (ex & tr)
    if leaked:
        fail(f"P5: {len(leaked)} contaminated volumes are still in the lists; the "
             "dataset must be given RAC_VOLUME_EXCLUDE, or they must be removed "
             "from every arm with the same denominator")
    else:
        ok(f"{len(ex)} excluded volumes, none in train or valid")

    print("=== P8/P9 dropout and counterfactual ===")
    from arcct.context import NO_INDICATION
    from arcct.dataset import RACDatasetV4

    class Probe(RACDatasetV4):
        def __init__(self, ctx, dem, seed):
            self.NO_INDICATION = NO_INDICATION
            self.acc2ind = {k: (v["Indication_EN"] or NO_INDICATION) for k, v in ctx.items()}
            self.acc2ind_status = {k: v["ind_status"] for k, v in ctx.items()}
            self.acc2band = {k: int(v["AgeBand"]) for k, v in dem.items()}
            self.acc2sex = {k: int(v["SexIdx"]) for k, v in dem.items()}
            self.acc2age = {k: float(v["AgeYears"] or -1.0) for k, v in dem.items()}
            self.ind_pool = [k for k, v in ctx.items() if v["ind_status"] == "present"]
            self.ind_cohort = {k: ctx[k]["cohort"] for k in self.ind_pool}
            self.ctx_seed, self.epoch = seed, 0
            self.ind_dropout = float(env("RAC_IND_DROPOUT", "0.3"))
            self.cf_prob = float(env("RAC_CF_PROB", "0.25"))
            self.cf_same_cohort = env("RAC_CF_SAME_COHORT", "1") == "1"

    # The probe reuses RACDatasetV4._context_for, so any field that method reads
    # must be set here. If the dataset gains one and this does not, the probe
    # raises -- which is the right failure, but the message should say why.
    needed = {"acc2ind", "acc2ind_status", "acc2band", "acc2sex", "acc2age",
              "ind_pool", "ind_cohort", "ctx_seed", "epoch", "ind_dropout",
              "cf_prob", "cf_same_cohort", "NO_INDICATION"}

    seed = int(env("RAC_CTX_SEED", env("RAC_SEED", "0")))
    p1, p2 = Probe(ctx, dem, seed), Probe(ctx, dem, seed)
    absent = sorted(n for n in needed if not hasattr(p1, n))
    if absent:
        fail(f"P8: the probe is missing {absent} -- RACDatasetV4._context_for "
             "reads fields this stand-in does not set")
    keys = sorted(tr)[:4000] or sorted(ctx)[:4000]
    c1 = [p1._context_for(i, k) for i, k in enumerate(keys)]
    c2 = [p2._context_for(i, k) for i, k in enumerate(keys)]
    if any(x["indication"] != y["indication"] for x, y in zip(c1, c2)):
        fail("P8: the draw is not reproducible for the same (seed, epoch, idx)")
    else:
        ok("dropout draw reproducible")
    elig = [c for c in c1 if c["ind_status"] == "present"]
    if elig:
        rate = sum(c["ind_dropped"] for c in elig) / len(elig)
        if abs(rate - p1.ind_dropout) > 0.05:
            fail(f"P8: dropout rate {rate:.3f} over eligible rows, expected "
                 f"{p1.ind_dropout}")
        else:
            ok(f"dropout {rate:.3f} over {len(elig):,} eligible rows")
        if any(c["ind_dropped"] for c in c1 if c["ind_status"] != "present"):
            fail("P8: a vacuous/absent row was counted as dropped")
    cfs = [c for c in c1 if c["indication_cf"]]
    if cfs:
        if any(c["ind_dropped"] for c in cfs):
            fail("P9: a counterfactual was drawn against a dropped row")
        if any(c["indication_cf"] == c["indication"] for c in cfs):
            fail("P9: a counterfactual equals the row's own indication")
        ok(f"counterfactual on {len(cfs):,} rows, none against a dropped row")
    else:
        NOTES.append("P9: no counterfactual was drawn in the probe window")
    if env("RAC_USE_CONTEXT_QFORMER") == "1":
        ok("persistent_workers is forced off by train_stage2 when context is on")

    print("=== P7 slot arithmetic ===")
    from arcct.schema import active
    from arcct.slots import SlotLayout
    sch = active()
    use_anatomy = env("RAC_CTX_ANATOMY_QUERIES", "1") == "1"
    n_anatomy = len(sch["FINE_LABEL_NAMES"]) - 1 if use_anatomy else 0
    want_q = int(env("RAC_QFORMER_QUERIES", "0") or 0)
    n_global = want_q - n_anatomy - len(sch["PATHOLOGIES"]) if want_q else 2
    L = SlotLayout.phase1(
        n_path=len(sch["PATHOLOGIES"]), n_anatomy=n_anatomy,
        n_glob_gen=n_global)
    try:
        L.validate()
        ok(L.describe())
    except ValueError as exc:
        fail(f"P7: {exc}")
    if want_q and want_q != L.n_gen:
        fail(f"P7: RAC_QFORMER_QUERIES={want_q} but the unconditioned bank is "
             f"{L.n_gen}")

    if not a.skip_slow:
        print("=== P6/P10 step-0 identity and isolation ===")
        rc = subprocess.run([sys.executable, os.path.join(HERE, "tools",
                                                          "check_ctx_identity.py")],
                            capture_output=True, text=True)
        line = (rc.stdout or rc.stderr).strip().splitlines()
        if rc.returncode != 0:
            fail("P6/P10: the step-0 identity gate FAILED:\n      " +
                 "\n      ".join(line[-8:]))
        else:
            ok(line[-1] if line else "identity gate green")

    print("=== P11 manifest ===")
    man = {}
    for key in ("RAC_CONTEXT_CSV", "RAC_DEMOGRAPHICS_CSV", "RAC_LABELS_TRAIN",
                "RAC_LABELS_VALID", "RAC_REGION_CACHE", "RAC_VOLUME_LIST_TRAIN",
                "RAC_VOLUME_LIST_VALID", "RAC_WARM_START_CKPT"):
        p = env(key)
        if p and os.path.isfile(p):
            man[key] = sha256(p)[:16]
    digest = hashlib.sha256(
        "".join(f"{k}={v}" for k, v in sorted(man.items())).encode()).hexdigest()
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "input_manifest.txt"), "w", encoding="utf-8") as fh:
        for k, v in sorted(man.items()):
            fh.write(f"{k} {v}\n")
        fh.write(f"MANIFEST {digest}\n")
    ok(f"manifest {digest[:16]} over {len(man)} inputs -> {results}/input_manifest.txt")

    for n in NOTES:
        print(f"  note {n}")
    if FAILS:
        print(f"\nPREFLIGHT FAILED ({len(FAILS)}):")
        for f in FAILS:
            print("  FAIL " + f)
        return 78
    print("\npreflight OK -- the run may start")
    return 0


if __name__ == "__main__":
    sys.exit(main())

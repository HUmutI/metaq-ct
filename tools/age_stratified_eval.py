#!/usr/bin/env python3
"""Per-class AUC by age group -- the table a pediatric paper is actually asked for.

The evaluator reports one AUC per class over the whole split. The question a
pediatric reader has is different: does this model work on infants, or only on
adolescents? A single pooled number cannot answer that, and a pediatric cohort
whose own age range is the contribution ought to say.

Runs on an existing predictions.npz. No retraining, no re-evaluation.

    python tools/age_stratified_eval.py \\
        --pred /temp_work/ch278233/eval_matrix/peds/V5/predictions.npz

WHY THE GROUPS ARE COARSE. The natural cut is the model's own ten bands, and it
does not survive contact with the data: over the 1,761-volume pediatric
validation split only 113 of 243 band-by-class cells clear a floor of ten
positives and ten negatives, and the infant bands give 6 and 3 classes out of 27.
Reported at that resolution the table would be more blank than filled, and the
filled cells would carry confidence intervals as wide as the classes themselves.
Five clinical groups bring it to 90 of 135, and the <18 / >=18 split -- the one
the design already commits to for the headline -- is near-complete at 24 and 25
classes of 27.

A cell below the floor is left BLANK. It is not filled with an AUC computed from
four positives, because that number would be quoted.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# (label, set of model age bands). Bands are 0:<1y 1:1-2 2:2-5 3:5-10 4:10-13
# 5:13-18 6:18-25 7:25-40 8:40-60 9:60+ 10:unknown.
GROUPS = [
    ("<2y infant/toddler", {0, 1}),
    ("2-10y preschool/school", {2, 3}),
    ("10-18y adolescent", {4, 5}),
    ("18-25y young adult", {6}),
    ("25y+ adult", {7, 8, 9}),
]
HEADLINE = [("<18y", {0, 1, 2, 3, 4, 5}), (">=18y", {6, 7, 8, 9})]


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Mann-Whitney AUC. None when one class is absent."""
    keep = np.isfinite(y) & np.isfinite(s)
    y, s = y[keep], s[keep]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties, or equal scores bias the statistic
    su = np.unique(s)
    if len(su) < len(s):
        for v in su:
            m = s == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def boot_ci(y, s, groups, n=1000, seed=0):
    """Patient-clustered percentile CI, resampling PATIENTS not volumes."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        a = auc(y[sel], s[sel])
        if a is not None:
            out.append(a)
    if len(out) < n // 4:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.npz from evaluate.py")
    ap.add_argument("--demographics", default="/temp_work/ch278233/CONTEXT/demographics.csv")
    ap.add_argument("--min-pos", type=int, default=10,
                    help="reportability floor: positives AND negatives per cell")
    ap.add_argument("--ci", action="store_true", help="bootstrap CIs (slower)")
    ap.add_argument("--csv", default="", help="also write the table here")
    a = ap.parse_args()

    d = np.load(a.pred, allow_pickle=True)
    pred, true = np.asarray(d["pred"], float), np.asarray(d["true"], float)
    names = [str(x) for x in d["pathologies"]]
    accs = [str(x) for x in d["accessions"]]
    print(f"{os.path.basename(os.path.dirname(a.pred))}: {pred.shape[0]:,} volumes, "
          f"{pred.shape[1]} classes, mode={d.get('mode', '?')}")

    csv.field_size_limit(10 ** 9)
    band = {}
    with open(a.demographics, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                band[r["VolumeName"]] = int(r["AgeBand"])
            except (KeyError, ValueError):
                pass
    b = np.array([band.get(x, 10) for x in accs])
    missing = int((b == 10).sum())
    if missing:
        print(f"  {missing} volumes have no age; they appear in the pooled column only")
    # Patient id: the first two underscore-separated tokens, matching the
    # evaluator's own clustering (ped_<subj>_<study>).
    pid = np.array(["_".join(x.split("_")[:2]) for x in accs])

    rows = []
    for gname, bands in GROUPS + HEADLINE + [("pooled", set(range(11)))]:
        sel = np.isin(b, list(bands))
        n = int(sel.sum())
        cells, resolved = {}, 0
        for j, c in enumerate(names):
            y, s = true[sel, j], pred[sel, j]
            npos = int(np.nansum(y == 1))
            nneg = int(np.nansum(y == 0))
            if npos < a.min_pos or nneg < a.min_pos:
                cells[c] = (None, npos, None, None)
                continue
            v = auc(y, s)
            lo = hi = None
            if a.ci and v is not None:
                lo, hi = boot_ci(y, s, pid[sel])
            cells[c] = (v, npos, lo, hi)
            if v is not None:
                resolved += 1
        vals = [v for v, *_ in cells.values() if v is not None]
        rows.append({"group": gname, "n": n, "resolved": resolved,
                     "macro": float(np.mean(vals)) if vals else None, "cells": cells})

    print(f"\n{'group':<24s} {'n':>5s} {'classes':>8s} {'macro AUC':>10s}")
    print("-" * 52)
    for r in rows:
        m = f"{r['macro']:.4f}" if r["macro"] is not None else "    --"
        print(f"{r['group']:<24s} {r['n']:>5d} {r['resolved']:>5d}/{len(names)} {m:>10s}")
    print("-" * 52)

    # A macro over each group's OWN resolved classes is not a comparison: a
    # group that only resolves its easy classes scores higher for that reason
    # alone. The common set is the apples-to-apples column.
    grp = rows[:len(GROUPS)]
    common = [c for c in names
              if all(r["cells"][c][0] is not None for r in grp)]
    print(f"\nsame {len(common)} classes in every group -- the comparable column:")
    for r in grp:
        vals = [r["cells"][c][0] for c in common]
        print(f"  {r['group']:<24s} {np.mean(vals):.4f}")
    if common:
        best = max(grp, key=lambda r: np.mean([r["cells"][c][0] for c in common]))
        worst = min(grp, key=lambda r: np.mean([r["cells"][c][0] for c in common]))
        spread = (np.mean([best["cells"][c][0] for c in common])
                  - np.mean([worst["cells"][c][0] for c in common]))
        print(f"  spread {spread:+.4f}  ({worst['group']} -> {best['group']})")
    print(f"\n  common classes: {', '.join(common)}")

    print(f"\nper class (blank = fewer than {a.min_pos} positives or negatives "
          "in that group)")
    hdr = "class".ljust(38) + "".join(f"{g[:11]:>13s}" for g, _ in GROUPS)
    print(hdr)
    print("-" * len(hdr))
    order = sorted(names, key=lambda c: -sum(
        1 for r in rows[:len(GROUPS)] if r["cells"][c][0] is not None))
    for c in order:
        line = c[:37].ljust(38)
        for r in rows[:len(GROUPS)]:
            v, npos, _, _ = r["cells"][c]
            line += f"{v:>9.4f}({npos:>2d})" if v is not None else f"{'--':>9s}({npos:>2d})"
        print(line)

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["group", "n_volumes", "class", "auc", "n_pos", "ci_lo", "ci_hi"])
            for r in rows:
                for c, (v, npos, lo, hi) in r["cells"].items():
                    w.writerow([r["group"], r["n"], c,
                                "" if v is None else f"{v:.6f}", npos,
                                "" if lo is None else f"{lo:.6f}",
                                "" if hi is None else f"{hi:.6f}"])
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

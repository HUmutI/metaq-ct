#!/usr/bin/env python3
"""Read the ladder's state and say what it means, not just what it printed.

Run it at any time; it reports whatever has finished so far and is explicit
about what has not. Nothing here interprets a partial ladder as a result -- a
rung with fewer than the configured seeds is marked incomplete and its mean is
shown with the count so it cannot be quoted as if it were three.

    python pipeline/08_analysis/ladder_report.py
    python pipeline/08_analysis/ladder_report.py --seeds 3 --json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import time

RUNS = "/temp_work/ch278233/runs"
LOGS = "/temp_work/ch278233/ts_logs"
RUNGS = ["ct_only", "concat", "c1_xattn", "c1_full", "c2",
         "fusion_gated", "isolation_off", "age_scalar"]
# What each rung is FOR. Printed with the numbers, because a ladder read without
# its hypotheses is just a table.
MEANS = {
    "ct_only": "baseline: today's model, context off",
    "concat": "the cheap fusion the cited study says can win at this data scale",
    "c1_xattn": "spatial conditioning only -- must beat concat to justify itself",
    "c1_full": "C1 complete: + channel gating",
    "c2": "HEADLINE: C1 + relevance weighting",
    "fusion_gated": "gated sum instead of concat fusion",
    "isolation_off": "UNSAFE by design: measures the leak, never a reportable model",
    "age_scalar": "scalar age instead of the ordinal band",
}
BEST_RE = re.compile(r"best_auc_update(\d+)\.txt$")


def is_finished(run_dir: str, quiet_secs: int = 1800) -> bool:
    """Has this run stopped writing, with a best checkpoint to show for it?

    Not "did it reach RAC_TOTAL_UPDATES": early stopping is on (patience 5), so
    a legitimate run ends at 3,600 or 4,000 of 6,000 and a progress threshold
    calls it unfinished. What actually distinguishes a finished run from a live
    one is that nothing is writing to its directory any more.
    """
    best = os.path.join(run_dir, "CTClip.best.pt")
    if not os.path.isfile(best):
        return False
    try:
        newest = max(os.path.getmtime(os.path.join(run_dir, f))
                     for f in os.listdir(run_dir))
    except (OSError, ValueError):
        return False
    return (time.time() - newest) > quiet_secs


def last_update(run_dir: str) -> int:
    """How far the run actually got, from its numbered checkpoints."""
    ns = []
    for path in glob.glob(os.path.join(run_dir, "CTClip.*.pt")):
        m = re.search(r"CTClip\.(\d+)\.pt$", path)
        if m:
            ns.append(int(m.group(1)))
    for path in glob.glob(os.path.join(run_dir, "best_auc_update*.txt")):
        m = BEST_RE.search(path)
        if m:
            ns.append(int(m.group(1)))
    return max(ns) if ns else 0


def best_auc(run_dir: str) -> tuple[float | None, int | None]:
    """Highest validation AUC this run reached, and at which update."""
    best, at = None, None
    for path in glob.glob(os.path.join(run_dir, "best_auc_update*.txt")):
        m = BEST_RE.search(path)
        try:
            vals = [float(ln.split(":")[1]) for ln in open(path, encoding="utf-8")
                    if ":" in ln and ln.split(":")[1].strip().replace(".", "").isdigit()]
        except (OSError, ValueError, IndexError):
            continue
        if vals:
            mean = sum(vals) / len(vals)
            if best is None or mean > best:
                best, at = mean, int(m.group(1)) if m else None
    return best, at


def log_tail_metrics(rung: str, seed: int) -> dict:
    """beta and the fusion ratio from the most recent log for this cell."""
    out = {}
    cand = sorted(glob.glob(os.path.join(LOGS, "ctxrung_*.out")),
                  key=os.path.getmtime, reverse=True)
    for path in cand[:60]:
        try:
            head = open(path, encoding="utf-8", errors="replace").read(4000)
        except OSError:
            continue
        if f"rung={rung} seed={seed}" not in head:
            continue
        try:
            text = open(path, encoding="utf-8", errors="replace").read()[-60000:]
        except OSError:
            continue
        for key, rx in (("beta", r"beta=(-?[\d.]+)"),
                        ("w_ratio", r"\|Wi\|/\|Wg\|=(-?[\d.]+)"),
                        ("update", r"update=\s*(\d+)")):
            hits = re.findall(rx, text)
            if hits:
                out[key] = float(hits[-1])
        out["log"] = os.path.basename(path)
        out["rc"] = None
        m = re.search(r"rc=(\d+)", text[-2000:])
        if m:
            out["rc"] = int(m.group(1))
        break
    return out


def queue_state() -> dict[str, int]:
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "ch278233"), "-h", "-o", "%j|%T"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:                                          # noqa: BLE001
        return {}
    st: dict[str, int] = {}
    for ln in out.splitlines():
        if "ctxrung" in ln:
            st[ln.split("|")[1].strip()] = st.get(ln.split("|")[1].strip(), 0) + 1
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    table, blob = [], {}
    for rung in RUNGS:
        cells = []
        for seed in range(a.seeds):
            d = os.path.join(RUNS, f"ctx_{rung}_seed{seed}")
            auc, at = best_auc(d) if os.path.isdir(d) else (None, None)
            reached = last_update(d) if os.path.isdir(d) else 0
            m = log_tail_metrics(rung, seed) if os.path.isdir(d) else {}
            done = is_finished(d) if os.path.isdir(d) else False
            cells.append({"seed": seed, "auc": auc, "at": at,
                          "reached": reached, "done": done, **m})
        # Only FINISHED seeds contribute to a rung's mean. A run at update 800 of
        # 6000 has a best_auc, and averaging it in would report a rung as worse
        # than it is purely because it started later.
        got = [c["auc"] for c in cells if c["auc"] is not None and c["done"]]
        row = {
            "rung": rung, "means": MEANS[rung], "cells": cells,
            "n": len(got),
            "mean": statistics.mean(got) if got else None,
            "sd": statistics.stdev(got) if len(got) > 1 else None,
            "complete": len(got) == a.seeds,
        }
        table.append(row)
        blob[rung] = row

    if a.json:
        print(json.dumps(blob, indent=1))
        return 0

    q = queue_state()
    print(f"queue: {q or 'no ctxrung tasks'}\n")
    print(f"{'rung':<15s} {'n':>2s} {'mean AUC':>9s} {'sd':>7s}  seeds  (x@n = still running at update n, excluded)")
    print("-" * 78)
    base = next((r for r in table if r["rung"] == "ct_only"), None)
    for r in table:
        seeds = " ".join(
            (f"{c['auc']:.4f}" if c["done"] else f"({c['auc']:.4f}@{c['reached']})")
            if c["auc"] is not None else "  --  " for c in r["cells"])
        mean = f"{r['mean']:.4f}" if r["mean"] is not None else "   --"
        sd = f"{r['sd']:.4f}" if r["sd"] is not None else "     --"
        flag = "" if r["complete"] else "  (incomplete)"
        print(f"{r['rung']:<15s} {r['n']:>2d} {mean:>9s} {sd:>7s}  {seeds}{flag}")
    print("-" * 78)

    if base and base["mean"] is not None:
        print("\ndelta vs ct_only (positive = better than today's model):")
        for r in table:
            if r["rung"] == "ct_only" or r["mean"] is None:
                continue
            d = r["mean"] - base["mean"]
            note = ""
            if r["rung"] == "isolation_off":
                note = "   <- if this leads, the reason is the leak, not the model"
            if r["rung"] == "concat" and any(
                    x["rung"] in ("c1_xattn", "c1_full") and x["mean"] is not None
                    and x["mean"] < r["mean"] for x in table):
                note = "   <- concat is beating cross-attention; the cited study's result"
            print(f"  {r['rung']:<15s} {d:+.4f}{note}")

    print("\nsigma, from Phase 0/8 (V1 recipe):")
    # A run still in progress has a best_auc from an early update, and folding
    # that into sigma inflates it -- which would then be used to decide how many
    # seeds the ladder needs. Only runs that reached the full schedule count.
    v1, partial = [], []
    for d in ("peds_finetune", "peds_finetune_v1_seed1", "peds_finetune_v1_seed2"):
        path = os.path.join(RUNS, d)
        auc, _ = best_auc(path)
        reached = last_update(path)
        if auc is None:
            continue
        (v1 if is_finished(path) else partial).append((d, auc, reached))
    for d, auc, reached in partial:
        print(f"  {d}: {auc:.4f}, last update {reached} -- STILL WRITING, "
              "excluded from sigma")
    if len(v1) > 1:
        vals = [a for _, a, _ in v1]
        sd = statistics.stdev(vals)
        print(f"  {len(v1)} completed seeds, sd = {sd:.4f}  -> a delta below "
              f"~{2*sd:.4f} is inside the seed band and is not a difference")
    else:
        print(f"  only {len(v1)} completed seed(s); sigma is not measurable yet")

    incomplete = [r["rung"] for r in table if not r["complete"]]
    if incomplete:
        print(f"\nNOT FINISHED, do not quote as a result: {', '.join(incomplete)}")
    alerts = os.path.join(LOGS, "watchdog_ctx_ALERTS.txt")
    if os.path.isfile(alerts):
        lines = open(alerts, encoding="utf-8").read().strip().splitlines()
        if lines:
            print(f"\nwatchdog alerts ({len(lines)}), last 5:")
            for ln in lines[-5:]:
                print("  " + ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

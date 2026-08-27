#!/usr/bin/env python3
"""Catch a run failing WHILE it runs, not when it exits.

A SLURM job that stalls does not fail. It holds the GPU, prints nothing, and is
discovered hours later when someone looks. The existing watchdog polls job state,
which is exactly the signal that does not move in that case. This one reads the
logs.

Every cycle (default 10 minutes) it looks at each tracked job and asks:

  ALIVE      has the log grown since the last cycle?
  PROGRESS   has update= advanced, or the eval progress bar moved?
  FINITE     is any loss NaN or inf?
  SANE       for a context run, is Z_gen still invariant, is beta >= 0, is
             |W_ind|/|W_gen| inside its band?
  ERRORS     any traceback, CUDA OOM, or NCCL/timeout text?

Anything that trips is written to the alert file immediately and printed. A
stalled job is optionally killed so the GPU goes back to the queue rather than
being held by a process that will never finish.

Run:
    python pipeline/07_ops/watchdog_ctx.py --interval 600
    python pipeline/07_ops/watchdog_ctx.py --once            # one cycle, for cron
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

LOG_DIR = "/temp_work/ch278233/ts_logs"
STATE = "/temp_work/ch278233/ts_logs/watchdog_ctx_state.json"
ALERTS = "/temp_work/ch278233/ts_logs/watchdog_ctx_ALERTS.txt"

UPDATE_RE = re.compile(r"update=\s*(\d+)")
LOSS_RE = re.compile(r"(clip|cls|align|ptok|ptok_ind|cf)=(-?[\d.]+|nan|inf|-inf)", re.I)
BETA_RE = re.compile(r"beta=(-?[\d.]+)")
WRATIO_RE = re.compile(r"\|Wi\|/\|Wg\|=(-?[\d.]+)")
AUC_RE = re.compile(r"mean_auc[= ]\s*([\d.]+)", re.I)
ERROR_PAT = [
    (re.compile(r"Traceback \(most recent call last\)"), "python traceback"),
    (re.compile(r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED"), "CUDA OOM"),
    (re.compile(r"RuntimeError|AssertionError|ValueError|KeyError"), "exception"),
    (re.compile(r"NCCL|watchdog timeout|Socket Timeout"), "distributed/timeout"),
    (re.compile(r"FATAL:"), "preflight gate"),
    (re.compile(r"Z_gen moved with the indication"), "ISOLATION BROKEN"),
    (re.compile(r"is not finite"), "non-finite tensor"),
    (re.compile(r"slurmstepd.*(CANCELLED|DUE TO)"), "slurm cancellation"),
]
# Anything the trainer prints that is not actually a problem.
BENIGN = re.compile(r"Refusing to start silently|expected|WARNING could not add")


def squeue() -> dict[str, dict]:
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "ch278233"), "-h",
             "-o", "%i|%j|%T|%M|%R"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    jobs = {}
    for ln in out.strip().splitlines():
        if not ln.strip():
            continue
        jid, name, state, elapsed, reason = (ln.split("|") + [""] * 5)[:5]
        jobs[jid.strip()] = {"name": name.strip(), "state": state.strip(),
                             "elapsed": elapsed.strip(), "reason": reason.strip()}
    return jobs


def find_log(jid: str) -> str | None:
    if not os.path.isdir(LOG_DIR):
        return None
    hits = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR)
            if jid in f and f.endswith(".out")]
    return max(hits, key=os.path.getmtime) if hits else None


def tail(path: str, nbytes: int = 200_000) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def inspect(jid: str, info: dict, prev: dict) -> tuple[list[str], dict]:
    """Return (alerts, new state) for one job."""
    alerts: list[str] = []
    path = find_log(jid)
    now = time.time()
    st = {"name": info["name"], "state": info["state"], "checked": now,
          "size": 0, "update": None, "log": path}

    if info["state"] == "PENDING":
        # Pending is not a failure, but a job pending on a dependency whose
        # parent already died would wait forever, and one held for a bad reason
        # is worth surfacing rather than discovering in the morning.
        if "DependencyNeverSatisfied" in info["reason"] or "launch failed" in info["reason"]:
            alerts.append(f"[{jid} {info['name']}] will never run: {info['reason']}")
        st["update"] = prev.get("update")
        st["size"] = prev.get("size", 0)
        return alerts, st

    if not path:
        alerts.append(f"[{jid} {info['name']}] RUNNING for {info['elapsed']} "
                      "but no log file exists yet")
        return alerts, st

    size = os.path.getsize(path)
    st["size"] = size
    text = tail(path)

    # -- ERRORS ------------------------------------------------------------
    for rx, label in ERROR_PAT:
        for m in rx.finditer(text):
            line = text[max(0, m.start() - 120):m.end() + 200].splitlines()
            snippet = " / ".join(x.strip() for x in line if x.strip())[:300]
            if BENIGN.search(snippet):
                continue
            alerts.append(f"[{jid} {info['name']}] {label}: {snippet}")
            break

    # -- PROGRESS ----------------------------------------------------------
    ups = UPDATE_RE.findall(text)
    cur = int(ups[-1]) if ups else None
    st["update"] = cur
    prev_size = prev.get("size", 0)
    prev_up = prev.get("update")
    prev_checked = prev.get("checked")
    if prev_checked is not None:
        quiet_for = (now - prev_checked) / 60.0
        if size == prev_size:
            alerts.append(f"[{jid} {info['name']}] STALLED: log has not grown in "
                          f"{quiet_for:.0f} min (elapsed {info['elapsed']})")
        elif cur is not None and prev_up is not None and cur == prev_up:
            # The log moved but the step counter did not. Validation and
            # checkpointing both do this legitimately, so it is a warning after
            # one cycle and an alert after two.
            if prev.get("no_update_cycles", 0) >= 1:
                alerts.append(f"[{jid} {info['name']}] no update in "
                              f"{quiet_for * 2:.0f} min, still at update={cur}")
            st["no_update_cycles"] = prev.get("no_update_cycles", 0) + 1
        else:
            st["no_update_cycles"] = 0

    # -- FINITE ------------------------------------------------------------
    for m in LOSS_RE.finditer(text[-40_000:]):
        val = m.group(2).lower()
        if val in ("nan", "inf", "-inf"):
            alerts.append(f"[{jid} {info['name']}] NON-FINITE loss: "
                          f"{m.group(1)}={m.group(2)}")
            break

    # -- SANE (context runs only) -----------------------------------------
    betas = BETA_RE.findall(text)
    if betas:
        b = float(betas[-1])
        st["beta"] = b
        if b < 0:
            alerts.append(f"[{jid} {info['name']}] beta={b} is NEGATIVE -- "
                          "softplus should make that impossible; w >= 1 is broken")
    ratios = WRATIO_RE.findall(text)
    if ratios:
        r = float(ratios[-1])
        st["w_ratio"] = r
        # The conditioned half taking over the fusion is the failure mode the
        # design worries about. It starts at exactly 0 and should grow slowly.
        if r > 3.0:
            alerts.append(f"[{jid} {info['name']}] |W_ind|/|W_gen|={r:.2f} -- the "
                          "conditioned path is dominating the fusion")
    aucs = AUC_RE.findall(text)
    if aucs:
        st["auc"] = float(aucs[-1])
    return alerts, st


def cycle(track: list[str], kill_stalled: int) -> list[str]:
    jobs = squeue()
    try:
        with open(STATE, encoding="utf-8") as fh:
            prev_all = json.load(fh)
    except (OSError, json.JSONDecodeError):
        prev_all = {}

    watch = {j: i for j, i in jobs.items()
             if not track or j in track or any(t in i["name"] for t in track)}
    alerts, new_all = [], {}
    for jid, info in sorted(watch.items()):
        a, st = inspect(jid, info, prev_all.get(jid, {}))
        alerts += a
        new_all[jid] = st
        flag = "!" if a else " "
        print(f" {flag} {jid:>9s} {info['name'][:12]:<12s} {info['state']:<9s} "
              f"{info['elapsed']:>9s}  update={st.get('update')}  "
              f"auc={st.get('auc')}  beta={st.get('beta')}  "
              f"|Wi/Wg|={st.get('w_ratio')}")

    # A job that vanished from the queue since the last cycle: report how it went.
    for jid, st in prev_all.items():
        if jid not in jobs and st.get("state") == "RUNNING":
            rc = subprocess.run(["sacct", "-j", jid, "-n", "-o", "State,ExitCode"],
                                capture_output=True, text=True).stdout.strip()
            first = rc.splitlines()[0].strip() if rc else "unknown"
            if "COMPLETED" not in first:
                alerts.append(f"[{jid} {st.get('name')}] LEFT THE QUEUE: {first}")
            else:
                print(f"   {jid} {st.get('name')} finished COMPLETED")

    for jid, st in new_all.items():
        if kill_stalled and any(f"[{jid} " in a and "STALLED" in a for a in alerts):
            print(f"   killing stalled {jid}")
            subprocess.run(["scancel", jid], check=False)

    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(new_all, fh, indent=1)
    if alerts:
        with open(ALERTS, "a", encoding="utf-8") as fh:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for a in alerts:
                fh.write(f"{ts}  {a}\n")
    return alerts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600, help="seconds between cycles")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--track", nargs="*", default=[],
                    help="job ids or name substrings; empty = every job")
    ap.add_argument("--kill-stalled", type=int, default=0,
                    help="1 = scancel a job whose log stopped growing")
    a = ap.parse_args()

    while True:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n=== watchdog {stamp} ===", flush=True)
        try:
            alerts = cycle(a.track, a.kill_stalled)
        except Exception as exc:                              # noqa: BLE001
            # A watchdog that dies is worse than no watchdog, because its
            # silence reads as "everything is fine".
            print(f"   watchdog error (continuing): {exc}", flush=True)
            alerts = []
        for al in alerts:
            print(f"   ALERT {al}", flush=True)
        if a.once:
            return 1 if alerts else 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

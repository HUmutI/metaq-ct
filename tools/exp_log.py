#!/usr/bin/env python3
"""The experiment ledger: every run we have done, are doing, or intend to do.

Two halves, deliberately separated.

The NUMBERS are harvested from disk on every render -- validation AUC out of each
run directory, held-out AUC out of each eval_matrix cell. Nothing here is a
number a person typed. A ledger whose figures are transcribed by hand drifts from
the runs it describes, and the drift is invisible precisely because the document
looks authoritative.

The MEANING lives in docs/experiments.yaml -- what an experiment was for, which
architecture it belongs to, what it settled, what is still queued. That cannot be
derived from a directory listing, so it is written by hand and reviewed.

    python tools/exp_log.py                 # rewrite docs/EXPERIMENTS.md
    python tools/exp_log.py --html out.html # also the page
    python tools/exp_log.py --check         # registry vs disk, non-zero on drift

--check is the part that keeps this honest: it fails when a run directory exists
that the registry has never heard of. Forgetting to log an experiment is the
failure mode this document exists to prevent, so it is made loud.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("RAC_WORK", "/temp_work/ch278233")
RUNS = os.path.join(WORK, "runs")
EVALS = os.path.join(WORK, "eval_matrix")
REGISTRY = os.path.join(ROOT, "docs", "experiments.yaml")

BEST_RE = re.compile(r"best_auc_update(\d+)\.txt$")
QUIET = 1800


# ----------------------------------------------------------------- harvesting

def _run_state(run_dir: str, min_update: int = 0) -> dict:
    """Validation AUC, how far it got, and whether it is still moving.

    'Finished' is not 'reached RAC_TOTAL_UPDATES' -- early stopping ends a
    legitimate run short, and a progress threshold would label it as still
    running forever. What separates a finished run from a live one is that
    nothing has written to its directory for half an hour.

    But quiet-with-a-checkpoint is not sufficient either, and this is where a
    ledger silently lies. A run killed at update 400 leaves exactly what a
    finished run leaves: a best.pt, a best_auc file, and no further writes. Nine
    of the first ladder's twenty-four tasks died that way on a TypeError, and
    averaging their early AUCs in reports the arm as WORSE than it is -- the
    dead seeds drag the mean down and the failure reads as a result.

    min_update is the earliest update at which early stopping could possibly
    have fired (patience x val_every, from the registry). Below it, quiet means
    died, not converged.
    """
    out = {"exists": os.path.isdir(run_dir), "auc": None, "at": None,
           "last_update": 0, "state": "missing", "started": None, "updated": None}
    if not out["exists"]:
        return out
    ns, best, at = [], None, None
    for path in glob.glob(os.path.join(run_dir, "CTClip.*.pt")):
        m = re.search(r"CTClip\.(\d+)\.pt$", path)
        if m:
            ns.append(int(m.group(1)))
    for path in glob.glob(os.path.join(run_dir, "best_auc_update*.txt")):
        m = BEST_RE.search(path)
        if m:
            ns.append(int(m.group(1)))
        try:
            vals = [float(ln.split(":")[1]) for ln in open(path, encoding="utf-8")
                    if ":" in ln and ln.split(":")[1].strip().replace(".", "").isdigit()]
        except (OSError, ValueError, IndexError):
            continue
        if vals:
            mean = sum(vals) / len(vals)
            if best is None or mean > best:
                best, at = mean, int(m.group(1)) if m else None
    out["auc"], out["at"] = best, at
    out["last_update"] = max(ns) if ns else 0
    try:
        mt = [os.path.getmtime(os.path.join(run_dir, f)) for f in os.listdir(run_dir)]
        newest, oldest = max(mt), min(mt)
    except (OSError, ValueError):
        return out
    # Dates come off the filesystem, like every other fact here. A hand-typed
    # start date is the first thing to go stale after a requeue.
    out["started"] = datetime.fromtimestamp(oldest).strftime("%Y-%m-%d")
    out["updated"] = datetime.fromtimestamp(newest).strftime("%Y-%m-%d")
    has_best = os.path.isfile(os.path.join(run_dir, "CTClip.best.pt"))
    quiet = (time.time() - newest) > QUIET
    if not quiet:
        out["state"] = "running"
    elif not has_best:
        out["state"] = "stalled"
    elif min_update and out["last_update"] < min_update:
        out["state"] = "died"
    else:
        out["state"] = "done"
    return out


def _eval_cell(cohort: str, cell: str) -> dict:
    """Held-out macro AUC for one eval_matrix cell, global and routed.

    Both readings are kept because their DIFFERENCE is a result: routing the
    read-out through the anatomy masks costs accuracy, and the size of that cost
    is one of the things this project set out to measure. When they are equal to
    the last digit no mask was supplied and the routed path silently fell back to
    global -- recorded as None rather than as a routed score, because reporting
    it as one would invent an ablation we never ran.
    """
    d = os.path.join(EVALS, cohort, cell)
    out = {"exists": os.path.isdir(d), "n": None, "classes": None,
           "global": None, "routed": None, "ci": None, "fellback": False}
    if not out["exists"]:
        return out
    for key, fn in (("global", "metric_bundle_global.json"),
                    ("routed", "metric_bundle_routed.json")):
        p = os.path.join(d, fn)
        if not os.path.isfile(p):
            continue
        try:
            b = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out[key] = b.get("macro", {}).get("auc")
        out["n"] = b.get("n_samples", out["n"])
        out["classes"] = b.get("n_classes", out["classes"])
        if key == "global":
            lo = b.get("macro", {}).get("auc_ci_lo")
            hi = b.get("macro", {}).get("auc_ci_hi")
            out["ci"] = (lo, hi) if lo is not None else None
    g, r = out["global"], out["routed"]
    if g is not None and r is not None and abs(g - r) < 1e-6:
        out["fellback"], out["routed"] = True, None
    return out


def harvest(reg: dict) -> dict:
    """Attach live disk state to every experiment in the registry."""
    for exp in reg["experiments"]:
        floor = int(exp.get("patience", 0)) * int(exp.get("val_every", 0))
        exp["_runs"] = [dict(name=n, **_run_state(os.path.join(RUNS, n), floor))
                        for n in exp.get("runs", [])]
        exp["_floor"] = floor
        exp["_died"] = [r["name"] for r in exp["_runs"] if r["state"] == "died"]
        exp["_evals"] = [dict(cohort=c, cell=k, **_eval_cell(c, k))
                         for c, k in (e.split("/", 1) for e in exp.get("evals", []))]
        seeds = [r["auc"] for r in exp["_runs"] if r["auc"] is not None
                 and r["state"] == "done"]
        exp["_mean"] = sum(seeds) / len(seeds) if seeds else None
        exp["_nseed"] = len(seeds)
        if len(seeds) > 1:
            m = exp["_mean"]
            exp["_sd"] = (sum((v - m) ** 2 for v in seeds) / (len(seeds) - 1)) ** 0.5
        else:
            exp["_sd"] = None
        st = [r["started"] for r in exp["_runs"] if r["started"]]
        up = [r["updated"] for r in exp["_runs"] if r["updated"]]
        exp["_started"], exp["_updated"] = (min(st) if st else None), (max(up) if up else None)
        if exp["_started"] and exp["_updated"] and exp["_started"] != exp["_updated"]:
            exp["_dates"] = f"{exp['_started']} \u2192 {exp['_updated']}"
        else:
            exp["_dates"] = exp["_started"] or "--"
        states = {r["state"] for r in exp["_runs"]}
        if exp.get("status") in ("queued", "planned"):
            exp["_state"] = exp["status"]
        elif "running" in states:
            exp["_state"] = "running"
        elif states and states <= {"done"}:
            exp["_state"] = "done"
        elif "stalled" in states or "died" in states:
            exp["_state"] = "partial" if seeds else "stalled"
        else:
            exp["_state"] = exp.get("status", "unknown")
    return reg


def orphans(reg: dict) -> tuple[list[str], list[str]]:
    """Run directories and eval cells the registry does not mention."""
    known_r = {n for e in reg["experiments"] for n in e.get("runs", [])}
    known_e = {v for e in reg["experiments"] for v in e.get("evals", [])}
    have_r = {d for d in os.listdir(RUNS)
              if os.path.isdir(os.path.join(RUNS, d)) and not d.startswith("_")} \
        if os.path.isdir(RUNS) else set()
    have_e = set()
    if os.path.isdir(EVALS):
        for c in os.listdir(EVALS):
            for k in os.listdir(os.path.join(EVALS, c)):
                if os.path.isdir(os.path.join(EVALS, c, k)):
                    have_e.add(f"{c}/{k}")
    return sorted(have_r - known_r), sorted(have_e - known_e)


# ------------------------------------------------------------------ rendering

def _auc(v, sd=None, n=None):
    if v is None:
        return "--"
    s = f"{v:.4f}"
    if sd is not None and n and n > 1:
        s += f" ±{sd:.4f}"
    return s


STATE_MARK = {"done": "done", "running": "RUNNING", "queued": "queued",
              "planned": "planned", "stalled": "STALLED", "died": "DIED",
              "partial": "partial", "unknown": "?"}


def render_md(reg: dict) -> str:
    L = []
    A = L.append
    A("# Experiment ledger")
    A("")
    A("Every run of this project: finished, in flight, and queued. **Generated** by")
    A("`tools/exp_log.py` -- do not edit the tables by hand, edit `docs/experiments.yaml`")
    A("and re-render. Every AUC below is read off disk at render time.")
    A("")
    A(f"Rendered {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.")
    A("")
    A("Two architectures appear here and they must never be compared casually:")
    A("")
    for k, v in reg["architectures"].items():
        A(f"- **{k}** -- {v}")
    A("")
    A("`val AUC` is the validation macro AUC the training loop selected on, mean over")
    A("completed seeds. It is a MODEL SELECTION number and is not a result. `held-out`")
    A("columns come from `eval_matrix` and are the reportable ones.")
    A("")

    for group in reg["groups"]:
        exps = [e for e in reg["experiments"] if e.get("group") == group["id"]]
        if not exps:
            continue
        A(f"## {group['title']}")
        A("")
        if group.get("note"):
            A(group["note"])
            A("")
        A("| experiment | dates | arch | cohort · classes | masks | seeds | val AUC | state |")
        A("|---|---|---|---|---|---|---|---|")
        for e in exps:
            A(f"| **{e['id']}** | {e['_dates']} | {e.get('arch','?')} | "
              f"{e.get('cohort','?')} · "
              f"{e.get('classes','?')} | {e.get('masks','--')} | "
              f"{e['_nseed']}/{len(e.get('runs',[]))} | "
              f"{_auc(e['_mean'], e['_sd'], e['_nseed'])} | "
              f"{STATE_MARK.get(e['_state'], e['_state'])} |")
        A("")
        ev = [(e, c) for e in exps for c in e["_evals"] if c["exists"]]
        if ev:
            A("Held-out evaluation:")
            A("")
            A("| experiment | cell | n | classes | global AUC | 95% CI | routed AUC | Δ routed |")
            A("|---|---|---|---|---|---|---|---|")
            for e, c in ev:
                ci = f"{c['ci'][0]:.4f}–{c['ci'][1]:.4f}" if c["ci"] else "--"
                dr = (f"{c['routed'] - c['global']:+.4f}"
                      if c["routed"] is not None and c["global"] is not None else "--")
                rt = "no mask" if c["fellback"] else _auc(c["routed"])
                A(f"| {e['id']} | `{c['cohort']}/{c['cell']}` | {c['n'] or '--'} | "
                  f"{c['classes'] or '--'} | {_auc(c['global'])} | {ci} | {rt} | {dr} |")
            A("")
        for e in exps:
            if e["_died"]:
                A(f"`{e['id']}`: {len(e['_died'])} seed(s) died before update "
                  f"{e['_floor']:,} and are excluded from the mean "
                  f"({', '.join('`' + d + '`' for d in e['_died'])}).")
                A("")
            if e.get("finding"):
                A(f"**{e['id']} -- what it settled.** {e['finding']}")
                A("")

    if reg.get("journal"):
        A("## Journal")
        A("")
        A("Dated record of what happened and what it changed. Newest first.")
        A("")
        for j in sorted(reg["journal"], key=lambda x: str(x["date"]), reverse=True):
            A(f"**{j['date']} — {j['title']}**  {j['text']}")
            A("")

    if reg.get("findings"):
        A("## Standing conclusions")
        A("")
        for f in reg["findings"]:
            A(f"**{f['title']}**  {f['text']}")
            A("")

    orf, ore = orphans(reg)
    if orf or ore:
        A("## Not yet in the registry")
        A("")
        A("These exist on disk and nothing above describes them. Either they belong")
        A("in `experiments.yaml` or they are debris that should be deleted.")
        A("")
        for x in orf:
            A(f"- run `{x}`")
        for x in ore:
            A(f"- eval `{x}`")
        A("")
    return "\n".join(L) + "\n"


def render_html(reg: dict) -> str:
    e_ = html.escape
    rows = []
    for group in reg["groups"]:
        exps = [e for e in reg["experiments"] if e.get("group") == group["id"]]
        if not exps:
            continue
        rows.append(f'<section><h2>{e_(group["title"])}</h2>')
        if group.get("note"):
            rows.append(f'<p class="note">{e_(group["note"])}</p>')
        rows.append('<div class="scroll"><table><thead><tr>'
                    '<th>experiment</th><th>dates</th><th>arch</th>'
                    '<th>cohort · classes</th>'
                    '<th>masks</th><th>seeds</th><th class="num">val AUC</th>'
                    '<th>state</th></tr></thead><tbody>')
        for e in exps:
            st = e["_state"]
            rows.append(
                f'<tr><td><strong>{e_(e["id"])}</strong></td>'
                f'<td class="dates">{e_(e["_dates"])}</td>'
                f'<td><span class="arch a-{e_(str(e.get("arch","?")))}">'
                f'{e_(str(e.get("arch","?")))}</span></td>'
                f'<td>{e_(str(e.get("cohort","?")))} · {e_(str(e.get("classes","?")))}</td>'
                f'<td>{e_(str(e.get("masks","--")))}</td>'
                f'<td class="num">{e["_nseed"]}/{len(e.get("runs",[]))}</td>'
                f'<td class="num">{e_(_auc(e["_mean"], e["_sd"], e["_nseed"]))}</td>'
                f'<td><span class="pill p-{st}">{e_(STATE_MARK.get(st, st))}</span></td></tr>')
        rows.append('</tbody></table></div>')
        ev = [(e, c) for e in exps for c in e["_evals"] if c["exists"]]
        if ev:
            rows.append('<h3>Held-out evaluation</h3><div class="scroll"><table><thead><tr>'
                        '<th>experiment</th><th>cell</th><th class="num">n</th>'
                        '<th class="num">classes</th><th class="num">global AUC</th>'
                        '<th class="num">95% CI</th><th class="num">routed AUC</th>'
                        '<th class="num">Δ routed</th></tr></thead><tbody>')
            for e, c in ev:
                ci = f"{c['ci'][0]:.4f}–{c['ci'][1]:.4f}" if c["ci"] else "--"
                d = (c["routed"] - c["global"]
                     if c["routed"] is not None and c["global"] is not None else None)
                dr = f'<span class="{"neg" if d < 0 else "pos"}">{d:+.4f}</span>' if d is not None else "--"
                rt = '<span class="muted">no mask</span>' if c["fellback"] else e_(_auc(c["routed"]))
                rows.append(
                    f'<tr><td>{e_(e["id"])}</td><td><code>{e_(c["cohort"])}/{e_(c["cell"])}</code></td>'
                    f'<td class="num">{c["n"] or "--"}</td><td class="num">{c["classes"] or "--"}</td>'
                    f'<td class="num">{e_(_auc(c["global"]))}</td><td class="num">{ci}</td>'
                    f'<td class="num">{rt}</td><td class="num">{dr}</td></tr>')
            rows.append('</tbody></table></div>')
        for e in exps:
            if e["_died"]:
                rows.append(
                    f'<p class="died"><strong>{e_(e["id"])}</strong>: '
                    f'{len(e["_died"])} seed(s) died before update {e["_floor"]:,} '
                    f'and are excluded from the mean &mdash; '
                    f'{e_(", ".join(e["_died"]))}</p>')
            if e.get("finding"):
                rows.append(f'<p class="finding"><strong>{e_(e["id"])}</strong> '
                            f'{e_(e["finding"])}</p>')
        rows.append('</section>')

    journal = "".join(
        f'<li><span class="jdate">{e_(str(j["date"]))}</span>'
        f'<div><strong>{e_(j["title"])}</strong> {e_(j["text"])}</div></li>'
        for j in sorted(reg.get("journal", []), key=lambda x: str(x["date"]), reverse=True))
    findings = "".join(
        f'<div class="card"><h3>{e_(f["title"])}</h3><p>{e_(f["text"])}</p></div>'
        for f in reg.get("findings", []))
    arches = "".join(f'<div class="card"><h3>{e_(k)}</h3><p>{e_(v)}</p></div>'
                     for k, v in reg["architectures"].items())
    orf, ore = orphans(reg)
    orph = ""
    if orf or ore:
        items = "".join(f"<li><code>{e_(x)}</code></li>" for x in orf + ore)
        orph = ('<section><h2>Not yet in the registry</h2><p class="note">On disk, '
                'described nowhere above. Either log them or delete them.</p>'
                f'<ul class="orph">{items}</ul></section>')

    return f"""<title>ARC-CT Experiment Ledger</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap">
<style>
:root {{
  --bg:#fbfaf7; --panel:#ffffff; --ink:#1c1a17; --ink2:#4a453d; --muted:#8b8378;
  --rule:#e2ddd3; --accent:#7a3b2e; --pos:#2f6b4f; --neg:#a3462f;
  --done:#2f6b4f; --run:#a06a10; --queue:#5a6b7a;
}}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15140f; --panel:#1d1c17; --ink:#eee9df; --ink2:#bdb6a8; --muted:#857e72;
    --rule:#302d26; --accent:#d99177; --pos:#7fc0a0; --neg:#e0917a;
    --done:#7fc0a0; --run:#e0b45e; --queue:#93a6b5;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15140f; --panel:#1d1c17; --ink:#eee9df; --ink2:#bdb6a8; --muted:#857e72;
  --rule:#302d26; --accent:#d99177; --pos:#7fc0a0; --neg:#e0917a;
  --done:#7fc0a0; --run:#e0b45e; --queue:#93a6b5;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:400 16px/1.6 Inter,system-ui,sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:56px 28px 96px; }}
h1 {{ font:600 40px/1.15 Newsreader,Georgia,serif; margin:0 0 6px; letter-spacing:-.01em; }}
.sub {{ color:var(--ink2); max-width:64ch; margin:0 0 8px; }}
.stamp {{ color:var(--muted); font:400 13px/1.5 "IBM Plex Mono",monospace; margin:0 0 40px; }}
h2 {{ font:600 25px/1.2 Newsreader,Georgia,serif; margin:52px 0 6px;
  padding-bottom:8px; border-bottom:2px solid var(--accent); letter-spacing:-.01em; }}
h3 {{ font:600 15px/1.3 Inter,sans-serif; margin:26px 0 8px;
  text-transform:uppercase; letter-spacing:.09em; color:var(--ink2); font-size:12px; }}
.note {{ color:var(--ink2); max-width:70ch; margin:10px 0 18px; }}
.scroll {{ overflow-x:auto; margin:14px 0 8px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px;
  background:var(--panel); border:1px solid var(--rule); }}
th {{ text-align:left; font:600 11px/1.4 Inter,sans-serif; text-transform:uppercase;
  letter-spacing:.07em; color:var(--muted); padding:11px 13px;
  border-bottom:1px solid var(--rule); white-space:nowrap; }}
td {{ padding:10px 13px; border-bottom:1px solid var(--rule); vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums;
  font-family:"IBM Plex Mono",monospace; white-space:nowrap; }}
code {{ font:400 13px/1.4 "IBM Plex Mono",monospace; color:var(--ink2); }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:2px;
  font:600 11px/1.7 "IBM Plex Mono",monospace; letter-spacing:.04em; }}
.p-done {{ color:var(--done); border:1px solid var(--done); }}
.p-running {{ color:var(--run); border:1px solid var(--run); }}
.p-queued,.p-planned {{ color:var(--queue); border:1px solid var(--queue); }}
.p-stalled,.p-unknown,.p-died {{ color:var(--neg); border:1px solid var(--neg); }}
.p-partial {{ color:var(--run); border:1px dashed var(--run); }}
.died {{ border-left:3px solid var(--neg); padding:2px 0 2px 16px; margin:14px 0;
  color:var(--ink2); font-size:14px; max-width:74ch; }}
.arch {{ font:600 12px/1.6 "IBM Plex Mono",monospace; }}
.a-ARC-CT {{ color:var(--muted); }}
.pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
.muted {{ color:var(--muted); }}
.finding {{ border-left:3px solid var(--accent); padding:2px 0 2px 16px;
  margin:18px 0; color:var(--ink2); max-width:74ch; }}
.cards {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  margin:18px 0 8px; }}
.card {{ background:var(--panel); border:1px solid var(--rule); padding:16px 18px; }}
.card h3 {{ margin:0 0 6px; color:var(--accent); }}
.card p {{ margin:0; color:var(--ink2); font-size:14px; }}
.orph {{ color:var(--ink2); }}
.dates {{ font:400 12px/1.5 "IBM Plex Mono",monospace; color:var(--muted);
  white-space:nowrap; }}
.journal {{ list-style:none; padding:0; margin:18px 0 8px; }}
.journal li {{ display:flex; gap:18px; padding:14px 0;
  border-bottom:1px solid var(--rule); }}
.journal li:last-child {{ border-bottom:none; }}
.jdate {{ flex:0 0 92px; font:600 12px/1.7 "IBM Plex Mono",monospace;
  color:var(--accent); }}
.journal div {{ color:var(--ink2); font-size:14px; }}
.journal strong {{ color:var(--ink); }}
</style>
<div class="wrap">
<h1>ARC-CT Experiment Ledger</h1>
<p class="sub">Every run of this project: finished, in flight, and queued. All figures
are read off disk at render time by <code>tools/exp_log.py</code>; nothing here is
transcribed by hand.</p>
<p class="stamp">rendered {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<h2>The two architectures</h2>
<p class="note">These are different models. A number from one is not a baseline for
the other unless the ledger says the split, the labels and the class set matched.</p>
<div class="cards">{arches}</div>
<p class="note"><strong>val AUC</strong> is the validation macro AUC the training
loop selected on, averaged over completed seeds &mdash; a model-selection number,
not a result. The <strong>held-out</strong> tables are the reportable ones.</p>
{"".join(rows)}
<h2>Journal</h2>
<p class="note">Dated record of what happened and what it changed. Newest first.</p>
<ol class="journal">{journal}</ol>
<h2>Standing conclusions</h2>
<div class="cards">{findings}</div>
{orph}
</div>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--md", default=os.path.join(ROOT, "docs", "EXPERIMENTS.md"))
    ap.add_argument("--html", default="")
    ap.add_argument("--check", action="store_true",
                    help="non-zero exit if disk holds runs the registry omits")
    a = ap.parse_args()

    reg = yaml.safe_load(open(a.registry, encoding="utf-8"))
    harvest(reg)

    if a.check:
        orf, ore = orphans(reg)
        for x in orf:
            print(f"UNLOGGED run   {x}")
        for x in ore:
            print(f"UNLOGGED eval  {x}")
        if orf or ore:
            print(f"\n{len(orf) + len(ore)} artifact(s) on disk that the ledger does "
                  "not describe. Add them to docs/experiments.yaml.")
            return 1
        print("ledger covers everything on disk")
        return 0

    open(a.md, "w", encoding="utf-8").write(render_md(reg))
    print(f"wrote {a.md}")
    if a.html:
        open(a.html, "w", encoding="utf-8").write(render_html(reg))
        print(f"wrote {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

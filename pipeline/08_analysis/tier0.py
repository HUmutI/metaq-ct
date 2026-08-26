#!/usr/bin/env python3
"""Tier 0: everything that can be decided about keyword-vs-Qwen with zero human time.

Four deliverables:
  1. matched-form frequency tables  - makes a broken lexicon visible at a glance
  2. ablation grid                  - attributes the 74.8 -> 31.5 swing to named knobs
  3. 2x2 + Cohen's kappa vs Qwen    - where the two methods actually differ
  4. citation-support audit         - per-label unsupported-citation rate, both methods

Prints aggregates and single matched tokens only. Never a sentence.
"""
from __future__ import annotations
import csv, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kw_baseline import PATHOLOGIES, Policy, score_report, matched_forms, compiled

D = "/temp_work/ch278233/BCH_DATASET/LABELS"


def load():
    reports = []
    with open(f"{D}/sentences.jsonl") as fh:
        for ln in fh:
            r = json.loads(ln)
            reports.append((r["VolumeName"], r["sentences"], r.get("n_findings_sentences", 0)))
    qwen = {}
    with open(f"{D}/labels.csv") as fh:
        for row in csv.DictReader(fh):
            qwen[row["VolumeName"]] = {p: int(row[p]) for p in PATHOLOGIES}
    return reports, qwen


def kappa(n11, n10, n01, n00):
    n = n11 + n10 + n01 + n00
    if n == 0:
        return float("nan")
    po = (n11 + n00) / n
    p1 = ((n11 + n10) / n) * ((n11 + n01) / n)
    p0 = ((n01 + n00) / n) * ((n10 + n00) / n)
    pe = p1 + p0
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main():
    reports, qwen = load()
    n = len(reports)
    print("=" * 78)
    print("TIER 0  -  %d reports, %d labels" % (n, len(PATHOLOGIES)))
    print("=" * 78)

    # ---- 1. matched-form tables -------------------------------------------
    print("\n1. MATCHED SURFACE FORMS (fixed lexicon, boundaries ON)")
    print("   a broken regex shows up here as a nonsense token with a huge count")
    forms = {l: Counter() for l in PATHOLOGIES}
    for _v, ss, _nf in reports:
        for l, fs in matched_forms(ss).items():
            forms[l].update(fs)
    for l in PATHOLOGIES:
        top = ", ".join("%s(%d)" % (w, c) for w, c in forms[l].most_common(6))
        print("   %-36s %s" % (l[:36], top or "-"))

    # ---- 2. ablation grid --------------------------------------------------
    print("\n2. ABLATION GRID  (Medical material %, the label that started this)")
    print("   %-34s %8s %8s" % ("policy", "MedMat%", "macro%"))
    grid = []
    for wb in (False, True):
        for neg in (False, True):
            for post in (False, True):
                for hist in (False, True):
                    grid.append(Policy(word_boundaries=wb, negation=neg,
                                       post_negation=post, history_as_negation=hist))
    for pol in grid:
        pos = Counter()
        for _v, ss, nf in reports:
            r = score_report(ss, nf, pol)
            for l in PATHOLOGIES:
                pos[l] += r[l]["p"]
        mm = 100.0 * pos["Medical material"] / n
        macro = 100.0 * sum(pos.values()) / (n * len(PATHOLOGIES))
        print("   %-34s %7.1f%% %7.1f%%" % (pol.tag(), mm, macro))

    # ---- 3. 2x2 vs Qwen ----------------------------------------------------
    print("\n3. KEYWORD (default policy) vs QWEN")
    print("   %-34s %7s %7s %7s %7s %6s" % ("label", "kw%", "qwen%", "kw_only", "qw_only", "kappa"))
    kwlab = {}
    for v, ss, nf in reports:
        kwlab[v] = score_report(ss, nf)
    rows = []
    for l in PATHOLOGIES:
        n11 = n10 = n01 = n00 = 0
        for v in qwen:
            if v not in kwlab:
                continue
            k = kwlab[v][l]["p"]; q = qwen[v][l]
            if k and q: n11 += 1
            elif q and not k: n10 += 1
            elif k and not q: n01 += 1
            else: n00 += 1
        kp = kappa(n11, n10, n01, n00)
        rows.append((l, n11, n10, n01, n00, kp))
        tot = n11 + n10 + n01 + n00
        print("   %-34s %6.1f%% %6.1f%% %7d %7d %6.2f"
              % (l[:34], 100.0*(n11+n01)/tot, 100.0*(n11+n10)/tot, n01, n10, kp))
    rows.sort(key=lambda r: r[5])
    print("\n   lowest agreement (context-dependent, LLM-native):  %s"
          % ", ".join("%s %.2f" % (r[0][:22], r[5]) for r in rows[:4]))
    print("   highest agreement (lexically crisp, keyword suffices): %s"
          % ", ".join("%s %.2f" % (r[0][:22], r[5]) for r in rows[-4:]))

    # ---- 4. citation-support audit ----------------------------------------
    print("\n4. CITATION SUPPORT  (does the cited sentence contain any term for that label?)")
    print("   symmetric: same broad lexicon applied to both methods' own evidence")
    ev = {}
    with open(f"{D}/evidence.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ev.setdefault(row["VolumeName"], []).append((row["label"], row["evidence_sentence"]))
    q_ok = q_tot = k_ok = k_tot = 0
    per = {}
    sent_by_vol = {v: ss for v, ss, _ in reports}
    for v, items in ev.items():
        for label, sent in items:
            if label not in PATHOLOGIES:
                continue
            q_tot += 1
            hit = bool(compiled(label).search(sent))
            q_ok += hit
            d = per.setdefault(label, [0, 0, 0, 0])
            d[0] += 1; d[1] += hit
    for v, kl in kwlab.items():
        for label, h in kl.items():
            if h["p"]:
                k_tot += 1
                ss = sent_by_vol.get(v, [])
                ok = 0 <= h["e"] < len(ss)
                k_ok += ok
                d = per.setdefault(label, [0, 0, 0, 0])
                d[2] += 1; d[3] += ok
    print("   %-34s %10s %10s" % ("label", "qwen_sup%", "kw_sup%"))
    for l in PATHOLOGIES:
        d = per.get(l, [0, 0, 0, 0])
        qs = 100.0*d[1]/d[0] if d[0] else float("nan")
        ks = 100.0*d[3]/d[2] if d[2] else float("nan")
        print("   %-34s %9.1f%% %9.1f%%" % (l[:34], qs, ks))
    print("\n   overall: qwen %d/%d = %.1f%%   keyword %d/%d = %.1f%%"
          % (q_ok, q_tot, 100.0*q_ok/max(q_tot,1), k_ok, k_tot, 100.0*k_ok/max(k_tot,1)))
    print("\n   NOTE: keyword's citation is its own match by construction, so its rate is")
    print("   near 100 by definition and is NOT evidence of quality. Qwen's rate is")
    print("   meaningful because its citation is an independent choice.")
    print("=" * 78)


if __name__ == "__main__":
    main()

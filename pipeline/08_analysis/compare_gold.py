#!/usr/bin/env python3
"""Compare Claude's gold-standard labels against Qwen V1 and V2.

The sample is pediatric, so there is no external ground truth: CT-RATE's
published labels are RadBERT predictions on *adult* reports and cover only
18 of the 27 classes. What this script measures is therefore agreement, plus
one thing that is closer to a verdict - on the 52 reports where V1 and V2
disagree, exactly one of them is right, so Claude's vote adjudicates.

Three numbers per class:
  agree_v2   - cells where Claude and the current labels (V2) match
  claude_pos - how often Claude says 1 where V2 says 0 (Qwen missed it)
  qwen_pos   - how often V2 says 1 where Claude says 0 (Qwen over-called it)

Cohen's kappa is reported alongside raw agreement because these classes are
rare: a class with 2% prevalence gets 98% agreement from a labeller that
always says 0, and that number means nothing.
"""
from __future__ import annotations
import glob, json, os, sys
from collections import defaultdict

GOLD = "/temp_work/ch278233/GOLD_SAMPLE/gold_sample.jsonl"
CLAUDE_GLOB = "/temp_work/ch278233/GOLD_SAMPLE/claude_batch*.jsonl"

sys.path.insert(0, "/home/ch278233/bch-arc-ct")
os.environ.setdefault("RAC_SCHEMA", "peds")
from arcct.schema import PEDS_PATHOLOGIES as P27


def kappa(a, b):
    """Cohen's kappa for two binary label vectors."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:                      # both constant and identical
        return float("nan")
    return (po - pe) / (1 - pe)


def load():
    gold = {}
    with open(GOLD) as fh:
        for line in fh:
            r = json.loads(line)
            gold[r["VolumeName"]] = r
    claude, dupes = {}, []
    for path in sorted(glob.glob(CLAUDE_GLOB)):
        with open(path) as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    print("  ! %s:%d unparseable: %s" % (path, ln, e))
                    continue
                v = r["VolumeName"]
                if v in claude:
                    dupes.append(v)
                claude[v] = r
    return gold, claude, dupes


def main():
    gold, claude, dupes = load()
    missing = [v for v in gold if v not in claude]
    extra = [v for v in claude if v not in gold]
    print("gold reports          : %d" % len(gold))
    print("claude reports        : %d" % len(claude))
    if missing:
        print("MISSING from claude   : %d  %s" % (len(missing), missing[:5]))
    if extra:
        print("NOT IN SAMPLE         : %d  %s" % (len(extra), extra[:5]))
    if dupes:
        print("DUPLICATED            : %d  %s" % (len(dupes), dupes[:5]))

    vols = [v for v in gold if v in claude]
    # every class must be present in every record, or the comparison silently
    # scores absent keys as 0 and flatters whichever side omitted them
    incomplete = [v for v in vols
                  if any(c not in claude[v].get("labels", {}) for c in P27)]
    if incomplete:
        print("INCOMPLETE label dicts: %d  %s" % (len(incomplete), incomplete[:5]))
        for v in incomplete[:3]:
            miss = [c for c in P27 if c not in claude[v]["labels"]]
            print("    %s -> missing %s" % (v, miss[:6]))
    print()

    rows = []
    tot = defaultdict(int)
    for c in P27:
        cl = [int(claude[v]["labels"].get(c, 0)) for v in vols]
        v1 = [int(gold[v]["labels_v1"][c]) for v in vols]
        v2 = [int(gold[v]["labels_v2"][c]) for v in vols]
        ag2 = sum(x == y for x, y in zip(cl, v2))
        c_only = sum(1 for x, y in zip(cl, v2) if x == 1 and y == 0)
        q_only = sum(1 for x, y in zip(cl, v2) if x == 0 and y == 1)
        rows.append((c, sum(cl), sum(v1), sum(v2),
                     ag2 / len(vols), kappa(cl, v2), c_only, q_only))
        tot["ag2"] += ag2
        tot["n"] += len(vols)
        tot["c_only"] += c_only
        tot["q_only"] += q_only

    print("%-42s %5s %5s %5s %7s %7s %6s %6s"
          % ("class", "CLA+", "V1+", "V2+", "agree", "kappa", "C-only", "Q-only"))
    print("-" * 96)
    for r in sorted(rows, key=lambda r: (r[5] if r[5] == r[5] else -9)):
        k = "%7.3f" % r[5] if r[5] == r[5] else "      -"
        print("%-42s %5d %5d %5d %6.1f%% %s %6d %6d"
              % (r[0], r[1], r[2], r[3], 100 * r[4], k, r[6], r[7]))
    print("-" * 96)
    print("%-42s %5d %5d %5d %6.1f%% %7s %6d %6d"
          % ("OVERALL (cells)",
             sum(r[1] for r in rows), sum(r[2] for r in rows),
             sum(r[3] for r in rows),
             100 * tot["ag2"] / tot["n"], "",
             tot["c_only"], tot["q_only"]))

    # ---- adjudication on the V1/V2 disagreements -------------------------
    print("\n=== ADJUDICATION: the cells where V1 and V2 disagree ===")
    per_class = defaultdict(lambda: [0, 0, 0])   # v1_right, v2_right, neither
    for v in vols:
        for c in P27:
            a, b = int(gold[v]["labels_v1"][c]), int(gold[v]["labels_v2"][c])
            if a == b:
                continue
            x = int(claude[v]["labels"].get(c, 0))
            slot = 0 if x == a else (1 if x == b else 2)
            per_class[c][slot] += 1
    if not per_class:
        print("(no disagreeing cells in this sample)")
        return
    print("%-42s %8s %8s" % ("class", "V1 right", "V2 right"))
    print("-" * 60)
    t1 = t2 = 0
    for c, (a, b, _) in sorted(per_class.items(), key=lambda kv: -sum(kv[1])):
        print("%-42s %8d %8d" % (c, a, b))
        t1 += a
        t2 += b
    print("-" * 60)
    print("%-42s %8d %8d" % ("TOTAL", t1, t2))


if __name__ == "__main__":
    main()

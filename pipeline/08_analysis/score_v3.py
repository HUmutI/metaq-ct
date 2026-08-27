#!/usr/bin/env python3
"""Score a re-extraction against the 200-report gold standard.

Takes a directory of labels.shard*.jsonl and reports, per class, agreement and
kappa against Claude's gold labels - alongside V2's, so the question the run was
launched to answer ("did the definition fix help, and did it break anything
that was working?") is answered in one table rather than two.

The second column is the one that matters. A definition change can raise the
class it targeted and quietly cost two others; only a side-by-side shows that.
"""
from __future__ import annotations
import argparse, glob, json, os, statistics, sys
from collections import defaultdict

sys.path.insert(0, "/home/ch278233/bch-arc-ct")
os.environ.setdefault("RAC_SCHEMA", "peds")
from arcct.schema import PEDS_PATHOLOGIES as P27

GOLD = "/temp_work/ch278233/GOLD_SAMPLE/gold_sample.jsonl"
CLA = "/temp_work/ch278233/GOLD_SAMPLE/claude_batch*.jsonl"


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    p1, p2 = sum(a) / n, sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def load_shards(d):
    out, bad = {}, 0
    # a single non-array task writes labels.jsonl; an 8-shard array writes
    # labels.shard{0..7}.jsonl. Accept both so the smoke and the full run score
    # through the same code path.
    paths = sorted(glob.glob(os.path.join(d, "labels.shard*.jsonl"))) or \
            sorted(glob.glob(os.path.join(d, "labels.jsonl")))
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") != "ok":
                    bad += 1
                    continue
                out[r["VolumeName"]] = {c: int(v["p"])
                                        for c, v in r["labels"].items()}
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/temp_work/ch278233/GOLD_SAMPLE/V3_SMOKE")
    a = ap.parse_args()

    gold = {json.loads(l)["VolumeName"]: json.loads(l) for l in open(GOLD)}
    cla = {}
    for p in glob.glob(CLA):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                cla[r["VolumeName"]] = r["labels"]
    v3, bad = load_shards(a.dir)

    vols = [v for v in gold if v in v3 and v in cla]
    print("gold 200 | v3 records %d (%d not ok) | scored %d"
          % (len(v3), bad, len(vols)))
    if len(vols) < len(gold):
        print("  missing from v3: %d" % (len(gold) - len(vols)))
    if not vols:
        return
    print()

    print("%-40s %13s %13s %8s" % ("class", "V2 kappa", "V3 kappa", "delta"))
    print("-" * 78)
    rows, d2, d3 = [], [], []
    for c in P27:
        g = [int(cla[v].get(c, 0)) for v in vols]
        k2 = kappa(g, [int(gold[v]["labels_v2"][c]) for v in vols])
        k3 = kappa(g, [int(v3[v].get(c, 0)) for v in vols])
        if k2 is None and k3 is None:
            continue
        rows.append((c, k2, k3))
    for c, k2, k3 in sorted(rows, key=lambda r: (r[2] or -9) - (r[1] or -9)):
        f = lambda k: "%.3f" % k if k is not None else "    -"
        dl = ("%+.3f" % (k3 - k2)) if (k2 is not None and k3 is not None) else "     -"
        mark = ""
        if k2 is not None and k3 is not None:
            mark = "  <<<" if k3 - k2 <= -0.03 else ("  ***" if k3 - k2 >= 0.03 else "")
        print("%-40s %13s %13s %8s%s" % (c, f(k2), f(k3), dl, mark))
        if k2 is not None and k3 is not None:
            d2.append(k2); d3.append(k3)
    print("-" * 78)
    print("%-40s %13.3f %13.3f %+8.3f"
          % ("MEAN (classes scored in both)", statistics.mean(d2),
             statistics.mean(d3), statistics.mean(d3) - statistics.mean(d2)))
    print("\n*** improved by >=0.03    <<< REGRESSED by >=0.03")


if __name__ == "__main__":
    main()

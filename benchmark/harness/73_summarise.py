#!/usr/bin/env python3
"""Collect the eval JSONs into the comparison table."""
import glob, json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/temp_work/ch278233/BENCHMARK_RUNS/eval"

def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(D, pat))):
        try:
            out.append((f, json.load(open(f))))
        except Exception as e:
            out.append((f, {"error": str(e)}))
    return out

print("\n=== selection sweep (ctrate_dev prefix) ===")
for f, d in load("sel_*.json"):
    m = d.get("macro_auroc")
    print(f"  {os.path.basename(f):34s} macro={('%.4f'%m) if m is not None else 'n/a':>7s} "
          f"({d.get('n_classes_evaluable','?')}/{d.get('n_classes','?')} cls, "
          f"{d.get('n_volumes','?')} vols)  {os.path.basename(str(d.get('ckpt','')))}")

print("\n=== HEADLINE: official CT-RATE validation, 18 classes, 3,002 studies ===")
print(f"  {'arm':34s} {'macro AUROC':>12s}  {'cls':>7s}  {'vols':>6s}")
for f, d in load("TEST_*.json"):
    if "peds" in f: continue
    m = d.get("macro_auroc")
    print(f"  {os.path.basename(f)[5:-5]:34s} {('%.4f'%m) if m is not None else 'FAILED':>12s}  "
          f"{d.get('n_classes_evaluable','?')}/{d.get('n_classes','?'):<5} {d.get('n_volumes','?'):>6}")

print("\n=== HEADLINE: pediatric age>=5, 23 classes, 1,507 studies ===")
print(f"  {'arm':34s} {'macro AUROC':>12s}  {'cls':>7s}  {'vols':>6s}")
for f, d in load("TEST_peds_*.json"):
    m = d.get("macro_auroc")
    print(f"  {os.path.basename(f)[10:-5]:34s} {('%.4f'%m) if m is not None else 'FAILED':>12s}  "
          f"{d.get('n_classes_evaluable','?')}/{d.get('n_classes','?'):<5} {d.get('n_volumes','?'):>6}")

print("\n=== per-class, best arm per cohort ===")
for tag, pat in (("CT-RATE 18", "TEST_grf_classifier.json"), ("pediatric 23", "TEST_peds_grf_classifier.json")):
    for f, d in load(pat):
        if "per_class" not in d: continue
        print(f"\n  [{tag}] {os.path.basename(f)}")
        for r in d["per_class"]:
            a = r.get("auc")
            print(f"    {r['class']:44s} {('%.4f'%a) if a is not None else 'skipped(no pos/neg)':>10s}  n_pos={r.get('n_pos')}")

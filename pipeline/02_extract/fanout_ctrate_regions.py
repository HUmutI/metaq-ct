#!/usr/bin/env python3
"""Fan the CT-RATE region cache from 22,976 reports out to all 47,149 volumes.

build_outputs.py keys region_cache.json by the VolumeName it saw, and the LLM
saw one representative volume per distinct report - the same study appears under
several reconstructions carrying identical text. The trainer looks the cache up
by the volume it is training on, so without this step 24,173 adult volumes would
miss and fall back to whole-report text for every organ query.

That fallback is the reason this matters at all: the combined run today has 0
adult keys in the cache out of 14,685 adult training volumes, so the per-organ
alignment term - the thing the anatomy Q-Former exists for - has been running
unrouted on the entire adult half.

Merging with the pediatric cache is deliberately NOT done here. Keys are
disjoint by construction (ped_* vs train_*), but a silent overwrite of the
pediatric side is exactly the kind of damage that shows up three hours into a
training run, so the merge is its own explicit step with its own assertion.
"""
from __future__ import annotations
import argparse, json, os, sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/temp_work/ch278233/CTRATE/LABELS27")
    ap.add_argument("--out", default="/temp_work/ch278233/CTRATE/region_cache_all.json")
    a = ap.parse_args()

    cache = json.load(open(os.path.join(a.dir, "region_cache.json")))
    fan = json.load(open(os.path.join(a.dir, "report_fanout.json")))
    print("  onbellek anahtari (temsilci): %d" % len(cache))
    print("  fan-out girisi              : %d" % len(fan))

    out, miss = {}, 0
    for rep_vol, vols in fan.items():
        obj = cache.get(rep_vol)
        if obj is None:
            miss += len(vols)
            continue
        for v in vols:
            out[v] = obj                      # shared reference: same report, same routing
    json.dump(out, open(a.out, "w"), ensure_ascii=False)

    total = sum(len(v) for v in fan.values())
    print("  yazilan hacim               : %d / %d  (onbellegi olmayan %d)"
          % (len(out), total, miss))
    # every region should be non-empty for a decent share of volumes; an all-zero
    # row means the region ids in the adult run did not match the schema
    cov = {str(k): 0 for k in range(1, 11)}
    for obj in list(out.values())[:5000]:
        for k, v in obj.items():
            if v:
                cov[k] = cov.get(k, 0) + 1
    n = min(len(out), 5000) or 1
    print("  bolge kapsami (ilk %d hacim):" % n)
    for k in sorted(cov, key=int):
        print("     bolge %-3s %5.1f%%" % (k, 100.0 * cov[k] / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())

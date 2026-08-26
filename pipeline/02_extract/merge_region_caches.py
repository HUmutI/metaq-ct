#!/usr/bin/env python3
"""Merge the pediatric and adult region caches into one the combined run can use.

Keys are disjoint by construction - ped_* against train_* - but "by
construction" is exactly the kind of assumption that stops being true after
someone renames something, and a silent overwrite of the pediatric half would
show up only as a slowly worse alignment loss three hours into a run. So the
disjointness is asserted, not trusted.
"""
from __future__ import annotations
import argparse, json, sys

PEDS = "/temp_work/ch278233/BCH_DATASET/LABELS27_V2/region_cache.json"
ADULT = "/temp_work/ch278233/CTRATE/region_cache_all.json"
OUT = "/temp_work/ch278233/COMBINED_region_cache_v2.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peds", default=PEDS)
    ap.add_argument("--adult", default=ADULT)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    p = json.load(open(a.peds))
    q = json.load(open(a.adult))
    both = set(p) & set(q)
    print("  pediatrik %d  yetiskin %d  ortak %d" % (len(p), len(q), len(both)))
    if both:
        print("  !! ANAHTAR CAKISMASI: %s" % sorted(both)[:5])
        return 2

    merged = dict(p)
    merged.update(q)
    tmp = a.out + ".tmp"
    json.dump(merged, open(tmp, "w"), ensure_ascii=False)
    import os
    os.replace(tmp, a.out)
    print("  birlesik anahtar: %d" % len(merged))

    npeds = sum(1 for k in merged if k.startswith("ped_"))
    print("  ped_ %d   train_ %d" % (npeds, len(merged) - npeds))
    empty = sum(1 for o in merged.values() if not any(o.values()))
    print("  hicbir bolgeye yonlenmeyen: %d (%.2f%%)"
          % (empty, 100.0 * empty / max(len(merged), 1)))
    print("  yazildi: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

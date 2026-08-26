#!/usr/bin/env python3
"""Run TotalSegmentator over many volumes in ONE process.

The CLI reloads the nnU-Net weights on every invocation, which the 20-volume
benchmark measured at a flat ~45 s/volume regardless of size - i.e. dominated by
startup, not by inference. The Python API keeps the model resident, so the
marginal cost per volume is the actual forward pass.

Prints per-volume timings and aggregate stats; never a path that identifies a
study beyond the de-identified stem already on disk.
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ch278233/pipeline/lib")   # shared modules: ts_roi, qwen_extract, prepare_reports
from ts_roi import PEDS as ROI          # 77 names -> pediatric regions 1-10, 12, 13


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=int(os.environ.get("SHARD_ID", "0")))
    ap.add_argument("--shard-n", type=int, default=int(os.environ.get("SHARD_N", "1")))
    a = ap.parse_args()

    from totalsegmentator.python_api import totalsegmentator

    os.makedirs(a.out_dir, exist_ok=True)
    # An explicit worklist beats re-listing the directory: shards start at
    # different times, and when the input set changes between those starts the
    # stride assignment shifts, so some volumes are claimed by nobody. That is
    # how the first pass finished 1275 short.
    wl = os.environ.get("TS_WORKLIST", "")
    if wl and os.path.exists(wl):
        files = [l.strip() for l in open(wl) if l.strip()]
    else:
        files = sorted(f for f in os.listdir(a.in_dir) if f.endswith(".nii.gz"))
    if a.shard_n > 1:
        files = files[a.shard_id::a.shard_n]
    if a.limit:
        files = files[:a.limit]
    print("[ts] shard %d/%d: %d volumes" % (a.shard_id, a.shard_n, len(files)), flush=True)

    times, ok, skip, fail = [], 0, 0, 0
    t_start = time.time()
    for i, fn in enumerate(files):
        dst = os.path.join(a.out_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1
            continue
        t0 = time.time()
        try:
            totalsegmentator(os.path.join(a.in_dir, fn), dst,
                             task="total", ml=True, device="gpu", quiet=True,
                             roi_subset=ROI, nr_thr_resamp=4, nr_thr_saving=4)
            dt = time.time() - t0
            times.append(dt)
            ok += 1
            if ok <= 5 or ok % 25 == 0:
                med = sorted(times)[len(times) // 2]
                print("[ts] %d/%d  last=%.1fs  median=%.1fs  eta=%.1fmin"
                      % (i + 1, len(files), dt, med,
                         med * (len(files) - i - 1) / 60), flush=True)
        except Exception as exc:
            fail += 1
            print("[ts] FAIL %s: %s" % (fn[:20], type(exc).__name__), flush=True)
    el = time.time() - t_start
    if times:
        s = sorted(times)
        print("[ts] DONE ok=%d skip=%d fail=%d  total=%.1fmin  "
              "per-volume median=%.1fs p90=%.1fs first=%.1fs"
              % (ok, skip, fail, el / 60, s[len(s) // 2], s[int(len(s) * 0.9)], times[0]))
    else:
        print("[ts] DONE ok=%d skip=%d fail=%d (nothing new)" % (ok, skip, fail))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

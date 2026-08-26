#!/usr/bin/env python3
"""List every npz in COMBINED_NPZ that cannot be read.

A single 0-byte file killed a 12,000-step run 13 minutes in. np.load on an npz
only reads the zip central directory unless an array is actually indexed, so
checking all 25k files costs a directory read each, not a decompress - cheap
enough to run before every launch rather than discovering it at hour three.

Checks the central directory AND that 'arr_0' is present: a file truncated
mid-write can still have a parseable header but no payload entry.
"""
import argparse, os, sys, glob, numpy as np
from multiprocessing import Pool


def check(p):
    # Atomic-write staging files are not corruption: the fetch job renames them
    # into place, so one that vanishes between the glob and the open was doing
    # exactly what it should. Counting them as bad hides the real failures in
    # noise - the tree had 2 "bad" files and only one was a genuine 0-byte npz.
    if ".tmp." in os.path.basename(p):
        return None
    try:
        if os.path.getsize(p) < 1024:
            return (p, "too small (%d bytes)" % os.path.getsize(p))
        with np.load(p) as z:
            if "arr_0" not in z.files:
                return (p, "no arr_0")
    except Exception as e:
        return (p, "%s: %s" % (type(e).__name__, e))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/temp_work/ch278233/COMBINED_NPZ")
    ap.add_argument("--out", default="/temp_work/ch278233/BAD_NPZ.txt")
    a = ap.parse_args()
    # the combined tree is patient/study/file, the fetch tree is flat - accept
    # either rather than silently finding nothing and reporting zero corruption
    files = (glob.glob(os.path.join(a.root, "*", "*", "*.npz"))
             or glob.glob(os.path.join(a.root, "*.npz")))
    if not files:
        print("[val] %s altinda npz bulunamadi" % a.root, flush=True)
        return 1
    print("[val] %d dosya (%s)" % (len(files), a.root), flush=True)
    with Pool(16) as pool:
        bad = [r for r in pool.imap_unordered(check, files, chunksize=64) if r]
    print("[val] BOZUK: %d" % len(bad), flush=True)
    with open(a.out, "w") as fh:
        for p, why in sorted(bad):
            fh.write(os.path.basename(p).replace(".npz", "") + "\n")
            print("  %s  <- %s" % (os.path.basename(p), why), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

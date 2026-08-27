#!/usr/bin/env python3
"""TotalSegmentator over CT-RATE npz volumes -> the 10-region mask arc-ct consumes.

The pediatric pipeline started from raw NIfTI and resampled; CT-RATE is already
stored resampled at 1.5x1.5x3.0 mm inside the npz, so the resample step is
skipped and the array is written straight out as NIfTI.

Two geometry facts this depends on, both verified against the loader rather than
assumed - getting either wrong is silent and produced a mask that was more
air-like inside the lung than outside it the first time round:

  * the npz is stored (D, H, W) and dataset._load_npz_hwd transposes (1, 2, 0)
    to (H, W, D). The NIfTI must be written in that same (H, W, D) order so the
    affine diag(1.5, 1.5, 3.0) puts the 3 mm axis last.
  * the npz holds HU/1000. TotalSegmentator wants HU, so multiply by 1000.

This stage stops at the raw TotalSegmentator output. Region mapping, the
small-structure dilation, the derived pleural shell and the pad/crop to
192x192x96 all live in build_peds_masks.py, which imports the CT loader's own
_pad_crop_hwd so the mask and the volume reach 192x192x96 by the same geometric
route - and that import pulls in monai, which does not exist in the
TotalSegmentator environment (it needs numpy 2.x and would break arcct's torch).
Copying the function here instead would let the two paths drift, which is
exactly the bug that once produced a mask more air-like inside the lung than
outside it. So the two stages stay separate processes, as they are for the
pediatric cohort.

The resampled NIfTI is still never kept: it is written to node-local disk, fed to
the segmenter, and deleted. Only the label volume survives, which compresses to
a fraction of the CT.
"""
from __future__ import annotations
import argparse, os, shutil, sys, tempfile, time
import numpy as np, nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
# ts_roi lives in pipeline/lib, reached from HERE rather than by absolute path:
# an absolute path here pointed into a SECOND copy of the pipeline tree, so this
# file imported its region table from a directory nothing else in the repo used.
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
from ts_roi import PEDS as ROI

SPACING = (1.5, 1.5, 3.0)

# Progress every 10 volumes, not 25. At ~90 s a volume the old interval put ~37
# minutes between log lines, and the watchdog calls a job stalled after 30 -- so a
# perfectly healthy shard raised an alert every cycle and the real alerts were
# harder to see among them.


def npz_to_nifti(npz_path: str, dst: str) -> None:
    arr = np.load(npz_path)["arr_0"]
    hwd = np.transpose(arr, (1, 2, 0)).astype(np.float32) * 1000.0   # -> HU
    aff = np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0]).astype(np.float64)
    nib.save(nib.Nifti1Image(hwd, affine=aff), dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worklist", default=os.environ.get("TS_WORKLIST", ""),
                    help="file of npz paths to process; overrides scanning --in-dir")
    ap.add_argument("--shard-id", type=int, default=int(os.environ.get("SHARD_ID", "0")))
    ap.add_argument("--shard-n", type=int, default=int(os.environ.get("SHARD_N", "1")))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from totalsegmentator.python_api import totalsegmentator

    # The two npz trees have different shapes: the fetch output is flat, the
    # CT-RATE release is patient/study/file. Accept either rather than silently
    # finding nothing - a listdir on the nested tree returns directories and
    # reports "0 volumes" without an error.
    import glob as _g
    if a.worklist:
        # Resuming a partial set. Scanning --in-dir and relying on the "output
        # already exists" skip below would still stat every one of 42k volumes
        # across 64 shards, and would silently reprocess anything whose mask
        # came from elsewhere and lives under a different root.
        with open(a.worklist, encoding="utf-8") as fh:
            files = [ln.strip() for ln in fh if ln.strip()]
        gone = [f for f in files if not os.path.exists(f)]
        if gone:
            print("[ts] worklist'te olmayan %d npz, ilki: %s"
                  % (len(gone), gone[0]), flush=True)
            return 1
        print("[ts] worklist %s: %d volumes" % (a.worklist, len(files)), flush=True)
    else:
        files = sorted(_g.glob(os.path.join(a.in_dir, "*", "*", "*.npz"))
                       or _g.glob(os.path.join(a.in_dir, "*.npz")))
    files = [f for f in files if ".tmp." not in os.path.basename(f)]
    if not files:
        print("[ts] %s altinda npz bulunamadi" % a.in_dir, flush=True)
        return 1
    if a.shard_n > 1:
        files = files[a.shard_id::a.shard_n]
    if a.limit:
        files = files[:a.limit]
    print("[ts] shard %d/%d: %d volumes" % (a.shard_id, a.shard_n, len(files)), flush=True)

    tmp = tempfile.mkdtemp(prefix="tsctr_", dir=os.environ.get("TMPDIR", "/tmp"))
    times, ok, skip, fail = [], 0, 0, 0
    t_start = time.time()
    try:
        for i, fn in enumerate(files):
            stem = os.path.basename(fn)[:-4]
            dst = os.path.join(a.out_dir, stem + ".nii.gz")
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                skip += 1
                continue
            t0 = time.time()
            src_nii = os.path.join(tmp, stem + ".nii.gz")
            try:
                npz_to_nifti(fn, src_nii)
                totalsegmentator(src_nii, dst, task="total", ml=True,
                                 device="gpu", quiet=True, roi_subset=ROI,
                                 nr_thr_resamp=4, nr_thr_saving=4)
                times.append(time.time() - t0)
                ok += 1
                if ok <= 3 or ok % 10 == 0:
                    med = sorted(times)[len(times) // 2]
                    print("[ts] %d/%d ok=%d last=%.1fs median=%.1fs eta=%.1fmin"
                          % (i + 1, len(files), ok, times[-1], med,
                             med * (len(files) - i - 1) / 60), flush=True)
            except Exception as exc:
                fail += 1
                print("[ts] FAIL %s: %s: %s" % (stem[:22], type(exc).__name__, exc), flush=True)
            finally:
                try: os.remove(src_nii)
                except OSError: pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    el = time.time() - t_start
    if times:
        s = sorted(times)
        print("[ts] DONE ok=%d skip=%d fail=%d total=%.1fmin median=%.1fs p90=%.1fs"
              % (ok, skip, fail, el / 60, s[len(s) // 2], s[int(len(s) * 0.9)]))
    else:
        print("[ts] DONE ok=%d skip=%d fail=%d (nothing new)" % (ok, skip, fail))
    return 0 if fail <= len(files) * 0.02 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download CT-RATE train volumes and convert to npz, batch by batch.

Raw CT-RATE is ~231 MB per volume; the whole 16k selection is 3.7 TB, which does
not fit anywhere convenient. So this works in batches: download N volumes with 8
parallel streams (measured optimum - 8 gives 229 MB/s, 16 only 250, so the
ceiling is not the connection count), convert each to npz, delete the raw file,
move on. Peak disk is one batch, not the whole set.

Preprocessing mirrors tools/prepare_data.py:preprocess_volume exactly:
rescale slope/intercept from the metadata CSV, resample to 1.5x1.5x3.0 mm,
transpose to (D,H,W), divide by 1000. Any drift here would silently train the
model on a different intensity scale than the adult checkpoint learned.
"""
from __future__ import annotations
import argparse, ast, csv, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np, torch, torch.nn.functional as F, nibabel as nib
from huggingface_hub import hf_hub_download

TARGET_SPACING = (1.5, 1.5, 3.0)
REPO = "ibrahimhamamci/CT-RATE"


def load_meta(path: str) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            out[r["VolumeName"]] = r
    return out


def hf_path(vol: str) -> str:
    # train_1_a_1.nii.gz -> dataset/train/train_1/train_1_a/train_1_a_1.nii.gz
    stem = vol.replace(".nii.gz", "")
    parts = stem.split("_")
    pid = "_".join(parts[:2])           # train_1
    sub = "_".join(parts[:3])           # train_1_a
    return f"dataset/train/{pid}/{sub}/{vol}"


class NotAChestCT(ValueError):
    """The source series is not an axial chest CT."""


def sanity_check(meta: dict) -> None:
    """Reject volumes CT-RATE's train split carries that are not chest CTs.

    Two turned up in the first 1839 conversions: a 512x914 "Thorax_Parankim MPR"
    (a reformat, not an axial acquisition) and a 699x512 "Sinus 1,00 Hr60 ax"
    with a 137 mm reconstruction diameter - a sinus study, not a chest. Pairing a
    sinus CT with a chest report teaches the model a wrong image-text pair, and
    nothing downstream would ever flag it.
    """
    try:
        rows, cols = int(float(meta.get("Rows", 0))), int(float(meta.get("Columns", 0)))
    except Exception:
        rows = cols = 0
    if rows and cols and abs(rows - cols) > 1:
        raise NotAChestCT("non-square acquisition %dx%d" % (rows, cols))
    try:
        diam = float(meta.get("ReconstructionDiameter", 0) or 0)
    except Exception:
        diam = 0.0
    if 0 < diam < 180:                      # an adult chest FOV is ~300-500 mm
        raise NotAChestCT("reconstruction diameter %.0f mm too small for a chest" % diam)
    desc = (meta.get("SeriesDescription") or "").lower()
    for bad in ("sinus", "mpr", "sagittal", "coronal", "head", "brain", "neck only"):
        if bad in desc:
            raise NotAChestCT("series description says %r" % bad)


def to_npz(nii_path: str, meta: dict, dst: str) -> tuple[int, int, int]:
    sanity_check(meta)
    nii = nib.load(nii_path)
    vol = nii.get_fdata().astype(np.float32)
    vol = vol * float(meta.get("RescaleSlope", 1.0)) + float(meta.get("RescaleIntercept", 0.0))
    try:
        xy = float(ast.literal_eval(str(meta.get("XYSpacing", "[1.0,1.0]")))[0])
    except Exception:
        xy = 1.0
    try:
        z = float(meta.get("ZSpacing", 1.0))
    except Exception:
        z = 1.0
    shape = tuple(max(1, round(s * so / st))
                  for s, so, st in zip(vol.shape, (xy, xy, z), TARGET_SPACING))
    t = F.interpolate(torch.from_numpy(vol)[None, None], size=shape,
                      mode="trilinear", align_corners=False)
    dhw = np.transpose(t.squeeze().numpy(), (2, 0, 1)).astype(np.float32) / 1000.0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Write to a private temp name and rename into place. Two processes can
    # legitimately target the same output directory here (one per account, to
    # keep the per-UID quota separate), and np.savez_compressed writing the same
    # path from both would interleave into a corrupt archive that still looks
    # like a file. rename() is atomic on POSIX, so a reader either sees the old
    # file or a complete new one, never a half-written one.
    tmp = "%s.tmp.%d" % (dst, os.getpid())
    np.savez_compressed(tmp, arr_0=dhw)
    os.replace(tmp + ".npz" if os.path.exists(tmp + ".npz") else tmp, dst)
    return dhw.shape



def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def own_usage_gb(root: str = "/temp_work/ch278233") -> float:
    """Bytes we own under our own space, in GB.

    The quota is per-uid, not per-directory: files this process writes into a
    colleague's temp_work are still owned by us and still count. The first run
    died with 'Disk quota exceeded' after 3.5 hours precisely because of that,
    so the budget has to be checked against what WE own, wherever it sits.
    """
    tot = 0
    for dp, _d, fn in os.walk(root):
        for f in fn:
            try:
                tot += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return tot / 1e9


def _convert_one(args) -> bool:
    """Top-level so ProcessPoolExecutor can pickle it."""
    src, meta, dst = args
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    try:
        to_npz(src, meta, dst)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="/temp_work/ch278233/tmp/ctrate_16k.txt")
    ap.add_argument("--meta", default="/temp_work/ch278233/CTRATE/ct_meta_hf/dataset/metadata/train_metadata.csv")
    ap.add_argument("--raw-dir", default="/temp_work/ch278452/CTRATE_RAW_TMP")
    ap.add_argument("--out-dir", default="/temp_work/ch278452/CTRATE_NPZ/train")
    ap.add_argument("--batch", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stop-below-gb", type=float, default=250.0,
                    help="stop before free space drops under this")
    ap.add_argument("--cworkers", type=int, default=14,
                    help="conversion processes; this is the real bottleneck")
    a = ap.parse_args()

    meta = load_meta(a.meta)
    vols = [l.strip() for l in open(a.list) if l.strip()]
    todo = [v for v in vols
            if not os.path.exists(os.path.join(a.out_dir, v.replace(".nii.gz", ".npz")))]
    print("[ct] %d volumes selected, %d still to do" % (len(vols), len(todo)), flush=True)

    t_start = time.time()
    done = fail = 0

    # Download batch N+1 while batch N converts. Measured serially: dl=227s,
    # conv=160s per 224-volume batch, so the two phases idle each other's
    # resource half the time - the network sits still during conversion and 14
    # cores sit still during download. Overlapping makes the batch cost
    # max(227, 160) instead of 387.
    def fetch_batch(vs, into):
        shutil.rmtree(into, ignore_errors=True)
        os.makedirs(into, exist_ok=True)

        def grab(v):
            # Retry with backoff instead of dropping the volume. HuggingFace
            # rate-limits anonymous traffic with HTTP 429, and the first version
            # counted every 429 as a permanent failure and moved on: one run
            # reported fail=4928 with total=0 and would have "finished"
            # successfully having downloaded nothing. A transient refusal must
            # not silently remove a volume from the dataset.
            delay = 5.0
            for attempt in range(5):
                try:
                    return v, hf_hub_download(REPO, hf_path(v), repo_type="dataset",
                                              local_dir=into)
                except Exception as exc:
                    msg = str(exc)
                    transient = ("429" in msg or "Rate limit" in msg.lower()
                                 or "timed out" in msg.lower() or "Connection" in msg)
                    if not transient or attempt == 4:
                        if attempt == 4:
                            print("[ct] giving up on %s after 5 tries: %s"
                                  % (v, type(exc).__name__), flush=True)
                        return v, None
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
            return v, None
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            return list(ex.map(grab, vs))

    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    dirs = [a.raw_dir + "_A", a.raw_dir + "_B"]
    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(fetch_batch, batches[0], dirs[0]) if batches else None

    for bi, batch in enumerate(batches):
        # Stop cleanly with room to spare rather than dying mid-batch on a quota
        # error. Leaving headroom matters: a full quota breaks every other job
        # too, including ones that only want to write a log line.
        if a.stop_below_gb > 0:
            fr = free_gb(a.out_dir)
            if fr < a.stop_below_gb:
                print("[ct] stopping: only %.0f GB free at %s (limit %.0f). "
                      "%d/%d done - rerun to continue."
                      % (fr, a.out_dir, a.stop_below_gb, done, len(todo)), flush=True)
                break
        t0 = time.time()
        got = pending.result()                       # this batch, already downloading
        t_dl = time.time() - t0
        if bi + 1 < len(batches):                    # start the next one immediately
            pending = pool.submit(fetch_batch, batches[bi + 1], dirs[(bi + 1) % 2])

        t0 = time.time()
        work = [(pth, meta[v], os.path.join(a.out_dir, v.replace(".nii.gz", ".npz")))
                for v, pth in got if pth and v in meta]
        fail += len(got) - len(work)
        nb = 0
        with ProcessPoolExecutor(max_workers=a.cworkers) as ex:
            for r in ex.map(_convert_one, work):
                if r:
                    done += 1
                    nb += 1
                else:
                    fail += 1
        t_cv = time.time() - t0
        shutil.rmtree(dirs[bi % 2], ignore_errors=True)

        el = time.time() - t_start
        rate = done / max(el, 1)
        eta = (len(todo) - done) / max(rate, 1e-6) / 3600
        print("[ct] batch %d/%d: +%d  wait_dl=%.0fs conv=%.0fs | total %d/%d fail=%d "
              "| %.2f vol/s eta=%.1fh"
              % (bi + 1, len(batches), nb, t_dl, t_cv, done, len(todo), fail,
                 rate, eta), flush=True)
    pool.shutdown(wait=False)

    print("[ct] DONE ok=%d fail=%d in %.1fh" % (done, fail, (time.time() - t_start) / 3600))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Watchdog that notices a job which has stopped producing, not just one that crashed.

The previous version counted outputs and reported all clear while eight Qwen
shards sat dead for 34 minutes: their files still existed, so a count-based check
saw nothing wrong. SLURM said RUNNING because the process had not exited. The
only visible signal was that the numbers had stopped moving.

So this keeps state between runs and compares. A job that is RUNNING but whose
output has not grown since the last check is the thing worth waking someone for.
"""
from __future__ import annotations
import glob, json, os, subprocess, sys, time
from collections import Counter

# Stall detection works by comparing against the PREVIOUS tick, so two
# watchdogs sharing one state file destroy it: each resets "t", the
# measured gap collapses to the offset between them, and dt never reaches
# STALL_MIN - the check silently never fires. That is the exact failure
# this watchdog exists to catch. One state file per instance.
STATE = os.environ.get("WD_STATE",
                      "/temp_work/ch278233/tmp/watchdog_state.json")
STALL_MIN = 20          # a running job with no new output for this long is stalled
PROBLEMS: list[str] = []


# (job name, log glob). The globs must not overlap: "train_peds_*.out" also
# matches "train_peds_v1s1_*.out", so the vanished-job check read the control
# run's log, found no "Done", and declared the long-finished pedtrain crashed.
RUNS = (
    ("pedtrn2", "train_peds2_*.out"),
    ("pedtrn3", "train_peds3_*.out"),
    ("pedtrn4", "train_peds4_*.out"),
    ("pedtrn5", "train_peds5_*.out"),
    ("pedv1s1", "train_peds_v1s1_*.out"),
    ("comb16k", "train_comb16k_*.out"),
    ("comb47k", "train_comb47k_*.out"),
    ("c47harm", "train_c47harm_*.out"),
)


def bad(m): PROBLEMS.append(m); print("  !! %s" % m)
def ok(m):  print("  ok %s" % m)


def squeue(user="ch278233"):
    try:
        out = subprocess.run(["squeue", "-u", user, "-h", "-o", "%i|%j|%t"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    return [l.split("|") for l in out.strip().splitlines() if l.strip()]


def count(pattern, needle=None):
    fs = glob.glob(pattern)
    if needle is None:
        return len(fs)
    n = 0
    for f in fs:
        try:
            with open(f, errors="replace") as fh:
                n += sum(1 for ln in fh if needle in ln)
        except OSError:
            pass
    return n


def main() -> int:
    now = time.time()
    prev = {}
    if os.path.exists(STATE):
        try:
            prev = json.load(open(STATE))
        except Exception:
            prev = {}
    cur = {"t": now}

    print("=" * 66)
    print("WATCHDOG  %s" % time.strftime("%H:%M:%S"))
    print("=" * 66)

    jobs = squeue()
    byname = Counter(j[1] for j in jobs)
    print("\n[jobs] " + (", ".join("%s x%d" % (k, v) for k, v in byname.items()) or "none"))
    # the colleague's fetch runs under a different account; report it if visible
    theirs = squeue("ch278452")
    if theirs:
        print("[jobs] ch278452: " + ", ".join("%s(%s)" % (j[1], j[2]) for j in theirs))

    # ---- a job that is simply GONE ----
    # The stall check only fires for jobs SLURM still lists as running, so a
    # crashed job is invisible to it: comb16k died 13 minutes in on a corrupt
    # npz and every counter still looked healthy because nothing was watching
    # for absence. Worse, the sbatch ended on an echo, so sacct said COMPLETED.
    # Trust the trainer's own last line instead of the exit code.
    print("\n[bekci]")
    for jobname, pat in RUNS:
        if byname.get(jobname):
            continue
        tl = sorted(glob.glob("/temp_work/ch278233/ts_logs/%s" % pat),
                    key=os.path.getmtime)
        if not tl:
            continue
        try:
            with open(tl[-1], errors="replace") as fh:
                sz = os.path.getsize(tl[-1])
                if sz > 100_000:
                    fh.seek(sz - 100_000)
                tail = fh.read()
        except OSError:
            continue
        if "[RAC] Done." in tail:
            ok("%s bitti" % jobname)
        else:
            bad("%s KUYRUKTA YOK ve 'Done' yazmamis - cokmus" % jobname)

    # ---- progress counters, each tied to the job that should be moving it ----
    D = "/temp_work/ch278233/BCH_DATASET/LABELS27"
    checks = [
        ("qwen_regions", f"{D}/regions.shard*.jsonl", '"status": "ok"', "pedextract", 8816),
        ("qwen_labels",  f"{D}/labels.shard*.jsonl",  '"status": "ok"', "pedextract", 8816),
        ("ctrate_npz",   "/temp_work/ch278452/CTRATE_NPZ/train/*.npz", None, "ctfetch", 47149),
        ("peds_masks",   "/temp_work/ch278233/PEDS_MASKS10_192/*.nii.gz", None, "pedmask", 8816),
        # CT-RATE organ segmentation. Adult volumes have never had masks - the
        # published adult protocol is mask-free - so the joint model trained
        # anatomy-routed on its pediatric third and as a plain Q-Former on the
        # adult two thirds. This closes that asymmetry, and lets us ask whether
        # the 0.03 mask-routing penalty measured in pediatrics also exists in
        # adults before committing ~4.5 days to the training split.
        ("ctrate_ts_valid", "/temp_work/ch278233/CTRATE_TS_RAW/valid/*.nii.gz",
         None, "ctvalts", 3002),
        ("ctrate_masks",    "/temp_work/ch278233/CTRATE_MASKS10_192/valid/*.nii.gz",
         None, "ctvalmask", 3002),
        # CT-RATE 27-class extraction: replaces the fabricated zeros the combined
        # dataset gave adults for the nine pediatric classes. Measured on a
        # 40-report smoke, 22.5% of adults have a bone lesion - the combined
        # labels said 0 of 16,400, and that class scored 0.373, below chance.
        # Pediatric re-extraction under the corrected prompt. It writes to
        # LABELS27_V2, never over LABELS27: the running combined training reads
        # the old labels and overwriting them mid-run would swap the training
        # target underneath it.
    ]
    print("\n[progress]")
    for key, pat, needle, jobname in [(c[0], c[1], c[2], c[3]) for c in checks]:
        tgt = dict((c[0], c[4]) for c in checks)[key]
        n = count(pat, needle)
        cur[key] = n
        was = prev.get(key)
        dt = (now - prev.get("t", now)) / 60 if prev else 0
        running = byname.get(jobname, 0) or (jobname == "ctfetch" and theirs)
        delta = "" if was is None else "  (+%d in %.0f min)" % (n - was, dt)
        print("  %-14s %7d / %-7d%s" % (key, n, tgt, delta))
        if running and was is not None and n == was and dt >= STALL_MIN and n < tgt:
            bad("%s: %s RUNNING but produced nothing in %.0f min - likely dead"
                % (key, jobname, dt))

    # ---- training: the tqdm counter is the only thing that says it is alive.
    # A stalled trainer looks exactly like a working one from the queue - which
    # is how eight Qwen shards sat dead for 34 minutes earlier today.
    # Both trainings, each keyed by its own job name. Watching only the
    # pediatric log left the combined run - the longer and more expensive of
    # the two - completely uninstrumented.
    for jobname, pat in RUNS:
        tag = jobname
        tl = sorted(glob.glob("/temp_work/ch278233/ts_logs/%s" % pat),
                    key=os.path.getmtime)
        if not tl:
            continue
        f = tl[-1]
        try:
            with open(f, errors="replace") as fh:
                sz = os.path.getsize(f)
                if sz > 200_000:
                    fh.seek(sz - 200_000)
                txt = fh.read().replace("\r", "\n")
        except OSError:
            txt = ""
        import re as _re
        steps = _re.findall(r"RAC-updates:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", txt)
        aucs = _re.findall(r"global_val_auc=([\d.]+).*?best=([\d.]+)", txt)
        empt = _re.findall(r"empty-region rate per anatomy query: (.+)", txt)
        print("\n[%s]" % jobname)
        skey = "step:" + tag
        if steps:
            cu, tot = int(steps[-1][0]), int(steps[-1][1])
            cur[skey] = cu
            was = prev.get(skey)
            dt = (now - prev.get("t", now)) / 60 if prev else 0
            d = "" if was is None else "  (+%d in %.0f min)" % (cu - was, dt)
            print("  step %d / %d%s" % (cu, tot, d))
            if byname.get(jobname, 0) and was is not None and cu == was and dt >= STALL_MIN:
                bad("%s: RUNNING ama adim %d'de %.0f dk takili" % (jobname, cu, dt))
        else:
            print("  (henuz adim yok - model yukleniyor)")
        if aucs:
            print("  son val AUC %s  (best %s)" % aucs[-1])
        if empt:
            print("  bos bolge orani: %s" % empt[-1][:90])
        for pat in ("CUDA out of memory", "Traceback", "RuntimeError"):
            if pat in txt:
                bad("%s logunda '%s'" % (jobname, pat))


    # ---- duplicates: a restarted array can reshuffle shard assignments, and
    # each shard only knows what IT already wrote, so a chunk can be redone by
    # a different shard. Harmless downstream (build_outputs keys by VolumeName)
    # but it inflates the counters, so report it rather than let the numbers lie.
    try:
        import json as _j
        seen, dup, tot = set(), 0, 0
        for f in glob.glob(f"{D}/regions.shard*.jsonl"):
            for ln in open(f, errors="replace"):
                try:
                    v = _j.loads(ln)["VolumeName"]
                except Exception:
                    continue
                tot += 1
                dup += v in seen
                seen.add(v)
        if tot:
            print("\n[dedup] regions: %d satir, %d benzersiz, %d mukerrer" % (tot, len(seen), dup))
            cur["qwen_regions_unique"] = len(seen)
    except Exception:
        pass

    # ---- can we still write? the failure that killed the Qwen shards ----
    print("\n[disk]")
    probe = "/temp_work/ch278233/tmp/_wd_probe"
    try:
        with open(probe, "wb") as fh:
            fh.write(b"0" * 1_000_000)
        os.remove(probe)
        ok("1 MB test yazimi gecti")
    except OSError as e:
        bad("YAZILAMIYOR: %s - kota dolmus olabilir" % e.strerror)
    st = os.statvfs("/temp_work/ch278233")
    ok("dosya sistemi bos: %.0f TB" % (st.f_bavail * st.f_frsize / 1e12))

    # ---- fresh errors in the last 30 minutes ----
    print("\n[logs]")
    pats = ("Disk quota exceeded", "CUDA out of memory", "Traceback",
            "OUT_OF_MEMORY", "Killed", "SchemaMismatch")
    hits = Counter()
    for f in glob.glob("/temp_work/ch278233/ts_logs/*.out") + \
             glob.glob("/temp_work/ch278233/BCH_DATASET/LABELS*/_slurm/*.out") + \
             glob.glob("/temp_work/ch278452/ctrate_fetch/logs/*.out"):
        try:
            if now - os.path.getmtime(f) > 1800:
                continue
            # Only the tail. A log that is still being appended to keeps its
            # whole history, so scanning the file entire re-reports an error
            # from an hour ago as if it just happened - which it did, once,
            # and was already dealt with.
            sz = os.path.getsize(f)
            # A job that crashed keeps a <30-min-old log for the next 30 min, so
            # the mtime window alone re-reports the same dead traceback every
            # tick. If the file has not GROWN since the previous tick, nothing
            # new happened in it - report once, then stay quiet. A still-running
            # job keeps growing, so real new errors are never suppressed.
            k = "log:" + f
            grew = prev.get(k) != sz
            cur[k] = sz
            if not grew:
                continue
            with open(f, errors="replace") as fh:
                if sz > 200_000:
                    fh.seek(sz - 200_000)
                txt = fh.read()
        except OSError:
            continue
        for p in pats:
            if p in txt:
                hits[p] += txt.count(p)
    if hits:
        for k, v in hits.most_common():
            bad("son 30 dk log'larda '%s' x%d" % (k, v))
    else:
        ok("son 30 dk aktif log'larda hata yok")

    json.dump(cur, open(STATE, "w"))
    print("\n" + "=" * 66)
    print("SORUN: %d" % len(PROBLEMS))
    for p in PROBLEMS:
        print("  - %s" % p)
    return min(len(PROBLEMS), 100)


if __name__ == "__main__":
    sys.exit(main())

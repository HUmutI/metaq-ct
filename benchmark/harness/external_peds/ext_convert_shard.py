#!/usr/bin/env python3
"""Shard-aware version of the download agent's run_convert loop, for Slurm.

Same per-patient convert() (dcm2niix + RTSTRUCT rasterisation in the CT's exact
geometry); resumable through logs/convert/<pid>.json. Each task polls for
patients whose CT+RTSTRUCT downloads carry a .ok marker and that hash to its
shard, and exits when the downloader writes '# DONE' and nothing is left."""
import glob, os, sys, time, zlib
X = "/temp_work/ch278233/EXTERNAL/PEDIATRIC_CT_SEG"
sys.path.insert(0, X)
from convert import convert, BYPID

SHARD = int(os.environ.get("SHARD_ID", "0")); N = int(os.environ.get("SHARD_N", "1"))
mine = lambda pid: zlib.crc32(pid.encode()) % N == SHARD

def ready():
    out = []
    for pid, d in BYPID.items():
        if not mine(pid) or os.path.exists(f"{X}/logs/convert/{pid}.json"): continue
        if all(os.path.exists(f"{X}/dicom/{pid}/{m}_{d[m]['SeriesInstanceUID']}/.ok") for m in ("CT", "RTSTRUCT")):
            out.append(pid)
    return sorted(out)

def dl_done():
    p = f"{X}/logs/download.log"
    return os.path.exists(p) and "# DONE" in open(p).read()

print(f"[conv] shard {SHARD}/{N}, {sum(1 for p in BYPID if mine(p))} patients assigned", flush=True)
n = 0
while True:
    todo = ready()
    if not todo:
        if dl_done(): break
        time.sleep(60); continue
    for pid in todo:
        t0 = time.time()
        r = convert(pid)
        n += 1
        print(f"[conv] {n} {pid} {r['status']} {time.time()-t0:.0f}s anom={r['anomalies'][:3]}", flush=True)
print(f"[conv] shard {SHARD} done, converted {n}; total json={len(glob.glob(f'{X}/logs/convert/*.json'))}", flush=True)

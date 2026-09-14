#!/bin/bash
#SBATCH -J merlin_ctrsmk
#SBATCH -p bch-compute,bch-compute-pe
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_ctrsmoke_%j.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_ctrsmoke_%j.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
$PY /home/ch278233/BENCHMARK/harness/peds_bench/merlin_extract_contrastive.py --limit 2 --batch-size 2 --workers 2 \
  --outdir /temp_work/ch278233/PEDS_BENCH/retrieval/_smoke
$PY - <<'PY'
import numpy as np
d=np.load("/temp_work/ch278233/PEDS_BENCH/retrieval/_smoke/shard00.npz")
print({k:d[k].shape for k in d.files}); print("stems",d["stems"],"ntok",d["ntok"],"ctx",d["ctx"])
print("img norms",np.linalg.norm(d["img_lat"],axis=1),"txt norms",np.linalg.norm(d["txt_lat"],axis=1))
print("img.txt cos diag",(d["img_lat"]*d["txt_lat"]).sum(1))
PY

#!/bin/bash
#SBATCH -J merlin_smkc
#SBATCH -p bch-compute,bch-compute-pe
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_smokecpu_%j.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_smokecpu_%j.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
$PY /home/ch278233/BENCHMARK/harness/peds_bench/merlin_extract_feats.py \
  --split valid --limit 2 --batch-size 1 --workers 2 \
  --outdir /temp_work/ch278233/PEDS_BENCH/merlin/_smokecpu
echo "=== inspect ==="
$PY - <<'PY'
import numpy as np,glob
for f in glob.glob("/temp_work/ch278233/PEDS_BENCH/merlin/_smokecpu/*.npz"):
    d=np.load(f,allow_pickle=True)
    print(f, {k:d[k].shape for k in d.files})
    print("feat dim:",d["feats"].shape,"norm:",np.linalg.norm(d["feats"],axis=1),
          "nonzero:",(d["feats"]!=0).sum(1), "vols:",d["vols"])
PY

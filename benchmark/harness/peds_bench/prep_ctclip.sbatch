#!/bin/bash
#SBATCH --job-name=pbPrepCTC
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --array=0-49
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/prep_ctclip_%A_%a.out
set -uo pipefail
/home/ch278233/BENCHMARK/envs/CT-CLIP/bin/python -u \
  /home/ch278233/BENCHMARK/harness/peds_bench/prep_ctclip.py \
  --vollist /temp_work/ch278233/PEDS_BENCH/peds_all_vollist.txt \
  --out-dir /temp_work/ch278233/PEDS_BENCH/ctclip_npz \
  --shard $SLURM_ARRAY_TASK_ID --nshards 50
echo "=== rc=$? ==="

#!/bin/bash
#SBATCH -J merlin_ctr_rsc
#SBATCH -p bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH --exclude=gpu-b11-3
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_ctr_rsc_%A_%a.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_ctr_rsc_%A_%a.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
# nshards=36, tasks {0,12,24} == indices 0 mod 12 == main-array shard 0 of 12 (verified)
srun $PY /home/ch278233/BENCHMARK/harness/peds_bench/merlin_extract_contrastive.py \
  --shard "$SLURM_ARRAY_TASK_ID" --nshards 36 --batch-size 2 --workers 6 \
  --outdir /temp_work/ch278233/PEDS_BENCH/retrieval/_merlin_shards_rescue

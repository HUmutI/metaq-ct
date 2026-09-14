#!/bin/bash
#SBATCH -J merlin_rsc
#SBATCH -p bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_rsc_%A_%a.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_rsc_%A_%a.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
# nshards=48 with tasks {1,9,17,25,33,41} exactly reproduces vols[1::8] (shard 1 of 8)
srun $PY /home/ch278233/BENCHMARK/harness/peds_bench/merlin_extract_feats.py \
  --split valid --shard "$SLURM_ARRAY_TASK_ID" --nshards 48 \
  --batch-size 2 --workers 6 \
  --outdir /temp_work/ch278233/PEDS_BENCH/merlin/feats_rescue

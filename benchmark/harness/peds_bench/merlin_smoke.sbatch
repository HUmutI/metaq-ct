#!/bin/bash
#SBATCH -J merlin_smoke
#SBATCH -p bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 0:40:00
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_smoke_%j.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_smoke_%j.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY /home/ch278233/BENCHMARK/harness/peds_bench/merlin_extract_feats.py \
  --split valid --limit 4 --batch-size 2 --workers 4 \
  --outdir /temp_work/ch278233/PEDS_BENCH/merlin/_smoke

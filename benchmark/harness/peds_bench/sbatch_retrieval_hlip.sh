#!/bin/bash
#SBATCH --job-name=hlip_retr
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/hlip_retrieval_%j.out
#SBATCH --error=/temp_work/ch278233/PEDS_BENCH/logs/hlip_retrieval_%j.err

set -euo pipefail
export HF_HOME=/temp_work/ch278233/PEDS_BENCH/hlip/_hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4

PY=/temp_work/ch278233/PEDS_BENCH/hlip/_venv/bin/python
SCRIPT=/home/ch278233/BENCHMARK/harness/peds_bench/hlip_retrieval_latents.py

echo "host=$(hostname)"; nvidia-smi -L || true
srun $PY $SCRIPT \
  --model clip_vit_base_singlescan_h2_token2744 \
  --use-cxr-bert \
  --resume /temp_work/ch278233/PEDS_BENCH/weights/hlip/chestct_clip_vit_base_singlescan_h2_token2744.pt \
  --data-root /temp_work/ch278233/PEDS_BENCH/hlip/pt/ --split valid \
  --labels /temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv \
  --input-info -1150 350 crop \
  --out /temp_work/ch278233/PEDS_BENCH/retrieval/hlip_valid.npz \
  --device cuda:0 --workers 4 "$@"

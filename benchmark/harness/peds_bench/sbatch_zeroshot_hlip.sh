#!/bin/bash
#SBATCH --job-name=hlip_zs_peds
# bch-gpu only: bch-gpu-pe is preemptable and cost us a run
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/hlip_zeroshot_%j.out
#SBATCH --error=/temp_work/ch278233/PEDS_BENCH/logs/hlip_zeroshot_%j.err

set -euo pipefail
export HF_HOME=/temp_work/ch278233/PEDS_BENCH/hlip/_hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4

PY=/temp_work/ch278233/PEDS_BENCH/hlip/_venv/bin/python
SCRIPT=/home/ch278233/BENCHMARK/harness/peds_bench/zeroshot_peds23_hlip.py

echo "host=$(hostname)"; nvidia-smi || true

srun $PY $SCRIPT \
  --model clip_vit_base_singlescan_h2_token2744 \
  --use-cxr-bert \
  --resume /temp_work/ch278233/PEDS_BENCH/weights/hlip/chestct_clip_vit_base_singlescan_h2_token2744.pt \
  --data-root /temp_work/ch278233/PEDS_BENCH/hlip/pt/ \
  --split valid \
  --labels /temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv \
  --input-info -1150 350 crop \
  --zeroshot-template volume \
  --out-npz /temp_work/ch278233/PEDS_BENCH/preds/hlip_peds_zeroshot_preds.npz \
  --results-dir /temp_work/ch278233/PEDS_BENCH/hlip/results/ \
  --device cuda:0 \
  --workers 6 "$@"

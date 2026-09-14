#!/bin/bash
#SBATCH --job-name=pbCTCsmoke
#SBATCH --partition=bch-gpu-pe,bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/ctclip_smoke_%j.out
set -uo pipefail
cd /home/ch278233/BENCHMARK/CT-CLIP/scripts
PYTHONPATH=/home/ch278233/BENCHMARK/CT-CLIP/scripts:/home/ch278233/BENCHMARK/harness/peds_bench \
/home/ch278233/BENCHMARK/envs/CT-CLIP/bin/python -u \
  /home/ch278233/BENCHMARK/harness/peds_bench/eval_ctclip_peds.py \
  --ckpt /temp_work/ch278233/BENCHMARK_WEIGHTS/CT-CLIP/CT-CLIP_v2.pt \
  --limit 8 \
  --out /temp_work/ch278233/PEDS_BENCH/preds/_smoke_ctclip.npz
echo "=== rc=$? ==="

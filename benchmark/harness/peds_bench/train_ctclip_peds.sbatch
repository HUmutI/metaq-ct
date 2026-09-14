#!/bin/bash
#SBATCH --job-name=pbCTCtrain
#SBATCH --partition=bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/ctclip_train_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
N=$(ls /temp_work/ch278233/PEDS_BENCH/ctclip_npz 2>/dev/null | wc -l)
echo "onbellekte $N hacim var (beklenen 7453)"
if [ "$N" -lt 7000 ]; then echo "FATAL: onisleme tamamlanmamis"; exit 78; fi
cd /home/ch278233/BENCHMARK/CT-CLIP/scripts
PYTHONPATH=/home/ch278233/BENCHMARK/CT-CLIP/scripts:/home/ch278233/BENCHMARK/harness/peds_bench \
/home/ch278233/BENCHMARK/envs/CT-CLIP/bin/python -u \
  /home/ch278233/BENCHMARK/harness/peds_bench/train_ctclip_peds.py \
  --steps 8000 --batch 8 --workers 6 \
  --out /temp_work/ch278233/PEDS_BENCH/runs/ctclip_peds
echo "=== rc=$? ==="

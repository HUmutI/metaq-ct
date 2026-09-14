#!/bin/bash
#SBATCH --job-name=pbCTCtrSmk
#SBATCH --partition=bch-gpu-xlarge
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/ctclip_trainsmoke_%j.out
set -uo pipefail
cd /home/ch278233/BENCHMARK/CT-CLIP/scripts
PYTHONPATH=/home/ch278233/BENCHMARK/CT-CLIP/scripts:/home/ch278233/BENCHMARK/harness/peds_bench \
/home/ch278233/BENCHMARK/envs/CT-CLIP/bin/python -u \
  /home/ch278233/BENCHMARK/harness/peds_bench/train_ctclip_peds.py \
  --train-vollist /temp_work/ch278233/PEDS_BENCH/smoke_train_vollist.txt \
  --valid-vollist /temp_work/ch278233/PEDS_BENCH/smoke_valid_vollist.txt \
  --steps 4 --batch 2 --workers 2 --save-every 1000 --val-every 1000 \
  --out /temp_work/ch278233/PEDS_BENCH/runs/_smoke_ctclip_train
echo "=== rc=$? ==="

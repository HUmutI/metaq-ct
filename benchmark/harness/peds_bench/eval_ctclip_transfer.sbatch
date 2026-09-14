#!/bin/bash
#SBATCH --job-name=pbCTCxfer
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/ctclip_transfer_%j.out
# CT-CLIP with the OFFICIAL CT-RATE weights, zero-shot onto pediatrics with no
# pediatric training at all. This is a legitimate and clearly-labelled row:
# "zero-shot transfer", distinct from the pediatric fine-tune row.
set -uo pipefail
N=$(comm -12 <(sed 's/\.nii\.gz$//' /temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt | sort) \
             <(ls /temp_work/ch278233/PEDS_BENCH/ctclip_npz | sed 's/\.npz$//' | sort) | wc -l)
echo "onbellekteki valid hacim: $N / 1507"
if [ "$N" -lt 1500 ]; then echo "FATAL: valid onbellegi eksik"; exit 78; fi
cd /home/ch278233/BENCHMARK/CT-CLIP/scripts
PYTHONPATH=/home/ch278233/BENCHMARK/CT-CLIP/scripts:/home/ch278233/BENCHMARK/harness/peds_bench \
/home/ch278233/BENCHMARK/envs/CT-CLIP/bin/python -u \
  /home/ch278233/BENCHMARK/harness/peds_bench/eval_ctclip_peds.py \
  --ckpt /temp_work/ch278233/BENCHMARK_WEIGHTS/CT-CLIP/CT-CLIP_v2.pt \
  --verify 3 \
  --out /temp_work/ch278233/PEDS_BENCH/preds/ctclip_transfer_peds_zeroshot_preds.npz
echo "=== rc=$? ==="

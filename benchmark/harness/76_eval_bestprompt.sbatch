#!/bin/bash
#SBATCH --job-name=bmEvBP
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/temp_work/ch278233/ts_logs/bmEvBP_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=$BM_RUNS/eval; mkdir -p $O
CTL=/temp_work/ch278233/CTRATE/multi_abnormality_labels/valid_predicted_labels.csv
CTRP=/temp_work/ch278233/CTRATE/ct_reports/valid_reports.csv

# The prompt sweep on ctrate_dev put GreenRFM's zero-shot at 0.6751 with the
# `null` pair against 0.5770 with its own default p1 (and 0.4080 with p4, below
# chance).  Choosing the prompt on dev is exactly what dev is for, so the
# reportable zero-shot number uses the dev-selected prompt.  MPS-CT's dev-best
# is also `null`, which is already its shipped default, so its number is unchanged.
echo "=== GreenRFM zero-shot, dev-selected prompt=null, official test ==="
$BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
  --model greenrfm --mode zeroshot --prompt null \
  --ckpt $BM_RUNS/greenrfm_ctrate/stage2_align/checkpoint_epoch_5.pt \
  --tree $BM_DATA/ctrate_test --reports $CTRP --labels $CTL \
  --batch 8 --workers 8 --out $O/TEST_grf_zeroshot_null.json

echo "=== GreenRFM zero-shot, prompt=null, pediatric ==="
PG=$(ls -1 $BM_RUNS/greenrfm_peds23/stage2_align/checkpoint_epoch_*.pt | sort -V | tail -1)
$BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
  --model greenrfm --mode zeroshot --prompt null --ckpt "$PG" \
  --tree $BM_DATA/peds_valid --reports /temp_work/ch278233/COMBINED_reports.csv \
  --labels $BM_DATA/peds23_labels_valid.csv \
  --batch 8 --workers 8 --out $O/TEST_peds_grf_zeroshot_null.json
echo "=== best-prompt eval done ==="

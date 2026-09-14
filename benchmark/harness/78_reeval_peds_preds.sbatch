#!/bin/bash
#SBATCH --job-name=pbReEval
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/reeval_%j.out
# Re-run the two already-trained pediatric baselines so 70_eval.py writes the
# *_preds.npz files the table needs. AUC alone is not enough: the table has
# Prec/Acc/F1 columns and those must come from ARC-CT's own metric code.
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=/temp_work/ch278233/PEDS_BENCH/preds; mkdir -p $O

PM=$(find $BM_RUNS/mpsct_peds23 -name 'CTClip.*.pt' 2>/dev/null \
     | sed 's/.*CTClip\.\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
GF=$BM_RUNS/greenrfm_peds23/stage2_align/checkpoint_epoch_10.pt
echo "MPS-CT ckpt : ${PM:-NONE}"
echo "GreenRFM ckpt: $GF"
[ -z "${PM:-}" ] && { echo "FATAL: no MPS-CT pediatric checkpoint"; exit 78; }
[ -f "$GF" ]     || { echo "FATAL: no GreenRFM pediatric checkpoint"; exit 78; }

rc=0
echo "### 1/3 MPS-CT zero-shot (its only readout)"
PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT \
$BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/70_eval.py \
  --model mpsct --mode zeroshot --ckpt "$PM" \
  --tree $BM_DATA/peds_valid --reports /temp_work/ch278233/COMBINED_reports.csv \
  --labels $BM_DATA/peds23_labels_valid.csv \
  --out $O/mpsct_peds_zeroshot.json --batch 8 --workers 8 || rc=$?

echo "### 2/3 GreenRFM classifier head (its native/published readout)"
PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/GreenRFM \
$BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
  --model greenrfm --mode classifier --ckpt "$GF" \
  --tree $BM_DATA/peds_valid --reports /temp_work/ch278233/COMBINED_reports.csv \
  --labels $BM_DATA/peds23_labels_valid.csv \
  --out $O/greenrfm_peds_classifier.json --batch 8 --workers 8 || rc=$?

echo "### 3/3 GreenRFM zero-shot (documented as defective; kept for the record)"
PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/GreenRFM \
$BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
  --model greenrfm --mode zeroshot --ckpt "$GF" \
  --tree $BM_DATA/peds_valid --reports /temp_work/ch278233/COMBINED_reports.csv \
  --labels $BM_DATA/peds23_labels_valid.csv \
  --out $O/greenrfm_peds_zeroshot.json --batch 8 --workers 8 || rc=$?

echo "=== rc=$rc ==="
exit $rc

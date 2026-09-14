#!/bin/bash
#SBATCH --job-name=bmEvMpsPeds
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/temp_work/ch278233/ts_logs/bmEvMpsPeds_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=$BM_RUNS/eval; mkdir -p $O
PM=$(find $BM_RUNS/mpsct_peds23 -name 'CTClip.*.pt' 2>/dev/null \
     | sed 's/.*CTClip\.\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
echo "final pediatric checkpoint: ${PM:-NONE}"
[ -z "${PM:-}" ] && { echo "FATAL: no MPS-CT pediatric checkpoint"; exit 78; }
PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT \
$BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/70_eval.py \
  --model mpsct --mode zeroshot --ckpt "$PM" \
  --tree $BM_DATA/peds_valid --reports /temp_work/ch278233/COMBINED_reports.csv \
  --labels $BM_DATA/peds23_labels_valid.csv \
  --out $O/TEST_peds_mpsct_zeroshot.json --batch 8 --workers 8
rc=$?
echo "=== rc=$rc ==="; exit $rc

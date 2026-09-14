#!/bin/bash
#SBATCH --job-name=bmSmokeMPS
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/bmSmokeMPS_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
export PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT

R=$BM_RUNS/smoke_mpsct_$SLURM_JOB_ID
export BM_DATA_TRAIN=$BM_DATA/smoke/ctrate_train
export BM_DATA_VALID=$BM_DATA/smoke/ctrate_dev
export BM_REPORTS_TRAIN=$CTR/reports.csv
export BM_REPORTS_VALID=$CTR/reports.csv
export BM_LABELS=$CTR/labels18.csv
export BM_RESULTS=$R
export BM_STEPS=6 BM_BATCH=2 BM_WORKERS=2 BM_SAVE_EVERY=4

echo "=== MPS-CT SMOKE job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
cd $BM_REPOS/MPS-CT
$BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/20_train_mpsct.py
rc=$?
echo "=== rc=$rc $(date) ==="
echo "--- artefacts written ---"; ls -l $R 2>/dev/null | head -20
if [ $rc -eq 0 ] && ls $R/*.pt >/dev/null 2>&1; then echo "SMOKE MPS OK"; else echo "SMOKE MPS FAIL"; fi
exit $rc

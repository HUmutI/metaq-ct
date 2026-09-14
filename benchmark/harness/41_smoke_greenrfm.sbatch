#!/bin/bash
#SBATCH --job-name=bmSmokeGRF
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/bmSmokeGRF_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh

R=$BM_RUNS/smoke_greenrfm_$SLURM_JOB_ID
export BM_DATA_TRAIN=$BM_DATA/smoke/ctrate_train
export BM_DATA_VALID=$BM_DATA/smoke/ctrate_dev
export BM_REPORTS_TRAIN=$CTR/reports.csv
export BM_REPORTS_VALID=$CTR/reports.csv
export BM_LABELS_TRAIN=$CTR/labels18.csv
export BM_LABELS_VALID=$CTR/labels18.csv
export BM_EPOCHS=1 BM_BATCH=2 BM_WORKERS=2 BM_LIMIT=8

echo "=== GreenRFM SMOKE job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
rc=0
for stage in image text align; do
  echo "----- stage $stage -----"
  export BM_STAGE=$stage BM_SAVE_DIR=$R/$stage
  $BM_REPOS/envs/GreenRFM/bin/python $BM_HARNESS/30_train_greenrfm.py || rc=1
done
echo "=== rc=$rc $(date) ==="
echo "--- artefacts ---"; find $R -type f 2>/dev/null | head -20
if [ $rc -eq 0 ]; then echo "SMOKE GRF OK"; else echo "SMOKE GRF FAIL"; fi
exit $rc

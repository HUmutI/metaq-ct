#!/bin/bash
#SBATCH --job-name=bmGrfS1
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --output=/temp_work/ch278233/ts_logs/bmGrfS1_%A_%a.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh

# Stage 1 of the GreenRFM recipe: the vision and text encoders are pre-trained
# independently against the label set before any alignment.  The two are
# independent, so they run as one array rather than in sequence.
STAGES=(image text)
export BM_STAGE=${STAGES[$SLURM_ARRAY_TASK_ID]}

export BM_DATA_TRAIN=$BM_DATA/ctrate_train
export BM_DATA_VALID=$BM_DATA/ctrate_dev
export BM_REPORTS_TRAIN=$CTR/reports.csv
export BM_REPORTS_VALID=$CTR/reports.csv
export BM_LABELS_TRAIN=$CTR/labels18.csv
export BM_LABELS_VALID=$CTR/labels18.csv
export BM_SAVE_DIR=$BM_RUNS/greenrfm_ctrate/stage1_$BM_STAGE
export BM_EPOCHS=5 BM_BATCH=10 BM_WORKERS=8 BM_LR=1e-4

echo "=== GreenRFM stage1/$BM_STAGE job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
$BM_REPOS/envs/GreenRFM/bin/python $BM_HARNESS/30_train_greenrfm.py
rc=$?
echo "=== stage1/$BM_STAGE rc=$rc $(date) ==="
ls -l $BM_SAVE_DIR 2>/dev/null | tail -5
exit $rc

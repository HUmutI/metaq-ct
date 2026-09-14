#!/bin/bash
#SBATCH --job-name=bmGrfAln
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/bmGrfAln_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh

export BM_STAGE=align
export BM_DATA_TRAIN=$BM_DATA/ctrate_train
export BM_DATA_VALID=$BM_DATA/ctrate_dev
export BM_REPORTS_TRAIN=$CTR/reports.csv
export BM_REPORTS_VALID=$CTR/reports.csv
export BM_LABELS_TRAIN=$CTR/labels18.csv
export BM_LABELS_VALID=$CTR/labels18.csv
export BM_SAVE_DIR=$BM_RUNS/greenrfm_ctrate/stage2_align
export BM_EPOCHS=5 BM_BATCH=10 BM_WORKERS=8 BM_LR=1e-5

# Stage 2 must start from stage 1, otherwise this is just contrastive training
# with extra steps and the "semantic-enforcing supervision" claim is untested.
# Pick the last epoch each stage-1 arm wrote; the driver aborts if a checkpoint
# matches zero tensors, so a silent no-op load cannot happen.
pick () { ls -1 $1/checkpoint_epoch_*.pt 2>/dev/null | sort -V | tail -1; }
export BM_PRETRAINED_VISION=$(pick $BM_RUNS/greenrfm_ctrate/stage1_image)
export BM_PRETRAINED_TEXT=$(pick $BM_RUNS/greenrfm_ctrate/stage1_text)
echo "vision ckpt: ${BM_PRETRAINED_VISION:-NONE}"
echo "text   ckpt: ${BM_PRETRAINED_TEXT:-NONE}"
if [ -z "${BM_PRETRAINED_VISION:-}" ] || [ -z "${BM_PRETRAINED_TEXT:-}" ]; then
  echo "FATAL: stage-1 checkpoints missing; refusing to run alignment from scratch"; exit 78
fi

echo "=== GreenRFM stage2/align job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
$BM_REPOS/envs/GreenRFM/bin/python $BM_HARNESS/30_train_greenrfm.py
rc=$?
echo "=== stage2/align rc=$rc $(date) ==="
ls -l $BM_SAVE_DIR 2>/dev/null | tail -5
exit $rc

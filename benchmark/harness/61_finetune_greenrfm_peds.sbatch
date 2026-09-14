#!/bin/bash
#SBATCH --job-name=bmGrfPeds
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/bmGrfPeds_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh

# Pediatric fine-tune: re-align on pediatrics starting from the CT-RATE aligned
# encoders.  Both encoders transfer; the classifier is rebuilt at 23 outputs
# because num_classes is detected from the label CSV, so this is a genuine
# 23-way head and not a 18-way model scored on a subset.
export BM_STAGE=align
pick () { ls -1 $1/checkpoint_epoch_*.pt 2>/dev/null | sort -V | tail -1; }
CK=$(pick $BM_RUNS/greenrfm_ctrate/stage2_align)
echo "warm start (CT-RATE aligned): ${CK:-NONE}"
if [ -z "${CK:-}" ]; then echo "FATAL: no CT-RATE aligned checkpoint"; exit 78; fi
export BM_PRETRAINED_VISION=$CK
export BM_PRETRAINED_TEXT=$CK

export BM_DATA_TRAIN=$BM_DATA/peds_train
export BM_DATA_VALID=$BM_DATA/peds_valid
export BM_REPORTS_TRAIN=/temp_work/ch278233/COMBINED_reports.csv
export BM_REPORTS_VALID=/temp_work/ch278233/COMBINED_reports.csv
export BM_LABELS_TRAIN=$BM_DATA/peds23_labels_train.csv
export BM_LABELS_VALID=$BM_DATA/peds23_labels_valid.csv
export BM_SAVE_DIR=$BM_RUNS/greenrfm_peds23/stage2_align
export BM_EPOCHS=${BM_EPOCHS:-10} BM_BATCH=10 BM_WORKERS=8 BM_LR=1e-5

echo "=== GreenRFM peds23 fine-tune job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
$BM_REPOS/envs/GreenRFM/bin/python $BM_HARNESS/30_train_greenrfm.py
rc=$?
echo "=== GreenRFM peds rc=$rc $(date) ==="
exit $rc

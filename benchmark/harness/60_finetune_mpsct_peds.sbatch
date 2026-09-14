#!/bin/bash
#SBATCH --job-name=bmMpsPeds
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/bmMpsPeds_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
export PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT

# Pediatric fine-tune of the CT-RATE pretrain -- the arm that is comparable to
# ARC-CT, which is itself warm-started rather than trained on pediatrics alone.
# MPS-CT trains a plain contrastive objective; the labels CSV is read only by
# eval_auc (CTCLIPTrainer.py:45), so moving 18 -> 23 classes changes the
# reported metric, not the architecture.  No head is resized.
pick () { ls -1 $BM_RUNS/mpsct_ctrate/attempt*/CTClip.*.pt 2>/dev/null \
          | sed 's/.*CTClip\.\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2; }
export BM_PRETRAINED=$(pick)
echo "warm start: ${BM_PRETRAINED:-NONE}"
if [ -z "${BM_PRETRAINED:-}" ]; then echo "FATAL: no CT-RATE checkpoint to fine-tune from"; exit 78; fi

export BM_DATA_TRAIN=$BM_DATA/peds_train
export BM_DATA_VALID=$BM_DATA/peds_valid
export BM_REPORTS_TRAIN=/temp_work/ch278233/COMBINED_reports.csv
export BM_REPORTS_VALID=/temp_work/ch278233/COMBINED_reports.csv
export BM_LABELS=$BM_DATA/peds23_labels_all.csv
BASE=$BM_RUNS/mpsct_peds23
A=${SLURM_RESTART_COUNT:-0}
export BM_RESULTS=$BASE/attempt$A
PN=$(ls -1 $BASE/attempt*/CTClip.*.pt 2>/dev/null \
     | sed 's/.*CTClip\.\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1)
if [ -n "$PN" ]; then
  export BM_PRETRAINED=$(echo "$PN" | cut -d' ' -f2)
  echo "resume: pediatric attempt=$A warm start $BM_PRETRAINED"
fi
export BM_STEPS=${BM_STEPS:-12000} BM_BATCH=10 BM_WORKERS=8 BM_LR=1e-5 BM_SAVE_EVERY=1000

echo "=== MPS-CT peds23 fine-tune job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
cd $BM_REPOS/MPS-CT
$BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/20_train_mpsct.py
rc=$?
echo "=== MPS-CT peds rc=$rc $(date) ==="
exit $rc

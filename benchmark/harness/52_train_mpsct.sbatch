#!/bin/bash
#SBATCH --job-name=bmMpsCT
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/bmMpsCT_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
export PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT

export BM_DATA_TRAIN=$BM_DATA/ctrate_train
export BM_DATA_VALID=$BM_DATA/ctrate_dev
export BM_REPORTS_TRAIN=$CTR/reports.csv
export BM_REPORTS_VALID=$CTR/reports.csv
export BM_LABELS=$CTR/labels18.csv
# Preemption is real on these partitions (PreemptMode=REQUEUE, all tiers equal).
# Each attempt gets a fresh results dir -- so the empty-dir guard, which is what
# stops upstream prompting on stdin, keeps holding -- and warm-starts from the
# newest checkpoint any earlier attempt wrote.
BASE=$BM_RUNS/mpsct_ctrate
A=${SLURM_RESTART_COUNT:-0}
export BM_RESULTS=$BASE/attempt$A
newest () { ls -1 $BASE/attempt*/CTClip.*.pt 2>/dev/null \
            | sed 's/.*CTClip\.\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1; }
N=$(newest)
if [ -n "$N" ]; then
  export BM_PRETRAINED=$(echo "$N" | cut -d' ' -f2)
  DONE=$(echo "$N" | cut -d' ' -f1)
  export BM_STEPS=$(( 30001 - DONE )); [ $BM_STEPS -lt 1000 ] && export BM_STEPS=1000
  echo "resume: attempt=$A warm start $BM_PRETRAINED (step $DONE), $BM_STEPS steps left"
else
  echo "resume: attempt=$A, no prior checkpoint -- fresh start"
fi
# Upstream train_clip.py: 30001 steps, batch 10, lr 1e-5, save every 1000.
export BM_STEPS=${BM_STEPS:-30001} BM_BATCH=10 BM_WORKERS=8 BM_LR=1e-5 BM_SAVE_EVERY=1000

echo "=== MPS-CT CT-RATE job=$SLURM_JOB_ID node=$(hostname) $(date) ==="
cd $BM_REPOS/MPS-CT
$BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/20_train_mpsct.py
rc=$?
echo "=== MPS-CT rc=$rc $(date) ==="
ls -l $BM_RESULTS 2>/dev/null | tail -5
exit $rc

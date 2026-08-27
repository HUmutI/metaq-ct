#!/bin/bash
#SBATCH --job-name=cttrts
#SBATCH --partition=bch-gpu-pe,bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/cttrts_%A_%a.out
set -uo pipefail
# The 15,575 CT-RATE training volumes that arrived without a mask. 31,570 came
# from another machine already in the target format; the remainder is generated
# HERE so the whole training set has one provenance.
#
# Why finish the set at all, when 71.6% coverage already trains: what is missing
# is not a random 33%. Sliced by patient id the gap runs 1.6% / 58% / 66% / 8% /
# 2% / 76% / 74% / 43% / 4% / 2% -- the signature of a shard array that stopped
# part way. L_org would then be trained on a subsample selected by acquisition
# order, and in a paper about anatomy routing the answer to that would have to be
# "probably fine".
#
# Preemption-safe by construction: the loop skips any volume whose output already
# exists, so a requeued shard resumes rather than restarts. bch-gpu-pe starts in
# minutes where bch-gpu is a week out, and gets preempted -- that trade is only
# worth taking because of the skip.
#
# 15,575 x 99.3 s median (A40) = 429 GPU-hours; 64 shards = ~6.7 h wall.
#
# Submit:  sbatch --array=0-63%64 pipeline/03_masks/52_ts_ctrate_train.sbatch
TS=/temp_work/ch278233/envs/totalseg
export ARCCT_ROOT=$HOME/bch-arc-ct
export LD_LIBRARY_PATH=$TS/lib:${LD_LIBRARY_PATH:-}   # some nodes lack CXXABI_1.3.15
export TOTALSEG_HOME_DIR=/temp_work/ch278233/totalseg_home
export TOTALSEG_WEIGHTS_PATH=$TOTALSEG_HOME_DIR/nnunet/results
export nnUNet_compile=0
export TMPDIR=/tmp/cttrts_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

WORKLIST=${TS_WORKLIST:-/temp_work/ch278233/CTRATE_TS_MISSING.txt}
[ -s "$WORKLIST" ] || { echo "FATAL: worklist $WORKLIST yok/bos"; exit 78; }

# Node-local weights. The same 1.4 GB copy took 2 s on an idle node and 75 s on a
# saturated one, and on that saturated node inference never got going at all.
# Staging time is therefore also the cheapest health check on the node we landed on.
LOCALW=$TMPDIR/weights
t0=$SECONDS; mkdir -p "$LOCALW"
cp -r "$TOTALSEG_WEIGHTS_PATH"/. "$LOCALW"/ && export TOTALSEG_WEIGHTS_PATH="$LOCALW"
STAGE=$((SECONDS-t0))
echo "=== shard ${SLURM_ARRAY_TASK_ID} on $(hostname): staged 1.4G in ${STAGE}s $(date) ==="
[ "$STAGE" -gt 180 ] && echo "WARNING: slow node, expect degraded throughput"
nvidia-smi --query-gpu=name --format=csv,noheader

export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
"$TS/bin/python" $ARCCT_ROOT/pipeline/03_masks/ts_ctrate.py \
    --worklist "$WORKLIST" \
    --in-dir  /temp_work/ch278233/COMBINED_NPZ \
    --out-dir /temp_work/ch278233/CTRATE_TS_RAW/train
rc=$?
echo "=== shard $SHARD_ID rc=$rc $(date) ==="
exit $rc

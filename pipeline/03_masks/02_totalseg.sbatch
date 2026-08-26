#!/bin/bash
#SBATCH --job-name=pedts
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --output=/temp_work/ch278233/ts_logs/ts_%A_%a.out
set -uo pipefail
TS=/temp_work/ch278233/envs/totalseg
export LD_LIBRARY_PATH=$TS/lib:${LD_LIBRARY_PATH:-}   # some nodes lack CXXABI_1.3.15
export TOTALSEG_HOME_DIR=/temp_work/ch278233/totalseg_home
export TOTALSEG_WEIGHTS_PATH=$TOTALSEG_HOME_DIR/nnunet/results
export nnUNet_compile=0
export TMPDIR=/tmp/pedts_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT

# Node-local weights. The same 1.4 GB copy took 2 s on an idle node and 75 s on a
# saturated one, and on that saturated node inference never got going at all
# (1m25s of CPU across 11 minutes of wall time). Staging time is therefore also
# the cheapest available health check on the node we landed on.
LOCALW=$TMPDIR/weights
t0=$SECONDS; mkdir -p "$LOCALW"
cp -r "$TOTALSEG_WEIGHTS_PATH"/. "$LOCALW"/ && export TOTALSEG_WEIGHTS_PATH="$LOCALW"
STAGE=$((SECONDS-t0))
echo "=== shard ${SLURM_ARRAY_TASK_ID} on $(hostname): staged 1.4G in ${STAGE}s ==="
[ "$STAGE" -gt 180 ] && echo "WARNING: slow node, expect degraded throughput"
nvidia-smi --query-gpu=name --format=csv,noheader

export TS_WORKLIST=${TS_WORKLIST:-}
export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
"$TS/bin/python" $ARCCT_ROOT/pipeline/03_masks/ts_batch.py \
    --in-dir /temp_work/ch278233/TS_RESAMPLED \
    --out-dir /temp_work/ch278233/TS_MASKS_PEDS
echo "=== shard $SHARD_ID done rc=$? ==="

#!/bin/bash
#SBATCH --job-name=ctcleanup
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-25-0
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/ctcleanup_%A_%a.out
set -uo pipefail
# The tail of the segmentation: volumes two shards did not finish.
#
# Both hung on gpu-25-0, one of them after a FileNotFoundError on its own
# node-local nnUNet scratch file and a volume that took 2,185 s against a 94 s
# median. They then sat RUNNING for eight and fourteen hours without writing a
# line, which is exactly what the watchdog is for -- and what SLURM will not tell
# you, since a hung job is not a failed one. gpu-25-0 is excluded here by name.
TS=/temp_work/ch278233/envs/totalseg
export ARCCT_ROOT=$HOME/bch-arc-ct
export LD_LIBRARY_PATH=$TS/lib:${LD_LIBRARY_PATH:-}
export TOTALSEG_HOME_DIR=/temp_work/ch278233/totalseg_home
export TOTALSEG_WEIGHTS_PATH=$TOTALSEG_HOME_DIR/nnunet/results
export nnUNet_compile=0
export TMPDIR=/tmp/ctcleanup_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
WORKLIST=${TS_WORKLIST:-/temp_work/ch278233/CTRATE_TS_CLEANUP.txt}
[ -s "$WORKLIST" ] || { echo "FATAL: worklist $WORKLIST yok/bos"; exit 78; }
LOCALW=$TMPDIR/weights; mkdir -p "$LOCALW"
t0=$SECONDS; cp -r "$TOTALSEG_WEIGHTS_PATH"/. "$LOCALW"/ && export TOTALSEG_WEIGHTS_PATH="$LOCALW"
echo "=== shard ${SLURM_ARRAY_TASK_ID:-0} on $(hostname): staged in $((SECONDS-t0))s $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
"$TS/bin/python" $ARCCT_ROOT/pipeline/03_masks/ts_ctrate.py \
    --worklist "$WORKLIST" \
    --in-dir  /temp_work/ch278233/COMBINED_NPZ \
    --out-dir /temp_work/ch278233/CTRATE_TS_RAW/train
rc=$?
echo "=== cleanup shard $SHARD_ID rc=$rc $(date) ==="
exit $rc

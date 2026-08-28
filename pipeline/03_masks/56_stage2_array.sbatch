#!/bin/bash
#SBATCH --job-name=cttrmask
#SBATCH --partition=bch-compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/cttrmask_%A_%a.out
set -uo pipefail
# Stage 2 as an ARRAY. The first attempt ran it as one process and measured 14.9
# volumes a minute -- 17.4 hours for 15,575, on the critical path of the run that
# is waiting for it. build_peds_masks.py already shards by SHARD_ID/SHARD_N and
# already skips any output that exists, so this is just using what was there.
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds
W=/temp_work/ch278233
DECIDE=$W/PLEURA_MODE.txt
PLEURA=lung
[ -r "$DECIDE" ] && PLEURA=$(tr -d '[:space:]' < "$DECIDE")
export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
echo "=== shard $SHARD_ID/$SHARD_N on $(hostname) pleura=$PLEURA $(date) ==="
$E/bin/python $ARCCT_ROOT/pipeline/03_masks/build_peds_masks.py \
    --in-dir  "$W/CTRATE_TS_RAW/train" \
    --out-dir "$W/CTRATE_MASKS10_192/train" \
    --schema 10 --no-map --pleura "$PLEURA"
rc=$?
echo "=== shard $SHARD_ID rc=$rc $(date) ==="
exit $rc

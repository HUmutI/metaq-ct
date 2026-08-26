#!/bin/bash
#SBATCH --job-name=ctvalmask
#SBATCH --partition=bch-compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/ctvalmask_%A_%a.out
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds

# Stage 2: raw TotalSegmentator labels -> the 10-region mask arc-ct consumes.
# Runs in the arcct env, not the TotalSegmentator one, because it imports the CT
# loader's own _pad_crop_hwd so mask and volume reach 192x192x96 by the same
# geometric route. --no-map because CT-RATE stems already are the VolumeName.
export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
$E/bin/python $ARCCT_ROOT/pipeline/03_masks/build_peds_masks.py \
    --in-dir  /temp_work/ch278233/CTRATE_TS_RAW/valid \
    --out-dir /temp_work/ch278233/CTRATE_MASKS10_192/valid \
    --schema 10 --no-map
rc=$?
echo "=== shard $SHARD_ID rc=$rc $(date) ==="
exit $rc

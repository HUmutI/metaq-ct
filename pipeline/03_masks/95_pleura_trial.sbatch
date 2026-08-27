#!/bin/bash
#SBATCH --job-name=pleuratr
#SBATCH --partition=bch-compute
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/pleuratr_%j.out
set -uo pipefail
# Build a SAMPLE with the ribcage-derived pleural region, so the fix is measured
# before 47k masks are committed to it.
#
# From CTRATE_TS_RAW/train, not /valid: the diagnostic resolves each mask to its
# npz under COMBINED_NPZ and its label row in COMBINED_labels_harmonised, and both
# hold train_* only. A first attempt built from /valid produced 700 masks that
# matched nothing, and the comparison silently re-measured the OLD masks twice.
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds
echo "=== $(hostname) $(date) ==="
$E/bin/python $ARCCT_ROOT/pipeline/03_masks/build_peds_masks.py \
    --in-dir  /temp_work/ch278233/CTRATE_TS_RAW/train \
    --out-dir /temp_work/ch278233/TRIAL_PLEURA_CAVITY \
    --schema 10 --no-map --pleura cavity
echo "=== rc=$? $(date) ==="

#!/bin/bash
#SBATCH --job-name=evalpeds
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/eval_peds_%j.out
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; set +a

# evaluate.py derives its valid npz dir and reports CSV from RAC_DATA_ROOT with
# fixed sub-paths, so the layout is reproduced with symlinks rather than by
# editing the evaluator. The 1,773-row labels CSV is what restricts this to the
# validation split: the loader skips any volume with no label row.
export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_PEDS
# Which label set a model is scored against is not neutral: V2 was trained
# and validated on the re-extracted labels, so scoring it against V1's
# gives V1 home advantage - it is why V2 read 0.7918 here against 0.7974
# in its own training log. Run both and check the ranking is stable.
export RAC_LABELS_VALID=${LBL:-/temp_work/ch278233/PEDS_VALID_labels.csv}
# _npz_to_mask_path takes a different branch depending on whether the npz
# path contains 'mps_ct_npz': with it, the mask name keeps the patient/
# study nesting; without it, just the basename. Training reads from
# PEDS_NPZ_NESTED and hits the flat branch, but evaluate.py hardcodes
# DATA_ROOT/mps_ct_npz/valid, so the same masks were looked up under a
# nested path and none were found - the first run reported 0 of 1,773
# with masks and would have measured an unmasked model.
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED
export EVAL_BATCH_SIZE=8
export EVAL_NUM_WORKERS=6

CKPT=${CKPT:?CKPT gerekli}
TAG=${TAG:?TAG gerekli}
export EVAL_CKPT=$CKPT
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_peds_only/$TAG
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== $TAG  $(date) ==="
echo "ckpt=$CKPT"
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" --expect-classes 27 --expect-schema peds || exit 78
$E/bin/python $ARCCT_ROOT/tools/evaluate.py
rc=$?
echo "=== eval rc=$rc $(date) ==="
exit $rc

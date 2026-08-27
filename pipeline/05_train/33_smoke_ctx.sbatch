#!/bin/bash
#SBATCH --job-name=ctxsmoke
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/ctxsmoke_%j.out
set -uo pipefail
# S8 gate. Two questions, in one job:
#
#   A. with the context flag OFF, is the training step still numerically what it
#      was? The new code adds branches, a wider tuple and two clip groups, and
#      "it should not have changed anything" is precisely the claim that has to
#      be measured. Losses are compared against the same run on the pinned
#      pre-change commit, printed side by side.
#
#   B. with the flag ON, does a step complete at all -- forward, the two
#      per-token terms, the counterfactual pass, a validation, a save, and a
#      RESUME from that save.
#
# RAC_TRAIN_LIMIT keeps both to minutes. This is a smoke test, not an experiment:
# no number it prints means anything about model quality.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/tmp/ctxsmoke_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; set +a

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID.txt

# Small, fixed, and no early stopping -- a smoke run must not exit on a metric.
export RAC_TRAIN_LIMIT=40
export RAC_VAL_LIMIT=40
export RAC_BATCH_SIZE=4
export RAC_ACCUM_STEPS=1
export RAC_TOTAL_UPDATES=6
export RAC_WARMUP_UPDATES=1
export RAC_VAL_EVERY=3
export RAC_SAVE_EVERY=3
export RAC_LOG_EVERY=1
export RAC_NUM_WORKERS=2
export RAC_EARLY_STOP_PATIENCE=0
export RAC_EARLY_STOP_MIN_UPDATES=999999
export RAC_SEED=0
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds27.pt

nvidia-smi --query-gpu=name --format=csv,noheader
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" \
    --expect-classes 27 --expect-schema peds || exit 78

BASE=${ARCCT_BASELINE_REF:-8c5fc0d}
WORK=$TMPDIR/baseline
run_and_grep () {   # $1 = label, $2 = results dir, rest = env already exported
    echo "=== $1 ==="
    $E/bin/python "$2" 2>&1 | grep -E "^\[RAC\] update=|^\[RAC\] Context|^\[RAC\] Q-Former|Traceback|Error" || true
}

# ---- A. flag OFF, current code vs the pinned baseline ---------------------
export RAC_USE_CONTEXT_QFORMER=0
export RAC_RESULTS_DIR=$TMPDIR/off_new
mkdir -p "$RAC_RESULTS_DIR"
run_and_grep "A1 flag OFF, current code" "$ARCCT_ROOT/tools/train_stage2.py" | tee "$TMPDIR/off_new.txt"

mkdir -p "$WORK"
git -C "$ARCCT_ROOT" archive "$BASE" | tar -x -C "$WORK"
export RAC_RESULTS_DIR=$TMPDIR/off_base
mkdir -p "$RAC_RESULTS_DIR"
SAVED_PP=$PYTHONPATH
export PYTHONPATH=$WORK           # the baseline must import ITS OWN arcct package
run_and_grep "A2 flag OFF, baseline $BASE" "$WORK/tools/train_stage2.py" \
    | tee "$TMPDIR/off_base.txt"
export PYTHONPATH=$SAVED_PP

echo "=== A: diff of the per-update loss lines ==="
if diff <(grep "update=" "$TMPDIR/off_new.txt") <(grep "update=" "$TMPDIR/off_base.txt"); then
    echo "S8-A OK: the flag-off path is line-for-line identical to $BASE"
else
    echo "S8-A FAIL: the flag-off path changed"
fi

# ---- B. flag ON ----------------------------------------------------------
set -a; . "$ARCCT_ROOT/configs/stage2_peds_ctx.env"; set +a
export RAC_CTX_DEBUG=1
export RAC_RESULTS_DIR=$TMPDIR/on
mkdir -p "$RAC_RESULTS_DIR"
# NOTE: the context CSVs do not exist yet, so every row carries NO_INDICATION
# and age_band/sex = -1. That still exercises tokenisation, C1, C2, the fusion
# and the two per-token terms, but NOT the counterfactual branch, which needs at
# least one row with ind_status=present. Re-run this once the extraction lands.
echo "=== B1 flag ON, 6 updates ==="
$E/bin/python $ARCCT_ROOT/tools/train_stage2.py; rc=$?
echo "B1 rc=$rc"
[ $rc -ne 0 ] && exit $rc

echo "=== B2 resume from the checkpoint B1 wrote ==="
export RAC_TOTAL_UPDATES=8
$E/bin/python $ARCCT_ROOT/tools/train_stage2.py; rc=$?
echo "B2 rc=$rc"
exit $rc

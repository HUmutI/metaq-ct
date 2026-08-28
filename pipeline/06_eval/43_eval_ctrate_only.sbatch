#!/bin/bash
#SBATCH --job-name=ctrEval
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --output=/temp_work/ch278233/ts_logs/ctrEval_%A_%a.out
set -uo pipefail
# The number the whole CT-RATE-only run exists to produce: 18-class macro AUC on
# the OFFICIAL 3,002-volume validation split, against ARC-CT's 0.8574.
#
# 12 hours, not 4: a previous evaluation timed out at four with the forward pass
# done in 48 minutes and the 1,000-resample patient-clustered bootstrap unfinished,
# so it was killed after all the expensive work and wrote nothing.
#
# Labels are CTRATE_VALID_labels27_full.csv, but under RAC_SCHEMA=ctrate only its
# first 18 columns are read -- and those were checked to be bit-identical to the
# published valid_predicted_labels.csv across all 3,002 volumes. Our own labeller
# does not enter the gate.
#
# Submit:  sbatch --array=0-5 --dependency=afterany:<train_array> \
#              pipeline/06_eval/43_eval_ctrate_only.sbatch
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; . "$ARCCT_ROOT/configs/stage2_ctrate_ctx.env"; set +a
export RAC_SCHEMA=ctrate

TASK=${SLURM_ARRAY_TASK_ID:-0}
CTR_RUNG=${CTR_RUNG:-c2}
if [ "$TASK" -lt 3 ]; then RUNG=ct_only; else RUNG=$CTR_RUNG; fi
SEED=$((TASK % 3))
RUN=/temp_work/ch278233/runs/ctrate_${RUNG}_seed${SEED}
CKPT=$RUN/CTClip.best.pt
[ -r "$CKPT" ] || { echo "FATAL: $CKPT yok -- egitim bitmemis olabilir"; exit 78; }

# ct_only checkpoints carry no context modules; evaluate.py takes its branch from
# the checkpoint's own query_layout, so this only has to match how it was trained.
case "$RUNG" in
  ct_only) export RAC_USE_CONTEXT_QFORMER=0 ;;
  *)       export RAC_USE_CONTEXT_QFORMER=1 ;;
esac

export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_CTRATE
export RAC_LABELS_VALID=/temp_work/ch278233/CTRATE_VALID_labels27_full.csv
export RAC_MASK_VALID=${CTMASK:-/temp_work/ch278233/CTRATE_MASKS10_192/valid_nested}
export EVAL_BATCH_SIZE=8
export EVAL_NUM_WORKERS=3
export EVAL_CKPT=$CKPT
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/ctrate18/${RUNG}_seed${SEED}
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== EVAL rung=$RUNG seed=$SEED  $(date) ==="
echo "    ckpt=$CKPT"
echo "    schema=$RAC_SCHEMA  ctx=$RAC_USE_CONTEXT_QFORMER"
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" \
    --expect-classes 18 --expect-schema ctrate || exit 78
$E/bin/python $ARCCT_ROOT/tools/evaluate.py
rc=$?
echo "=== EVAL rung=$RUNG seed=$SEED rc=$rc $(date) ==="
exit $rc

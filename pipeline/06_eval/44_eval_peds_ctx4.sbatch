#!/bin/bash
#SBATCH --job-name=evpCtx4
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
# gpu-b11-2 reported repeated uncorrectable ECC faults during the first audit
# submission. Keep the retry off that node; these are hardware failures, not
# model/evaluator failures.
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/evpCtx4_%A_%a.out
set -uo pipefail

# Pediatric single-checkpoint audit for the two decision arms:
#   task 0..8   standard C1: three seeds x true/none/shuffled indications
#   task 9..17  safety-controlled C1+C2: the same matrix
#
# The checkpoint does not persist all CtxConfig ablation flags, so this wrapper
# reconstructs each arm explicitly.  Never infer C1/C2 or age mode from tensor
# shapes: those shapes are deliberately identical across the ablations.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}

set -a
. "$ARCCT_ROOT/configs/stage2_peds.env"
. "$ARCCT_ROOT/configs/stage2_peds_ctx.env"
set +a

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
ARM=$((TASK / 9))
WITHIN=$((TASK % 9))
SEED=$((WITHIN / 3))
MODE_ID=$((WITHIN % 3))
case "$MODE_ID" in
  0) MODE=true ;;
  1) MODE=none ;;
  2) MODE=shuffled ;;
esac

export RAC_CTX_FUSION=class_logit
export RAC_CTX_C1=1
export RAC_CTX_SELFATTN=split
export RAC_CTX_SEED=$SEED
if [ "$ARM" = 0 ]; then
  RUNG=classlogit_c1_gatewarm
  export RAC_CTX_C2=0
  export RAC_AGE_MODE=scalar
else
  RUNG=classlogit_c2_gatewarm_safe
  export RAC_CTX_C2=1
  export RAC_AGE_MODE=band
fi

# Evaluation intervention must be assigned after sourcing the context config.
export RAC_EVAL_IND_MODE=$MODE
export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_PEDS
export RAC_LABELS_VALID=/temp_work/ch278233/PEDS_VALID_labels_v2.csv
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED
export EVAL_CKPT=/temp_work/ch278233/runs/ctx4_${RUNG}_seed${SEED}/CTClip.best.pt
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/peds_ctx4/${RUNG}_seed${SEED}_ind_${MODE}
export EVAL_BATCH_SIZE=8
export EVAL_NUM_WORKERS=3
# This first-pass matrix needs exact point estimates before tomorrow's meeting.
# Patient-clustered 1,000-replicate CIs will be generated only for the selected
# comparison from the saved predictions, avoiding repeated GPU forward passes.
export EVAL_BOOTSTRAP=200
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== ctx4 pediatric arm=$RUNG seed=$SEED indication=$MODE $(date) ==="
echo "ckpt=$EVAL_CKPT c2=$RAC_CTX_C2 age=$RAC_AGE_MODE"
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 27 --expect-schema peds || exit 78
$E/bin/python "$ARCCT_ROOT/tools/evaluate.py"
rc=$?
if [ "$rc" = 0 ]; then
  $E/bin/python "$ARCCT_ROOT/tools/age_stratified_eval.py" \
    --pred "$EVAL_RESULTS_DIR/predictions.npz" \
    --csv "$EVAL_RESULTS_DIR/auc_by_age.csv" \
    | tee "$EVAL_RESULTS_DIR/auc_by_age.txt"
fi
echo "=== rc=$rc $(date) ==="
exit $rc

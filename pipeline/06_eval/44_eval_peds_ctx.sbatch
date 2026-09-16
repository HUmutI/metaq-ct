#!/bin/bash
#SBATCH --job-name=pedCtxEval
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --output=/temp_work/ch278233/ts_logs/pedCtxEval_%A_%a.out
set -uo pipefail

# Held-out evaluation of the second-pass pediatric context ladder.
# Six rungs x three seeds.  Each task restores the exact ablation flags used
# for training before evaluate.py reconstructs the ContextQFormer.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
# Preserve a submission-time counterfactual request.  The context config has a
# safe production default (true), but sourcing it below must not silently
# replace --export=...,RAC_EVAL_IND_MODE=none|shuffled during causal evaluation.
REQUESTED_EVAL_IND_MODE=${RAC_EVAL_IND_MODE:-true}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG unreadable"; exit 78; }
set -a; . "$CFG"; . "$ARCCT_ROOT/configs/stage2_peds_ctx.env"; set +a
export RAC_SCHEMA=peds
export RAC_EVAL_IND_MODE=$REQUESTED_EVAL_IND_MODE

TASK=${SLURM_ARRAY_TASK_ID:-0}
RUNGS=(ct_only c1_xattn c1_full c2 fusion_gated age_scalar)
RUNG=${RUNGS[$((TASK / 3))]}
SEED=$((TASK % 3))
export RAC_SEED=$SEED
export RAC_CTX_SEED=$SEED

export RAC_USE_CONTEXT_QFORMER=1
export RAC_CTX_C1=1
export RAC_CTX_C2=1
export RAC_CTX_SELFATTN=split
export RAC_CTX_ZIND_COMBINE=mean
export RAC_CTX_FUSION=concat
export RAC_CTX_FILM_EPS=0.2
export RAC_AGE_MODE=band
case "$RUNG" in
  ct_only)      export RAC_USE_CONTEXT_QFORMER=0 ;;
  c1_xattn)     export RAC_CTX_C2=0; export RAC_CTX_FILM_EPS=0.0001 ;;
  c1_full)      export RAC_CTX_C2=0 ;;
  c2)           : ;;
  fusion_gated) export RAC_CTX_FUSION=gated ;;
  age_scalar)   export RAC_AGE_MODE=scalar ;;
  *) echo "FATAL: unknown rung=$RUNG"; exit 2 ;;
esac

RUN=/temp_work/ch278233/runs/ctx2_${RUNG}_seed${SEED}
export EVAL_CKPT=$RUN/CTClip.best.pt
[ -r "$EVAL_CKPT" ] || { echo "FATAL: missing $EVAL_CKPT"; exit 78; }

export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_PEDS
# The ctx2 ladder trained on LABELS27 (the original label family), so this is
# the like-for-like primary scoring target.  V1 scores 0.7961 on this same file.
export RAC_LABELS_VALID=${LBL:-/temp_work/ch278233/PEDS_VALID_labels.csv}
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED
export EVAL_BATCH_SIZE=8
export EVAL_NUM_WORKERS=3
ROOT=${EVAL_RESULTS_ROOT:-/temp_work/ch278233/eval_matrix/peds_ctx2}
export EVAL_RESULTS_DIR=$ROOT/${RUNG}_seed${SEED}${EVAL_TAG_SUFFIX:-}
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== PEDS CTX2 EVAL rung=$RUNG seed=$SEED $(date) ==="
echo "    ctx=$RAC_USE_CONTEXT_QFORMER c1=$RAC_CTX_C1 c2=$RAC_CTX_C2 age=$RAC_AGE_MODE fusion=$RAC_CTX_FUSION"
echo "    eval_ind_mode=$RAC_EVAL_IND_MODE ctx_seed=$RAC_CTX_SEED"
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" \
  --expect-classes 27 --expect-schema peds || exit 78
$E/bin/python $ARCCT_ROOT/tools/evaluate.py
rc=$?
if [ "$rc" = 0 ]; then
  AGE_CI_ARGS=""
  [ "${AGE_BOOTSTRAP_CI:-0}" = 1 ] && AGE_CI_ARGS="--ci"
  $E/bin/python $ARCCT_ROOT/tools/age_stratified_eval.py \
    --pred "$EVAL_RESULTS_DIR/predictions.npz" \
    --csv "$EVAL_RESULTS_DIR/auc_by_age.csv" $AGE_CI_ARGS \
    | tee "$EVAL_RESULTS_DIR/auc_by_age.txt"
fi
echo "=== PEDS CTX2 EVAL rung=$RUNG seed=$SEED rc=$rc $(date) ==="
exit $rc

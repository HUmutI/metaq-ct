#!/bin/bash
#SBATCH --job-name=evP23g5
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/evP23g5_%A_%a.out
set -uo pipefail

# Tasks 0..2: CT-only seeds.  Tasks 3..11: refined C1, three seeds x
# true/blank/shuffled indication interventions.  Every score is genuine peds23
# on the same age>=5 V2 validation cohort.
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
set -a
. "$ARCCT_ROOT/configs/stage2_peds.env"
. "$ARCCT_ROOT/configs/stage2_peds_ctx.env"
set +a
export RAC_SCHEMA=peds23
export RAC_QFORMER_QUERIES=35

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
if [ "$TASK" -lt 3 ]; then
  ARM=ct_only
  SEED=$TASK
  MODE=none
  export RAC_USE_CONTEXT_QFORMER=0
  export EVAL_CKPT=/temp_work/ch278233/runs/peds23_ge5_ct_only_seed${SEED}/CTClip.best.pt
else
  ARM=refined_c1
  WITHIN=$((TASK - 3))
  SEED=$((WITHIN / 3))
  case $((WITHIN % 3)) in 0) MODE=true ;; 1) MODE=none ;; 2) MODE=shuffled ;; esac
  export RAC_USE_CONTEXT_QFORMER=1
  export RAC_CTX_C1=1 RAC_CTX_C2=0 RAC_CTX_FUSION=class_logit
  export RAC_CTX_SELFATTN=split RAC_AGE_MODE=scalar RAC_CTX_SEED=$SEED
  export EVAL_CKPT=/temp_work/ch278233/runs/peds23_ge5_refined_c1_seed${SEED}/CTClip.best.pt
fi
export RAC_EVAL_IND_MODE=$MODE
export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_PEDS
export RAC_LABELS_VALID=/temp_work/ch278233/PEDS23_GE5_VALID_labels_v2.csv
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/peds23_ge5/${ARM}_seed${SEED}_ind_${MODE}
export EVAL_BATCH_SIZE=8 EVAL_NUM_WORKERS=3 EVAL_BOOTSTRAP=200
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== eval P23 age>=5 arm=$ARM seed=$SEED indication=$MODE $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
$E/bin/python "$ARCCT_ROOT/tools/evaluate.py"
rc=$?
if [ "$rc" = 0 ]; then
  $E/bin/python "$ARCCT_ROOT/tools/age_stratified_eval.py" \
    --pred "$EVAL_RESULTS_DIR/predictions.npz" \
    --csv "$EVAL_RESULTS_DIR/auc_by_age.csv" \
    | tee "$EVAL_RESULTS_DIR/auc_by_age.txt"
fi
echo "=== eval P23 age>=5 arm=$ARM seed=$SEED rc=$rc $(date) ==="
exit $rc

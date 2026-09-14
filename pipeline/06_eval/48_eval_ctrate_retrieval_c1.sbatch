#!/bin/bash
#SBATCH --job-name=ctrC1Ret
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --output=/temp_work/ch278233/ts_logs/ctrC1Ret_%A_%a.out
set -uo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
set -a
. "$ARCCT_ROOT/configs/stage2.env"
. "$ARCCT_ROOT/configs/stage2_ctrate_ctx.env"
set +a

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEED=$((TASK / 2))
MODE_ID=$((TASK % 2))
case "$MODE_ID" in
  0) EMBEDDING=general ;;
  1) EMBEDDING=mixed ;;
esac

# Reconstruct the selected adult Refined-C1 checkpoint configuration.
export RAC_SCHEMA=ctrate
export RAC_CTX_SEED=$SEED
export RAC_USE_CONTEXT_QFORMER=1
export RAC_QFORMER_QUERIES=30
export RAC_CTX_C1=1
export RAC_CTX_C2=0
export RAC_CTX_FUSION=class_logit
export RAC_AGE_MODE=scalar
export RAC_CTX_SELFATTN=split
export RAC_EVAL_IND_MODE=true
export RAC_IND_DROPOUT=0

export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_CTRATE
export RAC_LABELS_VALID=/temp_work/ch278233/CTRATE_VALID_labels27_full.csv
export RAC_MASK_VALID=/temp_work/ch278233/CTRATE_MASKS10_192/valid_nested
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export EVAL_CKPT=/temp_work/ch278233/runs/ctrate_refined_c1_seed${SEED}/CTClip.best.pt
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/ctrate18_retrieval/refined_c1_seed${SEED}_${EMBEDDING}
export EVAL_RUN_NAME=refined_c1_seed${SEED}_${EMBEDDING}
export EVAL_BATCH_SIZE=8
export EVAL_NUM_WORKERS=3
export EVAL_BOOTSTRAP=0
export RAC_RETRIEVAL_EMBEDDING=$EMBEDDING
mkdir -p "$EVAL_RESULTS_DIR"

[ -r "$EVAL_CKPT" ] || { echo "FATAL: $EVAL_CKPT missing"; exit 78; }

echo "=== CT-RATE retrieval seed=$SEED embedding=$EMBEDDING $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 18 --expect-schema ctrate || exit 78
$E/bin/python "$ARCCT_ROOT/tools/eval_retrieval.py"
rc=$?
echo "=== retrieval seed=$SEED embedding=$EMBEDDING rc=$rc $(date) ==="
exit $rc

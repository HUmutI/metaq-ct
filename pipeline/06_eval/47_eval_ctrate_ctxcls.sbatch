#!/bin/bash
#SBATCH --job-name=evCtrCtxCls
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --output=/temp_work/ch278233/ts_logs/evCtrCtxCls_%A_%a.out
set -uo pipefail

# Three seeds x correct/blank/shuffled indications. evaluate.py saves both our
# existing prompt-residual readout and the new supervised ctx_cls readout.
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
set -a
. "$ARCCT_ROOT/configs/stage2.env"
. "$ARCCT_ROOT/configs/stage2_ctrate_ctx.env"
set +a

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEED=$((TASK / 3))
case $((TASK % 3)) in 0) MODE=true ;; 1) MODE=none ;; 2) MODE=shuffled ;; esac
export RAC_SCHEMA=ctrate RAC_QFORMER_QUERIES=30
export RAC_CTX_C1=1 RAC_CTX_C2=0 RAC_CTX_FUSION=class_logit
export RAC_AGE_MODE=scalar RAC_CTX_SELFATTN=split RAC_CTX_CLS=1
export RAC_CTX_SEED=$SEED RAC_EVAL_IND_MODE=$MODE
export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_CTRATE
export RAC_LABELS_VALID=/temp_work/ch278233/CTRATE_VALID_labels27_full.csv
export RAC_MASK_VALID=/temp_work/ch278233/CTRATE_MASKS10_192/valid_nested
export EVAL_CKPT=/temp_work/ch278233/runs/ctrate_ctxcls_head_seed${SEED}/CTClip.best.pt
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/ctrate18_ctxcls/head_seed${SEED}_ind_${MODE}
export EVAL_BATCH_SIZE=8 EVAL_NUM_WORKERS=3 EVAL_BOOTSTRAP=200
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== official CT-RATE ctx_cls seed=$SEED indication=$MODE $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 18 --expect-schema ctrate || exit 78
$E/bin/python "$ARCCT_ROOT/tools/evaluate.py"
rc=$?
if [ "$rc" = 0 ]; then
  $E/bin/python "$ARCCT_ROOT/tools/eval_indication_subsets.py" \
    --pred "$EVAL_RESULTS_DIR/predictions_ctx_cls.npz" \
    --context /temp_work/ch278233/CONTEXT/indication.csv \
    --out "$EVAL_RESULTS_DIR/ctx_cls_indication_subsets.json"
fi
echo "=== CT-RATE ctx_cls eval seed=$SEED mode=$MODE rc=$rc $(date) ==="
exit $rc

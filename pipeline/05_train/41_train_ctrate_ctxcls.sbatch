#!/bin/bash
#SBATCH --job-name=ctrCtxCls
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/ctrCtxCls_%A_%a.out
set -uo pipefail

# Transfer experiment: add the independent environment's supervised ctx_cls
# readout to our completed refined-C1 representation.  Only the new head learns;
# CTCLIP, C1/FiLM, query tokens, and the existing class-logit branch stay frozen.
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/ctrctxcls_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"

set -a
. "$ARCCT_ROOT/configs/stage2.env"
. "$ARCCT_ROOT/configs/stage2_ctrate_ctx.env"
set +a
export RAC_SCHEMA=ctrate
export RAC_SEED=${SLURM_ARRAY_TASK_ID:-0}
export RAC_CTX_SEED=$RAC_SEED
export RAC_CTX_C1=1 RAC_CTX_C2=0 RAC_CTX_FUSION=class_logit
export RAC_AGE_MODE=scalar RAC_CTX_SELFATTN=split RAC_CTX_FILM_EPS=0.2
export RAC_IND_DROPOUT=0.10 RAC_CF_PROB=0.0 RAC_CF_WEIGHT=0.0
export RAC_CTX_CLS=1 RAC_CTX_CLS_WEIGHT=1.0 RAC_CTX_CLS_HEAD_ONLY=1
export RAC_CTX_FREEZE_BASE=1 RAC_CTX_CLASS_GATE_WARMUP=0
export RAC_TOTAL_UPDATES=4000 RAC_WARMUP_UPDATES=400
export RAC_VAL_EVERY=400 RAC_SAVE_EVERY=400
export RAC_EARLY_STOP_PATIENCE=4 RAC_EARLY_STOP_MIN_UPDATES=800

D=/temp_work/ch278233/CTRATE_ONLY
export RAC_DATA_TRAIN=/temp_work/ch278233/COMBINED_NPZ
export RAC_DATA_VALID=/temp_work/ch278233/COMBINED_NPZ
export RAC_VOLUME_LIST_TRAIN=$D/VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=$D/VOLLIST_VALID.txt
export RAC_LABELS_TRAIN=$D/labels18.csv RAC_LABELS_VALID=$D/labels18.csv
export RAC_REPORTS_TRAIN=$D/reports.csv RAC_REPORTS_VALID=$D/reports.csv
export RAC_REGION_CACHE=/temp_work/ch278233/COMBINED_region_cache_v2.json
export RAC_MASK_TRAIN=/temp_work/ch278233/COMBINED_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/COMBINED_MASKS10_192
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_WARM_START_CKPT=/temp_work/ch278233/runs/ctrate_refined_c1_seed${RAC_SEED}/CTClip.best.pt
export RAC_RESULTS_DIR=/temp_work/ch278233/runs/ctrate_ctxcls_head_seed${RAC_SEED}
export RAC_NUM_WORKERS=4 RAC_RESPECT_EMPTY_REGION=1

if [ "${CTR_CTXCLS_SMOKE:-0}" = 1 ]; then
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/ctrate_ctxcls_head_smoke_seed${RAC_SEED}
  export RAC_TRAIN_LIMIT=48 RAC_VAL_LIMIT=32 RAC_BATCH_SIZE=2 RAC_NUM_WORKERS=0
  export RAC_TOTAL_UPDATES=2 RAC_WARMUP_UPDATES=1 RAC_VAL_EVERY=2 RAC_SAVE_EVERY=2
  export RAC_EARLY_STOP_PATIENCE=0
fi
mkdir -p "$RAC_RESULTS_DIR"

echo "=== CT-RATE C1 + ctx_cls head-only seed=$RAC_SEED smoke=${CTR_CTXCLS_SMOKE:-0} $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 18 --expect-schema ctrate || exit 78
$E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" || exit 78
$E/bin/python "$ARCCT_ROOT/tools/train_stage2.py"
rc=$?
echo "=== CT-RATE ctx_cls seed=$RAC_SEED rc=$rc $(date) ==="
exit $rc

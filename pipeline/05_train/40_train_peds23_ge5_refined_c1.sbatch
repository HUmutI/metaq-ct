#!/bin/bash
#SBATCH --job-name=p23g5C1
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/p23g5C1_%A_%a.out
set -uo pipefail

# Matched refined-C1 arm.  It begins from the seed-matched, age>=5, genuine
# peds23 CT-only checkpoint above and changes only the metadata pathway.
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/p23g5c1_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"

set -a
. "$ARCCT_ROOT/configs/stage2_peds.env"
. "$ARCCT_ROOT/configs/stage2_peds_ctx.env"
set +a
export RAC_SCHEMA=peds23
export RAC_QFORMER_QUERIES=35
export RAC_SEED=${SLURM_ARRAY_TASK_ID:-0}
export RAC_CTX_SEED=$RAC_SEED
export RAC_USE_CONTEXT_QFORMER=1
export RAC_CTX_C1=1
export RAC_CTX_C2=0
export RAC_CTX_FUSION=class_logit
export RAC_CTX_FREEZE_BASE=1
export RAC_AGE_MODE=scalar
export RAC_CTX_SELFATTN=split
export RAC_CTX_FILM_EPS=0.2
export RAC_CTX_LR_MULT=20.0
export RAC_CTX_PTOK_WEIGHT=1.0
export RAC_CTX_CLASS_GATE_WARMUP=400
export RAC_IND_DROPOUT=0.10
export RAC_CF_PROB=0.0
export RAC_CF_WEIGHT=0.0
export RAC_TOTAL_UPDATES=5000
export RAC_WARMUP_UPDATES=400
export RAC_VAL_EVERY=200
export RAC_SAVE_EVERY=200
export RAC_EARLY_STOP_PATIENCE=10
export RAC_EARLY_STOP_MIN_UPDATES=600

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS23_GE5_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS23_GE5_VOLLIST_VALID.txt
export RAC_WARM_START_CKPT=/temp_work/ch278233/runs/peds23_ge5_ct_only_seed${RAC_SEED}/CTClip.best.pt
export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_ge5_refined_c1_seed${RAC_SEED}

if [ "${P23_GE5_SMOKE:-0}" = 1 ]; then
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_ge5_smoke_refined_c1_seed${RAC_SEED}
  export RAC_TRAIN_LIMIT=40 RAC_VAL_LIMIT=32 RAC_BATCH_SIZE=2 RAC_NUM_WORKERS=0
  export RAC_TOTAL_UPDATES=2 RAC_WARMUP_UPDATES=1 RAC_VAL_EVERY=2 RAC_SAVE_EVERY=2
  export RAC_EARLY_STOP_PATIENCE=0
fi
mkdir -p "$RAC_RESULTS_DIR"

echo "=== P23 age>=5 refined-C1 seed=$RAC_SEED smoke=${P23_GE5_SMOKE:-0} $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
$E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" || exit 78
$E/bin/python "$ARCCT_ROOT/tools/train_stage2.py"
rc=$?
echo "=== P23 age>=5 refined-C1 seed=$RAC_SEED rc=$rc $(date) ==="
exit $rc

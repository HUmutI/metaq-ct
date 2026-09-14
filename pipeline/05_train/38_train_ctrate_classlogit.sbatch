#!/bin/bash
#SBATCH --job-name=ctrRefC1
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/ctrRefC1_%A_%a.out
set -uo pipefail

# Refined C1 on the classical adult benchmark only:
#   train: CT-RATE train, published 18 labels
#   select: held-out subset of CT-RATE train
#   final: official CT-RATE validation, submitted separately after training
# No pediatric volume, label, report, mask, or checkpoint is in this job.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/ctrrefc1_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"

set -a
. "$ARCCT_ROOT/configs/stage2.env"
. "$ARCCT_ROOT/configs/stage2_ctrate_ctx.env"
set +a

TASK=${SLURM_ARRAY_TASK_ID:-0}
SEED=$TASK
export RAC_SCHEMA=ctrate
export RAC_SEED=$SEED
export RAC_CTX_SEED=$SEED

# Match the winning pediatric refinement structurally, without importing any
# pediatric data: C1 only, scalar age, standard indication supervision, and a
# zero-start per-class residual on a frozen trained CT-RATE-only checkpoint.
export RAC_USE_CONTEXT_QFORMER=1
export RAC_QFORMER_QUERIES=30
export RAC_CTX_C1=1
export RAC_CTX_C2=0
export RAC_CTX_FUSION=class_logit
export RAC_CTX_FREEZE_BASE=1
export RAC_AGE_MODE=scalar
export RAC_CTX_SELFATTN=split
export RAC_CTX_FILM_EPS=0.2
export RAC_CTX_LR_MULT=20.0
export RAC_CTX_PTOK_WEIGHT=1.0
export RAC_CTX_CLASS_GATE_WARMUP=1200
export RAC_IND_DROPOUT=0.10
export RAC_CF_PROB=0.0
export RAC_CF_WEIGHT=0.0

# Retain the adult recipe and its train-side development set. The official
# valid split remains untouched until the selected checkpoints are evaluated.
export RAC_TOTAL_UPDATES=14400
export RAC_WARMUP_UPDATES=1200
export RAC_VAL_EVERY=1200
export RAC_SAVE_EVERY=1200
export RAC_EARLY_STOP_PATIENCE=4
export RAC_EARLY_STOP_MIN_UPDATES=1200

D=/temp_work/ch278233/CTRATE_ONLY
export RAC_DATA_TRAIN=/temp_work/ch278233/COMBINED_NPZ
export RAC_DATA_VALID=/temp_work/ch278233/COMBINED_NPZ
export RAC_VOLUME_LIST_TRAIN=$D/VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=$D/VOLLIST_VALID.txt
export RAC_LABELS_TRAIN=$D/labels18.csv
export RAC_LABELS_VALID=$D/labels18.csv
export RAC_REPORTS_TRAIN=$D/reports.csv
export RAC_REPORTS_VALID=$D/reports.csv
export RAC_REGION_CACHE=/temp_work/ch278233/COMBINED_region_cache_v2.json
export RAC_MASK_TRAIN=/temp_work/ch278233/COMBINED_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/COMBINED_MASKS10_192
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_WARM_START_CKPT=/temp_work/ch278233/runs/ctrate_ct_only_seed${SEED}/CTClip.best.pt
export RAC_RESULTS_DIR=/temp_work/ch278233/runs/ctrate_refined_c1_seed${SEED}
export RAC_NUM_WORKERS=4
export RAC_RESPECT_EMPTY_REGION=1

if [ "${CTR_REFINED_SMOKE:-0}" = 1 ]; then
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/ctrate_refined_c1_smoke_seed${SEED}
  export RAC_TRAIN_LIMIT=48
  export RAC_VAL_LIMIT=32
  export RAC_BATCH_SIZE=2
  export RAC_NUM_WORKERS=0
  export RAC_TOTAL_UPDATES=2
  export RAC_WARMUP_UPDATES=1
  export RAC_VAL_EVERY=2
  export RAC_SAVE_EVERY=2
  export RAC_EARLY_STOP_PATIENCE=0
  export RAC_CTX_CLASS_GATE_WARMUP=1
fi
mkdir -p "$RAC_RESULTS_DIR"

echo "=== CT-RATE refined C1 seed=$SEED smoke=${CTR_REFINED_SMOKE:-0} $(date) ==="
echo "schema=$RAC_SCHEMA classes=18 train=CT-RATE-only warmstart=$RAC_WARM_START_CKPT"
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_mask_coverage.py" \
  --vollist "$RAC_VOLUME_LIST_TRAIN" --masks "$RAC_MASK_TRAIN" --min 99 || exit 78
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 18 --expect-schema ctrate || exit 78
$E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" || exit 78
$E/bin/python "$ARCCT_ROOT/tools/train_stage2.py"
rc=$?
echo "=== CT-RATE refined C1 seed=$SEED rc=$rc $(date) ==="
exit $rc

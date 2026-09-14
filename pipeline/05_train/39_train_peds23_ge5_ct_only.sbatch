#!/bin/bash
#SBATCH --job-name=p23g5Base
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/p23g5Base_%A_%a.out
set -uo pipefail

# Genuine 23-output CT-only control, trained and validated exclusively on age
# >=5.  The four excluded classes do not contribute gradients and do not exist
# in the Q-Former bank; this is not a 27-class model selectively reported on 23.
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/p23g5base_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"

set -a
. "$ARCCT_ROOT/configs/stage2_peds.env"
set +a
export RAC_SCHEMA=peds23
export RAC_QFORMER_QUERIES=35
export RAC_USE_CONTEXT_QFORMER=0
export RAC_SEED=${SLURM_ARRAY_TASK_ID:-0}

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS23_GE5_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS23_GE5_VOLLIST_VALID.txt
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds23.pt
export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_ge5_ct_only_seed${RAC_SEED}
export RAC_EARLY_STOP_PATIENCE=8

if [ "${P23_GE5_SMOKE:-0}" = 1 ]; then
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_ge5_smoke_ct_only_seed${RAC_SEED}
  export RAC_TRAIN_LIMIT=40 RAC_VAL_LIMIT=32 RAC_BATCH_SIZE=2 RAC_NUM_WORKERS=0
  export RAC_TOTAL_UPDATES=2 RAC_WARMUP_UPDATES=1 RAC_VAL_EVERY=2 RAC_SAVE_EVERY=2
  export RAC_EARLY_STOP_PATIENCE=0
fi
mkdir -p "$RAC_RESULTS_DIR"

echo "=== P23 age>=5 CT-only seed=$RAC_SEED smoke=${P23_GE5_SMOKE:-0} $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
$E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" --skip-slow || exit 78
$E/bin/python "$ARCCT_ROOT/tools/train_stage2.py"
rc=$?
echo "=== P23 age>=5 CT-only seed=$RAC_SEED rc=$rc $(date) ==="
exit $rc

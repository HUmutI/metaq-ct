#!/bin/bash
#SBATCH --job-name=p23AllAge
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-26-3
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/p23AllAge_%A_%a.out
set -euo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/p23allage_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -a
. "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"
set +a

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=$RAC_REPORTS_TRAIN
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=$RAC_LABELS_TRAIN
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN_clean.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID_clean.txt
export RAC_REGION_CACHE=
export RAC_STAGE1_CKPT=/temp_work/ch278233/runs/peds23_allage_stage1_seed31/stage1_best.pt
export RAC_WARM_START_CKPT=

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
SEED=$((TASK % 3))
export RAC_SEED=$SEED RAC_CTX_SEED=$SEED
if [ "$TASK" -lt 3 ]; then
  ARM=full_seed${SEED}
else
  ARM=ct_only_seed${SEED}
  export RAC_USE_CONTEXT_QFORMER=0
fi
export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_allage/${ARM}
mkdir -p "$RAC_RESULTS_DIR"

echo "=== PEDS23 all-age arm=$ARM seed=$SEED $(date) ==="
"$E/bin/python" "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23
if [ "$RAC_USE_CONTEXT_QFORMER" = 1 ]; then
  "$E/bin/python" "$ARCCT_ROOT/tools/preflight_context.py" --skip-slow
fi
"$E/bin/python" "$ARCCT_ROOT/tools/train_stage2.py"

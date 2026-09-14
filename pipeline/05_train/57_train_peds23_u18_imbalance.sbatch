#!/bin/bash
#SBATCH --job-name=p23Imb
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/p23Imb_%A_%a.out
set -euo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/p23imb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -a
. "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"
set +a

# Same U18 cohort, Stage-1 initialization, architecture, and dev split as the
# primary full_seed0 run. Only the Stage-2 classification loss changes.
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
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS23_U18_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS23_U18_VOLLIST_DEV.txt
export RAC_REGION_CACHE=
export RAC_STAGE1_CKPT=/temp_work/ch278233/runs/peds23_u18_stage1_seed31/stage1_best.pt
export RAC_WARM_START_CKPT=
export RAC_SEED=0 RAC_CTX_SEED=0

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
case "$TASK" in
  0) ARM=asl;        export RAC_STAGE2_CLS_LOSS=asl RAC_BNM_WEIGHT=0 ;;
  1) ARM=bnm001;     export RAC_STAGE2_CLS_LOSS=bce RAC_BNM_WEIGHT=0.001 ;;
  2) ARM=bnm003;     export RAC_STAGE2_CLS_LOSS=bce RAC_BNM_WEIGHT=0.003 ;;
  3) ARM=bnm010;     export RAC_STAGE2_CLS_LOSS=bce RAC_BNM_WEIGHT=0.010 ;;
  4) ARM=asl_bnm001; export RAC_STAGE2_CLS_LOSS=asl RAC_BNM_WEIGHT=0.001 ;;
  5) ARM=asl_bnm003; export RAC_STAGE2_CLS_LOSS=asl RAC_BNM_WEIGHT=0.003 ;;
  6) ARM=asl_bnm010; export RAC_STAGE2_CLS_LOSS=asl RAC_BNM_WEIGHT=0.010 ;;
  *) echo "FATAL unknown task $TASK"; exit 78 ;;
esac

RUN_ROOT=${PEDS_IMB_RUN_ROOT:-/temp_work/ch278233/runs/peds23_u18_imbalance}
export RAC_RESULTS_DIR=$RUN_ROOT/$ARM
if [ "${PEDS_IMB_SMOKE:-0}" = 1 ]; then
  export RAC_STAGE1_CKPT=/temp_work/ch278233/runs/peds23_u18_stage1_smoke/stage1_best.pt
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_u18_imbalance_smoke/$ARM
  export RAC_TRAIN_LIMIT=16 RAC_VAL_LIMIT=16 RAC_BATCH_SIZE=4 RAC_ACCUM_STEPS=1
  export RAC_NUM_WORKERS=0 RAC_TOTAL_UPDATES=2 RAC_WARMUP_UPDATES=1
  export RAC_VAL_EVERY=2 RAC_SAVE_EVERY=2 RAC_EARLY_STOP_PATIENCE=0
fi

mkdir -p "$RAC_RESULTS_DIR"
echo "=== U18 imbalance arm=$ARM loss=$RAC_STAGE2_CLS_LOSS bnm=$RAC_BNM_WEIGHT smoke=${PEDS_IMB_SMOKE:-0} $(date) ==="
"$E/bin/python" "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23
"$E/bin/python" "$ARCCT_ROOT/tools/preflight_context.py" --skip-slow
"$E/bin/python" "$ARCCT_ROOT/tools/train_stage2.py"

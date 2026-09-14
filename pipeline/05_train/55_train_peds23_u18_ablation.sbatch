#!/bin/bash
#SBATCH --job-name=p23U18Abl
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/p23U18Abl_%A_%a.out
set -uo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/p23u18_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -a; . "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"; set +a

COHORT=${PEDS_U18_COHORT:-u18}
ARCH=${PEDS_U18_ARCH:-r2plus1d_18}
if [ "$COHORT" = ge5 ]; then
  LIST_PREFIX=PEDS23_U18_GE5
  RUN_ROOT=${PEDS_U18_RUN_ROOT:-/temp_work/ch278233/runs/peds23_u18_ge5_${ARCH}}
  STAGE1_TAG=peds23_u18_ge5_${ARCH}_stage1_seed31
else
  LIST_PREFIX=PEDS23_U18
  RUN_ROOT=${PEDS_U18_RUN_ROOT:-/temp_work/ch278233/runs/peds23_u18_${ARCH}}
  STAGE1_TAG=peds23_u18_${ARCH}_stage1_seed31
fi
export RAC_R3D_ARCH=$ARCH

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
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/${LIST_PREFIX}_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/${LIST_PREFIX}_VOLLIST_DEV.txt
export RAC_REGION_CACHE=
export RAC_STAGE1_CKPT=/temp_work/ch278233/runs/${STAGE1_TAG}/stage1_best.pt
export RAC_WARM_START_CKPT=
export RAC_SEED=0 RAC_CTX_SEED=0

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
case "$TASK" in
  0) ARM=full_seed0 ;;
  1) ARM=full_seed1; export RAC_SEED=1 RAC_CTX_SEED=1 ;;
  2) ARM=full_seed2; export RAC_SEED=2 RAC_CTX_SEED=2 ;;
  3) ARM=ct_only_seed0; export RAC_USE_CONTEXT_QFORMER=0 ;;
  4) ARM=ct_only_seed1; export RAC_USE_CONTEXT_QFORMER=0 RAC_SEED=1 RAC_CTX_SEED=1 ;;
  5) ARM=ct_only_seed2; export RAC_USE_CONTEXT_QFORMER=0 RAC_SEED=2 RAC_CTX_SEED=2 ;;
  6) ARM=c2_off; export RAC_CTX_C2=0 ;;
  7) ARM=c1_off; export RAC_CTX_C1=0 ;;
  8) ARM=indication_only; export RAC_METADATA_MODE=indication ;;
  9) ARM=demographics_only; export RAC_METADATA_MODE=demographics ;;
 10) ARM=metadata_none; export RAC_METADATA_MODE=none ;;
 11) ARM=anatomy_queries_off; export RAC_CTX_ANATOMY_QUERIES=0 RAC_QFORMER_QUERIES=25 ;;
 12) ARM=routing_off; export RAC_SUPPRESS_ROUTING_MASK=1 ;;
 13) ARM=region_align_off; export RAC_ALIGN_WEIGHT=0 ;;
 14) ARM=maskfree; export RAC_SUPPRESS_ROUTING_MASK=1 RAC_ALIGN_WEIGHT=0 RAC_REQUIRE_MASK=0 ;;
 15) ARM=age_scalar; export RAC_AGE_MODE=scalar ;;
 16) ARM=xattn_only; export RAC_CTX_XATTN=1 RAC_CTX_FILM=0 ;;
 17) ARM=film_only; export RAC_CTX_XATTN=0 RAC_CTX_FILM=1 ;;
 18) ARM=drop00; export RAC_IND_DROPOUT=0.00 ;;
 19) ARM=drop10; export RAC_IND_DROPOUT=0.10 ;;
 20) ARM=drop40; export RAC_IND_DROPOUT=0.40 ;;
 21) ARM=no_augmentation; export RAC_RANDOM_CROP=0 RAC_ENABLE_ANATOMY_FLIPS=0 RAC_INTENSITY_JITTER_HU=0 ;;
 22) ARM=ctxcls_aux; export RAC_CTX_CLS=1 RAC_CTX_CLS_WEIGHT=0.5 ;;
 23) ARM=counterfactual005; export RAC_CF_PROB=0.25 RAC_CF_WEIGHT=0.05 ;;
 24) ARM=qwen_region_text; export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json ;;
 25) ARM=naive_metadata_fusion; export RAC_CTX_C1=0 RAC_CTX_C2=0 RAC_CTX_NAIVE_FUSION=1 RAC_CTX_FUSION=concat ;;
 32) ARM=l2_on; export RAC_NO_L2_CLIP=0 RAC_NO_L2_PROMPT=0 ;;
 33) ARM=standard_negative; export RAC_PROMPT_NEGATIVE=standard ;;
 34) ARM=ind_weight1; export RAC_CTX_W_IND=1.0 ;;
 35) ARM=ind_weight4; export RAC_CTX_W_IND=4.0 ;;
 36) ARM=one_global_query; export RAC_QFORMER_QUERIES=34 ;;
 37) ARM=six_global_queries; export RAC_QFORMER_QUERIES=39 ;;
 38) ARM=counterfactual015; export RAC_CF_PROB=0.25 RAC_CF_WEIGHT=0.15 ;;
 39) ARM=general_selection; export RAC_VAL_SELECTION=general ;;
 40) ARM=pathology_class_logit; export RAC_CTX_FUSION=class_logit RAC_CTX_PTOK_WEIGHT=1.0 RAC_CTX_CLASS_GATE_WARMUP=400 ;;
  *) echo "FATAL unknown task $TASK"; exit 78 ;;
esac

export RAC_RESULTS_DIR=${RUN_ROOT}/${ARM}
if [ "${PEDS_U18_SMOKE:-0}" = 1 ]; then
  export RAC_STAGE1_CKPT=/temp_work/ch278233/runs/peds23_u18_stage1_smoke/stage1_best.pt
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_u18_smoke/${ARM}
  export RAC_TRAIN_LIMIT=16 RAC_VAL_LIMIT=16 RAC_BATCH_SIZE=1 RAC_ACCUM_STEPS=1
  export RAC_NUM_WORKERS=0 RAC_TOTAL_UPDATES=2 RAC_WARMUP_UPDATES=1
  export RAC_VAL_EVERY=2 RAC_SAVE_EVERY=2 RAC_EARLY_STOP_PATIENCE=0
fi
mkdir -p "$RAC_RESULTS_DIR"
echo "=== U18 arm=$ARM task=$TASK seed=$RAC_SEED $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
if [ "$RAC_USE_CONTEXT_QFORMER" = 1 ]; then
  $E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" --skip-slow || exit 78
fi
$E/bin/python "$ARCCT_ROOT/tools/train_stage2.py"

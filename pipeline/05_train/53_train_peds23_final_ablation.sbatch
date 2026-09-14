#!/bin/bash
#SBATCH --job-name=p23FinalAbl
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/p23FinalAbl_%A_%a.out
set -uo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/p23final_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p "$TMPDIR"
set -a; . "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"; set +a
export RAC_R3D_ARCH=${PEDS_FINAL_ARCH:-r2plus1d_18}

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS23_GE5_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS23_GE5_VOLLIST_VALID.txt
export RAC_REGION_CACHE=
export RAC_STAGE1_CKPT=/temp_work/ch278233/runs/peds23_final_${RAC_R3D_ARCH}_stage1_seed31/stage1_best.pt
export RAC_WARM_START_CKPT=

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
export RAC_SEED=0 RAC_CTX_SEED=0
case "$TASK" in
  0) ARM=final_drop03_band ;;
  1) ARM=drop00; export RAC_IND_DROPOUT=0.00 ;;
  2) ARM=drop10; export RAC_IND_DROPOUT=0.10 ;;
  3) ARM=drop40; export RAC_IND_DROPOUT=0.40 ;;
  4) ARM=age_scalar; export RAC_AGE_MODE=scalar ;;
  5) ARM=c2_off; export RAC_CTX_C2=0 ;;
  6) ARM=c1_off; export RAC_CTX_C1=0 ;;
  7) ARM=indication_only; export RAC_METADATA_MODE=indication ;;
  8) ARM=demographics_only; export RAC_METADATA_MODE=demographics ;;
  9) ARM=metadata_none; export RAC_METADATA_MODE=none ;;
 10) ARM=anatomy_queries_off; export RAC_CTX_ANATOMY_QUERIES=0 RAC_QFORMER_QUERIES=25 ;;
 11) ARM=routing_off; export RAC_SUPPRESS_ROUTING_MASK=1 ;;
 12) ARM=region_align_off; export RAC_ALIGN_WEIGHT=0 ;;
 13) ARM=fully_maskfree; export RAC_SUPPRESS_ROUTING_MASK=1 RAC_ALIGN_WEIGHT=0 RAC_REQUIRE_MASK=0 ;;
 14) ARM=l2_on; export RAC_NO_L2_CLIP=0 RAC_NO_L2_PROMPT=0 ;;
 15) ARM=standard_negative; export RAC_PROMPT_NEGATIVE=standard ;;
 16) ARM=ctxcls_aux; export RAC_CTX_CLS=1 RAC_CTX_CLS_WEIGHT=0.5 RAC_VAL_SELECTION=conditioned ;;
 17) ARM=counterfactual005; export RAC_CF_PROB=0.25 RAC_CF_WEIGHT=0.05 ;;
 18) ARM=ind_weight1; export RAC_CTX_W_IND=1.0 ;;
 19) ARM=ind_weight4; export RAC_CTX_W_IND=4.0 ;;
 20) ARM=qwen_region_text; export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json ;;
 21) ARM=one_global_query; export RAC_QFORMER_QUERIES=34 ;;
 22) ARM=six_global_queries; export RAC_QFORMER_QUERIES=39 ;;
 23) ARM=no_augmentation; export RAC_RANDOM_CROP=0 RAC_ENABLE_ANATOMY_FLIPS=0 RAC_INTENSITY_JITTER_HU=0 ;;
 24) ARM=final_drop03_band_seed1; export RAC_SEED=1 RAC_CTX_SEED=1 ;;
 25) ARM=final_drop03_band_seed2; export RAC_SEED=2 RAC_CTX_SEED=2 ;;
  26|27|28)
      ARM=ct_only_seed$((TASK-26)); export RAC_SEED=$((TASK-26))
      export RAC_CTX_SEED=$RAC_SEED RAC_USE_CONTEXT_QFORMER=0
      ;;
 29) ARM=indication_present_train; export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS23_GE5_IND_PRESENT_TRAIN.txt ;;
 30) ARM=general_selection; export RAC_VAL_SELECTION=general ;;
  *) echo "FATAL unknown task $TASK"; exit 78 ;;
esac

if [ "$RAC_R3D_ARCH" != r2plus1d_18 ]; then
  ARM=${ARM}_${RAC_R3D_ARCH}
fi

export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_biltir_final/${ARM}
if [ "${PEDS_FINAL_SMOKE:-0}" = 1 ]; then
  export RAC_STAGE1_CKPT=/temp_work/ch278233/runs/peds23_final_r2plus1d_18_stage1_smoke/stage1_best.pt
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/peds23_biltir_final_smoke/${ARM}
  export RAC_TRAIN_LIMIT=16 RAC_VAL_LIMIT=16 RAC_BATCH_SIZE=1 RAC_ACCUM_STEPS=1
  export RAC_NUM_WORKERS=0 RAC_TOTAL_UPDATES=2 RAC_WARMUP_UPDATES=1
  export RAC_VAL_EVERY=2 RAC_SAVE_EVERY=2 RAC_EARLY_STOP_PATIENCE=0
  export RAC_CTX_ASSERT_INVARIANCE=1
fi
mkdir -p "$RAC_RESULTS_DIR"
echo "=== pediatric final arm=$ARM task=$TASK seed=$RAC_SEED smoke=${PEDS_FINAL_SMOKE:-0} $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
if [ "$RAC_USE_CONTEXT_QFORMER" = 1 ]; then
  if [ "$RAC_CTX_ANATOMY_QUERIES" = 1 ]; then
    $E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" || exit 78
  else
    $E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" --skip-slow || exit 78
  fi
fi
$E/bin/python "$ARCCT_ROOT/tools/train_stage2.py"
rc=$?
echo "=== pediatric final arm=$ARM rc=$rc $(date) ==="
exit $rc

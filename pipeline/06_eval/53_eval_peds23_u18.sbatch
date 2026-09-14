#!/bin/bash
#SBATCH --job-name=evP23U18
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/evP23U18_%A_%a.out
set -uo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
set -a; . "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"; set +a

COHORT=${PEDS_U18_COHORT:-u18}
ARCH=${PEDS_U18_ARCH:-r2plus1d_18}
if [ "$COHORT" = ge5 ]; then
  LIST_PREFIX=PEDS23_U18_GE5
  RUN_ROOT=${PEDS_U18_RUN_ROOT:-/temp_work/ch278233/runs/peds23_u18_ge5_${ARCH}}
  EVAL_ROOT=${PEDS_U18_EVAL_ROOT:-/temp_work/ch278233/eval_matrix/peds23_u18_ge5_${ARCH}_test}
else
  LIST_PREFIX=PEDS23_U18
  RUN_ROOT=${PEDS_U18_RUN_ROOT:-/temp_work/ch278233/runs/peds23_u18_${ARCH}}
  EVAL_ROOT=${PEDS_U18_EVAL_ROOT:-/temp_work/ch278233/eval_matrix/peds23_u18_${ARCH}_test}
fi
export RAC_R3D_ARCH=$ARCH

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
case "$TASK" in
  0) ARM=full_seed0 ;; 1) ARM=full_seed1 ;; 2) ARM=full_seed2 ;;
  3) ARM=ct_only_seed0 ;; 4) ARM=ct_only_seed1 ;; 5) ARM=ct_only_seed2 ;;
  6) ARM=c2_off ;; 7) ARM=c1_off ;; 8) ARM=indication_only ;;
  9) ARM=demographics_only ;; 10) ARM=metadata_none ;;
 11) ARM=anatomy_queries_off ;; 12) ARM=routing_off ;;
 13) ARM=region_align_off ;; 14) ARM=maskfree ;; 15) ARM=age_scalar ;;
 16) ARM=xattn_only ;; 17) ARM=film_only ;; 18) ARM=drop00 ;;
 19) ARM=drop10 ;; 20) ARM=drop40 ;; 21) ARM=no_augmentation ;;
 22) ARM=ctxcls_aux ;; 23) ARM=counterfactual005 ;;
 24) ARM=qwen_region_text ;; 25) ARM=naive_metadata_fusion ;;
 26) ARM=full_seed0; RAC_EVAL_IND_MODE=none ;;
 27) ARM=full_seed0; RAC_EVAL_IND_MODE=shuffled ;;
 28) ARM=full_seed1; RAC_EVAL_IND_MODE=none ;;
 29) ARM=full_seed1; RAC_EVAL_IND_MODE=shuffled ;;
30) ARM=full_seed2; RAC_EVAL_IND_MODE=none ;;
31) ARM=full_seed2; RAC_EVAL_IND_MODE=shuffled ;;
32) ARM=l2_on ;; 33) ARM=standard_negative ;;
34) ARM=ind_weight1 ;; 35) ARM=ind_weight4 ;;
36) ARM=one_global_query ;; 37) ARM=six_global_queries ;;
38) ARM=counterfactual015 ;; 39) ARM=general_selection ;;
40) ARM=pathology_class_logit ;;
  *) echo "FATAL unknown task $TASK"; exit 78 ;;
esac
export RAC_EVAL_IND_MODE=${RAC_EVAL_IND_MODE:-true}
export EVAL_CKPT=${RUN_ROOT}/${ARM}/CTClip.best.pt
TAG=${ARM}_ind-${RAC_EVAL_IND_MODE}
export EVAL_RESULTS_DIR=${EVAL_ROOT}/${TAG}
export RAC_DATA_ROOT=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_VALID=/temp_work/ch278233/${LIST_PREFIX}_TEST_labels.csv
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/${LIST_PREFIX}_VOLLIST_TEST.txt
export EVAL_BATCH_SIZE=8 EVAL_NUM_WORKERS=3 EVAL_BOOTSTRAP=2
mkdir -p "$EVAL_RESULTS_DIR"

$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
$E/bin/python "$ARCCT_ROOT/tools/evaluate.py"
rc=$?
if [ "$rc" = 0 ]; then
  $E/bin/python "$ARCCT_ROOT/tools/age_stratified_eval.py" \
    --pred "$EVAL_RESULTS_DIR/predictions.npz" \
    --csv "$EVAL_RESULTS_DIR/auc_by_age.csv" --ci \
    | tee "$EVAL_RESULTS_DIR/auc_by_age.txt"
  $E/bin/python "$ARCCT_ROOT/tools/analyze_predictions_paper.py" \
    --pred "$EVAL_RESULTS_DIR/predictions.npz" \
    --out "$EVAL_RESULTS_DIR" --bootstrap 1000
  if [ -f "$EVAL_RESULTS_DIR/predictions_ctx_cls.npz" ]; then
    $E/bin/python "$ARCCT_ROOT/tools/analyze_predictions_paper.py" \
      --pred "$EVAL_RESULTS_DIR/predictions_ctx_cls.npz" \
      --out "$EVAL_RESULTS_DIR/ctx_cls_paper" --bootstrap 1000
  fi
fi
exit $rc

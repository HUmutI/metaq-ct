#!/bin/bash
#SBATCH --job-name=evP23Final
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/evP23Final_%A_%a.out
set -uo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
set -a; . "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"; set +a

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
export RAC_EVAL_IND_MODE=true
unset RAC_EVAL_METADATA_MODE
case "$TASK" in
  0) ARM=final_drop03_band ;;
  1) ARM=drop00 ;; 2) ARM=drop10 ;; 3) ARM=drop40 ;; 4) ARM=age_scalar ;;
  5) ARM=c2_off ;; 6) ARM=c1_off ;; 7) ARM=indication_only ;;
  8) ARM=demographics_only ;; 9) ARM=metadata_none ;;
 10) ARM=anatomy_queries_off ;; 11) ARM=routing_off ;;
 12) ARM=region_align_off ;; 13) ARM=fully_maskfree ;; 14) ARM=l2_on ;;
 15) ARM=standard_negative ;; 16) ARM=ctxcls_aux ;;
 17) ARM=counterfactual005 ;; 18) ARM=ind_weight1 ;; 19) ARM=ind_weight4 ;;
 20) ARM=qwen_region_text ;; 21) ARM=one_global_query ;;
 22) ARM=six_global_queries ;; 23) ARM=no_augmentation ;;
 24) ARM=final_drop03_band_seed1 ;; 25) ARM=final_drop03_band_seed2 ;;
 26) ARM=ct_only_seed0 ;; 27) ARM=ct_only_seed1 ;; 28) ARM=ct_only_seed2 ;;
 29) ARM=indication_present_train ;; 30) ARM=general_selection ;;
 31) ARM=final_drop03_band; RAC_EVAL_IND_MODE=none ;;
 32) ARM=final_drop03_band; RAC_EVAL_IND_MODE=shuffled ;;
 33) ARM=final_drop03_band_seed1; RAC_EVAL_IND_MODE=none ;;
 34) ARM=final_drop03_band_seed1; RAC_EVAL_IND_MODE=shuffled ;;
 35) ARM=final_drop03_band_seed2; RAC_EVAL_IND_MODE=none ;;
 36) ARM=final_drop03_band_seed2; RAC_EVAL_IND_MODE=shuffled ;;
 37) ARM=final_drop03_band; RAC_EVAL_METADATA_MODE=indication ;;
 38) ARM=final_drop03_band; RAC_EVAL_METADATA_MODE=demographics ;;
 39) ARM=final_drop03_band; RAC_EVAL_METADATA_MODE=none ;;
  *) echo "FATAL unknown task $TASK"; exit 78 ;;
esac
if [ -n "${PEDS_EVAL_ARM_OVERRIDE:-}" ]; then ARM=$PEDS_EVAL_ARM_OVERRIDE; fi
export RAC_EVAL_IND_MODE
if [ -n "${RAC_EVAL_METADATA_MODE:-}" ]; then export RAC_EVAL_METADATA_MODE; fi

export EVAL_CKPT=/temp_work/ch278233/runs/peds23_biltir_final/${ARM}/CTClip.best.pt
TAG=${ARM}_ind-${RAC_EVAL_IND_MODE}_meta-${RAC_EVAL_METADATA_MODE:-checkpoint}
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/peds23_biltir_final/${TAG}
if [ "${PEDS_FINAL_SMOKE:-0}" = 1 ]; then
  export EVAL_CKPT=/temp_work/ch278233/runs/peds23_biltir_final_smoke/${ARM}/CTClip.best.pt
  export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/peds23_biltir_final_smoke/${TAG}
  export RAC_VAL_LIMIT=16 EVAL_BOOTSTRAP=2 EVAL_NUM_WORKERS=0 EVAL_BATCH_SIZE=2
fi
export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_PEDS
export RAC_DATA_VALID=/temp_work/ch278233/EVAL_PEDS/mps_ct_npz/valid
export RAC_REPORTS_VALID=/temp_work/ch278233/EVAL_PEDS/ct_reports/valid_reports.csv
export RAC_LABELS_VALID=/temp_work/ch278233/PEDS23_GE5_VALID_labels_v2.csv
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS23_GE5_VOLLIST_VALID.txt
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8} EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-3} EVAL_BOOTSTRAP=${EVAL_BOOTSTRAP:-200}
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== final pediatric eval task=$TASK arm=$ARM ind=$RAC_EVAL_IND_MODE meta=${RAC_EVAL_METADATA_MODE:-checkpoint} $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
$E/bin/python "$ARCCT_ROOT/tools/evaluate.py"
rc=$?
if [ "$rc" = 0 ]; then
  $E/bin/python "$ARCCT_ROOT/tools/age_stratified_eval.py" \
    --pred "$EVAL_RESULTS_DIR/predictions.npz" \
    --csv "$EVAL_RESULTS_DIR/auc_by_age.csv" \
    | tee "$EVAL_RESULTS_DIR/auc_by_age.txt"
fi
echo "=== final pediatric eval rc=$rc $(date) ==="
exit $rc

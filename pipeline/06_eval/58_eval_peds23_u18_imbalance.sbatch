#!/bin/bash
#SBATCH --job-name=evP23Imb
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-26-3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/evP23Imb_%A_%a.out
set -euo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
set -a
. "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"
set +a

TASK=${SLURM_ARRAY_TASK_ID:?array task required}
case "$TASK" in
  0) ARM=asl ;; 1) ARM=bnm001 ;; 2) ARM=bnm003 ;; 3) ARM=bnm010 ;;
  4) ARM=asl_bnm001 ;; 5) ARM=asl_bnm003 ;; 6) ARM=asl_bnm010 ;;
  *) echo "FATAL unknown task $TASK"; exit 78 ;;
esac
RUN_ROOT=${PEDS_IMB_RUN_ROOT:-/temp_work/ch278233/runs/peds23_u18_imbalance}
EVAL_ROOT=${PEDS_IMB_EVAL_ROOT:-/temp_work/ch278233/eval_matrix/peds23_u18_imbalance_test}
export EVAL_CKPT=$RUN_ROOT/$ARM/CTClip.best.pt
export EVAL_RESULTS_DIR=$EVAL_ROOT/$ARM
export RAC_DATA_ROOT=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_VALID=/temp_work/ch278233/PEDS23_U18_TEST_labels.csv
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS23_U18_VOLLIST_TEST.txt
export RAC_EVAL_IND_MODE=true EVAL_BATCH_SIZE=8 EVAL_NUM_WORKERS=3 EVAL_BOOTSTRAP=2
mkdir -p "$EVAL_RESULTS_DIR"

"$E/bin/python" "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23
"$E/bin/python" "$ARCCT_ROOT/tools/evaluate.py"
"$E/bin/python" "$ARCCT_ROOT/tools/age_stratified_eval.py" \
  --pred "$EVAL_RESULTS_DIR/predictions.npz" \
  --csv "$EVAL_RESULTS_DIR/auc_by_age.csv" --ci \
  | tee "$EVAL_RESULTS_DIR/auc_by_age.txt"
"$E/bin/python" "$ARCCT_ROOT/tools/analyze_predictions_paper.py" \
  --pred "$EVAL_RESULTS_DIR/predictions.npz" \
  --out "$EVAL_RESULTS_DIR" --bootstrap 1000

#!/bin/bash
#SBATCH --job-name=evalmx
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
# The combined cohort loads a 25k-row reports CSV plus the region cache in
# every DataLoader worker, and 96G was killed at 75% of the evaluation.
#SBATCH --mem=220G
#SBATCH --output=/temp_work/ch278233/ts_logs/evalmx_%j.out
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$HOME/arc-ct:${PYTHONPATH:-}
set -a; . $HOME/arc-ct/configs/stage2_peds.env; set +a

# SET selects which validation cohort to score on. evaluate.py derives its npz
# dir and reports CSV from RAC_DATA_ROOT with fixed sub-paths, so each cohort is
# a symlink layout rather than an edit to the evaluator.
case "${SET:?SET gerekli}" in
  peds)
    export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_PEDS
    export RAC_LABELS_VALID=/temp_work/ch278233/PEDS_VALID_labels_v2.csv
    export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED ;;
  ctrate)
    # CT-RATE has no pediatric-schema masks and never did: mask-free is the
    # published adult protocol.
    #
    # All 27 classes are real here. Leaving the nine pediatric-only ones blank
    # would have meant never measuring them on adults at all - and the whole
    # pipeline for extracting them from reports already existed, having been run
    # over the 22,976 training reports. The 3,039 validation reports cost ten
    # minutes on four GPUs. Published labels still win on the shared 18, so the
    # adult reproducibility gate stays a like-for-like measurement.
    export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_CTRATE
    export RAC_LABELS_VALID=/temp_work/ch278233/CTRATE_VALID_labels27_full.csv
    export RAC_MASK_VALID=/nonexistent ;;
  combined)
    export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_COMBINED
    # LBL lets one model be scored against both label sets. Which set a model
    # is judged on is not neutral: a run trained on harmonised labels scored
    # against the mixed ones is being marked by its rival's marking scheme.
    # The pediatric side already showed this - V2 read 0.7974 on its own
    # labels and 0.7918 on V1's. Run both; trust the result only if the
    # ranking survives.
    export RAC_LABELS_VALID=${LBL:-/temp_work/ch278233/COMBINED_VALID_labels.csv}
    export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED ;;
  *) echo "bilinmeyen SET=$SET"; exit 2 ;;
esac

export EVAL_BATCH_SIZE=8
export EVAL_NUM_WORKERS=${EVAL_NUM_WORKERS:-3}
export EVAL_CKPT=${CKPT:?CKPT gerekli}
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/${SET}/${TAG:?TAG gerekli}
mkdir -p "$EVAL_RESULTS_DIR"

echo "=== SET=$SET TAG=$TAG $(date) ==="
$E/bin/python $HOME/arc-ct/tools/evaluate.py
rc=$?
echo "=== rc=$rc $(date) ==="
exit $rc

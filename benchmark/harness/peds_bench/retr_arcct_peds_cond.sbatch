#!/bin/bash
#SBATCH --job-name=pbRetrArcC
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --array=0-5
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/retr_arcct_cond_%A_%a.out
# Pediatric report<->CT retrieval for our own models, mirroring exactly how
# 46_eval_peds23_ge5.sbatch scored classification (same env, same checkpoints,
# same data root) and how 48_eval_ctrate_retrieval_c1.sbatch invoked retrieval.
# Tasks 0-2: CT-only seeds 0-2.  Tasks 3-5: Refined-C1 seeds 0-2.  Embedding =
# general (z_gen), the fair equal-input readout. Bootstrap is done downstream
# at patient level by retrieval_metrics.py, so EVAL_BOOTSTRAP=0 here.
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
set -a
. "$ARCCT_ROOT/configs/stage2_peds.env"
. "$ARCCT_ROOT/configs/stage2_peds_ctx.env"
set +a
TASK=${SLURM_ARRAY_TASK_ID:?}
SEED=$((TASK % 3)); MODE_ID=$((TASK / 3)); ARM_ID=1
export RAC_SCHEMA=peds23
export RAC_QFORMER_QUERIES=35
if [ "$ARM_ID" -eq 0 ]; then
  ARM=ct_only
  export RAC_USE_CONTEXT_QFORMER=0
  export EVAL_CKPT=/temp_work/ch278233/runs/peds23_ge5_ct_only_seed${SEED}/CTClip.best.pt
else
  ARM=refined_c1
  export RAC_USE_CONTEXT_QFORMER=1
  export RAC_CTX_C1=1 RAC_CTX_C2=0 RAC_CTX_FUSION=class_logit
  export RAC_CTX_SELFATTN=split RAC_AGE_MODE=scalar RAC_CTX_SEED=$SEED
  export EVAL_CKPT=/temp_work/ch278233/runs/peds23_ge5_refined_c1_seed${SEED}/CTClip.best.pt
fi
export RAC_EVAL_IND_MODE=true
export RAC_DATA_ROOT=/temp_work/ch278233/EVAL_PEDS
export RAC_LABELS_VALID=/temp_work/ch278233/PEDS23_GE5_VALID_labels_v2.csv
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192_NESTED
case "$MODE_ID" in 0) EMB=mixed ;; 1) EMB=conditioned ;; esac
export RAC_RETRIEVAL_EMBEDDING=$EMB
export EVAL_RESULTS_DIR=/temp_work/ch278233/eval_matrix/peds23_retrieval/${ARM}_seed${SEED}_${EMB}
export EVAL_RUN_NAME=${ARM}_seed${SEED}_${EMB}
export EVAL_BATCH_SIZE=8 EVAL_NUM_WORKERS=3 EVAL_BOOTSTRAP=0
mkdir -p "$EVAL_RESULTS_DIR"
[ -f "$EVAL_CKPT" ] || { echo "FATAL: checkpoint yok: $EVAL_CKPT"; exit 78; }
echo "=== peds retrieval arm=$ARM emb=$EMB seed=$SEED $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23 || exit 78
$E/bin/python "$ARCCT_ROOT/tools/eval_retrieval.py"
rc=$?
echo "=== arm=$ARM seed=$SEED rc=$rc $(date) ==="
exit $rc

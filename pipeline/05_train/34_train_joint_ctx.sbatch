#!/bin/bash
#SBATCH --job-name=jointctx
#SBATCH --partition=bch-gpu-pe,bch-gpu
#SBATCH --gres=gpu:large:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/jointctx_%A_%a.out
set -uo pipefail
# The model that actually gets presented: the winning configuration retrained on
# the combined 49,545-volume split, alongside ct_only on the SAME split.
#
# Running ct_only jointly is not redundant. The ctx2 ladder picks a winner on
# 7,001 pediatric volumes; in the combined split CT-RATE outnumbers pediatrics
# six to one, and a configuration that wins in one regime is not thereby a
# winner in the other. Without a joint baseline the headline number would have
# nothing to be a delta AGAINST except a pediatric-only run trained on a
# seventh of the data -- which would flatter the new model for reasons that have
# nothing to do with the indication.
#
#   array index 0,1,2  -> ct_only         seeds 0,1,2   the joint baseline
#   array index 3,4,5  -> $JOINT_RUNG     seeds 0,1,2   the winner from ctx2
#
# Submit:  sbatch --export=ALL,JOINT_RUNG=c2 --array=0-5%3 \
#              pipeline/05_train/34_train_joint_ctx.sbatch
#
# 320 GB of memory, not 96: the combined reports table and the region cache are
# copied into every DataLoader worker and 96 GB was killed at 75% of a previous
# joint run. RAC_NUM_WORKERS=4 for the same reason.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/tmp/jointctx_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; . "$ARCCT_ROOT/configs/stage2_peds_ctx.env"; set +a

TASK=${SLURM_ARRAY_TASK_ID:-0}
JOINT_RUNG=${JOINT_RUNG:?JOINT_RUNG gerekli -- ctx2 kazanani, orn. c2}
if [ "$TASK" -lt 3 ]; then RUNG=ct_only; else RUNG=$JOINT_RUNG; fi
export RAC_SEED=$((TASK % 3))
export RAC_CTX_SEED=$RAC_SEED

export RAC_USE_CONTEXT_QFORMER=1
export RAC_CTX_C1=1
export RAC_CTX_C2=1
export RAC_CTX_SELFATTN=split
export RAC_CTX_ZIND_COMBINE=mean
export RAC_CTX_FILM_EPS=0.2
export RAC_AGE_MODE=band
case "$RUNG" in
  ct_only)       export RAC_USE_CONTEXT_QFORMER=0 ;;
  concat)        export RAC_CTX_C1=0; export RAC_CTX_C2=0 ;;
  c1_xattn)      export RAC_CTX_C2=0; export RAC_CTX_FILM_EPS=0.0001 ;;
  c1_full)       export RAC_CTX_C2=0 ;;
  c2)            : ;;
  fusion_gated)  export RAC_CTX_FUSION=gated ;;
  isolation_off) export RAC_CTX_SELFATTN=none ;;
  age_scalar)    export RAC_AGE_MODE=scalar ;;
  *) echo "FATAL: bilinmeyen RUNG=$RUNG"; exit 2 ;;
esac

LADDER_TAG=${LADDER_TAG:-joint}
OUT=/temp_work/ch278233/runs/${LADDER_TAG}_${RUNG}_seed${RAC_SEED}
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT

# 24,000 updates and a longer warmup: seven times the data per epoch, so the
# pediatric schedule would see each adult volume a fraction as often.
export RAC_BATCH_SIZE=20
export RAC_TOTAL_UPDATES=24000
export RAC_WARMUP_UPDATES=1000
export RAC_VAL_EVERY=800
export RAC_SAVE_EVERY=800
export RAC_EARLY_STOP_PATIENCE=8
export RAC_NUM_WORKERS=4
export RAC_RESPECT_EMPTY_REGION=1

export RAC_DATA_TRAIN=/temp_work/ch278233/COMBINED_NPZ
export RAC_DATA_VALID=/temp_work/ch278233/COMBINED_NPZ
# Pediatric masks only. CT-RATE has no training masks yet, so the adult half
# runs mask-free and masked/batch in the log reports how much of each batch was
# actually routed. When the CT-RATE masks land this becomes the ctx3 arm.
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/COMBINED_reports.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/COMBINED_reports.csv
# One labeller for both cohorts. The mixed-labeller version gave the same class
# two thresholds across the two halves -- Lymphadenopathy 25.9% adult against
# 7.9% pediatric, of which 10.8 points were the annotator rather than disease.
# The PUBLISHED CT-RATE labels stay the adult EVALUATION target, so the 0.8574
# gate remains a like-for-like measurement.
export RAC_LABELS_TRAIN=/temp_work/ch278233/COMBINED_labels_harmonised.csv
export RAC_LABELS_VALID=/temp_work/ch278233/COMBINED_labels_harmonised.csv
export RAC_REGION_CACHE=/temp_work/ch278233/COMBINED_region_cache_v2.json
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/COMBINED_VOLLIST_TRAIN_clean.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/COMBINED_VOLLIST_VALID_clean.txt
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds27.pt

echo "=== JOINT rung=$RUNG seed=$RAC_SEED task=$TASK on $(hostname) $(date) ==="
echo "    tag=$LADDER_TAG ctx=$RAC_USE_CONTEXT_QFORMER c1=$RAC_CTX_C1 c2=$RAC_CTX_C2 age=$RAC_AGE_MODE"
echo "    beta_init=$RAC_CTX_BETA_INIT lr_mult=$RAC_CTX_LR_MULT updates=$RAC_TOTAL_UPDATES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" \
    --expect-classes 27 --expect-schema peds || exit 78
PF_ARGS=""
[ "$RAC_USE_CONTEXT_QFORMER" = "0" ] && PF_ARGS="--skip-slow"
$E/bin/python $ARCCT_ROOT/tools/preflight_context.py $PF_ARGS || exit 78

$E/bin/python $ARCCT_ROOT/tools/train_stage2.py
rc=$?
echo "=== JOINT rung=$RUNG seed=$RAC_SEED rc=$rc $(date) ==="
exit $rc

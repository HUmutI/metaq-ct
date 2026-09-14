#!/bin/bash
#SBATCH --job-name=pedCtx4
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/pedCtx4_%A_%a.out
set -uo pipefail

# Refined pediatric metadata experiment, two arms x three seeds:
#   0..2  class-specific residual logits with C1
#   3..5  the same head with C1+C2 patient-specific relevance
#
# The inherited ARC-CT image/text/Q-Former path is frozen, including BatchNorm
# statistics. Every class gate starts at exactly zero, so update 0 reproduces
# the normal CT-only prompt margin. Only the additive metadata pathway learns.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/pedctx4_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"

set -a
. "$ARCCT_ROOT/configs/stage2_peds.env"
. "$ARCCT_ROOT/configs/stage2_peds_ctx.env"
set +a

TASK=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((TASK % 3))
if [ "$TASK" -lt 3 ]; then
  RUNG=classlogit_c1
  export RAC_CTX_C2=0
else
  RUNG=classlogit_c2
  export RAC_CTX_C2=1
fi
if [ -n "${CTX4_RUN_TAG:-}" ]; then
  RUNG=${RUNG}_${CTX4_RUN_TAG}
fi
export RAC_SEED=$SEED
export RAC_CTX_SEED=$SEED
GEN_TAG=${CTX_GEN_TAG:-ctx4}

export RAC_USE_CONTEXT_QFORMER=1
export RAC_CTX_C1=1
export RAC_CTX_FUSION=class_logit
export RAC_CTX_FREEZE_BASE=1
export RAC_AGE_MODE=${CTX4_AGE_MODE:-scalar}
export RAC_CTX_SELFATTN=split
export RAC_CTX_FILM_EPS=0.2
export RAC_CTX_LR_MULT=20.0
export RAC_CTX_CLASS_GATE_WARMUP=${CTX4_GATE_WARMUP:-400}

# Pediatric indications are present for 99.6% of the cohort. The previous 30%
# dropout plus counterfactual invariance strongly regularised away the signal we
# are trying to measure. Keep a small missing-context exposure but remove the
# mismatched-indication penalty in this efficacy arm. Robustness remains a
# separate evaluation intervention (true/blank/shuffled).
export RAC_IND_DROPOUT=${CTX4_IND_DROPOUT:-0.10}
export RAC_CF_PROB=${CTX4_CF_PROB:-0.0}
export RAC_CF_WEIGHT=${CTX4_CF_WEIGHT:-0.0}
# A command-line ablation may strengthen supervision of the conditioned
# pathology tokens without changing the zero-start prediction gate. The
# default remains the preregistered 1.0 control.
export RAC_CTX_PTOK_WEIGHT=${CTX4_IND_WEIGHT:-1.0}

# The old runs peaked near update 1600 and then deteriorated. Validate twice as
# often and stop on the single-model metric; never select an ensemble.
export RAC_TOTAL_UPDATES=5000
export RAC_WARMUP_UPDATES=400
export RAC_VAL_EVERY=200
export RAC_SAVE_EVERY=200
export RAC_EARLY_STOP_PATIENCE=10
export RAC_EARLY_STOP_MIN_UPDATES=600

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN_clean.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID_clean.txt
# Build the new path on the completed, seed-matched pediatric CT-only model.
# arcct_seed0.peds27.pt is only the 27-class surgery/start checkpoint (its frozen
# validation AUC is ~0.731), not the ~0.79 trained baseline.
export RAC_WARM_START_CKPT=/temp_work/ch278233/runs/ctx2_ct_only_seed${SEED}/CTClip.best.pt
export RAC_RESULTS_DIR=/temp_work/ch278233/runs/${GEN_TAG}_${RUNG}_seed${SEED}
if [ "${CTX4_SMOKE:-0}" = 1 ]; then
  export RAC_RESULTS_DIR=/temp_work/ch278233/runs/${GEN_TAG}_smoke_${RUNG}_seed${SEED}
  export RAC_TRAIN_LIMIT=40
  export RAC_VAL_LIMIT=32
  export RAC_BATCH_SIZE=2
  export RAC_NUM_WORKERS=0
  export RAC_TOTAL_UPDATES=2
  export RAC_WARMUP_UPDATES=1
  export RAC_VAL_EVERY=2
  export RAC_SAVE_EVERY=2
  export RAC_EARLY_STOP_PATIENCE=0
fi
mkdir -p "$RAC_RESULTS_DIR"

echo "=== PED CTX4 rung=$RUNG seed=$SEED task=$TASK $(date) ==="
echo "    fusion=$RAC_CTX_FUSION freeze_base=$RAC_CTX_FREEZE_BASE c2=$RAC_CTX_C2"
echo "    dropout=$RAC_IND_DROPOUT cf=$RAC_CF_WEIGHT ptok_ind=$RAC_CTX_PTOK_WEIGHT gate_warmup=$RAC_CTX_CLASS_GATE_WARMUP"

$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 27 --expect-schema peds || exit 78
$E/bin/python "$ARCCT_ROOT/tools/preflight_context.py" || exit 78
$E/bin/python "$ARCCT_ROOT/tools/train_stage2.py"
rc=$?
echo "=== PED CTX4 rung=$RUNG seed=$SEED rc=$rc $(date) ==="
exit $rc

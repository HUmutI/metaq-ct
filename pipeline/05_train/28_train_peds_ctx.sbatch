#!/bin/bash
#SBATCH --job-name=ctxrung
# Both GPU partitions. bch-gpu alone was estimating a start a WEEK out while
# bch-gpu-pe could start in minutes; letting SLURM take whichever frees
# first is the difference between results tomorrow and results next week.
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
# gpu:large, not gpu:1. Adding bch-gpu-pe to the partition list brought its
# gpu:medium nodes with it, and a 24 GB TITAN RTX cannot hold this recipe --
# ct_only seed=1 landed on gpu-1-0 and OOMed at 22.4 of 23.5 GiB. bch-gpu had
# only large and xlarge cards, so this never came up before tonight.
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0
# gpu:large excluded the A100s. They are gpu:xlarge, so a gpu:large request can
# never match one, and bch-gpu-xlarge was not in the partition list either --
# 24 A100s, the pool this account contends least for, were unreachable twice
# over. gpu:large was chosen to avoid the 24 GB TITANs, which OOM; those are
# gpu-1-0 and gpu-2-0 and they are the ONLY medium nodes across all three
# partitions, so excluding them by name does the same job without the cost.
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/ctxrung_%A_%a.out
set -uo pipefail
# The Phase 1 ablation ladder, one array task per (rung x seed).
#
# The ladder is the design's, not an invention here: the study the advisor cited
# found simple concatenation BEATING cross-attention on 3D chest CT at modest
# data scale, and our pediatric training split is 7,042 volumes. So the cheap
# fusion has to be measured before the expensive one is justified.
#
#   0  ct_only         today's model, context flag OFF -- the baseline
#   1  concat          indication as extra context tokens, no C1, no C2
#   2  c1_xattn        + cross-attention conditioning, FiLM off
#   3  c1_full         + FiLM channel gating          <- C1 complete
#   4  c2              + relevance weighting          <- C1+C2, the headline
#   5  fusion_gated    Z_gen + g*Z_ind instead of the concat fusion
#   6  isolation_off   self-attention group mask REMOVED -- measures the leak
#   7  age_scalar      scalar age instead of the ordinal band
#
# Rung 6 is deliberately the unsafe configuration. If it scores BETTER than
# rung 4, the reason is the leak, and that is a result worth having measured
# rather than argued about.
#
# Submit:  sbatch --array=0-23%6 pipeline/05_train/28_train_peds_ctx.sbatch
#          (8 rungs x 3 seeds; %6 caps concurrency so the queue stays usable)

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/tmp/ctxrung_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p "$TMPDIR"
# The first context rungs OOMed with 14.7 GB reserved-but-unallocated -- that is
# fragmentation, not a real memory shortfall, and the allocator's own error
# message names the fix. The 67-slot bank plus the counterfactual pass allocate
# in a pattern the default caching allocator handles badly.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; . "$ARCCT_ROOT/configs/stage2_peds_ctx.env"; set +a

TASK=${SLURM_ARRAY_TASK_ID:-0}
RUNGS=(ct_only concat c1_xattn c1_full c2 fusion_gated isolation_off age_scalar)
RUNG=${RUNGS[$((TASK / 3))]}
export RAC_SEED=$((TASK % 3))
export RAC_CTX_SEED=$RAC_SEED

# Defaults for every rung; each case below turns things OFF, never on, so a
# missing case is a full-strength run rather than a silently crippled one.
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
  c2)            : ;;                       # C1 + C2, the headline configuration
  fusion_gated)  export RAC_CTX_FUSION=gated ;;
  isolation_off) export RAC_CTX_SELFATTN=none ;;
  age_scalar)    export RAC_AGE_MODE=scalar ;;
esac

# LADDER_TAG keeps successive ladders apart. The first one (tag "ctx") ran with
# the context modules effectively switched off -- beta inert at 0.0025 and the
# conditioned half of the fusion at 2.7% of the general half -- so it is not a
# failed run to be overwritten but a measurement in its own right: the
# ARCHITECTURE'S contribution with the indication pathway closed. Keeping it is
# what lets the next ladder attribute its gain to the indication rather than to
# the wider bank.
LADDER_TAG=${LADDER_TAG:-ctx}
OUT=/temp_work/ch278233/runs/${LADDER_TAG}_${RUNG}_seed${RAC_SEED}
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json
# The _clean lists have the 54 report-contaminated volumes removed: 41 from
# train, 13 from valid. Every rung uses the SAME lists, or the arms are being
# scored on different denominators and cannot be compared. RAC_VOLUME_EXCLUDE
# stays set as well -- the list is what runs, the env var is the guard that
# catches a run that forgot.
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN_clean.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID_clean.txt
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds27.pt
# patience 5 x VAL_EVERY 400 = 2,000 updates, and the first ladder stopped at
# 3,200-4,000 of 6,000 with its best at 1,200-2,000 -- exactly on that boundary.
# The conditioning starts at zero and needs training time to become anything, so
# stopping it 2,000 updates after a best set while the gates were still closed
# measures the wrong thing. Applied to EVERY rung, including ct_only, so the
# ladder stays internally comparable.
export RAC_EARLY_STOP_PATIENCE=${RAC_EARLY_STOP_PATIENCE_OVERRIDE:-8}

echo "=== rung=$RUNG seed=$RAC_SEED task=$TASK on $(hostname) $(date) ==="
echo "    tag=$LADDER_TAG beta_init=$RAC_CTX_BETA_INIT lr_mult=$RAC_CTX_LR_MULT patience=$RAC_EARLY_STOP_PATIENCE"
echo "    ctx=$RAC_USE_CONTEXT_QFORMER c1=$RAC_CTX_C1 c2=$RAC_CTX_C2 "\
"selfattn=$RAC_CTX_SELFATTN age=$RAC_AGE_MODE film_eps=$RAC_CTX_FILM_EPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" \
    --expect-classes 27 --expect-schema peds || exit 78

# Preflight runs PER TASK, not once for the array: each task has its own
# RAC_RESULTS_DIR and its own manifest, and a task that starts from a different
# input set than its siblings is exactly what the manifest exists to catch.
# --skip-slow on rungs that do not build the context bank, where the identity
# gate has nothing to assert.
PF_ARGS=""
[ "$RAC_USE_CONTEXT_QFORMER" = "0" ] && PF_ARGS="--skip-slow"
$E/bin/python $ARCCT_ROOT/tools/preflight_context.py $PF_ARGS || exit 78

$E/bin/python $ARCCT_ROOT/tools/train_stage2.py
rc=$?
echo "=== rung=$RUNG seed=$RAC_SEED rc=$rc $(date) ==="
exit $rc

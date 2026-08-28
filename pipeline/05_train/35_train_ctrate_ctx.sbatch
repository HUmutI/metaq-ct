#!/bin/bash
#SBATCH --job-name=ctrctx
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0
# gpu:large excluded the A100s. They are gpu:xlarge, so a gpu:large request can
# never match one, and bch-gpu-xlarge was not in the partition list either --
# 24 A100s, the pool this account contends least for, were unreachable twice
# over. gpu:large was chosen to avoid the 24 GB TITANs, which OOM; those are
# gpu-1-0 and gpu-2-0 and they are the ONLY medium nodes across all three
# partitions, so excluding them by name does the same job without the cost.
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/ctrctx_%A_%a.out
set -uo pipefail
# The run that has to clear the adult gate: ARC-CT's architecture plus the
# indication/age/sex pathway, trained on CT-RATE ALONE, on the PUBLISHED 18-class
# labels, warm-started from the published adult checkpoint.
#
# No pediatric data. Our best joint attempt reaches 0.8442 against a 0.8574 gate,
# and the mix is a plausible cause: CT-RATE outnumbers pediatrics six to one in
# that split, but the pediatric half still pulls the shared representation toward
# a different age range and label distribution. Remove it and the architecture is
# the only thing that differs from the published model.
#
#   array index 0,1,2  -> ct_only        seeds 0,1,2   the control
#   array index 3,4,5  -> $CTR_RUNG      seeds 0,1,2   default c2 (C1+C2 on)
#
# The control is not optional and not a formality. 0.8574 came from THEIR training
# run; ours is a different one. If the context arm lands at 0.855 we could not say
# whether the architecture failed to help or our recipe simply does not reproduce
# ARC-CT. The control answers that: it must come back near 0.8574 for the context
# arm's delta to mean anything.
#
# Submit:  sbatch --array=0-5%3 pipeline/05_train/35_train_ctrate_ctx.sbatch
#          sbatch --export=ALL,CTR_RUNG=c1_full --array=3-5%3 ...   (one arm only)
#
# Checkpoint selection runs on 4,589 volumes held out of CT-RATE's TRAIN half.
# The official valid_* split is never trained on and never selected on -- it stays
# the held-out set the 0.8574 gate is measured on, and selecting there would turn
# the gate into a training signal.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/tmp/ctrctx_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}

# The ADULT recipe, unchanged. Sourced first so the context file cannot silently
# alter a hyperparameter the published number depends on.
CFG=$ARCCT_ROOT/configs/stage2.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; . "$ARCCT_ROOT/configs/stage2_ctrate_ctx.env"; set +a

export RAC_SCHEMA=ctrate

# An env file that is missing, empty, or half-sourced does not fail -- it leaves
# the code defaults in place, and the code defaults are RAC_CTX_BETA_INIT=-6.0 and
# RAC_CTX_LR_MULT=1.0: precisely the two settings that made the first pediatric
# ladder measure nothing at all. beta moved 0.00012 over an entire run at -6, and
# the zero-init modules never left zero on the warm-start rate. This file WAS
# emptied once, by a sync loop that redirected it onto itself, and the identity
# gate passed on the empty version without complaint because every default is
# individually legal.
for v in RAC_CTX_BETA_INIT RAC_CTX_LR_MULT RAC_CTX_FILM_EPS RAC_IND_DROPOUT \
         RAC_CF_PROB RAC_CTX_PTOK_WEIGHT RAC_CTX_SELFATTN RAC_FOCAL_CLASSES \
         RAC_QFORMER_QUERIES RAC_TOTAL_UPDATES RAC_BATCH_SIZE RAC_LR; do
  eval "val=\${$v:-}"
  [ -n "$val" ] || { echo "FATAL: $v bos -- config sourcelanmamis"; exit 78; }
done
[ "$RAC_CTX_BETA_INIT" = "-6.0" ] && { echo "FATAL: beta_init=-6.0, olu gradyan"; exit 78; }
[ "$RAC_CTX_LR_MULT" = "1.0" ] && { echo "FATAL: lr_mult=1.0, sifir-init moduller ogrenmez"; exit 78; }
echo "    config dogrulandi: beta_init=$RAC_CTX_BETA_INIT lr_mult=$RAC_CTX_LR_MULT"

TASK=${SLURM_ARRAY_TASK_ID:-0}
CTR_RUNG=${CTR_RUNG:-c2}
if [ "$TASK" -lt 3 ]; then RUNG=ct_only; else RUNG=$CTR_RUNG; fi
export RAC_SEED=$((TASK % 3))
export RAC_CTX_SEED=$RAC_SEED

case "$RUNG" in
  ct_only)   export RAC_USE_CONTEXT_QFORMER=0 ;;
  concat)    export RAC_CTX_C1=0; export RAC_CTX_C2=0 ;;
  c1_xattn)  export RAC_CTX_C2=0; export RAC_CTX_FILM_EPS=0.0001 ;;
  c1_full)   export RAC_CTX_C2=0 ;;
  c2)        : ;;
  *) echo "FATAL: bilinmeyen RUNG=$RUNG"; exit 2 ;;
esac

OUT=/temp_work/ch278233/runs/ctrate_${RUNG}_seed${RAC_SEED}
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT

# Can this run actually write a checkpoint? The filesystem reported 158 TB free
# while the user quota was exhausted, so df says nothing useful here. Six runs
# started, one reached update 1,200 and a full validation (auc 0.8711), and then
# every one of them died -- one on "CTClip.1200.pt cannot be opened" and the rest
# on a cuDNN workspace failure downstream of the same cause. An hour of GPU each,
# to discover something a 2 GB write would have found in a second.
TESTF=$OUT/.quota_probe
if ! dd if=/dev/zero of="$TESTF" bs=1M count=2048 status=none 2>/dev/null; then
  rm -f "$TESTF"
  echo "FATAL: $OUT icine 2 GB yazilamiyor (kota?). Checkpoint 1.3 GB, bu kosu"
  echo "       ilk kayitta olurdu. df bos gosterse bile kota dolu olabilir."
  exit 78
fi
rm -f "$TESTF"
echo "    disk yazma testi gecti (2 GB)"

D=/temp_work/ch278233/CTRATE_ONLY
export RAC_DATA_TRAIN=/temp_work/ch278233/COMBINED_NPZ
export RAC_DATA_VALID=/temp_work/ch278233/COMBINED_NPZ
export RAC_VOLUME_LIST_TRAIN=$D/VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=$D/VOLLIST_VALID.txt
# PUBLISHED labels on both sides. Our 27-class labeller appears nowhere in this
# run: training on our labels and reporting against a published gate would be
# marking our own homework.
export RAC_LABELS_TRAIN=$D/labels18.csv
export RAC_LABELS_VALID=$D/labels18.csv
export RAC_REPORTS_TRAIN=$D/reports.csv
export RAC_REPORTS_VALID=$D/reports.csv
export RAC_REGION_CACHE=/temp_work/ch278233/COMBINED_region_cache_v2.json
# CT-RATE masks only; no pediatric mask is in scope here. Coverage is 67% of the
# training half until the 15,575 in flight land, and the log's masked/batch line
# reports what each batch actually got.
export RAC_MASK_TRAIN=/temp_work/ch278233/COMBINED_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/COMBINED_MASKS10_192
# The PUBLISHED adult checkpoint, not the pediatric re-head. queries is (30,768)
# there and the context bank appends exactly one row.
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.pt

export RAC_NUM_WORKERS=4
export RAC_RESPECT_EMPTY_REGION=1

echo "=== CTRATE-ONLY rung=$RUNG seed=$RAC_SEED task=$TASK on $(hostname) $(date) ==="
echo "    schema=$RAC_SCHEMA queries=$RAC_QFORMER_QUERIES ctx=$RAC_USE_CONTEXT_QFORMER"
echo "    c1=$RAC_CTX_C1 c2=$RAC_CTX_C2 beta_init=$RAC_CTX_BETA_INIT lr_mult=$RAC_CTX_LR_MULT"
echo "    updates=$RAC_TOTAL_UPDATES batch=$RAC_BATCH_SIZE patience=$RAC_EARLY_STOP_PATIENCE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Mask coverage first. RAC_REQUIRE_MASK=1 makes an incomplete mask set a
# SMALLER COHORT rather than an error, and that is invisible in the result.
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_mask_coverage.py \
    --vollist "$RAC_VOLUME_LIST_TRAIN" --masks "$RAC_MASK_TRAIN" \
    --min "${MIN_MASK_COV:-99}" || exit 78
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" \
    --expect-classes 18 --expect-schema ctrate || exit 78
PF_ARGS=""
[ "$RAC_USE_CONTEXT_QFORMER" = "0" ] && PF_ARGS="--skip-slow"
$E/bin/python $ARCCT_ROOT/tools/preflight_context.py $PF_ARGS || exit 78

$E/bin/python $ARCCT_ROOT/tools/train_stage2.py
rc=$?
echo "=== CTRATE-ONLY rung=$RUNG seed=$RAC_SEED rc=$rc $(date) ==="
exit $rc

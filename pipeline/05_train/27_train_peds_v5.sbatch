#!/bin/bash
#SBATCH --job-name=pedtrn5
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/train_peds5_%j.out
set -uo pipefail
# --requeue: if the node is drained or the job is preempted SLURM restarts it
# SLURM restarts it and it picks up the last checkpoint. Worth having even for a
# few-hour run - node maintenance does not ask first.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}   # some nodes lack CXXABI_1.3.15
export TMPDIR=/tmp/pedtrain5_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; set +a

OUT=/temp_work/ch278233/runs/peds_finetune_v5
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT
# V3 differs from V2 in exactly one thing. When the region cache has an
# entry for a volume but an organ's sentence list is empty, that is the
# LLM's verdict - the report says nothing about this organ - and V3
# honours it. V2 keeps the old behaviour: fall through to keyword
# matching, which returns the whole findings text truncated to 400 chars
# when no keyword hits, so the organ is aligned to the entire report.
# 7,815 of 88,170 pediatric (volume, organ) pairs are affected - 8.9%.
export RAC_RESPECT_EMPTY_REGION=1
# V5 = V3 + the multi-scale feature pyramid. This is the real test of the
# resolution hypothesis. Connecting top-K pooling (V4) moved Lung nodule
# by +0.0028 against a band of 0.0118 - nothing - which argues the problem
# is not HOW the tokens are pooled but that the token is too big: at 12^3
# one token covers 24x16x16 mm and a 5 mm nodule is inside a single one.
# Multi-scale gives a 24^3 grid, so 12x8x8 mm per token, 8x as many.
# ms_project is zero-initialised as a pure layer4 pass-through, so this
# starts as an exact no-op and learns to use layer3 from there.
export RAC_USE_MULTISCALE=1
# The 24^3 grid is 8x the tokens and OOMed at batch 20 on a 44 GiB card
# (tried to allocate 5.27 GiB with 1.97 GiB free). Cutting the batch alone
# would confound the comparison with V3, so hold the EFFECTIVE batch at 20
# and pay for it in accumulation steps: 4 x 5 = 20, identical optimisation,
# fewer activations alive at once.
export RAC_BATCH_SIZE=5
export RAC_ACCUM_STEPS=4
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds27.pt
# train_stage2.py resumes on its own: it scans RAC_RESULTS_DIR for an existing
# checkpoint and picks up from it, so a requeue needs no extra wiring.

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27_V2/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27_V2/labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27_V2/region_cache.json
# train and valid read the SAME folder, so the split MUST come from these lists.
# Without them the two datasets are identical and val AUC measures nothing.
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID.txt

echo "=== $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "schema=$RAC_SCHEMA queries=$RAC_QFORMER_QUERIES require_mask=$RAC_REQUIRE_MASK"
echo "train list: $(wc -l < $RAC_VOLUME_LIST_TRAIN)  valid list: $(wc -l < $RAC_VOLUME_LIST_VALID)"

$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" --expect-classes 27 --expect-schema peds || exit 78
$E/bin/python $ARCCT_ROOT/tools/train_stage2.py
rc=$?
echo "=== TRAIN rc=$rc $(date) ==="
# exit with the trainer's code, not the echo's. Without this SLURM logs
# COMPLETED for a crashed run and --requeue never fires.
exit $rc

#!/bin/bash
#SBATCH --job-name=pedv1s2
# Both GPU partitions. bch-gpu alone was estimating a start a WEEK out while
# bch-gpu-pe could start in minutes; letting SLURM take whichever frees
# first is the difference between results tomorrow and results next week.
#SBATCH --partition=bch-gpu-pe,bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/train_peds_v1s2_%j.out
set -uo pipefail
# --requeue: if the node is drained or the job is preempted SLURM restarts it
# SLURM restarts it and it picks up the last checkpoint. Worth having even for a
# few-hour run - node maintenance does not ask first.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}   # some nodes lack CXXABI_1.3.15
export TMPDIR=/tmp/pedv1s2_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; set +a

OUT=/temp_work/ch278233/runs/peds_finetune_v1_seed2
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT
# Phase 0/8: the third seed of the V1 recipe. Seeds 0 and 1 are already on
# disk (runs/peds_finetune, runs/peds_finetune_v1_seed1), so this single run
# completes the triplet and gives sigma with a usable degree of freedom.
#
# sigma decides how many seeds each rung of the ablation ladder needs: at
# sigma = 0.002 a delta of 0.005 is resolvable at 80% power with 3 seeds per
# arm, at sigma = 0.005 it takes 16. Nothing about the Phase 1 code depends on
# this number, so the run is queued now and read when the ladder starts.
#
# Everything is identical to the 0.7961 seed-0 run except RAC_SEED.
export RAC_SEED=2
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds27.pt
# train_stage2.py resumes on its own: it scans RAC_RESULTS_DIR for an existing
# checkpoint and picks up from it, so a requeue needs no extra wiring.

export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json
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

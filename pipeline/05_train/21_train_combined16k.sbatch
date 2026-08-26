#!/bin/bash
#SBATCH --job-name=comb16k
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/train_comb16k_%j.out
set -uo pipefail
# Joint pediatric + adult run. One GPU on purpose: train_stage2.py has no DDP or
# DataParallel anywhere, so a second card would sit idle.
#
# A40/L40S rather than A100: the a100-80g nodes sit in bch-compute, where our
# QOS refuses GPU jobs (QOSMaxGRESPerJob), and bch-gpu has no xlarge nodes. So
# 46 GB, which fixes the batch size at the same 20 the pediatric run uses.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/tmp/comb16k_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
CFG=$ARCCT_ROOT/configs/stage2_peds.env
[ -r "$CFG" ] || { echo "FATAL: $CFG okunamiyor"; exit 78; }
set -a; . "$CFG"; set +a

# Same batch as the pediatric run - 46 GB does not allow more, and keeping it
# equal makes the two runs comparable step for step.
export RAC_BATCH_SIZE=20
export RAC_TOTAL_UPDATES=12000
export RAC_VAL_EVERY=800
export RAC_SAVE_EVERY=800
export RAC_WARMUP_UPDATES=1000

OUT=/temp_work/ch278233/runs/combined_16k
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds27.pt

export RAC_DATA_TRAIN=/temp_work/ch278233/COMBINED_NPZ
export RAC_DATA_VALID=/temp_work/ch278233/COMBINED_NPZ
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/COMBINED_reports.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/COMBINED_reports.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/COMBINED_labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/COMBINED_labels.csv
export RAC_REGION_CACHE=/temp_work/ch278233/BCH_DATASET/LABELS27/region_cache.json
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/COMBINED_VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/COMBINED_VOLLIST_VALID.txt
# Adult volumes have no pediatric-schema mask, so a hard require would drop all
# of them. The pediatric half still gets masked; masked/batch in the log shows
# the actual fraction.
export RAC_REQUIRE_MASK=0

echo "=== $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "schema=$RAC_SCHEMA queries=$RAC_QFORMER_QUERIES batch=$RAC_BATCH_SIZE require_mask=$RAC_REQUIRE_MASK"
echo "train: $(wc -l < $RAC_VOLUME_LIST_TRAIN)  valid: $(wc -l < $RAC_VOLUME_LIST_VALID)"

$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" --expect-classes 27 --expect-schema peds || exit 78
$E/bin/python $ARCCT_ROOT/tools/train_stage2.py
rc=$?
echo "=== rc=$rc $(date) ==="
# The echo used to be the last command, so the script always exited 0 and SLURM
# recorded COMPLETED for a run that had crashed 13 minutes in on a corrupt npz.
# That is worse than a plain failure: --requeue never fires, sacct reads clean,
# and anything chained with afterok launches on a dead run.
exit $rc

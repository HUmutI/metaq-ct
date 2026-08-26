#!/bin/bash
#SBATCH --job-name=pedv1s1
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/train_peds_v1s1_%j.out
set -uo pipefail
# --requeue: if the node is drained or the job is preempted SLURM restarts it
# SLURM restarts it and it picks up the last checkpoint. Worth having even for a
# few-hour run - node maintenance does not ask first.

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}   # some nodes lack CXXABI_1.3.15
export TMPDIR=/tmp/pedv1s1_$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export PYTHONPATH=$HOME/arc-ct:${PYTHONPATH:-}

set -a; . $HOME/arc-ct/configs/stage2_peds.env; set +a

OUT=/temp_work/ch278233/runs/peds_finetune_v1_seed1
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT
# Noise control. V1 labels, V1 region cache, everything identical to the
# 0.7961 run except the seed. Needed because V2 is NOT a clean one-variable
# comparison after all: the region routing also moved between the two
# extractions - 8,107 of 88,170 (volume, region) pairs, 9.19% - purely from
# vLLM batch-composition nondeterminism, since the region prompt never
# changed. The drift is symmetric (14,564 indices only in V1 vs 14,206 only
# in V2, empty-entry rate 8.86% vs 8.87%), so it is noise rather than a
# shift - but it means a small V2-V1 gap cannot be attributed to the labels
# without knowing the run-to-run band. This run measures that band.
export RAC_SEED=1
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

$E/bin/python $HOME/arc-ct/tools/train_stage2.py
rc=$?
echo "=== TRAIN rc=$rc $(date) ==="
# exit with the trainer's code, not the echo's. Without this SLURM logs
# COMPLETED for a crashed run and --requeue never fires.
exit $rc

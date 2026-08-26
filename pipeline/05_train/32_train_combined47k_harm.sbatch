#!/bin/bash
#SBATCH --job-name=c47harm
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
# 96G was not enough and the job was killed by the OOM killer 37 minutes in,
# during its first validation. The dataset is not the problem - the npz are
# read lazily - the in-memory tables are: a 155 MB region cache and an 85 MB
# reports CSV become multi-GB Python dicts, and every DataLoader worker ends
# up with its own copy once refcounting un-shares the forked pages. With one
# main process, 6 train workers and 4 validation workers alive at the same
# time that is 11 copies. The 16k run fit because its tables were half the
# size and its validation set half as large.
#SBATCH --mem=320G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/train_c47harm_%j.out
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
export PYTHONPATH=$HOME/arc-ct:${PYTHONPATH:-}

set -a; . $HOME/arc-ct/configs/stage2_peds.env; set +a

# Same batch as the pediatric run - 46 GB does not allow more, and keeping it
# equal makes the two runs comparable step for step.
export RAC_BATCH_SIZE=20
export RAC_TOTAL_UPDATES=24000
export RAC_VAL_EVERY=800
export RAC_SAVE_EVERY=800
export RAC_WARMUP_UPDATES=1000

OUT=/temp_work/ch278233/runs/combined_47k_harmonised
mkdir -p "$OUT"
export RAC_RESULTS_DIR=$OUT
# The 16k run measured 0.7843 on the pediatric split against 0.7961 for
# pediatric-only training - adding adults HURT pediatrics. That run was
# trained, and its checkpoint selected, on a validation set where all
# 16,400 adults were labelled 0 for nine classes they actually have
# (bone lesions in 21.9% of them). This run uses the extracted adult
# labels instead, so it is the test of whether the harm was the adults
# or the fabricated labels.
export RAC_RESPECT_EMPTY_REGION=1
# Fewer copies of those tables. Throughput is bound by the GPU here, not
# by npz decoding, so 4 workers loses very little.
export RAC_NUM_WORKERS=4
export RAC_REGION_CACHE=/temp_work/ch278233/COMBINED_region_cache_v2.json
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.peds27.pt

export RAC_DATA_TRAIN=/temp_work/ch278233/COMBINED_NPZ
export RAC_DATA_VALID=/temp_work/ch278233/COMBINED_NPZ
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/COMBINED_reports.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/COMBINED_reports.csv
# One labeller for both cohorts. The previous joint runs mixed two: our LLM on
# the pediatric half, CT-RATE's published RadBERT labels on the adult half, so
# the same class name carried two thresholds - Lymphadenopathy 25.9% adult vs
# 7.9% pediatric, of which 10.8 points were annotator, not disease. Switching
# the adult 18 to our LLM changes 3.9% of cells and closes the cross-cohort gap
# on 12 of 18 classes (Lymphadenopathy 18.1 -> 7.3 points), while leaving the
# genuinely epidemiological gaps open (Emphysema 18.8, median age 14 vs 47).
# The published labels stay the adult EVALUATION target, so the 0.8524 gate
# remains a like-for-like measurement.
export RAC_LABELS_TRAIN=/temp_work/ch278233/COMBINED_labels_harmonised.csv
export RAC_LABELS_VALID=/temp_work/ch278233/COMBINED_labels_harmonised.csv
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

$E/bin/python $HOME/arc-ct/tools/train_stage2.py
rc=$?
echo "=== rc=$rc $(date) ==="
# The echo used to be the last command, so the script always exited 0 and SLURM
# recorded COMPLETED for a run that had crashed 13 minutes in on a corrupt npz.
# That is worse than a plain failure: --requeue never fires, sacct reads clean,
# and anything chained with afterok launches on a dead run.
exit $rc

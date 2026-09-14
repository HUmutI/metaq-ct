#!/bin/bash
#SBATCH --job-name=p23AllS1
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/p23AllS1_%j.out
set -euo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export TMPDIR=/tmp/p23alls1_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# No age filter: 0.01--69 years in the current clean cohort.
export RAC_SCHEMA=peds23 RAC_R3D_ARCH=r2plus1d_18 RAC_SEED=31
export RAC_STAGE1_EPOCHS=8 RAC_STAGE1_BATCH_SIZE=8 RAC_STAGE1_LR=1e-4
export RAC_STAGE1_WEIGHT_DECAY=1e-4 RAC_STAGE1_LOSS=asl
export RAC_KINETICS_PRETRAINED=1 RAC_KINETICS_REQUIRED=1 RAC_AMP=1
export RAC_NUM_WORKERS=6 RAC_MAX_GRAD_NORM=1.0
export RAC_RANDOM_CROP=1 RAC_ENABLE_ANATOMY_FLIPS=1
export RAC_INTENSITY_JITTER_HU=30 RAC_JITTER_PER_CHANNEL=0
export RAC_GAUSSIAN_NOISE_SIGMA=0 RAC_HU_LUNG_LO=-1500 RAC_HU_LUNG_HI=500
export RAC_DATA_TRAIN=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_MASK_TRAIN=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_REPORTS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_REPORTS_VALID=$RAC_REPORTS_TRAIN
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=$RAC_LABELS_TRAIN
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN_clean.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID_clean.txt
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_STAGE1_RESULTS_DIR=/temp_work/ch278233/runs/peds23_allage_stage1_seed31
mkdir -p "$RAC_STAGE1_RESULTS_DIR"

echo "=== PEDS23 all-age Stage1 $(date) ==="
"$E/bin/python" "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23
"$E/bin/python" "$ARCCT_ROOT/tools/train_stage1.py"


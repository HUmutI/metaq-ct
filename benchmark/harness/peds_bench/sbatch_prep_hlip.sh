#!/bin/bash
#SBATCH --job-name=hlip_prep
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/hlip_prep_%A_%a.out
#SBATCH --error=/temp_work/ch278233/PEDS_BENCH/logs/hlip_prep_%A_%a.err

# VALIDATION SPLIT ONLY (the 5946 train volumes are deliberately NOT preprocessed:
# the benchmark table does not need them and disk is budgeted).
# Set VOLLIST to override, e.g. VOLLIST="<train.txt> <valid.txt>" sbatch ...
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

VOLLIST=${VOLLIST:-/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt}
MAXGB=${MAXGB:-40}

PY=/temp_work/ch278233/PEDS_BENCH/hlip/_venv/bin/python
SCRIPT=/home/ch278233/BENCHMARK/harness/peds_bench/prep_hlip.py

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT} vollist=${VOLLIST} cap=${MAXGB}GB"
srun $PY $SCRIPT \
  --vollist ${VOLLIST} \
  --volume-map /temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv \
  --save-dir /temp_work/ch278233/PEDS_BENCH/hlip/pt/ \
  --info-dir /temp_work/ch278233/PEDS_BENCH/hlip/info/ \
  --save-astype float16 \
  --spacing 3 1 1 \
  --max-total-gb ${MAXGB} \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards ${SLURM_ARRAY_TASK_COUNT}

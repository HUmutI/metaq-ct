#!/bin/bash
#SBATCH --job-name=pedmask
#SBATCH --partition=bch-compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/mask_%A_%a.out
set -uo pipefail
export TMPDIR=/temp_work/ch278233/tmp
export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
$HOME/micromamba/envs/arcct/bin/python $ARCCT_ROOT/pipeline/03_masks/build_peds_masks.py \
    --in-dir /temp_work/ch278233/TS_MASKS_PEDS \
    --schema 10 --out-dir /temp_work/ch278233/PEDS_MASKS10_192

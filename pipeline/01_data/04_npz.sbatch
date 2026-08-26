#!/bin/bash
#SBATCH --job-name=pednpz
#SBATCH --partition=bch-compute
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/npz_%A_%a.out
set -uo pipefail
export TMPDIR=/temp_work/ch278233/tmp
export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
$HOME/micromamba/envs/arcct/bin/python $HOME/pipeline/01_data/make_peds_npz.py

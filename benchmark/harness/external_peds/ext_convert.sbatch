#!/bin/bash
#SBATCH --job-name=extConv
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --array=0-11
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/temp_work/ch278233/EXTERNAL/logs/conv_%A_%a.out
set -uo pipefail
X=/temp_work/ch278233/EXTERNAL/PEDIATRIC_CT_SEG
export PATH=/home/ch278233/micromamba/envs/ct/bin:$PATH   # dcm2niix
export SHARD_ID=${SLURM_ARRAY_TASK_ID} SHARD_N=${SLURM_ARRAY_TASK_COUNT}
command -v dcm2niix >/dev/null || { echo "FATAL: dcm2niix yok"; exit 78; }
$X/venv/bin/python /home/ch278233/BENCHMARK/harness/external_peds/ext_convert_shard.py
echo "=== rc=$? ==="

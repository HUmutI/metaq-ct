#!/bin/bash
#SBATCH --job-name=extDice
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --array=0-7
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/temp_work/ch278233/EXTERNAL/logs/dice_%A_%a.out
set -uo pipefail
$HOME/micromamba/envs/arcct/bin/python \
  /home/ch278233/BENCHMARK/harness/external_peds/ext_dice.py \
  --shard-id ${SLURM_ARRAY_TASK_ID} --shard-n ${SLURM_ARRAY_TASK_COUNT}
echo "=== rc=$? ==="

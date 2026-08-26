#!/bin/bash
#SBATCH --job-name=ctfetch
#SBATCH --partition=bch-compute
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/ctfetch_%j.out
set -uo pipefail
export HF_HOME=/temp_work/ch278233/hf_cache
export TMPDIR=/temp_work/ch278233/tmp
export OMP_NUM_THREADS=8
$HOME/micromamba/envs/arcct/bin/python $HOME/pipeline/01_data/ctrate_fetch_npz.py \
    --batch 224 --workers 8 --cworkers 14 --stop-below-gb 250
echo "=== CTFETCH rc=$? ==="

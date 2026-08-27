#!/bin/bash
#SBATCH --job-name=subsetm
#SBATCH --partition=bch-compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/subsetm_%j.out
set -uo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds
echo "=== $(hostname) $(date) ==="
exec $E/bin/python $ARCCT_ROOT/tools/subset_metrics.py --n-boot 1000

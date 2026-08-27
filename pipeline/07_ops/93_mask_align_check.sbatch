#!/bin/bash
#SBATCH --job-name=maskchk
#SBATCH --partition=bch-compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/temp_work/ch278233/ts_logs/maskchk_%j.out
set -uo pipefail
# Runs mask_align_check.py on a compute node. Each check reads a full npz off
# NFS and the login node cannot carry it: three attempts there were killed
# mid-import, and OpenBLAS ran out of threads once several were racing.
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
echo "=== $(hostname) $(date) ==="
exec $E/bin/python $ARCCT_ROOT/pipeline/07_ops/mask_align_check.py

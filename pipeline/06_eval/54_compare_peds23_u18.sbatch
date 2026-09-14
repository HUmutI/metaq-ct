#!/bin/bash
#SBATCH --job-name=cmpP23U18
#SBATCH --partition=bch-compute
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/cmpP23U18_%j.out
set -euo pipefail
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct PYTHONPATH=$HOME/bch-arc-ct/tools:$HOME/bch-arc-ct:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 RAC_SCHEMA=peds23
$E/bin/python "$ARCCT_ROOT/tools/compare_peds23_u18.py"
$E/bin/python "$ARCCT_ROOT/tools/render_peds23_u18_tex.py"

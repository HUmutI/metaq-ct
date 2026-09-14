#!/bin/bash
#SBATCH --job-name=cmpP23Age
#SBATCH --partition=bch-compute
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/cmpP23Age_%j.out
set -euo pipefail
E=$HOME/micromamba/envs/arcct
export PYTHONPATH=$HOME/bch-arc-ct/tools:$HOME/bch-arc-ct:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 RAC_SCHEMA=peds23
$E/bin/python $HOME/bch-arc-ct/tools/compare_peds23_age_factorial.py

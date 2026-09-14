#!/bin/bash
#SBATCH --job-name=radP23U18
#SBATCH --partition=bch-compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/temp_work/ch278233/ts_logs/radP23U18_%j.out
set -euo pipefail
E=$HOME/micromamba/envs/arcct
export PYTHONPATH=$HOME/bch-arc-ct:${PYTHONPATH:-} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
$E/bin/python $HOME/bch-arc-ct/tools/prepare_peds23_u18_radiologist_review.py

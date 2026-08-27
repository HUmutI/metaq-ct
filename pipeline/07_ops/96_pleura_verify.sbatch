#!/bin/bash
#SBATCH --job-name=pleuraok
#SBATCH --partition=bch-compute
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/pleuraok_%j.out
set -uo pipefail
# Does the ribcage-derived pleural region remove the inversion the lung-derived
# shell had? Both are measured in one job, on the same 400-volume stratified
# sample, so the comparison cannot drift on sampling.
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds
echo "###### ESKI: akcigerden turetilen kabuk"
DIAG_MASKS=/temp_work/ch278233/COMBINED_MASKS10_192 DIAG_N=400 \
  $E/bin/python $ARCCT_ROOT/pipeline/07_ops/region_diagnose.py
echo
echo "###### YENI: kaburga kafesinden turetilen bosluk"
DIAG_MASKS=/temp_work/ch278233/TRIAL_PLEURA_CAVITY DIAG_N=400 \
  $E/bin/python $ARCCT_ROOT/pipeline/07_ops/region_diagnose.py

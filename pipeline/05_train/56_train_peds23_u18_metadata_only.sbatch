#!/bin/bash
#SBATCH --job-name=p23U18Meta
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/temp_work/ch278233/ts_logs/p23U18Meta_%A_%a.out
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct PYTHONPATH=$HOME/bch-arc-ct:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export RAC_SCHEMA=peds23 META_ONLY_SEEDS=0,1,2
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) export META_ONLY_MODE=indication ;;
  1) export META_ONLY_MODE=demographics ;;
  2) export META_ONLY_MODE=all ;;
  *) exit 78 ;;
esac
export META_ONLY_RESULTS=/temp_work/ch278233/runs/peds23_u18_metadata_only/$META_ONLY_MODE
$E/bin/python "$ARCCT_ROOT/tools/train_metadata_only.py"

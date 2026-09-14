#!/bin/bash
#SBATCH --job-name=rgRadGraph
#SBATCH --partition=bch-gpu-pe,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-1,gpu-b11-2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --output=/temp_work/ch278233/ts_logs/rgRadGraph_%A_%a.out
set -euo pipefail

E=$HOME/micromamba/envs/radgraph
ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/temp_work/ch278233/hf_cache
TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) DIR=ct_only_seed1_true ;;
  1) DIR=refined_c1_seed1_true ;;
  2) DIR=refined_c1_seed1_none ;;
  3) DIR=refined_c1_seed1_shuffled ;;
  *) echo "unknown task $TASK"; exit 64 ;;
esac
SUFFIX=${REPORTGEN_SUFFIX:-}
"$E/bin/python" "$ROOT/tools/eval_report_radgraph.py" \
  --predictions "/temp_work/ch278233/reportgen/eval/${DIR}${SUFFIX}/predictions.json"

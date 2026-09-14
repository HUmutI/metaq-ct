#!/bin/bash
#SBATCH --job-name=rgEval
#SBATCH --partition=bch-gpu-pe,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-1,gpu-b11-2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=/temp_work/ch278233/ts_logs/rgEval_%A_%a.out
set -euo pipefail

E=$HOME/micromamba/envs/qwen
ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
MODEL=/temp_work/ch278233/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) ARM=ct_only; TOKEN_MODE=general; MODE=true ;;
  1) ARM=refined_c1; TOKEN_MODE=all; MODE=true ;;
  2) ARM=refined_c1; TOKEN_MODE=all; MODE=none ;;
  3) ARM=refined_c1; TOKEN_MODE=all; MODE=shuffled ;;
  *) echo "unknown task $TASK"; exit 64 ;;
esac
SUFFIX=${REPORTGEN_SUFFIX:-}
LIMIT_ARGS=()
if [ "${REPORTGEN_SMOKE:-0}" = 1 ]; then LIMIT_ARGS=(--limit 4 --num-beams 1 --max-new-tokens 64); fi
"$E/bin/python" "$ROOT/tools/eval_report_generator.py" \
  --cache /temp_work/ch278233/reportgen/cache/seed1_dev_${MODE}${SUFFIX} \
  --run /temp_work/ch278233/reportgen/runs/${ARM}_seed1${SUFFIX} \
  --base-model "$MODEL" --token-mode "$TOKEN_MODE" \
  --out /temp_work/ch278233/reportgen/eval/${ARM}_seed1_${MODE}${SUFFIX} \
  --batch-size 2 --workers 2 "${LIMIT_ARGS[@]}"

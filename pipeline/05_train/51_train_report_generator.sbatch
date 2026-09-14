#!/bin/bash
#SBATCH --job-name=rgQwen
#SBATCH --partition=bch-gpu-pe,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-1,gpu-b11-2
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/temp_work/ch278233/ts_logs/rgQwen_%A_%a.out
set -euo pipefail

E=$HOME/micromamba/envs/qwen
ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/temp_work/ch278233/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) ARM=ct_only; TOKEN_MODE=general ;;
  1) ARM=refined_c1; TOKEN_MODE=all ;;
  *) echo "unknown task $TASK"; exit 64 ;;
esac
SUFFIX=${REPORTGEN_SUFFIX:-}
LIMIT_ARGS=()
if [ "${REPORTGEN_SMOKE:-0}" = 1 ]; then
  LIMIT_ARGS=(--train-limit 8 --dev-limit 8 --epochs 1 --batch-size 1 --grad-accum 2 --eval-batches 2)
fi
OUT=/temp_work/ch278233/reportgen/runs/${ARM}_seed1${SUFFIX}
mkdir -p "$OUT"
"$E/bin/python" "$ROOT/tools/train_report_generator.py" \
  --train-cache /temp_work/ch278233/reportgen/cache/seed1_train_true${SUFFIX} \
  --dev-cache /temp_work/ch278233/reportgen/cache/seed1_dev_true${SUFFIX} \
  --model "$MODEL" --out "$OUT" --token-mode "$TOKEN_MODE" \
  --batch-size 2 --grad-accum 16 --epochs 3 --eval-batches 500 --workers 2 "${LIMIT_ARGS[@]}"

#!/bin/bash
#SBATCH --job-name=rgCache
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-1,gpu-b11-2
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=/temp_work/ch278233/ts_logs/rgCache_%A_%a.out
set -euo pipefail

E=$HOME/micromamba/envs/arcct
ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
set -a
. "$ROOT/configs/stage2.env"
. "$ROOT/configs/stage2_ctrate_ctx.env"
set +a

export RAC_SCHEMA=ctrate
export RAC_USE_CONTEXT_QFORMER=1
export RAC_QFORMER_QUERIES=30
export RAC_CTX_C1=1
export RAC_CTX_C2=0
export RAC_CTX_FUSION=class_logit
export RAC_AGE_MODE=scalar
export RAC_CTX_SELFATTN=split
export RAC_IND_DROPOUT=0

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) SPLIT=train; MODE=true; LIST=/temp_work/ch278233/CTRATE_ONLY/VOLLIST_TRAIN.txt ;;
  1) SPLIT=dev; MODE=true; LIST=/temp_work/ch278233/CTRATE_ONLY/VOLLIST_VALID.txt ;;
  2) SPLIT=dev; MODE=none; LIST=/temp_work/ch278233/CTRATE_ONLY/VOLLIST_VALID.txt ;;
  3) SPLIT=dev; MODE=shuffled; LIST=/temp_work/ch278233/CTRATE_ONLY/VOLLIST_VALID.txt ;;
  *) echo "unknown task $TASK"; exit 64 ;;
esac
LIMIT=${REPORTGEN_CACHE_LIMIT:-0}
SUFFIX=${REPORTGEN_CACHE_SUFFIX:-}
OUT=/temp_work/ch278233/reportgen/cache/seed1_${SPLIT}_${MODE}${SUFFIX}
rm -rf "$OUT.tmp"
mkdir -p "$OUT.tmp"

"$E/bin/python" "$ROOT/tools/cache_report_tokens.py" \
  --ckpt /temp_work/ch278233/runs/ctrate_refined_c1_seed1/CTClip.best.pt \
  --data /temp_work/ch278233/COMBINED_NPZ \
  --reports /temp_work/ch278233/CTRATE_ONLY/reports.csv \
  --labels /temp_work/ch278233/CTRATE_ONLY/labels18.csv \
  --volume-list "$LIST" \
  --context /temp_work/ch278233/CONTEXT/indication.csv \
  --demographics /temp_work/ch278233/CONTEXT/demographics.csv \
  --indication-mode "$MODE" --out "$OUT.tmp" --batch-size 6 --workers 4 \
  --limit "$LIMIT"
mv "$OUT.tmp" "$OUT"

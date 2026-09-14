#!/bin/bash
#SBATCH --job-name=p23Heatmap
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/p23Heatmap_%j.out
set -euo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
set -a
. "$ARCCT_ROOT/configs/stage2_peds_biltir_final.env"
set +a

export RAC_DATA_VALID=/temp_work/ch278233/PEDS_NPZ_NESTED
export RAC_REPORTS_VALID=/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv
export RAC_LABELS_VALID=/temp_work/ch278233/PEDS23_U18_TEST_labels.csv
export RAC_MASK_VALID=/temp_work/ch278233/PEDS_MASKS10_192
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_VOLUME_EXCLUDE=/temp_work/ch278233/LEAKAGE_EXCLUDE.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS23_U18_VOLLIST_TEST.txt

CKPT=${HEATMAP_CKPT:-/temp_work/ch278233/runs/peds23_u18/full_seed0/CTClip.best.pt}
PRED=${HEATMAP_PREDICTIONS:-/temp_work/ch278233/eval_matrix/peds23_u18_r2plus1d_18_test/full_seed0_ind-true/predictions.npz}
OUT=${HEATMAP_OUTPUT:-/temp_work/ch278233/PEDS_HEATMAPS/context_qformer_seed0}
MAX_CASES=${HEATMAP_MAX_CASES:-5}

ARGS=(
  --checkpoint "$CKPT"
  --output "$OUT"
  --predictions "$PRED"
  --max-cases "$MAX_CASES"
  --workers 2
  --dpi 220
)

if [ -n "${HEATMAP_CASE:-}" ]; then
  ARGS+=(--case "$HEATMAP_CASE")
fi

mkdir -p "$OUT"
echo "[heatmap] checkpoint=$CKPT"
echo "[heatmap] output=$OUT"
"$E/bin/python" "$ARCCT_ROOT/tools/visualize_peds_context_heatmaps.py" "${ARGS[@]}"

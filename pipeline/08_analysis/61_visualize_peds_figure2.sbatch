#!/bin/bash
#SBATCH --job-name=p23Fig2
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/p23Fig2_%j.out
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

NOVEL=${FIG2_NOVEL_CKPT:-/temp_work/ch278233/runs/peds23_u18/full_seed0/CTClip.best.pt}
CONTROL=${FIG2_CONTROL_CKPT:-/temp_work/ch278233/runs/peds23_u18/ct_only_seed0/CTClip.best.pt}
OUT=${FIG2_OUTPUT:-/temp_work/ch278233/PEDS_HEATMAPS/figure2_style_seed0}

mkdir -p "$OUT"
ARGS=(
  --novel-checkpoint "$NOVEL"
  --control-checkpoint "$CONTROL"
  --output "$OUT"
  --workers 2 --dpi 300
)
if [ -n "${FIG2_CASE:-}" ]; then
  ARGS+=(--case "$FIG2_CASE")
else
  ARGS+=(
    --case 'ped_01393_1.nii.gz|Cardiomegaly|Pleural effusion|Consolidation|Medical material'
    --case 'ped_02194_3.nii.gz|Medical material|Pleural effusion|Consolidation|Pulmonary metastases'
    --case 'ped_01687_1.nii.gz|Cardiomegaly|Lung opacity|Consolidation|Interlobular septal thickening'
    --case 'ped_01687_1.nii.gz|Cardiomegaly|Lung opacity|Pulmonary fibrotic sequela|Interlobular septal thickening'
  )
fi
"$E/bin/python" "$ARCCT_ROOT/tools/visualize_peds_figure2_gradcam.py" "${ARGS[@]}"

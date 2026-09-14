#!/bin/bash
#SBATCH --job-name=indOnly
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/temp_work/ch278233/ts_logs/indOnly_%j.out
set -uo pipefail

E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv
export RAC_LABELS_VALID=/temp_work/ch278233/PEDS_VALID_labels_v2.csv
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN_clean.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID_clean.txt
export RAC_CTX_MAX_IND_LEN=64
export RAC_IND_DROPOUT=0.10
export IND_ONLY_SEEDS=0,1,2
export IND_ONLY_RESULTS=/temp_work/ch278233/runs/indication_only_peds27
mkdir -p "$IND_ONLY_RESULTS"

echo "=== indication-only control $(date) ==="
$E/bin/python "$ARCCT_ROOT/tools/train_indication_only.py"
rc=$?
echo "=== indication-only rc=$rc $(date) ==="
exit $rc

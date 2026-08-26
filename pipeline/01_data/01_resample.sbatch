#!/bin/bash
#SBATCH --job-name=pedresamp
#SBATCH --partition=bch-compute
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/resamp_%A_%a.out
set -uo pipefail
# TotalSegmentator must see the 1.5x1.5x3.0 mm grid, because prepare_data's
# preprocess_mask pad/crops the mask without ever resampling by spacing. Feeding
# it a native 512x512x~1000 volume would centre-crop a sliver of chest, silently.
export TMPDIR=/temp_work/ch278233/tmp
N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-0}
LIST=$TMPDIR/resamp_chunk_${SLURM_JOB_ID}_$I.txt
awk -v n="$N" -v i="$I" 'NR % n == i' /temp_work/ch278233/all_nii.txt > "$LIST"
trap 'rm -f "$LIST"' EXIT
echo "=== shard $I/$N: $(wc -l < "$LIST") volumes on $(hostname) ==="
$HOME/micromamba/envs/arcct/bin/python $HOME/pipeline/01_data/resample_for_ts.py \
    --list "$LIST" --out-dir /temp_work/ch278233/TS_RESAMPLED
echo "=== shard $I done rc=$? ==="

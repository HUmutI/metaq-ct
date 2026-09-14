#!/bin/bash
#SBATCH --job-name=extResamp
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/EXTERNAL/logs/resamp_%A_%a.out
# Pediatric-CT-SEG -> 1.5x1.5x3.0 mm, using the SAME script that produced BCH's
# TS_RESAMPLED (pipeline/01_data/resample_for_ts.py: trilinear, HU untouched).
# Input list = chest-covering (CAP) NIfTIs from the download agent's index.
set -uo pipefail
X=/temp_work/ch278233/EXTERNAL/PEDIATRIC_CT_SEG
LIST=${EXT_LIST:-$X/lists/cap_nii.txt}
[ -s "$LIST" ] || { echo "FATAL: liste yok/bos: $LIST"; exit 78; }
mkdir -p $X/ts_resampled
SH=/tmp/claude-ext-shard-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.txt
awk -v s=${SLURM_ARRAY_TASK_ID} -v n=${SLURM_ARRAY_TASK_COUNT} 'NR%n==s' "$LIST" > $SH
echo "shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}: $(wc -l < $SH) hacim"
$HOME/micromamba/envs/arcct/bin/python $HOME/pipeline/01_data/resample_for_ts.py --list "$SH" --out-dir $X/ts_resampled
echo "=== rc=$? ==="; rm -f $SH

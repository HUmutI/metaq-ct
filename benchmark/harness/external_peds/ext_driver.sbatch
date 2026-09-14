#!/bin/bash
#SBATCH --job-name=extDrive
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/temp_work/ch278233/EXTERNAL/logs/driver_%j.out
# Runs after conversion: builds the inclusion list, then submits
# resample -> TotalSegmentator -> Dice as a dependency chain.
set -uo pipefail
X=/temp_work/ch278233/EXTERNAL/PEDIATRIC_CT_SEG
H=/home/ch278233/BENCHMARK/harness/external_peds
PY=$HOME/micromamba/envs/arcct/bin/python
$PY $H/ext_make_lists.py || exit 1
N=$(wc -l < $X/lists/cap_nii.txt)
echo "[driver] $N calisma dahil edildi"
[ "$N" -ge 50 ] || { echo "[driver] FATAL: cok az hacim ($N) - zincir baslatilmadi"; exit 1; }
J1=$(sbatch --parsable --export=ALL,EXT_LIST=$X/lists/cap_nii.txt $H/ext_resample.sbatch)
echo "[driver] resample=$J1"
# TS worklist is written by a tiny wrapper job so it sees the finished resample dir
J2=$(sbatch --parsable --dependency=afterany:$J1 --wrap "ls $X/ts_resampled | grep '\.nii\.gz$' | sort > $X/lists/ts_worklist.txt; wc -l < $X/lists/ts_worklist.txt" \
     --job-name=extWL --partition=bch-compute,bch-compute-pe --time=00:10:00 --mem=2G \
     --output=/temp_work/ch278233/EXTERNAL/logs/wl_%j.out)
echo "[driver] worklist=$J2"
J3=$(sbatch --parsable --dependency=afterok:$J2 --export=ALL,TS_WORKLIST=$X/lists/ts_worklist.txt $H/ext_ts.sbatch)
echo "[driver] totalsegmentator=$J3"
J4=$(sbatch --parsable --dependency=afterany:$J3 $H/ext_dice.sbatch)
echo "[driver] dice=$J4"
echo "[driver] zincir: $J1 -> $J2 -> $J3 -> $J4"

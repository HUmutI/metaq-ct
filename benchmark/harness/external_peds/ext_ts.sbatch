#!/bin/bash
#SBATCH --job-name=extTS
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --output=/temp_work/ch278233/EXTERNAL/logs/ts_%A_%a.out
# TotalSegmentator on the external cohort with the IDENTICAL call BCH used
# (pipeline/03_masks/ts_batch.py: task=total, ml=True, roi_subset=PEDS, same
# env, same weights). Mirrors 02_totalseg.sbatch line for line.
set -uo pipefail
X=/temp_work/ch278233/EXTERNAL/PEDIATRIC_CT_SEG
TS=/temp_work/ch278233/envs/totalseg
export ARCCT_ROOT=$HOME/pipeline
export LD_LIBRARY_PATH=$TS/lib:${LD_LIBRARY_PATH:-}
export TOTALSEG_HOME_DIR=/temp_work/ch278233/totalseg_home
export TOTALSEG_WEIGHTS_PATH=$TOTALSEG_HOME_DIR/nnunet/results
export nnUNet_compile=0
export TMPDIR=/tmp/extts_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $TMPDIR
LOCALW=$TMPDIR/weights; mkdir -p $LOCALW
cp -r "$TOTALSEG_WEIGHTS_PATH"/. "$LOCALW"/ && export TOTALSEG_WEIGHTS_PATH="$LOCALW"
nvidia-smi --query-gpu=name --format=csv,noheader
export TS_WORKLIST=${TS_WORKLIST:-}
export SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
mkdir -p $X/ts_masks
"$TS/bin/python" $ARCCT_ROOT/03_masks/ts_batch.py --in-dir $X/ts_resampled --out-dir $X/ts_masks
rc=$?; rm -rf $TMPDIR; echo "=== rc=$rc ==="; exit $rc

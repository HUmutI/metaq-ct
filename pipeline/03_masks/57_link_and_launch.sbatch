#!/bin/bash
#SBATCH --job-name=linklaunch
#SBATCH --partition=bch-compute
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/temp_work/ch278233/ts_logs/linklaunch_%j.out
set -uo pipefail
# The tail of the chain, split out from the stage-2 work so that work can be an
# array: link the new masks where the loader looks, re-check coverage, submit.
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
W=/temp_work/ch278233
BUILT=$(ls "$W/CTRATE_MASKS10_192/train" 2>/dev/null | wc -l)
echo "=== $(hostname) $(date)  2.asama ciktisi: $BUILT ==="
if [ "$BUILT" -lt 15000 ]; then
  echo "FATAL: yalnizca $BUILT maske uretilmis, zincir durdu"; exit 78
fi
n=0
for f in "$W/CTRATE_MASKS10_192/train"/*.nii.gz; do
  b=$(basename "$f")
  [ -e "$W/COMBINED_MASKS10_192/$b" ] || { ln -s "$f" "$W/COMBINED_MASKS10_192/$b" && n=$((n+1)); }
done
echo "--- $n yeni link"
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_mask_coverage.py \
    --vollist "$W/CTRATE_ONLY/VOLLIST_TRAIN.txt" \
    --masks   "$W/COMBINED_MASKS10_192" --min 99 || {
  echo "FATAL: kapsam esigin altinda, egitim BASLATILMADI"; exit 78; }
cd $HOME/pipeline && sbatch --array=0-5%3 05_train/35_train_ctrate_ctx.sbatch
echo "=== zincir tamam $(date) ==="

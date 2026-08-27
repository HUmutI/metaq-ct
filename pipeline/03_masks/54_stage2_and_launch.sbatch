#!/bin/bash
#SBATCH --job-name=maskchain
#SBATCH --partition=bch-compute
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/maskchain_%j.out
set -uo pipefail
# Runs when the TotalSegmentator array finishes: reduces the raw labels to the
# 10-region masks the loader consumes, links them where the loader looks, and
# only then submits the CT-RATE training array.
#
# Chained rather than done by hand because the segmentation finishes overnight and
# nothing should wait on someone being awake. Each step checks the previous one;
# a failure stops the chain instead of launching a run on a half-built mask set.
#
# PLEURA is read from a decision file. The ribcage-derived pleural region is being
# verified against the lung-derived one, and stage 2 is where that choice is
# applied -- the expensive segmentation above is identical either way. Absent the
# file, the current definition is used, because launching with the unverified one
# would be the worse mistake.
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
W=/temp_work/ch278233
DECIDE=$W/PLEURA_MODE.txt
PLEURA=lung
[ -r "$DECIDE" ] && PLEURA=$(tr -d '[:space:]' < "$DECIDE")
case "$PLEURA" in lung|cavity) ;; *) echo "FATAL: gecersiz PLEURA=$PLEURA"; exit 78 ;; esac

RAW=$(ls "$W/CTRATE_TS_RAW/train" 2>/dev/null | wc -l)
echo "=== $(hostname) $(date)  ham TS: $RAW  pleura=$PLEURA ==="
if [ "$RAW" -lt 15000 ]; then
  echo "FATAL: ham TS maskesi $RAW < 15000; segmentasyon tamamlanmamis, zincir durdu"
  exit 78
fi

echo "--- 2. asama: 10 bolgeye indirgeme"
$E/bin/python $ARCCT_ROOT/pipeline/03_masks/build_peds_masks.py \
    --in-dir  "$W/CTRATE_TS_RAW/train" \
    --out-dir "$W/CTRATE_MASKS10_192/train" \
    --schema 10 --no-map --pleura "$PLEURA"
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: 2. asama rc=$rc"; exit $rc; }

echo "--- loader'in baktigi duz dizine baglama"
n=0
for f in "$W/CTRATE_MASKS10_192/train"/*.nii.gz; do
  b=$(basename "$f")
  [ -e "$W/COMBINED_MASKS10_192/$b" ] || { ln -s "$f" "$W/COMBINED_MASKS10_192/$b" && n=$((n+1)); }
done
echo "    $n yeni link"

echo "--- kapsam kontrolu"
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_mask_coverage.py \
    --vollist "$W/CTRATE_ONLY/VOLLIST_TRAIN.txt" \
    --masks   "$W/COMBINED_MASKS10_192" --min 99 || {
  echo "FATAL: kapsam esigin altinda, egitim BASLATILMADI"; exit 78; }

echo "--- CT-RATE egitim dizisi gonderiliyor"
cd $HOME/pipeline && sbatch --array=0-5%3 05_train/35_train_ctrate_ctx.sbatch
echo "=== zincir tamam $(date) ==="

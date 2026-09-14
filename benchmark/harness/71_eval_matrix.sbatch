#!/bin/bash
#SBATCH --job-name=bmEvalMx
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/bmEvalMx_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=$BM_RUNS/eval
mkdir -p $O

# Selection uses ctrate_dev -- the 4,589 train-side studies -- and reporting uses
# ctrate_test, the untouched official 3,002.  Neither baseline validates during
# training, so without this step we would be comparing their last step against
# ARC-CT's selected best.  SEL_N fixes a prefix of dev; the same volumes are used
# for every candidate, so the comparison between checkpoints is apples-to-apples.
SEL_N=1500

mpsct () {  # $1=mode(ignored) $2=ckpt $3=tree $4=reports $5=labels $6=out
  PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT \
  $BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/70_eval.py \
    --model mpsct --mode zeroshot --ckpt "$2" --tree "$3" \
    --reports "$4" --labels "$5" --out "$6" --batch 8 --workers 8 ${7:-}
}
grf () {    # $1=mode $2=ckpt $3=tree $4=reports $5=labels $6=out
  $BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
    --model greenrfm --mode "$1" --ckpt "$2" --tree "$3" \
    --reports "$4" --labels "$5" --out "$6" --batch 8 --workers 8 ${7:-}
}
best_of () {  # prints "<macro> <ckpt>" of the highest macro_auroc among json files
  $BM_REPOS/envs/GreenRFM/bin/python - "$@" <<'PY'
import json,sys
rows=[]
for f in sys.argv[1:]:
    try:
        d=json.load(open(f))
        if d.get("macro_auroc") is not None: rows.append((d["macro_auroc"], d["ckpt"]))
    except Exception: pass
if rows:
    m,c=max(rows); print(f"{m:.6f} {c}")
PY
}

echo "################ A. selection on ctrate_dev (first $SEL_N) ################"
# MPS-CT: 31 checkpoints exist; a coarse grid of 10 is plenty to locate the peak.
for s in 3000 6000 9000 12000 15000 18000 21000 24000 27000 30000; do
  C=$BM_RUNS/mpsct_ctrate/attempt0/CTClip.$s.pt
  [ -f "$C" ] || continue
  echo "--- mpsct step $s ---"
  mpsct zeroshot "$C" $BM_DATA/ctrate_dev $CTR/reports.csv $CTR/labels18.csv \
        $O/sel_mpsct_$s.json "--limit $SEL_N"
done
for e in 1 2 3 4 5; do
  C=$BM_RUNS/greenrfm_ctrate/stage2_align/checkpoint_epoch_$e.pt
  [ -f "$C" ] || continue
  for m in zeroshot classifier; do
    echo "--- greenrfm epoch $e / $m ---"
    grf $m "$C" $BM_DATA/ctrate_dev $CTR/reports.csv $CTR/labels18.csv \
        $O/sel_grf_${m}_$e.json "--limit $SEL_N"
  done
done

BM=$(best_of $O/sel_mpsct_*.json      | cut -d' ' -f2)
BZ=$(best_of $O/sel_grf_zeroshot_*.json  | cut -d' ' -f2)
BC=$(best_of $O/sel_grf_classifier_*.json | cut -d' ' -f2)
echo "################ selected ################"
echo "  mpsct  zeroshot   : ${BM:-NONE}"
echo "  grf    zeroshot   : ${BZ:-NONE}"
echo "  grf    classifier : ${BC:-NONE}"

echo "################ B. official CT-RATE test (3,002) ################"
CTL=/temp_work/ch278233/CTRATE/multi_abnormality_labels/valid_predicted_labels.csv
CTRP=/temp_work/ch278233/CTRATE/ct_reports/valid_reports.csv
[ -n "${BM:-}" ] && mpsct zeroshot "$BM" $BM_DATA/ctrate_test $CTRP $CTL $O/TEST_mpsct_zeroshot.json
[ -n "${BZ:-}" ] && grf zeroshot   "$BZ" $BM_DATA/ctrate_test $CTRP $CTL $O/TEST_grf_zeroshot.json
[ -n "${BC:-}" ] && grf classifier "$BC" $BM_DATA/ctrate_test $CTRP $CTL $O/TEST_grf_classifier.json

echo "################ C. pediatric 23-class (1,507) ################"
PL=$BM_DATA/peds23_labels_valid.csv
PR=/temp_work/ch278233/COMBINED_reports.csv
# No clean pediatric selection split exists that the model did not train on, so
# the FINAL checkpoint is the primary number and no selection is performed here.
PG=$(ls -1 $BM_RUNS/greenrfm_peds23/stage2_align/checkpoint_epoch_*.pt 2>/dev/null | sort -V | tail -1)
[ -n "${PG:-}" ] && for m in zeroshot classifier; do
  grf $m "$PG" $BM_DATA/peds_valid $PR $PL $O/TEST_peds_grf_$m.json
done
# MPS-CT's pediatric fine-tune is still training; scoring a partial checkpoint
# here would put a half-trained number in the table. It is evaluated by
# 74_eval_mpsct_peds.sbatch, chained afterok on that training job.

echo "################ SUMMARY ################"
$BM_REPOS/envs/GreenRFM/bin/python $BM_HARNESS/73_summarise.py $O
echo "=== eval matrix done $(date) ==="

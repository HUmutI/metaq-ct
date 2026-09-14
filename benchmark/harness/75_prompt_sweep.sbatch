#!/bin/bash
#SBATCH --job-name=bmPrompt
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/temp_work/ch278233/ts_logs/bmPrompt_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=$BM_RUNS/eval/prompts; mkdir -p $O
N=1500

# GreenRFM zero-shot came out near chance (0.56 on a 64-volume probe) while its
# classifier head read 0.83 on the same volumes. Its loss and its own zero_shot
# are both explicitly un-normalised, so our scoring matches them -- this sweep
# checks the remaining cheap explanation, that the result is an artefact of one
# prompt pair rather than a property of the alignment.
for p in null p1 p3 p4; do
  echo "--- greenrfm zeroshot prompt=$p ---"
  $BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
    --model greenrfm --mode zeroshot --prompt $p \
    --ckpt $BM_RUNS/greenrfm_ctrate/stage2_align/checkpoint_epoch_5.pt \
    --tree $BM_DATA/ctrate_dev --reports $CTR/reports.csv --labels $CTR/labels18.csv \
    --limit $N --batch 8 --workers 8 --out $O/grf_$p.json
  echo "--- mpsct zeroshot prompt=$p ---"
  PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT \
  $BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/70_eval.py \
    --model mpsct --mode zeroshot --prompt $p \
    --ckpt $BM_RUNS/mpsct_ctrate/attempt0/CTClip.30000.pt \
    --tree $BM_DATA/ctrate_dev --reports $CTR/reports.csv --labels $CTR/labels18.csv \
    --limit $N --batch 8 --workers 8 --out $O/mpsct_$p.json
done
echo "=== prompt sweep summary (dev prefix $N) ==="
$BM_REPOS/envs/GreenRFM/bin/python - $O <<'PY'
import glob,json,os,sys
for f in sorted(glob.glob(os.path.join(sys.argv[1],"*.json"))):
    d=json.load(open(f)); m=d.get("macro_auroc")
    print(f"  {os.path.basename(f)[:-5]:18s} macro={('%.4f'%m) if m is not None else 'n/a':>7s}  ({d.get('n_classes_evaluable')}/{d.get('n_classes')} cls)")
PY
echo "=== prompt sweep done ==="

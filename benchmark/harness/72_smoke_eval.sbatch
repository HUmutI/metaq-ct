#!/bin/bash
#SBATCH --job-name=bmEvSmoke
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/bmEvSmoke_%j.out
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=$BM_RUNS/eval_smoke_$SLURM_JOB_ID
rc=0

echo "########## MPS-CT zero-shot (18-class, CT-RATE) ##########"
PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT \
$BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/70_eval.py \
  --model mpsct --mode zeroshot \
  --ckpt $BM_RUNS/mpsct_ctrate/attempt0/CTClip.30000.pt \
  --tree $BM_DATA/ctrate_dev --reports $CTR/reports.csv --labels $CTR/labels18.csv \
  --limit 64 --batch 4 --out $O/mpsct.json || rc=1

echo "########## GreenRFM zero-shot + classifier (18-class, CT-RATE) ##########"
for m in zeroshot classifier; do
  $BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
    --model greenrfm --mode $m \
    --ckpt $BM_RUNS/greenrfm_ctrate/stage2_align/checkpoint_epoch_5.pt \
    --tree $BM_DATA/ctrate_dev --reports $CTR/reports.csv --labels $CTR/labels18.csv \
    --limit 64 --batch 4 --out $O/greenrfm_$m.json || rc=1
done
echo "=== smoke eval rc=$rc ==="
[ $rc -eq 0 ] && echo "EVAL SMOKE OK" || echo "EVAL SMOKE FAIL"
exit $rc

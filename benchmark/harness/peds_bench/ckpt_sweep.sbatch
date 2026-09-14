#!/bin/bash
#SBATCH --job-name=pbCkptSweep
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/ckpt_sweep_%j.out
# ARC-CT selects its reported checkpoint on validation macro AUC (every 500 updates,
# CTClip.best.pt, early-stop patience 6 -- BILTIR handoff lines 277-278). The two
# baselines were never given that: their training scripts contain no selection logic
# at all, and we scored whatever checkpoint happened to be last. That is a real
# advantage for ARC-CT. This sweep scores EVERY saved pediatric checkpoint so the
# baselines get the same best-of-N protocol.
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=/temp_work/ch278233/PEDS_BENCH/sweep; mkdir -p $O
R=/temp_work/ch278233/COMBINED_reports.csv
L=$BM_DATA/peds23_labels_valid.csv
T=$BM_DATA/peds_valid

echo "################ GreenRFM: 10 epoch checkpointi, classifier readout"
for e in 1 2 3 4 5 6 7 8 9 10; do
  C=$BM_RUNS/greenrfm_peds23/stage2_align/checkpoint_epoch_${e}.pt
  [ -f "$C" ] || { echo "atlandi (yok): $C"; continue; }
  [ -f "$O/grf_ep${e}.json" ] && { echo "atlandi (zaten var): epoch $e"; continue; }
  echo "--- GreenRFM epoch $e ---"
  PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/GreenRFM \
  $BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
    --model greenrfm --mode classifier --ckpt "$C" \
    --tree $T --reports $R --labels $L \
    --out $O/grf_ep${e}.json --batch 8 --workers 8 2>&1 | grep -E "MACRO|FATAL|Error loading"
done

echo "################ MPS-CT: adim checkpointleri, zero-shot null readout"
for s in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000; do
  C=$BM_RUNS/mpsct_peds23/attempt0/CTClip.${s}.pt
  [ -f "$C" ] || { echo "atlandi (yok): $C"; continue; }
  [ -f "$O/mps_${s}.json" ] && { echo "atlandi (zaten var): step $s"; continue; }
  echo "--- MPS-CT step $s ---"
  PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT \
  $BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/70_eval.py \
    --model mpsct --mode zeroshot --ckpt "$C" \
    --tree $T --reports $R --labels $L \
    --out $O/mps_${s}.json --batch 8 --workers 8 2>&1 | grep -E "MACRO|FATAL|Error loading"
done
echo "=== rc=$? ==="

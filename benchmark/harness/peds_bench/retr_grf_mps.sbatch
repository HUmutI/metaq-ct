#!/bin/bash
#SBATCH --job-name=pbRetrGM
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/retr_grfmps_%j.out
# Same best-of-N checkpoints as the classification table (GreenRFM epoch 9, MPS-CT update 3000).
set -uo pipefail
. /home/ch278233/BENCHMARK/harness/env.sh
O=/temp_work/ch278233/PEDS_BENCH/retrieval; S=/temp_work/ch278233/PEDS_BENCH/retrieval/_scratch; mkdir -p $O $S
R=/temp_work/ch278233/COMBINED_reports.csv; L=$BM_DATA/peds23_labels_valid.csv; T=$BM_DATA/peds_valid
echo "### MPS-CT step 3000"
PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/MPS-CT $BM_REPOS/envs/MPS-CT/bin/python -u $BM_HARNESS/70_eval.py \
  --model mpsct --mode zeroshot --ckpt $BM_RUNS/mpsct_peds23/attempt0/CTClip.3000.pt \
  --tree $T --reports $R --labels $L --out $S/mpsct.json --batch 8 --workers 8 \
  --dump-retrieval $O/mpsct_valid.npz 2>&1 | grep -E "MACRO|retrieval latents|FATAL|Error"
echo "### GreenRFM epoch 9"
PYTHONPATH=$BM_HARNESS/shims:$BM_REPOS/GreenRFM $BM_REPOS/envs/GreenRFM/bin/python -u $BM_HARNESS/70_eval.py \
  --model greenrfm --mode zeroshot --ckpt $BM_RUNS/greenrfm_peds23/stage2_align/checkpoint_epoch_9.pt \
  --tree $T --reports $R --labels $L --out $S/greenrfm.json --batch 8 --workers 8 \
  --dump-retrieval $O/greenrfm_valid.npz 2>&1 | grep -E "MACRO|retrieval latents|FATAL|Error"
echo "=== rc=$? ==="

#!/bin/bash
#SBATCH --job-name=pbRetrCTC
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/retr_ctclip_%j.out
set -uo pipefail
P=/home/ch278233/BENCHMARK/envs/CT-CLIP/bin/python
S=/home/ch278233/BENCHMARK/harness/peds_bench/ctclip_text_latents.py
W=/temp_work/ch278233/BENCHMARK_WEIGHTS/CT-CLIP; L=/temp_work/ch278233/PEDS_BENCH/latents; O=/temp_work/ch278233/PEDS_BENCH/retrieval
mkdir -p $O
$P -u $S --ckpt $W/CT-CLIP_v2.pt      --latents $L/ctclip_v2_valid.npz        --out $O/ctclip_valid.npz
$P -u $S --ckpt $W/CT_VocabFine_v2.pt --latents $L/ctclip_vocabfine_valid.npz --out $O/ctclip_vocabfine_valid.npz
echo "=== rc=$? ==="

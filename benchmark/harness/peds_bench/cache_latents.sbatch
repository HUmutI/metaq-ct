#!/bin/bash
#SBATCH --job-name=pbLatents
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0,gpu-b11-2,gpu-b11-3
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/PEDS_BENCH/logs/latents_%j.out
set -uo pipefail
cd /home/ch278233/BENCHMARK/CT-CLIP/scripts
P=/home/ch278233/BENCHMARK/envs/CT-CLIP/bin/python
S=/home/ch278233/BENCHMARK/harness/peds_bench/cache_ctclip_latents.py
W=/temp_work/ch278233/BENCHMARK_WEIGHTS/CT-CLIP
O=/temp_work/ch278233/PEDS_BENCH/latents; mkdir -p $O
export PYTHONPATH=/home/ch278233/BENCHMARK/CT-CLIP/scripts:/home/ch278233/BENCHMARK/harness/peds_bench

echo "### 1/2 CT-CLIP (zero-shot, resmi CT-RATE agirliklari)"
$P -u $S --ckpt $W/CT-CLIP_v2.pt      --out $O/ctclip_v2_valid.npz
echo "### 2/2 CT-CLIP VocabFine (ayri yayinlanmis varyant)"
$P -u $S --ckpt $W/CT_VocabFine_v2.pt --out $O/ctclip_vocabfine_valid.npz
echo "=== rc=$? ==="

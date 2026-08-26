#!/bin/bash
#SBATCH --job-name=pedqwendl
#SBATCH --partition=bch-compute
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/temp_work/ch278233/BCH_DATASET/LABELS/_slurm/00_download_%j.out
set -uo pipefail
# Downloading from a login-node background process kept dying; as a SLURM job it
# is immune to logouts, VPN switches and login-node cleanup.
export HF_HOME=/temp_work/ch278233/hf_cache
export TMPDIR=/temp_work/ch278233/tmp
mkdir -p "$TMPDIR"
HF=$HOME/micromamba/envs/qwen/bin/hf
MODEL=${PED_QWEN_MODEL:-Qwen/Qwen3-8B}

rm -rf "/temp_work/ch278233/hf_cache/hub/.locks/models--Qwen--Qwen3-8B" 2>/dev/null
for attempt in 1 2 3 4 5; do
    echo "[dl] attempt $attempt for $MODEL at $(date)"
    $HF download "$MODEL" && break
    echo "[dl] attempt $attempt failed, retrying"; sleep 30
done

n=$(find "$HF_HOME" -path '*Qwen3-8B*' -name '*.safetensors' | wc -l)
inc=$(find "$HF_HOME" -name '*.incomplete' | wc -l)
echo "[dl] safetensors=$n incomplete=$inc size=$(du -sh "$HF_HOME" | cut -f1)"
# the model card lists 5 shards; require all of them and no partial files
if [ "$n" -ge 5 ] && [ "$inc" -eq 0 ]; then
    echo "[dl] COMPLETE"; exit 0
fi
echo "[dl] INCOMPLETE"; exit 1

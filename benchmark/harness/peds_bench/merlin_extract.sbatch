#!/bin/bash
#SBATCH -J merlin_ext
#SBATCH -p bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_ext_%A_%a.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_ext_%A_%a.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
: "${SPLIT:?set SPLIT}"; : "${NSHARDS:?set NSHARDS}"
echo "host=$(hostname) split=$SPLIT shard=$SLURM_ARRAY_TASK_ID/$NSHARDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun $PY /home/ch278233/BENCHMARK/harness/peds_bench/merlin_extract_feats.py \
  --split "$SPLIT" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS" \
  --batch-size 2 --workers 6

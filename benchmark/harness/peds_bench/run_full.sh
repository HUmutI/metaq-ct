#!/bin/bash
#SBATCH -J fvPedsZS
#SBATCH -p bch-gpu-pe,bch-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=72G
#SBATCH -t 08:00:00
#SBATCH -a 0-3
#SBATCH -o /home/ch278233/BENCHMARK/harness/peds_bench/logs/full_%A_%a.out
#SBATCH -e /home/ch278233/BENCHMARK/harness/peds_bench/logs/full_%A_%a.err
cd /temp_work/ch278233/PEDS_BENCH/fvlm/run
nvidia-smi --query-gpu=name --format=csv,noheader
B=/home/ch278233/BENCHMARK/harness/peds_bench
/temp_work/ch278233/PEDS_BENCH/fvlm/_venv/bin/python $B/infer_peds.py \
  --out    $B/preds_raw/preds_${SLURM_ARRAY_TASK_ID}.jsonl \
  --qc-out $B/preds_raw/qc_${SLURM_ARRAY_TASK_ID}.jsonl \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 4 --workers 6

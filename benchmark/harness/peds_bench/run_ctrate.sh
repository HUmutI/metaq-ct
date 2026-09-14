#!/bin/bash
#SBATCH -J fvCtrCtl
#SBATCH -p bch-gpu-pe,bch-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=72G
#SBATCH -t 06:00:00
#SBATCH -a 0-3
#SBATCH -o /home/ch278233/BENCHMARK/harness/peds_bench/logs/ctr_%A_%a.out
#SBATCH -e /home/ch278233/BENCHMARK/harness/peds_bench/logs/ctr_%A_%a.err
cd /temp_work/ch278233/PEDS_BENCH/fvlm/run
B=/home/ch278233/BENCHMARK/harness/peds_bench
/temp_work/ch278233/PEDS_BENCH/fvlm/_venv/bin/python $B/ctrate_control.py \
  --out    $B/ctrate_raw/preds_${SLURM_ARRAY_TASK_ID}.jsonl \
  --qc-out $B/ctrate_raw/qc_${SLURM_ARRAY_TASK_ID}.jsonl \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 4 --n 1000 --workers 6

#!/bin/bash
#SBATCH -J fvSmoke
#SBATCH -p bch-gpu-pe,bch-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /home/ch278233/BENCHMARK/harness/peds_bench/logs/smoke_%j.out
#SBATCH -e /home/ch278233/BENCHMARK/harness/peds_bench/logs/smoke_%j.err
cd /temp_work/ch278233/PEDS_BENCH/fvlm/run
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/temp_work/ch278233/PEDS_BENCH/fvlm/_venv/bin/python \
  /home/ch278233/BENCHMARK/harness/peds_bench/infer_peds.py \
  --out  /home/ch278233/BENCHMARK/harness/peds_bench/logs/_smoke_preds.jsonl \
  --qc-out /home/ch278233/BENCHMARK/harness/peds_bench/logs/_smoke_qc.jsonl \
  --limit 6 --workers 3

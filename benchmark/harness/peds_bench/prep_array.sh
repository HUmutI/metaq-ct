#!/bin/bash
#SBATCH -J fvPrepPeds
#SBATCH -p bch-compute,bch-compute-pe
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -a 0-23
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/prep_%A_%a.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/prep_%A_%a.err
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/ch278233/micromamba/envs/arcct/bin/python \
  /home/ch278233/BENCHMARK/harness/peds_bench/preprocess_peds.py \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 24

#!/bin/bash
#SBATCH -J merlin_probe
#SBATCH -p bch-compute,bch-compute-pe
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_probe_%j.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_probe_%j.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
$PY /home/ch278233/BENCHMARK/harness/peds_bench/merlin_linear_probe.py

#!/bin/bash
#SBATCH --job-name=bmWatch
#SBATCH --partition=bch-compute
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/temp_work/ch278233/ts_logs/bmWatch_%j.out
bash /home/ch278233/BENCHMARK/harness/91_watchdog.sh 40

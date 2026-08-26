#!/bin/bash
#SBATCH --job-name=report
#SBATCH --partition=bch-compute
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/temp_work/ch278233/ts_logs/report_%j.out
set -uo pipefail
# afterany, not afterok: if the training died the report is MORE useful, not
# less. A chain that only reports on success leaves the failure case silent,
# which is the case somebody actually needs to wake up to.
$HOME/micromamba/envs/arcct/bin/python $HOME/pipeline/06_eval/morning_report.py
exit 0

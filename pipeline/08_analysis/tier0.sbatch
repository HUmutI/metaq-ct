#!/bin/bash
#SBATCH --job-name=tier0
#SBATCH --partition=bch-compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/temp_work/ch278233/BCH_DATASET/LABELS/_slurm/tier0_%j.out
set -uo pipefail
export TMPDIR=/temp_work/ch278233/tmp
$HOME/micromamba/envs/arcct/bin/python $HOME/pipeline/08_analysis/test_kw_baseline.py || exit 1
echo
$HOME/micromamba/envs/arcct/bin/python $HOME/pipeline/08_analysis/tier0.py
echo "TIER0 rc=$?"

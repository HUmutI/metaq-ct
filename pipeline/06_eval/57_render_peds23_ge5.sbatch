#!/bin/bash
#SBATCH --job-name=renderP23G5
#SBATCH --partition=bch-compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/temp_work/ch278233/ts_logs/renderP23G5_%j.out
set -euo pipefail
$HOME/micromamba/envs/arcct/bin/python $HOME/bch-arc-ct/tools/render_peds23_ge5_tex.py

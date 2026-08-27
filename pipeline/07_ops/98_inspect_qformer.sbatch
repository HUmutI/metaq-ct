#!/bin/bash
#SBATCH --job-name=qfins
#SBATCH --partition=bch-compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/qfins_%j.out
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
C=/temp_work/ch278233/arcct_checkpoints
exec $E/bin/python $ARCCT_ROOT/tools/inspect_ckpt.py \
    $C/arcct_seed0.pt $C/arcct_seed0.peds27.pt

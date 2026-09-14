#!/bin/bash
#SBATCH --job-name=prepP23g5
#SBATCH --partition=bch-compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/prepP23g5_%j.out
set -euo pipefail

E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export RAC_SCHEMA=peds23
export RAC_QFORMER_QUERIES=35

echo "=== prepare genuine peds23 age>=5 $(date) ==="
$E/bin/python "$ARCCT_ROOT/pipeline/lib/check_env.py" --root "$ARCCT_ROOT" \
  --expect-classes 23 --expect-schema peds23
$E/bin/python "$ARCCT_ROOT/tools/prepare_peds23_ge5.py"
echo "=== complete $(date) ==="

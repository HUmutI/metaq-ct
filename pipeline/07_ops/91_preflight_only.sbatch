#!/bin/bash
#SBATCH --job-name=ctxpre
#SBATCH --partition=bch-compute
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/ctxpre_%j.out
set -uo pipefail
# Preflight on its own, on a compute node. The login node's NFS makes the
# monai/scipy import take minutes; inside a job it is warm and the check is
# seconds. Run this once before submitting the ladder array.
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
set -a; . $ARCCT_ROOT/configs/stage2_peds.env; . $ARCCT_ROOT/configs/stage2_peds_ctx.env; set +a
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/PEDS_VOLLIST_TRAIN_clean.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/PEDS_VOLLIST_VALID_clean.txt
export RAC_RESULTS_DIR=/temp_work/ch278233/runs/_preflight_probe
mkdir -p "$RAC_RESULTS_DIR"
$E/bin/python $ARCCT_ROOT/pipeline/lib/check_env.py --root "$ARCCT_ROOT" --expect-classes 27 --expect-schema peds || exit 78
$E/bin/python $ARCCT_ROOT/tools/preflight_context.py
echo "PREFLIGHT_RC=$?"

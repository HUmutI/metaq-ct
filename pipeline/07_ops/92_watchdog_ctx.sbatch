#!/bin/bash
#SBATCH --job-name=ctxdog
#SBATCH --partition=bch-compute
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --output=/temp_work/ch278233/ts_logs/ctxdog_%j.out
set -uo pipefail
# The watchdog as a job, not a login-node background process. A shell that goes
# away takes a nohup'd process with it often enough that the monitor is the
# first thing to die, and its silence then reads as "everything is fine".
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
echo "=== watchdog starting $(date) on $(hostname) ==="
exec $E/bin/python $ARCCT_ROOT/pipeline/07_ops/watchdog_ctx.py \
    --interval 600 --ignore pedwatch ctxdog

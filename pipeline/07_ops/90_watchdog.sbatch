#!/bin/bash
#SBATCH --job-name=pedwatch
#SBATCH --partition=bch-compute
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/temp_work/ch278233/ts_logs/watchdog_%j.out
set -uo pipefail
export ARCCT_ROOT=${ARCCT_ROOT:-$HOME/bch-arc-ct}
# Runs as a SLURM job so it outlives any login session, VPN drop or laptop.
# Re-submits itself before the 2-day wall clock so the watch never lapses.
LOG=/temp_work/ch278233/WATCHDOG.log
END=$(( $(date +%s) + 46*3600 ))          # stop looping 2h before the limit
while [ "$(date +%s)" -lt "$END" ]; do
    {
        echo "----- $(date '+%F %T') -----"
        $HOME/micromamba/envs/arcct/bin/python $ARCCT_ROOT/pipeline/07_ops/watchdog2.py 2>&1 \
            | grep -E "^\[|^  (ok|!!|step |son val|bos bolge)|^SORUN"
    } >> "$LOG" 2>&1
    # keep the file readable: last 4000 lines is a couple of days of history
    tail -4000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
    sleep 600
done
sbatch $ARCCT_ROOT/pipeline/07_ops/90_watchdog.sbatch          # hand over before the wall
echo "resubmitted at $(date)"

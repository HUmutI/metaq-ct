#!/bin/bash
# Detect hung benchmark jobs. Detect only -- never requeue.
#
# Tonight a job sat 6 minutes on gpu-b11-3 without printing its first line while
# imports take 17s; SLURM still showed it RUNNING. EXPERIMENTS.md records the
# same thing twice. A hung job is not a failed one and SLURM will not tell you.
#
# Safe to use a tight threshold because neither trainer validates: there is no
# silent eval pass that could look like a hang (see the handoff doc). Still,
# this only reports -- a false positive that requeued a run would throw away
# hours, which is a worse failure than a late diagnosis.
LOG=/temp_work/ch278233/BENCHMARK_RUNS/WATCHDOG.log
TS=/temp_work/ch278233/ts_logs
QUIET_MIN=${1:-40}

while true; do
  now=$(date '+%Y-%m-%d %H:%M:%S')
  squeue -u ch278233 -h -o "%i %j %T" | grep -E "bmGrf|bmMps" | while read jid name state; do
    [ "$state" = "RUNNING" ] || continue
    f=$(ls -t $TS/${name}_${jid%%_*}*.out 2>/dev/null | head -1)
    [ -n "$f" ] || { echo "$now  $jid $name RUNNING but no log file yet" >> $LOG; continue; }
    age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
    if [ $age -ge $QUIET_MIN ]; then
      echo "$now  STALL? $jid $name silent ${age}min on $(squeue -j ${jid%%_*} -h -o %N) -- $(basename $f)" >> $LOG
    fi
  done
  # note any job that left the queue since last pass
  squeue -u ch278233 -h -o "%i" | grep -q . || true
  sleep 600
done

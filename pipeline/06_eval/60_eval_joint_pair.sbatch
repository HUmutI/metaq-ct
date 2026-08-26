#!/bin/bash
#SBATCH --job-name=jointcmp
#SBATCH --partition=bch-compute
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/temp_work/ch278233/ts_logs/jointcmp_%j.out
set -uo pipefail
# Driver: submits the evaluation pair once BOTH joint runs have finished, one at
# a time. Sequential on purpose - running six 6,362-volume evaluations in
# parallel exhausted both node memory and the system-wide file descriptor table,
# and lost ~9 GPU-hours to OOM and thrashing.
P=$HOME/pipeline
CMB=/temp_work/ch278233
declare -A M=( [C47mix]=combined_47k [C47harm]=combined_47k_harmonised )
PREV=""
for tag in C47mix C47harm; do
  ck=$CMB/runs/${M[$tag]}/CTClip.best.pt
  [ -f "$ck" ] || { echo "!! $tag checkpoint yok: $ck"; continue; }
  for spec in "peds:" "ctrate:" \
              "combined:$CMB/COMBINED_VALID_labels.csv" \
              "combined:$CMB/COMBINED_VALID_labels_harmonised.csv"; do
    set_name="${spec%%:*}"; lbl="${spec#*:}"
    suffix=""
    [ "$set_name" = "combined" ] && case "$lbl" in *harmonised*) suffix="_harmlbl";; *) suffix="_mixlbl";; esac
    dep=""; [ -n "$PREV" ] && dep="--dependency=afterany:$PREV"
    j=$(sbatch $dep --parsable \
        --export=ALL,CKPT=$ck,TAG=${tag}${suffix},SET=$set_name,LBL=$lbl,EVAL_NUM_WORKERS=2 \
        $P/41_eval_matrix.sbatch)
    echo "  $tag / $set_name${suffix} -> $j"
    PREV=$j
  done
done
echo "zincir sonu: $PREV"

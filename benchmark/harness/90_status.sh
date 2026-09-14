#!/bin/bash
# One-shot dashboard for the MPS-CT / GreenRFM benchmark.
R=/temp_work/ch278233/BENCHMARK_RUNS
L=/temp_work/ch278233/ts_logs

echo "===== queue ====="
squeue -u ch278233 -o "%.13i %.14j %.9T %.10M %.10L %R" | grep -E "JOBID|bmGrf|bmMps|bmSmoke" || echo "  (no benchmark jobs)"

echo
echo "===== progress (last step line per run) ====="
for f in $L/bmGrfS1_*.out $L/bmGrfAln_*.out $L/bmMpsCT_*.out $L/bmGrfPeds_*.out $L/bmMpsPeds_*.out; do
  [ -e "$f" ] || continue
  last=$(tail -c 4000 "$f" 2>/dev/null | tr '\r' '\n' | grep -aE "Epoch|Training:|loss=|stage complete|rc=" | tail -1)
  printf "%-34s %s\n" "$(basename $f)" "${last:0:110}"
done

echo
echo "===== checkpoints on disk ====="
for d in $R/greenrfm_ctrate/stage1_image $R/greenrfm_ctrate/stage1_text \
         $R/greenrfm_ctrate/stage2_align $R/greenrfm_peds23/stage2_align; do
  n=$(ls -1 $d/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
  printf "  %-46s %s ckpt  %s\n" "${d#$R/}" "$n" "$(ls -1 $d/checkpoint_epoch_*.pt 2>/dev/null | sort -V | tail -1 | xargs -r basename)"
done
for d in $R/mpsct_ctrate $R/mpsct_peds23; do
  n=$(find $d -name "CTClip.*.pt" 2>/dev/null 
  printf "  %-46s %s ckpt  %s\n" "${d#$R/}" "$n" "$(ls -1 $d/CTClip.*.pt 2>/dev/null | sed 's/.*CTClip\.\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2 | xargs -r basename)"
done

echo
echo "===== failures / stalls ====="
grep -alE "FATAL|Traceback|CUDA out of memory|rc=[1-9]" $L/bmGrf*.out $L/bmMps*.out 2>/dev/null | while read f; do
  echo "  !! $(basename $f): $(grep -aoE 'FATAL.*|CUDA out of memory|rc=[1-9][0-9]*' "$f" | tail -1)"
done
for f in $L/bmGrfS1_*.out $L/bmGrfAln_*.out $L/bmMpsCT_*.out $L/bmGrfPeds_*.out $L/bmMpsPeds_*.out; do
  [ -e "$f" ] || continue
  age=$(( ($(date +%s) - $(stat -c %Y "$f")) / 60 ))
  jid=$(basename "$f" .out | sed 's/^[a-zA-Z0-9]*_//')
  if [ $age -gt 30 ] && squeue -u ch278233 -h -o "%i" | grep -q "${jid%%_*}"; then
    echo "  !! $(basename $f) RUNNING but silent for ${age} min -- a hung job is not a failed one"
  fi
done

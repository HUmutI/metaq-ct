#!/bin/bash
# Pediatric ARC-CT pipeline status.  Run: bash ~/status.sh
W=/temp_work/ch278233
PY=$HOME/micromamba/envs/arcct/bin/python

printf '\n%s\n' "=================== ISLER ==================="
squeue -u ch278233 -o "%.11i %.11j %.2t %.9M %.4C %.14N" -h 2>/dev/null | head -12
[ "$(squeue -u ch278233 -h 2>/dev/null | wc -l)" -eq 0 ] && echo "  (bizim kuyrukta is yok)"
th=$(squeue -u ch278452 -h -o "%j(%t)" 2>/dev/null | tr '\n' ' ')
[ -n "$th" ] && echo "  ch278452 (Ege): $th"

printf '\n%s\n' "================= ILERLEME ================="
bar() { p=$(( $1 * 28 / ($2>0?$2:1) )); printf '['
        for i in $(seq 1 28); do [ $i -le $p ] && printf '#' || printf '.'; done
        printf '] %6d/%-6d %3d%%\n' "$1" "$2" $(( $1 * 100 / ($2>0?$2:1) )); }
npz=$(ls $W/PEDS_NPZ 2>/dev/null | wc -l)
mk=$(ls $W/PEDS_MASKS10_192 2>/dev/null | wc -l)
ql=$(cat $W/BCH_DATASET/LABELS27/labels.shard*.jsonl 2>/dev/null | grep -c '"status": "ok"')
ct=$(ls /temp_work/ch278452/CTRATE_NPZ/train 2>/dev/null | wc -l)
printf '  %-20s ' "pediatrik npz";   bar "$npz" 8816
printf '  %-20s ' "10-bolge maske";  bar "$mk"  8816
printf '  %-20s ' "Qwen 27 label";   bar "$ql"  8816
printf '  %-20s ' "CT-RATE npz";     bar "$ct"  47149

printf '\n%s\n' "================= EGITIM ==================="
f=$(ls -t $W/ts_logs/train_peds_*.out 2>/dev/null | head -1)
if [ -n "$f" ]; then
  st=$(tr '\r' '\n' < "$f" | grep -oE "RAC-updates: *[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]" | tail -1)
  echo "  ${st:-  (henuz adim yok)}"
  tr '\r' '\n' < "$f" | grep -E "global_val_auc" | tail -2 | sed 's/^/  /'
  tr '\r' '\n' < "$f" | grep -oE "empty-region rate per anatomy query: .*" | tail -1 | cut -c1-100 | sed 's/^/  /'
  tr '\r' '\n' < "$f" | grep -E "skipped_nomask|Warm-started Q-Former" | tail -2 | cut -c1-118 | sed 's/^/  /'
else echo "  (egitim baslamadi)"; fi

printf '\n%s\n' "================= CT-RATE =================="
g=$(ls -t /temp_work/ch278452/ctrate_fetch/logs/*.out 2>/dev/null | head -1)
[ -n "$g" ] && grep -oE "batch [0-9]+/[0-9]+: \+[0-9]+.*eta=[0-9.]+h" "$g" 2>/dev/null | tail -2 | sed 's/^/  /'

printf '\n%s\n' "================= BEKCI ===================="
$PY $HOME/pipeline/07_ops/watchdog2.py 2>/dev/null | grep -E "^  (ok|!!)|^SORUN|^  step |^  son val" | sed 's/^/  /'
age=$(( ($(date +%s) - $(stat -c %Y $W/tmp/watchdog_state.json 2>/dev/null || echo 0)) / 60 ))
echo "  (arka plan bekcisi $age dk once kosdu)"

printf '\n%s\n' "================= DISK ====================="
for d in PEDS_NPZ PEDS_MASKS10_192 runs; do
  printf '  %-24s %s\n' "$d" "$(timeout 20 du -sh --apparent-size $W/$d 2>/dev/null | cut -f1)"
done
printf '  %-24s %s\n' "CTRATE_NPZ (Ege)" "$(timeout 25 du -sh --apparent-size /temp_work/ch278452/CTRATE_NPZ 2>/dev/null | cut -f1)"
echo

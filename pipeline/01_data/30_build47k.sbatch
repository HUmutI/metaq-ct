#!/bin/bash
#SBATCH --job-name=build47k
#SBATCH --partition=bch-compute
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/temp_work/ch278233/ts_logs/build47k_%j.out
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export PYTHONPATH=$HOME/arc-ct:${PYTHONPATH:-}

# The CT-RATE fetch runs under a different account, so SLURM cannot express the
# dependency - poll instead. Stop on a target count OR on the fetch job leaving
# the queue, because a fetch that dies at 46k must not leave this waiting for a
# number that will never arrive.
SRC=/temp_work/ch278452/CTRATE_NPZ/train
TARGET=47149
for i in $(seq 1 720); do
    # `grep -c` already prints 0 when nothing matches, but it also EXITS 1, so
    # `|| echo 0` appended a second line and the integer test below died with
    # "[: 0\n0: integer expression expected" - leaving the loop to spin for its
    # full 60 hours instead of noticing the fetch had finished.
    n=$(ls -1 "$SRC" 2>/dev/null | grep -c '\.npz$')
    running=$(squeue -u ch278452 -h -n ctfetch452 -o "%t" 2>/dev/null | grep -c R)
    echo "$(date +%H:%M) npz=$n/$TARGET fetch_kosuyor=$running"
    [ "${n:-0}" -ge "$TARGET" ] && { echo "hedefe ulasildi"; break; }
    [ "${running:-0}" -eq 0 ] && { echo "fetch bitti/durdu, elde $n ile devam"; break; }
    sleep 300
done

# Validate before assembling: one 0-byte npz killed a 12,000-step run 13 minutes
# in, and the assembler hardlinks whatever it finds.
echo "=== $(date) npz dogrulama ==="
$E/bin/python $HOME/pipeline/01_data/validate_npz.py --root "$SRC" --out /temp_work/ch278233/BAD_NPZ_FETCH.txt || true

echo "=== $(date) birlesik veri seti (47k) ==="
$E/bin/python $HOME/pipeline/04_dataset/build_combined.py
rc=$?
echo "=== build47k rc=$rc $(date) ==="
exit $rc

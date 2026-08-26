#!/bin/bash
#SBATCH --job-name=prepv2
#SBATCH --partition=bch-compute
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/prep_v2_%j.out
set -uo pipefail
E=$HOME/micromamba/envs/arcct
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$HOME/arc-ct:${PYTHONPATH:-}

echo "=== $(date) build_outputs on LABELS27_V2 ==="
$E/bin/python $HOME/pipeline/02_extract/build_outputs.py --dir /temp_work/ch278233/BCH_DATASET/LABELS27_V2
rc=$?; [ $rc -ne 0 ] && { echo "build_outputs rc=$rc"; exit $rc; }

echo "=== $(date) build_outputs on CTRATE/LABELS27 ==="
$E/bin/python $HOME/pipeline/02_extract/build_outputs.py --dir /temp_work/ch278233/CTRATE/LABELS27
rc=$?; [ $rc -ne 0 ] && echo "!! ctrate build_outputs rc=$rc (47k icin; pediatrik egitimi bloklamaz)"

echo "=== $(date) CT-RATE bolge onbellegi fan-out ==="
$E/bin/python $HOME/pipeline/02_extract/fanout_ctrate_regions.py
rc=$?; [ $rc -ne 0 ] && echo "!! fanout rc=$rc (47k icin; pediatrik egitimi bloklamaz)"

echo "=== $(date) CT-RATE 27 sutunlu etiketler (yayin 18 + llm 9) ==="
$E/bin/python $HOME/pipeline/02_extract/apply_ctrate_labels.py
rc=$?; [ $rc -ne 0 ] && echo "!! apply_ctrate_labels rc=$rc (47k icin; pediatrik egitimi bloklamaz)"

# The gate is last and its exit code is the job's: the pediatric training is
# chained afterok, so a structural problem in the new labels stops the run
# before a GPU is taken rather than three hours into it.
echo "=== $(date) KAPI: V2 pediatrik etiket dogrulama ==="
$E/bin/python $HOME/pipeline/04_dataset/prep_v2.py
rc=$?
echo "=== prep rc=$rc $(date) ==="
exit $rc

#!/bin/bash
#SBATCH --job-name=ctrgate
#SBATCH --partition=bch-gpu-pe,bch-gpu,bch-gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-1-0,gpu-2-0
# gpu:large excluded the A100s. They are gpu:xlarge, so a gpu:large request can
# never match one, and bch-gpu-xlarge was not in the partition list either --
# 24 A100s, the pool this account contends least for, were unreachable twice
# over. gpu:large was chosen to avoid the 24 GB TITANs, which OOM; those are
# gpu-1-0 and gpu-2-0 and they are the ONLY medium nodes across all three
# partitions, so excluding them by name does the same job without the cost.
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/ts_logs/ctrgate_%j.out
set -uo pipefail
# Step-0 identity on the ADULT schema, before anything is launched. The pediatric
# gate passing at 27 classes proves nothing here: SlotLayout derives the whole
# bank from len(PATHOLOGIES), so 18 classes is a different bank (49 slots over 31
# rows, not 67 over 40) and every index vector is recomputed.
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
set -a; . $ARCCT_ROOT/configs/stage2.env; . $ARCCT_ROOT/configs/stage2_ctrate_ctx.env; set +a
export RAC_SCHEMA=ctrate
export RAC_CONTEXT_CSV=/temp_work/ch278233/CONTEXT/indication.csv
export RAC_DEMOGRAPHICS_CSV=/temp_work/ch278233/CONTEXT/demographics.csv
export RAC_LABELS_TRAIN=/temp_work/ch278233/CTRATE_ONLY/labels18.csv
export RAC_LABELS_VALID=/temp_work/ch278233/CTRATE_ONLY/labels18.csv
export RAC_REPORTS_TRAIN=/temp_work/ch278233/CTRATE_ONLY/reports.csv
export RAC_REPORTS_VALID=/temp_work/ch278233/CTRATE_ONLY/reports.csv
export RAC_VOLUME_LIST_TRAIN=/temp_work/ch278233/CTRATE_ONLY/VOLLIST_TRAIN.txt
export RAC_VOLUME_LIST_VALID=/temp_work/ch278233/CTRATE_ONLY/VOLLIST_VALID.txt
export RAC_DATA_TRAIN=/temp_work/ch278233/COMBINED_NPZ
export RAC_DATA_VALID=/temp_work/ch278233/COMBINED_NPZ
export RAC_MASK_TRAIN=/temp_work/ch278233/COMBINED_MASKS10_192
export RAC_MASK_VALID=/temp_work/ch278233/COMBINED_MASKS10_192
export RAC_REGION_CACHE=/temp_work/ch278233/COMBINED_region_cache_v2.json
export RAC_WARM_START_CKPT=/temp_work/ch278233/arcct_checkpoints/arcct_seed0.pt
echo "=== $(hostname) $(date)  schema=$RAC_SCHEMA queries=$RAC_QFORMER_QUERIES ==="
echo "###### 1) slot aritmetigi 18 sinifta"
$E/bin/python - <<'PYEOF'
import sys; sys.path.insert(0,"/home/ch278233/bch-arc-ct")
from arcct.slots import SlotLayout
L=SlotLayout.phase1(n_path=18, n_anatomy=10, n_glob_gen=2)
print(f"   n_slots={L.n_slots} n_rows={L.n_rows} n_gen={L.n_gen}")
assert (L.n_slots,L.n_rows,L.n_gen)==(49,31,30), (L.n_slots,L.n_rows,L.n_gen)
d=[i for i,(a,b) in enumerate(zip(L.slot_to_row,L.role_row)) if a!=b]
print(f"   slot_to_row != role_row yalnizca {d} pozisyonunda")
assert d==[48], d
L.validate(); print("   validate() gecti")
PYEOF
echo "###### 2) adim-0 ozdeslik (fp32, deterministik)"
$E/bin/python $ARCCT_ROOT/tools/check_ctx_identity.py
rc1=$?
echo "###### 3) preflight"
$E/bin/python $ARCCT_ROOT/tools/preflight_context.py
rc2=$?
echo "=== identity rc=$rc1  preflight rc=$rc2  $(date) ==="
exit $(( rc1 || rc2 ))

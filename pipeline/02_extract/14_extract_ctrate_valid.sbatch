#!/bin/bash
#SBATCH --job-name=ctvalext
#SBATCH --partition=bch-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/temp_work/ch278233/CTRATE/LABELS27_VALID/_slurm/ctvalext_%A_%a.out
set -uo pipefail
# One shard per array task, one GPU each. Each shard appends to its own
# regions.shardN.jsonl / labels.shardN.jsonl and is independently resumable:
# resubmitting the same array skips reports that already have an "ok" record.
export HF_HOME=/temp_work/ch278233/hf_cache
export HF_HUB_OFFLINE=1              # weights are pre-fetched by 00_download
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# vLLM assumes flashinfer is always installed (requirements/cuda.txt) and its
# "is it supported" probe imports it unguarded. flashinfer needs Python 3.12 for
# an import-time annotation and this env is 3.11, so it was removed; this flag
# makes the probe return False before that import. We decode greedily anyway.
export VLLM_USE_FLASHINFER_SAMPLER=0

# The conda env ships libstdc++ 6.0.35 (has CXXABI_1.3.15); some compute nodes
# only have an older /lib64/libstdc++.so.6 without it, and the env's libicui18n
# then fails to load. The smoke happened to land on a new-enough node while all
# eight shards did not. Put the env's lib dir first so this is node-independent.
PYBIN=$(cat $ARCCT_ROOT/pipeline/07_ops/PYENV)
export LD_LIBRARY_PATH=$(dirname $(dirname "$PYBIN"))/lib:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Never use the login node's /tmp: it is shared, small and has filled to 100%
# mid-run before. Node-local scratch under /temp_work is safe and roomy.
export TMPDIR=/temp_work/ch278233/tmp/ctvalext_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export VLLM_CACHE_ROOT=$TMPDIR/vllm
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT"
trap 'rm -rf "$TMPDIR"' EXIT

# Stage the weights on node-local disk. Eight shards reading the same 16 GB of
# safetensors off NFS at once collapsed to ~1 MB/s - the model load alone was
# taking 50 minutes per shard file, versus 3 seconds when only one shard ran.
# A single bulk copy is far kinder to NFS than vLLM's parallel reads, and the
# stagger keeps the copies from starting simultaneously.
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 90 ))
SNAP=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
LOCAL=/tmp/qwen3_8b_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
if [ -n "$SNAP" ]; then
    mkdir -p "$LOCAL"
    echo "staging model to $LOCAL at $(date)"
    # Cap the copy: a staging step that hangs on NFS otherwise holds a GPU
    # doing nothing. 16 GB took 37 s on a healthy node, so 10 minutes is very
    # generous; past that, fall back to loading straight from the cache.
    if timeout 600 cp -rL "$SNAP". "$LOCAL"/; then
        export PED_QWEN_MODEL="$LOCAL"
    else
        echo "staging timed out or failed - falling back to the NFS cache"
        rm -rf "$LOCAL"
    fi
    echo "staged $(du -sh "$LOCAL" | cut -f1) at $(date)"
fi
trap 'rm -rf "$TMPDIR" "$LOCAL"' EXIT

export RAC_COHORT=adult
OUT=/temp_work/ch278233/CTRATE/LABELS27_VALID
export RAC_SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
export RAC_SHARD_N=${SLURM_ARRAY_TASK_COUNT:-1}
echo "=== shard $RAC_SHARD_ID/$RAC_SHARD_N on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Call the env's python directly: `micromamba run` takes a global lock that
# serialises simultaneous array tasks on the same node.
$(cat $ARCCT_ROOT/pipeline/07_ops/PYENV) $ARCCT_ROOT/pipeline/lib/qwen_extract.py \
    --sentences "$OUT/sentences.jsonl" \
    --out-dir   "$OUT" \
    --model     "${PED_QWEN_MODEL:-Qwen/Qwen3-8B}" \
    --task      "${PED_TASK:-both}"
rc=$?
echo "=== shard $RAC_SHARD_ID finished rc=$rc at $(date) ==="
exit $rc

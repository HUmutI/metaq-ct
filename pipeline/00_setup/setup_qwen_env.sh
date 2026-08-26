#!/bin/bash
# Build the `qwen` micromamba env: vLLM for local, in-network LLM inference.
#
# Deliberately SEPARATE from the `arcct` env. vLLM pins its own (much newer)
# torch, and installing it alongside arc-ct's torch==2.4.1 would break training.
# Label extraction is an offline preprocessing step, so a second env is free.
#
# No report text ever leaves this machine - that is the whole point of running
# the model locally instead of tools/extract_labels.py's default Ark endpoint.
set -euo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MM=$HOME/bin/micromamba

# Login-node /tmp is a 4 GB local disk that runs ~85% full; pip and HF both
# need far more than that for multi-GB wheels and weights.
export TMPDIR=$HOME/tmp
export PIP_CACHE_DIR=$HOME/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ -d "$MAMBA_ROOT_PREFIX/envs/qwen" ]; then
    echo "existing qwen env found - removing"
    rm -rf "$MAMBA_ROOT_PREFIX/envs/qwen"
fi

echo "=== create env ==="
$MM create -y -n qwen -c conda-forge python=3.11 pip

echo "=== install vllm ==="
$MM run -n qwen pip install --no-input vllm

echo "=== versions ==="
$MM run -n qwen python -c "
import torch, vllm
print('vllm ', vllm.__version__)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
"
echo "=== done ==="

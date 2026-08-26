#!/bin/bash
# Build the `arcct` micromamba env for the arc-ct repo on e3.
# torch 2.4.1 stock PyPI wheel is cu121; e3 GPUs are A40 (sm_86), driver 580 -> fine.
set -euo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MM=$HOME/bin/micromamba

# The login node's /tmp is a 4 GB local disk that sits ~86% full; pip buffers
# the 797 MB torch wheel there and dies with "[Errno 28] No space left on
# device". Home is NFS with terabytes free, so point both temp and cache there.
export TMPDIR=$HOME/tmp
export PIP_CACHE_DIR=$HOME/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "=== create env ==="
# micromamba's own recreate path does "remove all" first, which fails on NFS if
# anything still holds a file handle (.nfsXXXX) and leaves a half-deleted env.
# Remove it ourselves and only create when genuinely absent.
if [ -d "$MAMBA_ROOT_PREFIX/envs/arcct" ]; then
    echo "existing arcct env found - removing"
    rm -rf "$MAMBA_ROOT_PREFIX/envs/arcct"
fi
$MM create -y -n arcct -c conda-forge python=3.11 pip

echo "=== pip install requirements ==="
$MM run -n arcct pip install --no-input -r $HOME/arc-ct/requirements.txt

echo "=== pip install huggingface_hub ==="
# The README says `pip install -U huggingface_hub`, but that pulls 1.x and
# transformers==4.44.2 hard-requires <1.0 - it raises ImportError on import,
# not a warning. The `hf` CLI the README uses for weights arrived in 0.34, so
# that is the usable window.
$MM run -n arcct pip install --no-input "huggingface_hub>=0.34,<1.0"

echo "=== versions ==="
$MM run -n arcct python -c "
import torch, transformers, monai, nibabel, numpy
print('torch       ', torch.__version__)
print('cuda build  ', torch.version.cuda)
print('transformers', transformers.__version__)
print('monai       ', monai.__version__)
print('nibabel     ', nibabel.__version__)
print('numpy       ', numpy.__version__)
"
echo "=== done ==="

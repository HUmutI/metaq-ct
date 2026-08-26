#!/bin/bash
#SBATCH --job-name=tssetup
#SBATCH --partition=bch-compute
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/temp_work/ch278233/ts_logs/ts_setup_%j.out
set -uo pipefail
# Separate env on purpose: TotalSegmentator resolves numpy 2.x, and arcct's
# torch 2.4.1 is built against the NumPy 1.x C ABI. They share nothing but a
# .nii.gz file, so isolation costs nothing.
# Everything on /temp_work: $HOME hit its quota once already this project.
export MAMBA_ROOT_PREFIX=$HOME/micromamba
export TMPDIR=/temp_work/ch278233/tmp
export PIP_CACHE_DIR=/temp_work/ch278233/pip_cache
export TOTALSEG_HOME_DIR=/temp_work/ch278233/totalseg_home
export TOTALSEG_WEIGHTS_PATH=$TOTALSEG_HOME_DIR/nnunet/results
TS=/temp_work/ch278233/envs/totalseg
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TOTALSEG_WEIGHTS_PATH"

echo "=== create env $(date) ==="
rm -rf "$TS"
$HOME/bin/micromamba create -y -p "$TS" -c conda-forge python=3.11 pip || exit 1

echo "=== install TotalSegmentator 2.18.0 ==="
"$TS/bin/pip" install --no-input "TotalSegmentator==2.18.0" || exit 1

# Telemetry POSTs to stats.totalsegmentator.com on first run. The payload holds
# no image data, but a hospital cluster should not phone home. Pre-seeding the
# file wins because it is only created if absent.
cat > "$TOTALSEG_HOME_DIR/config.json" <<'JSON'
{"send_usage_stats": false, "prediction_counter": 0}
JSON
echo "telemetry disabled"

echo "=== versions ==="
export LD_LIBRARY_PATH=$TS/lib:${LD_LIBRARY_PATH:-}
"$TS/bin/python" -c "
import torch, numpy, totalsegmentator
print('totalsegmentator', totalsegmentator.__version__ if hasattr(totalsegmentator,'__version__') else 'n/a')
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('numpy', numpy.__version__)
"

echo "=== pre-download weights (once, so array tasks do not race) ==="
"$TS/bin/totalseg_download_weights" -t total || exit 1
du -sh "$TOTALSEG_WEIGHTS_PATH"
ls "$TOTALSEG_WEIGHTS_PATH"

echo "=== arcct env untouched? ==="
$HOME/micromamba/envs/arcct/bin/python -c "import numpy,torch;print('arcct numpy',numpy.__version__,'torch',torch.__version__)"
echo "=== TS SETUP DONE $(date) ==="

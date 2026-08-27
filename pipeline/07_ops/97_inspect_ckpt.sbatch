#!/bin/bash
#SBATCH --job-name=ckptins
#SBATCH --partition=bch-compute
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/temp_work/ch278233/ts_logs/ckptins_%j.out
set -uo pipefail
# What is actually inside the published adult checkpoint: it is the warm start
# for the CT-RATE-only run, and the query bank has to be 10 anatomy + 18
# pathology + 2 global for that to load without remapping. torch.load of these
# was killed on the login node, so it runs here.
E=$HOME/micromamba/envs/arcct
export ARCCT_ROOT=$HOME/bch-arc-ct
export PYTHONPATH=$ARCCT_ROOT:${PYTHONPATH:-}
$E/bin/python - <<'PYEOF'
import torch, os
for n in ("arcct_seed0.pt", "arcct_seed0.peds27.pt"):
    p = f"/temp_work/ch278233/arcct_checkpoints/{n}"
    if not os.path.isfile(p):
        print(f"{n}: yok"); continue
    pkg = torch.load(p, map_location="cpu", weights_only=False)
    print(f"\n===== {n}   ({os.path.getsize(p)/1e6:.0f} MB)")
    if isinstance(pkg, dict):
        print("  ust duzey anahtarlar:", list(pkg)[:12])
    sd = pkg.get("model", pkg.get("state_dict", pkg)) if isinstance(pkg, dict) else pkg
    if not isinstance(sd, dict):
        print("  state_dict cikarilamadi:", type(sd)); continue
    hits = [(k, tuple(v.shape)) for k, v in sd.items()
            if hasattr(v, "shape") and ("quer" in k.lower() or "qformer" in k.lower())]
    print(f"  qformer/query anahtari: {len(hits)}")
    for k, s in hits[:12]:
        print(f"     {k:58s} {s}")
    if not hits:
        print("  ilk 15 anahtar:", list(sd)[:15])
    cfg = pkg.get("config") if isinstance(pkg, dict) else None
    if isinstance(cfg, dict):
        for k in ("schema", "n_classes", "query_layout", "pathologies",
                  "use_anatomy_qformer", "num_global"):
            if k in cfg:
                v = cfg[k]
                print(f"  config.{k}: {len(v) if isinstance(v,(list,tuple)) else v}")
PYEOF

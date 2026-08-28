#!/usr/bin/env python3
"""How much of the fusion has the conditioned half actually taken?

The training log prints ||W_ind||/||W_gen|| to three decimals, and ||W_gen|| is
the norm of a 768x768 identity, sqrt(768) = 27.7. The ratio therefore has to reach
0.014 before it prints as anything but 0.000 -- so a log full of zeros cannot tell
"the gradient never arrives" from "it is training, slowly". This reads the actual
tensor and answers that.
"""
from __future__ import annotations

import argparse
import os

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    for p in a.ckpts:
        if not os.path.isfile(p):
            print(f"{os.path.basename(p)}: yok"); continue
        pkg = torch.load(p, map_location="cpu", weights_only=False)
        qm = pkg.get("qformer_module")
        if not isinstance(qm, dict):
            print(f"{os.path.basename(p)}: qformer_module yok"); continue
        W = next((v for k, v in qm.items() if k.endswith("fusion.weight")), None)
        if W is None:
            print(f"{os.path.basename(p)}: fusion.weight yok "
                  f"(anahtarlar: {[k for k in qm if 'fus' in k][:3]})"); continue
        d = W.shape[1] // 2
        gen, ind = W[:, :d], W[:, d:]
        gn, inn = float(gen.norm()), float(ind.norm())
        eye = torch.eye(d)
        print(f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}  "
              f"step={pkg.get('update_step')}")
        print(f"   ||W_gen||={gn:.6f}   ||W_ind||={inn:.3e}   oran={inn/max(gn,1e-8):.3e}")
        print(f"   W_gen birimden sapma={float((gen - eye).norm()):.3e}   "
              f"W_ind maks|w|={float(ind.abs().max()):.3e}   "
              f"tam sifir mi={bool((ind == 0).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""What do the two DERIVED regions actually contain, on volumes with and without
the pathology that routes to them?

Pneumothorax routes to region 8 (pleural_space, derived by dilating the lung union
and keeping unclaimed background) and Bone lesion to region 9 (chest wall and
skeleton). Both fall BELOW CHANCE under routed read-out -- 0.695 to 0.317 and
0.616 to 0.347 -- in every cohort and every checkpoint. Below 0.5 is inversion,
not misplaced attention; a query pointed at the wrong place scores 0.5.

Pleural thickening routes to the SAME region 8 and improves (+0.018), so region 8
is not uniformly broken and a geometric argument alone does not explain it. This
measures rather than argues:

  * how much of the region survives to the 12x12x12 feature grid, which is what
    build_role_mask actually restricts attention with. A region that reaches zero
    tokens becomes an UNRESTRICTED query, so its class is not routed at all.
  * mean HU inside the region, split by label. If the derived pleural shell
    captures the pneumothorax air, positives should be markedly more air-like
    than negatives. If it captures the chest wall instead, they will not differ.

Runs on a compute node; each volume reads a full npz off NFS.
"""
from __future__ import annotations

import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ch278233/bch-arc-ct")
from arcct.dataset import _pad_crop_hwd                      # noqa: E402

MASKS = os.environ.get("DIAG_MASKS", "/temp_work/ch278233/COMBINED_MASKS10_192")
LABELS = "/temp_work/ch278233/COMBINED_labels_harmonised.csv"
WANT = int(os.environ.get("DIAG_N", "300"))
PROBE = [("Pneumothorax", 8), ("Bone lesion or fracture", 9),
         ("Pleural thickening or nodule", 8), ("Pleural effusion", 8)]


def npz_for(name: str) -> str:
    pat = "_".join(name.split("_")[:2])
    return f"/temp_work/ch278233/COMBINED_NPZ/{pat}/{name}/{name}.npz"


def tokens(m: np.ndarray, region: int) -> int:
    """Voxels surviving to the 12x12x12 grid, by build_role_mask's own mechanism.

    A voxel count is not a proxy for this: nearest sampling can keep a structure
    smaller than a token or drop a larger one that misses the sample points.
    """
    lo = F.interpolate(torch.from_numpy(m.transpose(2, 0, 1)).float()[None, None],
                       size=(12, 12, 12), mode="nearest")[:, 0].long()
    return int((lo == region).sum())


def main() -> int:
    lab: dict[str, dict[str, int]] = {}
    csv.field_size_limit(10 ** 9)
    with open(LABELS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = r.get("VolumeName") or r.get("volume_name") or ""
            lab[k.replace(".nii.gz", "")] = r

    names = sorted(f[:-7] for f in os.listdir(MASKS) if f.endswith(".nii.gz"))
    names = [n for n in names if n in lab and os.path.exists(npz_for(n))]

    # Stratify on the rare class. An unstratified sample of 300 volumes gave 11
    # pneumothorax positives, which is not enough to design a region fix against:
    # the positive mean is what the whole question turns on.
    def pos(n, cls="Pneumothorax"):
        try:
            return int(float(lab[n].get(cls, "") or 0)) == 1
        except (TypeError, ValueError):
            return False
    p_, n_ = [x for x in names if pos(x)], [x for x in names if not pos(x)]
    names = p_[:WANT // 2] + n_[::max(1, len(n_) // max(WANT // 2, 1))][:WANT // 2]
    print(f"{len(names)} hacim: {len(p_[:WANT // 2])} pnomotoraks pozitif "
          f"(havuzda {len(p_)}), gerisi negatif\n", flush=True)

    acc: dict[tuple, list] = {}
    empty: dict[int, int] = {8: 0, 9: 0}
    n = 0
    for name in names:
        try:
            m = np.asanyarray(nib.load(f"{MASKS}/{name}.nii.gz").dataobj)
            hu = np.transpose(np.load(npz_for(name))["arr_0"], (1, 2, 0))
            hu = _pad_crop_hwd(hu.astype(np.float32) * 1000.0, (192, 192, 96))
        except Exception:                                     # noqa: BLE001
            continue
        n += 1
        try:
            pos_now = int(float(lab[name].get("Pneumothorax", "") or 0)) == 1
        except (TypeError, ValueError):
            pos_now = False
        for region in (8, 9):
            if tokens(m, region) == 0:
                empty[region] += 1
        # Where is the air the shell was supposed to capture? Air outside the
        # lung labels but inside the body, and how much of it region 8 claims.
        # If the shell misses it, the fix is the shell; if there is none to miss,
        # the fix is elsewhere.
        air = (hu < -400) & (m != 0) & ~np.isin(m, (1, 2, 3, 4, 5))
        body = hu > -900
        air_out = air & body
        got = air_out & (m == 8)
        acc.setdefault(("__air__", 0, int(pos_now)), []).append(
            (float(air_out.sum()), float(got.sum()),
             100.0 * got.sum() / max(air_out.sum(), 1)))
        for cls, region in PROBE:
            v = lab[name].get(cls, "")
            try:
                y = int(float(v))
            except (TypeError, ValueError):
                continue
            sel = m == region
            if not sel.any():
                continue
            acc.setdefault((cls, region, y), []).append(
                (float(hu[sel].mean()), int(sel.sum()), tokens(m, region)))

    print(f"{n} hacim okundu")
    for y, tag in ((1, "pnomotoraks POZITIF"), (0, "pnomotoraks negatif")):
        r = acc.get(("__air__", 0, y), [])
        if r:
            print(f"  {tag:22s} akciger disi hava {np.mean([x[0] for x in r]):9.0f} voksel, "
                  f"bolge 8'in aldigi {np.mean([x[1] for x in r]):8.0f} "
                  f"(%{np.mean([x[2] for x in r]):.1f})")
    print()
    for region in (8, 9):
        print(f"  bolge {region}: 12^3 izgarasinda SIFIR token olan hacim "
              f"{empty[region]}/{n} (%{100 * empty[region] / max(n, 1):.1f}) "
              f"-> o hacimlerde sorgu KISITSIZ, yani yonlendirilmiyor")
    print()
    hdr = (f"{'sinif':32s} {'bolge':>5s} {'etiket':>6s} {'n':>5s} "
           f"{'ort HU':>8s} {'voksel':>9s} {'token':>6s}")
    print(hdr); print("-" * len(hdr))
    for cls, region in PROBE:
        for y in (1, 0):
            rows = acc.get((cls, region, y), [])
            if not rows:
                print(f"{cls[:31]:32s} {region:5d} {y:6d} {0:5d}"); continue
            print(f"{cls[:31]:32s} {region:5d} {y:6d} {len(rows):5d} "
                  f"{np.mean([r[0] for r in rows]):8.0f} "
                  f"{np.mean([r[1] for r in rows]):9.0f} "
                  f"{np.mean([r[2] for r in rows]):6.1f}")
        p = [r[0] for r in acc.get((cls, region, 1), [])]
        q = [r[0] for r in acc.get((cls, region, 0), [])]
        if p and q:
            print(f"{'':32s} {'':5s} {'fark':>6s} {'':5s} "
                  f"{np.mean(p) - np.mean(q):+8.0f} HU  (pozitif - negatif)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

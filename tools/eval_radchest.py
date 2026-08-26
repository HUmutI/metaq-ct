"""RadChest external 18-class eval with CT-RATE-MATCHED preprocessing.

Fixes the preprocessing mismatch in scripts/eval_radchest.py. The original eval
force-resized every RadChest volume to a fixed 240x240x120 grid with
F.interpolate, which makes the effective voxel spacing depend on the native
array size (a 331^2 scan lands at ~1.1mm, a 450^2 scan at ~1.5mm) and matches
CT-RATE's 1.5mm training spacing for none of them. The CNN is not
scale-invariant, so anatomy hit the encoder at the wrong (and inconsistent)
scale.

This version reproduces the CT-RATE cache pipeline exactly
(scripts/batched_ctrate_amax5.py::preproc_npz_one + dataset.py::_load_npz_hwd):

  1. RadChest npz `ct` is 0.8mm isotropic HU, layout (D, H, W) (D = slices).
  2. Resample to CT-RATE cache spacing (1.5, 1.5, 3.0) mm via trilinear, so
     axis0(Z)->3.0mm, axis1/axis2(in-plane)->1.5mm -- yielding the same
     (D, H, W)@(3.0,1.5,1.5) layout as the cache `arr_0`.
  3. transpose -> (H, W, D), then the IDENTICAL dataset.py pad/crop
     (_pad_crop_hwd) to (192,192,96) and 3-channel HU window (_to_3ch_hwd).

The ONLY intended difference from eval_radchest.py is preprocessing, so AUC is
directly comparable (threshold-independent) as a clean A/B test.

NOTE: RadChest HU is clipped to [-1000, 1000] in the public release, so the
bone window [300,2000] is inherently compressed for this dataset. That is a
source-data limitation and is NOT addressed here (no pre-clip source exists).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))

from arcct.dataset import PATHOLOGIES, _pad_crop_hwd, _to_3ch_hwd  # noqa: E402
from evaluate import build_model_for_eval, encode_prompts, prompt_probs  # noqa: E402


# CT-RATE cache spacing (x, y, z) mm -- batched_ctrate_amax5.NPZ_TARGET_SPACING.
CTRATE_SPACING_XYZ = (1.5, 1.5, 3.0)
# RadChest public release: 0.8mm isotropic (CT_Scan_Metadata final_spacing==0.8 for all).
RADCHEST_SPACING = 0.8
SPATIAL_HWD = (192, 192, 96)

# In-plane orientation correction. A spine-landmark analysis over 60 volumes of
# each dataset showed the axial axes are TRANSPOSED between the two datasets:
# CT-RATE has the anterior-posterior axis along W (spine col 0.74, row 0.49),
# while RadChest has A-P along H (spine row 0.69, col 0.49). The model was
# trained on the CT-RATE convention, so RadChest must be transposed (swap H,W)
# to align, optionally with a flip to fix left-right handedness (a bare
# transpose mirrors chirality). The model input is (C,D,H,W); D=dim1, H=dim2,
# W=dim3. Correction order: TRANSPOSE (swap H,W) then FLIP.
#   RAC_RC_TRANSPOSE=1  -> swap H,W (permute dims 2,3)
#   RAC_RC_FLIP="H"     -> flip listed axes AFTER the transpose
# Confirmed correction is transpose + flipH: on the full 3,630-scan cohort it
# lifts 3-seed macro AUC 0.675 -> 0.709 and every position-dependent class
# (cardiomegaly 0.70->0.79, pleural effusion 0.73->0.91, lung nodule up too),
# a uniform gain characteristic of a true orientation fix. Defaults below apply
# it; set RAC_RC_TRANSPOSE=0 RAC_RC_FLIP="" for the raw (native) orientation.
_FLIP_AXIS = {"D": 1, "H": 2, "W": 3}
def _flip_dims_from_env() -> list[int]:
    spec = os.environ.get("RAC_RC_FLIP", "H").strip().upper()
    return [_FLIP_AXIS[c] for c in spec if c in _FLIP_AXIS]
def _transpose_from_env() -> bool:
    return os.environ.get("RAC_RC_TRANSPOSE", "1") == "1"


def _resample_to_ctrate(ct_dhw: np.ndarray) -> np.ndarray:
    """Resample a RadChest (D,H,W) 0.8mm-iso HU volume to CT-RATE (1.5,1.5,3.0)mm.

    Returns (D',H',W') HU, matching the cache arr_0 layout/spacing (before the
    x1000 that the cache omits because it stores HU/1000).
    """
    d, h, w = ct_dhw.shape
    sp_x, sp_y, sp_z = CTRATE_SPACING_XYZ
    td = max(1, round(d * RADCHEST_SPACING / sp_z))   # slices -> 3.0mm
    th = max(1, round(h * RADCHEST_SPACING / sp_y))   # in-plane -> 1.5mm
    tw = max(1, round(w * RADCHEST_SPACING / sp_x))   # in-plane -> 1.5mm
    t = torch.from_numpy(ct_dhw.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(td, th, tw), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


class RadChestDatasetFixed(Dataset):
    """RadChest volumes preprocessed identically to CT-RATE stage-2 training."""

    def __init__(self, npz_dir: str, labels_df: pd.DataFrame, limit: int = 0):
        self.npz_dir = npz_dir
        self.flip_dims = _flip_dims_from_env()
        self.transpose_hw = _transpose_from_env()
        self.samples = []
        for _, row in labels_df.iterrows():
            acc = str(row["NoteAcc_DEID"])
            npz_path = os.path.join(npz_dir, f"{acc}.npz")
            if not os.path.exists(npz_path):
                continue
            lab = np.array([row[p] if not pd.isna(row[p]) else 0.0 for p in PATHOLOGIES], dtype=np.float32)
            mask = np.array([0.0 if pd.isna(row[p]) else 1.0 for p in PATHOLOGIES], dtype=np.float32)
            self.samples.append((npz_path, acc, lab, mask))
            if limit and len(self.samples) >= limit:
                break
        print(f"[radchest-fixed] matched {len(self.samples)} scans against npz dir {npz_dir}; "
              f"transpose_HW={self.transpose_hw} (RAC_RC_TRANSPOSE), flip dims={self.flip_dims} "
              f"(RAC_RC_FLIP='{os.environ.get('RAC_RC_FLIP', '')}')")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, acc, lab, mask = self.samples[idx]
        try:
            ct = np.load(npz_path)["ct"].astype(np.float32)  # (D, H, W) HU, 0.8mm iso
            vol_dhw = _resample_to_ctrate(ct)                # (D',H',W') HU @ (3.0,1.5,1.5)
            del ct
            arr_hwd = np.transpose(vol_dhw, (1, 2, 0))       # (H, W, D) -- cache arr_0 layout post-transpose
            arr_hwd = _pad_crop_hwd(arr_hwd, SPATIAL_HWD)    # identical to dataset._load_npz_hwd
            ct3 = _to_3ch_hwd(arr_hwd)                        # identical windows (lung/soft/bone)
            ct_t = torch.from_numpy(ct3).permute(0, 3, 1, 2).contiguous()  # (3, 96, 192, 192) = (C,D,H,W)
            if self.transpose_hw:                      # swap H(dim1),W(dim2) of (C,D,H,W)
                ct_t = ct_t.permute(0, 1, 3, 2).contiguous()
            if self.flip_dims:                         # flips applied AFTER transpose
                ct_t = torch.flip(ct_t, dims=self.flip_dims).contiguous()
        except Exception as e:
            print(f"[radchest-fixed] WARN {acc}: {e}; excluding (mask=0)", flush=True)
            return torch.zeros((3, 96, 192, 192)), acc, torch.from_numpy(lab), torch.zeros_like(torch.from_numpy(mask))
        return ct_t, acc, torch.from_numpy(lab), torch.from_numpy(mask)


def _radchest_collate(batch):
    cts, accs, labs, masks = zip(*batch)
    return torch.stack(cts), list(accs), torch.stack(labs), torch.stack(masks)


def _bootstrap_macro(pred, true, mask, n_bootstrap, seed, kind, threshold=0.5):
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals = []
        for j in range(pred.shape[1]):
            keep = mask[idx, j] > 0
            yt = true[idx, j][keep]
            ys = pred[idx, j][keep]
            if len(np.unique(yt)) < 2:
                continue
            if kind == "auc":
                vals.append(roc_auc_score(yt, ys))
            else:
                vals.append(f1_score(yt.astype(int), (ys >= threshold).astype(int), zero_division=0))
        if vals:
            boot.append(float(np.mean(vals)))
    boot = np.asarray(boot)
    return float(np.mean(boot)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("EVAL_CKPT", ""))
    ap.add_argument("--out_dir", default=os.environ.get("EVAL_RESULTS_DIR", ""))
    ap.add_argument("--rc_npz_dir", default=os.environ.get(
        "RAC_RADCHEST_NPZ", "/mnt/amax5_drive/alp_ozaydin_0/data/rad_chest_ct/extracted"))
    ap.add_argument("--rc_labels_csv", default=os.environ.get(
        "RAC_RADCHEST_LABELS",
        str(ROOT / "data" / "radchest_ctrate_aligned_full.csv")))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "4")))
    ap.add_argument("--num_workers", type=int, default=int(os.environ.get("EVAL_NUM_WORKERS", "4")))
    ap.add_argument("--n_bootstrap", type=int, default=int(os.environ.get("EVAL_BOOTSTRAP", "1000")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("EVAL_LIMIT", "0")))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert args.ckpt and args.out_dir, "EVAL_CKPT and EVAL_RESULTS_DIR (or --ckpt/--out_dir) required"

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, tokenizer = build_model_for_eval(args.ckpt)
    clip = clip.to(device).eval()
    pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)

    labels_df = pd.read_csv(args.rc_labels_csv)
    ds = RadChestDatasetFixed(args.rc_npz_dir, labels_df, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_radchest_collate, pin_memory=True)

    all_pred, all_true, all_mask, all_acc = [], [], [], []
    with torch.no_grad():
        for ct, acc, lab, mask in tqdm.tqdm(loader, desc="RadChest-fixed eval"):
            ct = ct.to(device, non_blocking=True)
            feat_map = clip.visual_transformer.forward_spatial(ct)
            raw = clip.visual_transformer.global_pool(feat_map)
            img_lat = F.normalize(clip.to_visual_latent(raw), dim=-1)
            probs = prompt_probs(img_lat, pos_embs, neg_embs).float().cpu().numpy()
            all_pred.append(probs)
            all_true.append(lab.numpy())
            all_mask.append(mask.numpy())
            all_acc.extend(acc)

    pred = np.concatenate(all_pred, 0)
    true = np.concatenate(all_true, 0)
    mask = np.concatenate(all_mask, 0)
    accs = np.asarray(all_acc)

    THRESHOLD = 0.5
    per_class, aucs = {}, []
    for j, p in enumerate(PATHOLOGIES):
        keep = mask[:, j] > 0
        yt = true[keep, j].astype(int)
        ys = pred[keep, j]
        n_keep = int(keep.sum())
        if len(np.unique(yt)) < 2:
            per_class[p] = {"auc": float("nan"), "acc": float("nan"), "precision": float("nan"), "f1": float("nan"), "n": n_keep}
            aucs.append(float("nan"))
            continue
        y_hat = (ys >= THRESHOLD).astype(int)
        auc = float(roc_auc_score(yt, ys))
        per_class[p] = {
            "auc": auc,
            "acc": float(accuracy_score(yt, y_hat)),
            "precision": float(precision_score(yt, y_hat, zero_division=0)),
            "f1": float(f1_score(yt, y_hat, zero_division=0)),
            "n": n_keep,
        }
        aucs.append(auc)

    mean_auc = float(np.nanmean(aucs))
    mean_acc = float(np.nanmean([v["acc"] for v in per_class.values() if not np.isnan(v["acc"])]))
    mean_prec = float(np.nanmean([v["precision"] for v in per_class.values() if not np.isnan(v["precision"])]))
    mean_f1 = float(np.nanmean([v["f1"] for v in per_class.values() if not np.isnan(v["f1"])]))
    auc_bm, auc_lo, auc_hi = _bootstrap_macro(pred, true, mask, args.n_bootstrap, args.seed, "auc")
    f1_bm, f1_lo, f1_hi = _bootstrap_macro(pred, true, mask, args.n_bootstrap, args.seed, "f1", THRESHOLD)

    report = {
        "ckpt": args.ckpt, "preproc": "ctrate_matched_resample_1.5_1.5_3.0",
        "n_samples": int(pred.shape[0]), "n_classes": int(pred.shape[1]),
        "n_bootstrap": int(args.n_bootstrap), "bootstrap_unit": "scan", "threshold": THRESHOLD,
        "labels_csv": args.rc_labels_csv, "npz_dir": args.rc_npz_dir,
        "macro": {
            "auc": mean_auc, "auc_bootstrap_mean": auc_bm, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
            "acc": mean_acc, "precision": mean_prec, "f1": mean_f1,
            "f1_bootstrap_mean": f1_bm, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
        },
        "per_class": per_class,
    }
    (Path(args.out_dir) / "radchest_fixed.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(Path(args.out_dir) / "radchest_predictions.npz",
                        pred=pred.astype(np.float32), true=true.astype(np.int8),
                        mask=mask.astype(np.int8), pathologies=np.array(PATHOLOGIES), accessions=accs)

    txt = Path(args.out_dir) / "radchest_fixed_auc.txt"
    with txt.open("w") as f:
        f.write(f"ckpt={args.ckpt}\npreproc=ctrate_matched (resample 0.8mm->1.5/1.5/3.0mm, then dataset pad/crop+window)\n")
        f.write(f"n_samples={pred.shape[0]}  n_classes={pred.shape[1]}  threshold={THRESHOLD}\n\n")
        f.write(f"{'pathology':45s}: {'auc':>8s} {'acc':>8s} {'prec':>8s} {'f1':>8s}  {'n':>6s}\n")
        for p in PATHOLOGIES:
            s = per_class[p]
            def fmt(v):
                return "     n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:8.4f}"
            f.write(f"{p:45s}: {fmt(s['auc'])} {fmt(s['acc'])} {fmt(s['precision'])} {fmt(s['f1'])}  {s['n']:>6d}\n")
        f.write(f"\n{'Mean AUC':45s}: {mean_auc:8.4f}\n{'Mean Acc':45s}: {mean_acc:8.4f}\n")
        f.write(f"{'Mean Precision':45s}: {mean_prec:8.4f}\n{'Mean F1':45s}: {mean_f1:8.4f}\n")
        f.write(f"{'AUC 95% CI (scan bootstrap)':45s}: [{auc_lo:.4f}, {auc_hi:.4f}]\n")
    print(f"[radchest-fixed] Mean AUC={mean_auc:.4f}  saved {txt}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dice between TotalSegmentator output (BCH route, ts_masks/<pid>.nii.gz) and the
Pediatric-CT-SEG expert contours, on the 1.5x1.5x3.0 mm grid the masks were built on.

The BCH resample (pipeline/01_data/resample_for_ts.py) is index-based: it computes
shape = round(in_shape * in_spacing / target) and calls F.interpolate(trilinear,
align_corners=False), discarding the affine. Expert masks therefore go through the
SAME index-based operation with nearest-exact sampling (same sample centres as
align_corners=False), never through an affine registration.

Per patient writes dice/<pid>.json: per-organ Dice (expert structure vs TS class set),
per-region Dice (expert union vs FINE_LABEL_MAP_PEDS10 region), volumes in ml, and
whether each region reaches the 12x12x12 routing grid after build_peds_masks'
nearest downsample (the routing-relevant coverage check). Resumable."""
import argparse, glob, json, os, sys
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
sys.path.insert(0, "/home/ch278233/pipeline/lib")
from ts_roi import FINE_LABEL_MAP_PEDS10, PEDS10_NAMES

X = "/temp_work/ch278233/EXTERNAL/PEDIATRIC_CT_SEG"
TARGET = (1.5, 1.5, 3.0)
# expert structure -> TotalSegmentator 'total' class ids (v2 map; ids as in ts_roi)
ORGAN_MAP = {
    "Lung_L": [10, 11], "Lung_R": [12, 13, 14], "Heart": [51], "Esophagus": [15],
    "Liver": [5], "Spleen": [1], "Kidney_Left": [3], "Kidney_Right": [2],
    "Adrenal_Left": [9], "Adrenal_Right": [8], "Gall_Bladder": [4], "Stomach": [6], "Pancreas": [7],
}
LATERAL = {"Lungs": ("Lung_L", "Lung_R"), "Kidneys": ("Kidney_Left", "Kidney_Right"),
           "Adrenals": ("Adrenal_Left", "Adrenal_Right")}
# expert union -> our merged region id (only regions with a clean expert counterpart)
REGION_MAP = {
    "lungs_all_lobes(1-5)": (["Lung_L", "Lung_R"], [1, 2, 3, 4, 5]),
    "heart_and_great_vessels(7)": (["Heart"], [7]),
    "upper_abdomen(10)": (["Liver", "Spleen", "Kidney_Left", "Kidney_Right", "Adrenal_Left", "Adrenal_Right",
                           "Gall_Bladder", "Stomach", "Pancreas"], [10]),
}
BONE_CLASSES = list(range(92, 118)) + list(range(27, 51)) + [69, 70, 71, 72, 73, 74]   # region 9 minus autochthon

def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")

def resample_mask(m, shape):
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.interpolate(t, size=shape, mode="nearest-exact")[0, 0].numpy() > 0.5

def reaches_grid(m):
    # anatomy_qformer.build_role_mask: (D,H,W) nearest-downsampled to 12x12x12
    lo = F.interpolate(torch.from_numpy(m.transpose(2, 0, 1)).float()[None, None], size=(12, 12, 12), mode="nearest")
    return int((lo > 0).sum())

def run(pid):
    out = f"{X}/dice/{pid}.json"
    if os.path.exists(out): return json.load(open(out))
    ct = nib.load(f"{X}/cap_nii/{pid}.nii.gz")
    zx, zy, zz = (float(v) for v in ct.header.get_zooms()[:3])
    shape = tuple(max(1, round(s * o / t)) for s, o, t in zip(ct.shape[:3], (zx, zy, zz), TARGET))
    ts_img = nib.load(f"{X}/ts_masks/{pid}.nii.gz")
    ts = np.asanyarray(ts_img.dataobj).astype(np.int16)
    R = {"patient_id": pid, "grid_shape": list(shape), "ts_shape": list(ts.shape), "organ": {}, "lateral": {}, "region": {}}
    if tuple(ts.shape) != shape:
        R["status"] = f"shape_mismatch ts={ts.shape} expected={shape}"; json.dump(R, open(out, "w"), indent=1); return R
    vox_ml = float(np.prod(TARGET)) / 1000.0
    reg = np.zeros(ts.shape, np.int16)
    for k, v in FINE_LABEL_MAP_PEDS10.items(): reg[ts == k] = v
    exp = {}
    for name in ORGAN_MAP:
        p = f"{X}/masks/{pid}/{name}.nii.gz"
        if not os.path.exists(p): continue
        m = resample_mask(np.asanyarray(nib.load(p).dataobj) > 0, shape)
        exp[name] = m
        t = np.isin(ts, ORGAN_MAP[name])
        R["organ"][name] = {"dice": dice(m, t), "expert_ml": float(m.sum() * vox_ml), "ts_ml": float(t.sum() * vox_ml),
                            "ts_empty": bool(t.sum() == 0)}
    for lname, (a, b) in LATERAL.items():
        if a in exp and b in exp:
            m = exp[a] | exp[b]; t = np.isin(ts, ORGAN_MAP[a] + ORGAN_MAP[b])
            R["lateral"][lname] = {"dice": dice(m, t), "expert_ml": float(m.sum() * vox_ml), "ts_ml": float(t.sum() * vox_ml)}
    for rname, (names, rids) in REGION_MAP.items():
        have = [n for n in names if n in exp]
        if not have: continue
        m = np.zeros(shape, bool)
        for n in have: m |= exp[n]
        t = np.isin(reg, rids)
        R["region"][rname] = {"dice": dice(m, t), "expert_structures": have, "expert_ml": float(m.sum() * vox_ml),
                              "ts_ml": float(t.sum() * vox_ml)}
    # chest-restricted skeleton: expert 'Bones' vs TS bone classes within the lung z-range
    pb = f"{X}/masks/{pid}/Bones.nii.gz"
    if os.path.exists(pb) and "Lung_L" in exp and "Lung_R" in exp:
        bones = resample_mask(np.asanyarray(nib.load(pb).dataobj) > 0, shape)
        zs = np.where((exp["Lung_L"] | exp["Lung_R"]).any(axis=(0, 1)))[0]
        sl = slice(int(zs.min()), int(zs.max()) + 1)
        t = np.isin(ts[..., sl], BONE_CLASSES)
        R["region"]["skeleton_chest_restricted(9-bone)"] = {"dice": dice(bones[..., sl], t), "expert_ml": float(bones[..., sl].sum() * vox_ml),
                                                            "ts_ml": float(t.sum() * vox_ml), "z_slices": [sl.start, sl.stop]}
    R["region_tokens_12x12x12"] = {PEDS10_NAMES[i]: reaches_grid(reg == i) for i in range(1, 11) if i != 8}
    R["region_ml"] = {PEDS10_NAMES[i]: float((reg == i).sum() * vox_ml) for i in range(1, 11) if i != 8}
    R["status"] = "ok"
    json.dump(R, open(out, "w"), indent=1); return R

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pids", nargs="*"); ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-n", type=int, default=1); a = ap.parse_args()
    os.makedirs(f"{X}/dice", exist_ok=True)
    pids = a.pids or sorted(os.path.basename(p)[:-7] for p in glob.glob(f"{X}/ts_masks/*.nii.gz"))
    pids = pids[a.shard_id::a.shard_n]
    print(f"[dice] {len(pids)} patients (shard {a.shard_id}/{a.shard_n})", flush=True)
    for i, pid in enumerate(pids):
        try:
            r = run(pid)
            lu = r["lateral"].get("Lungs", {}).get("dice", float("nan"))
            print(f"[dice] {i+1}/{len(pids)} {pid} {r['status']} lungs={lu:.3f}", flush=True)
        except Exception as ex:
            print(f"[dice] FAIL {pid}: {type(ex).__name__}: {ex}", flush=True)

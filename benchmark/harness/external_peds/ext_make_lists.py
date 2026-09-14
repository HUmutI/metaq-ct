#!/usr/bin/env python3
"""Select chest-covering Pediatric-CT-SEG studies from the converter's per-patient
JSON and stage them for the BCH mask route.

Inclusion: conversion ok, NIfTI verified (shape + HU range), expert masks written
in the NIfTI's exact geometry, and both expert lung contours present. Writes
  cap_nii/<pid>.nii.gz  -> symlink to nii/<pid>/<SeriesUID>.nii.gz  (stable stems)
  lists/cap_nii.txt     -> input list for resample_for_ts.py
  index.csv             -> one row per included study (age, sex, contrast, scanner...)
  index_excluded.csv    -> what was dropped and why
Re-runnable; only adds."""
import csv, glob, json, os
X = "/temp_work/ch278233/EXTERNAL/PEDIATRIC_CT_SEG"
os.makedirs(f"{X}/cap_nii", exist_ok=True); os.makedirs(f"{X}/lists", exist_ok=True)
rows, excl = [], []
for jf in sorted(glob.glob(f"{X}/logs/convert/*.json")):
    R = json.load(open(jf)); pid = R["patient_id"]
    why = []
    if R.get("status") != "ok": why.append("convert:" + R.get("status", "?"))
    if not R.get("nii_verified"): why.append("nii_unverified")
    if not R.get("mask_affine_match"): why.append("mask_affine_mismatch")
    for k in ("Lung_L", "Lung_R"):
        if R.get("roi", {}).get(k, {}).get("status") != "ok": why.append(f"no_{k}")
    if why:
        excl.append({"patient_id": pid, "reason": ";".join(why)}); continue
    link = f"{X}/cap_nii/{pid}.nii.gz"
    if not os.path.lexists(link): os.symlink(R["nii_path"], link)
    rows.append({
        "patient_id": pid, "age_years": R.get("age_years"), "age_dicom": R.get("patient_age_dicom"),
        "sex": R.get("sex"), "contrast_agent": R.get("contrast_agent"), "manufacturer": R.get("manufacturer"),
        "model": R.get("model"), "kvp": R.get("kvp"), "slice_thickness_mm": R.get("slice_thickness_mm"),
        "pixel_spacing_mm": R.get("pixel_spacing_mm", [None])[0], "shape": "x".join(map(str, R.get("nii_shape", []))),
        "z_extent_mm": R.get("z_extent_mm"), "lung_top_margin_slices": R.get("lung_top_margin_slices"),
        "lung_volume_ml": R.get("lung_volume_ml"), "lung_mean_hu": R.get("lung_mean_hu"),
        "n_roi_ok": sum(1 for e in R["roi"].values() if e.get("status") == "ok"),
        "anomalies": " | ".join(R.get("anomalies", [])), "nii_path": R["nii_path"], "cap_nii": link,
    })
with open(f"{X}/lists/cap_nii.txt", "w") as f:
    for r in rows: f.write(r["cap_nii"] + "\n")
for name, data in (("index.csv", rows), ("index_excluded.csv", excl)):
    if data:
        with open(f"{X}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print(f"included={len(rows)} excluded={len(excl)}  -> {X}/lists/cap_nii.txt")
if excl:
    from collections import Counter
    print("exclusion reasons:", Counter(e["reason"] for e in excl).most_common())

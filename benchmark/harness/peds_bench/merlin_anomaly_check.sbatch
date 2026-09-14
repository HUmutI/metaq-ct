#!/bin/bash
#SBATCH -J merlin_anom
#SBATCH -p bch-compute,bch-compute-pe
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 0:30:00
#SBATCH -o /temp_work/ch278233/PEDS_BENCH/logs/merlin_anom_%j.out
#SBATCH -e /temp_work/ch278233/PEDS_BENCH/logs/merlin_anom_%j.out
source /home/ch278233/BENCHMARK/harness/peds_bench/merlin_env.sh
$PY - <<'PY'
import csv, numpy as np, nibabel as nib
m={}
for r in csv.DictReader(open("/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv"),delimiter="\t"):
    m[r["volume_name"]]=r["nii_path"]
from merlin.data.monai_transforms import ImageTransforms
for name in ["ped_02194_4.nii.gz","ped_02059_7.nii.gz","ped_00002_1.nii.gz"]:
    p=m.get(name)
    print("="*60); print(name, p)
    if not p: print("  NOT IN MAP"); continue
    try:
        img=nib.load(p)
        print("  raw shape",img.shape,"pixdim",np.round(img.header['pixdim'][1:4],4),
              "sform_zooms",np.round(nib.affines.voxel_sizes(img.affine),4),
              "axcodes",nib.aff2axcodes(img.affine))
        d=img.dataobj[...,:1] if img.ndim==3 else None
        out=ImageTransforms({"image":p})["image"]
        a=out.as_tensor().numpy() if hasattr(out,'as_tensor') else np.asarray(out)
        print("  TRANSFORM OK shape",a.shape,"min",float(a.min()),"max",float(a.max()),"mean",round(float(a.mean()),4))
    except Exception as e:
        print("  TRANSFORM FAILED:",type(e).__name__,e)
PY

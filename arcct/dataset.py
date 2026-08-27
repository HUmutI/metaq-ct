"""Dataset and anatomy helpers for the final RAC-CLIP TIA recipe.

The executable pipeline uses the local CT-RATE NPZ cache under
``/mnt/amax5_drive/alp_ozaydin_0/data``. Volumes are stored as HU / 1000 and
are cropped to a paper-faithful CT-RATE spatial grid:

    input cache: [H, W, D] = [192, 192, 96]
    model input: [C, D, H, W] = [3, 96, 192, 192]

The three channels are clinical HU windows: lung, soft tissue, and bone. Fine
TotalSegmentator masks are loaded with the same spatial crop and returned in
model order as [1, D, H, W].
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
import monai.transforms as mt


from arcct import schema as _schema

_ACTIVE = _schema.active()
# Bound from arcct/schema.py, selected by RAC_SCHEMA (default "ctrate").
# Keeping these as module-level names means every existing importer is
# unchanged, while a pediatric run gets 27 labels and the pediatric region
# meanings without a second copy of either list.
PATHOLOGIES = _ACTIVE["PATHOLOGIES"]
N_PATH = len(PATHOLOGIES)


FINE_LABEL_NAMES = _ACTIVE["FINE_LABEL_NAMES"]
N_FINE = 10


PATHOLOGY_FINE_ORGANS = _ACTIVE["PATHOLOGY_FINE_ORGANS"]

# R-Map v2 (T1.15 from FINAL_PIPELINE_2026-06-13.md §3.5). Env-gated, default OFF
# so v10/v11/v12/v13/v14 scripts are unaffected. Three under-routed classes get
# anatomically-faithful organ unions:
#   Lymphadenopathy: [9] -> []        (distributed mediastinal/hilar/axillary)
#   Emphysema:       [1, 3] -> [1-5]  (bilateral, all lobes)
#   Pulm fibrotic:   [2, 5] -> [1-5]  (UIP/NSIP not exclusively basal)
if os.environ.get("RAC_PATHOLOGY_MAP_V2", "0") == "1":
    PATHOLOGY_FINE_ORGANS["Lymphadenopathy"] = []
    PATHOLOGY_FINE_ORGANS["Emphysema"] = [1, 2, 3, 4, 5]
    PATHOLOGY_FINE_ORGANS["Pulmonary fibrotic sequela"] = [1, 2, 3, 4, 5]
    print(f"[RAC] PATHOLOGY_MAP_V2 ENABLED (R-Map): Lymph->[], Emphysema->[1-5], Pulm fibrotic->[1-5]")

ORGAN_TO_PATHIDX: dict[int, list[int]] = {}
for _pname, _organs in PATHOLOGY_FINE_ORGANS.items():
    _pidx = PATHOLOGIES.index(_pname)
    for _organ in _organs:
        ORGAN_TO_PATHIDX.setdefault(_organ, []).append(_pidx)


REGION_KEYWORDS = {
    1: ["left upper", "upper left", "left lung upper", "lingula", "left apic", "left upper lobe"],
    2: ["left lower", "lower left", "left lung lower", "left base", "left basal", "left hemidiaphragm", "left lower lobe"],
    3: ["right upper", "upper right", "right lung upper", "right apic", "right upper lobe"],
    4: ["right middle", "middle lobe", "right middle lobe"],
    5: ["right lower", "lower right", "right lung lower", "right base", "right basal", "right hemidiaphragm", "right lower lobe"],
    6: ["trachea", "carina", "main bronch", "tracheal"],
    7: ["heart", "cardiac", "pericardial", "cardiomegal", "atrial", "ventricle", "myocardial", "pericardium"],
    8: ["aorta", "aortic", "coronary"],
    9: ["pulmonary artery", "pulmonary vein", "vascular", "mediastinal vessel", "lymph", "lymphadenopathy", "lymph node", "subclavian", "brachiocephalic"],
    10: ["esophagus", "esophageal", "hiatal hernia", "gastroesophageal"],
}

if _ACTIVE.get("REGION_KEYWORDS"):
    REGION_KEYWORDS = _ACTIVE["REGION_KEYWORDS"]
LUNG_GENERAL_KEYWORDS = [
    "lung", "pulmonary", "pleural", "bronch", "emphysema", "consolidation",
    "atelectasis", "nodule", "opacity", "fibrosis", "fibrotic", "interlobular",
    "mosaic", "septal", "air space", "airspace", "lobe",
]


def localize_findings(findings_text: str, organ_label: int, max_chars: int = 400) -> str:
    """Fallback rule-based organ sentence extraction."""
    sentences = re.split(r"(?<=[.!?])\s+", findings_text.strip())
    keywords = REGION_KEYWORDS.get(organ_label, [])
    relevant = []
    for sent in sentences:
        s = sent.lower()
        if any(k in s for k in keywords):
            relevant.append(sent)
        elif organ_label in range(1, 6) and any(k in s for k in LUNG_GENERAL_KEYWORDS):
            relevant.append(sent)
    text = " ".join(relevant) if relevant else findings_text
    return text[:max_chars]


def _coerce_region_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, dict):
        for key in ("text", "sentence", "sentences", "findings"):
            if key in value:
                return _coerce_region_text(value[key])
    return str(value)


def load_region_cache(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        cache = json.load(f)
    print(f"[RAC] Loaded region cache: {path} ({len(cache):,} top-level keys)")
    return cache


# When the region cache has an entry for this volume but the organ's list is
# empty, treat that as "the report says nothing here" instead of falling back
# to keyword matching over the whole findings text.
RESPECT_EMPTY_REGION = os.environ.get("RAC_RESPECT_EMPTY_REGION", "0") == "1"


def region_text_from_cache(cache: dict[str, Any], accession: str, organ_label: int, findings_text: str) -> str:
    """Return cached LLM organ text if present, otherwise keyword fallback."""
    keys = (
        f"{accession}|{organ_label}",
        f"{accession}:{organ_label}",
        f"{accession.replace('.nii.gz', '')}|{organ_label}",
        f"{accession.replace('.nii.gz', '')}:{organ_label}",
    )
    for key in keys:
        text = _coerce_region_text(cache.get(key))
        if text.strip():
            return text

    nested = cache.get(accession) or cache.get(accession.replace(".nii.gz", ""))
    hit_nested = isinstance(nested, dict)
    if hit_nested:
        for key in (str(organ_label), organ_label, FINE_LABEL_NAMES.get(organ_label, "")):
            text = _coerce_region_text(nested.get(key))
            if text.strip():
                return text

    # An entry that EXISTS and is empty is a verdict, not a miss: the LLM read
    # this report and said it says nothing about this organ. Falling through to
    # localize_findings overrules that verdict with keyword matching, and when
    # no keyword matches localize_findings returns the whole findings text
    # truncated to 400 characters - so the organ gets aligned to the entire
    # report. Measured on the pediatric cache: 7,815 of 88,170 (volume, organ)
    # pairs are empty lists, 8.9%, touching 3,615 of 8,817 volumes.
    #
    # Off by default: it changes the align target for ~9% of pairs, so it must
    # not silently move a number somebody is comparing against.
    if hit_nested and RESPECT_EMPTY_REGION:
        region_text_from_cache.n_empty_respected += 1
        return ""

    return localize_findings(findings_text, organ_label)


region_text_from_cache.n_empty_respected = 0


def _npz_to_mask_path(npz_path: str, mask_root: str) -> str:
    p = Path(npz_path)
    try:
        idx = list(p.parts).index("mps_ct_npz")
        rel = os.path.join(*p.parts[idx + 2:])
    except ValueError:
        rel = p.name
    return os.path.join(mask_root, rel.replace(".npz", ".nii.gz"))


def _pad_crop_hwd(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    t = mt.Compose([
        mt.ResizeWithPadOrCrop(spatial_size=(240, 240, 120)),
        mt.CenterSpatialCrop(roi_size=target_shape),
    ])
    return np.asarray(t(arr[np.newaxis].astype(np.float32))[0])


def _load_fine_mask(mask_path: str, target_shape=(192, 192, 96), fail_fast=True):
    try:
        img = nib.load(mask_path)
        data = np.round(img.get_fdata()).astype(np.uint8)
        if data.ndim == 4:
            data = data[..., 0]
        if data.shape != target_shape:
            data = np.round(_pad_crop_hwd(data, target_shape)).astype(np.uint8)
        return data
    except Exception:
        if fail_fast:
            raise
        return None


def _window_hu(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    hu = np.clip(hu, lo, hi)
    return ((hu - lo) / max(hi - lo, 1.0)).astype(np.float32)


def _to_3ch_hwd(hu: np.ndarray) -> np.ndarray:
    lung_lo = float(os.environ.get("RAC_HU_LUNG_LO", "-1000"))
    lung_hi = float(os.environ.get("RAC_HU_LUNG_HI", "400"))
    return np.stack([
        _window_hu(hu, lung_lo, lung_hi),
        _window_hu(hu, -160.0, 240.0),
        _window_hu(hu, 300.0, 2000.0),
    ], axis=0)


class RACDatasetV4(Dataset):
    """CT-RATE NPZ dataset with 3-channel HU windows and optional TS masks."""

    SPATIAL_HWD = (192, 192, 96)
    PAD_HWD = (240, 240, 120)

    def __init__(
        self,
        data_folder,
        reports_csv,
        labels_csv,
        mask_root=None,
        region_cache_path="",
        is_train=True,
        limit=0,
        fail_fast=True,
        volume_list_path="",
        context_csv="",
        demographics_csv="",
        ctx_seed=None,
        volume_exclude_path="",
    ):
        self.data_folder = data_folder
        self.mask_root = mask_root
        self.is_train = is_train
        self.fail_fast = fail_fast
        self.region_cache = load_region_cache(region_cache_path)
        self.volume_list = None
        if volume_list_path and os.path.isfile(volume_list_path):
            with open(volume_list_path) as f:
                self.volume_list = {ln.strip() for ln in f if ln.strip()}
            print(f"[CTRATE] volume_list filter active: {len(self.volume_list)} vols ({volume_list_path})", flush=True)
        # -- clinical context ------------------------------------------------
        # Loaded here, looked up by accession in __getitem__. The sample tuples
        # are indexed positionally elsewhere (s[5]), so widening them is the
        # riskier edit; a side table matches how acc2text/acc2label already work.
        from arcct.context import NO_INDICATION, age_to_band     # noqa: PLC0415
        self.NO_INDICATION = NO_INDICATION
        self.acc2ind, self.acc2ind_status = {}, {}
        self.acc2band, self.acc2sex = {}, {}
        self.ind_pool = []
        self.ind_cohort = {}
        context_csv = context_csv or os.environ.get("RAC_CONTEXT_CSV", "")
        demographics_csv = demographics_csv or os.environ.get("RAC_DEMOGRAPHICS_CSV", "")
        if context_csv and os.path.isfile(context_csv):
            df_ctx = pd.read_csv(context_csv, keep_default_na=False)
            for _, row in df_ctx.iterrows():
                acc = row["VolumeName"]
                status = str(row.get("ind_status", "") or "absent")
                txt = str(row.get("Indication_EN", "") or "").strip()
                if status != "present" or not txt:
                    txt, status = NO_INDICATION, status or "absent"
                self.acc2ind[acc] = txt
                self.acc2ind_status[acc] = status
                if status == "present":
                    self.ind_pool.append(acc)
                    self.ind_cohort[acc] = str(row.get("cohort", "") or "")
            print(f"[ctx] indications: {len(self.acc2ind)} rows, "
                  f"{len(self.ind_pool)} usable ({context_csv})", flush=True)
        if demographics_csv and os.path.isfile(demographics_csv):
            df_dem = pd.read_csv(demographics_csv, keep_default_na=False)
            for _, row in df_dem.iterrows():
                acc = row["VolumeName"]
                try:
                    self.acc2band[acc] = int(row["AgeBand"])
                except (KeyError, TypeError, ValueError):
                    self.acc2band[acc] = age_to_band(row.get("AgeYears"))
                try:
                    self.acc2sex[acc] = int(row["SexIdx"])
                except (KeyError, TypeError, ValueError):
                    self.acc2sex[acc] = 2
            print(f"[ctx] demographics: {len(self.acc2band)} rows ({demographics_csv})",
                  flush=True)

        # Dropout and the counterfactual partner are drawn from a seed, an epoch
        # and the sample index -- never from random.random(). This trainer
        # auto-resumes on every SLURM requeue, and Python's global RNG state is
        # not in the checkpoint, so a naive draw would silently change schedule
        # mid-run and differ with the worker count.
        self.ctx_seed = int(os.environ.get("RAC_CTX_SEED", os.environ.get("RAC_SEED", "0"))
                            if ctx_seed is None else ctx_seed)
        self.epoch = 0
        self.ind_dropout = float(os.environ.get("RAC_IND_DROPOUT", "0.0")) if is_train else 0.0
        self.cf_prob = float(os.environ.get("RAC_CF_PROB", "0.0")) if is_train else 0.0
        self.cf_same_cohort = os.environ.get("RAC_CF_SAME_COHORT", "1") == "1"
        # Evaluation conditions (§ four conditions): none | true | shuffled | off.
        # Applied once, at construction, so an eval always re-runs to the same
        # number; never as per-batch randomness.
        self.eval_ind_mode = os.environ.get("RAC_EVAL_IND_MODE", "true").lower()
        if not is_train and self.eval_ind_mode in ("none", "off"):
            self.acc2ind = {k: NO_INDICATION for k in self.acc2ind}
        elif not is_train and self.eval_ind_mode == "shuffled":
            import random as _r                                   # noqa: PLC0415
            keys = sorted(self.ind_pool)
            vals = [self.acc2ind[k] for k in keys]
            _r.Random(self.ctx_seed).shuffle(vals)
            self.acc2ind.update(dict(zip(keys, vals)))
            print(f"[ctx] EVAL indication mode=shuffled over {len(keys)} rows", flush=True)

        self.exclude = set()
        volume_exclude_path = volume_exclude_path or os.environ.get("RAC_VOLUME_EXCLUDE", "")
        if volume_exclude_path and os.path.isfile(volume_exclude_path):
            with open(volume_exclude_path) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln and not ln.startswith("#"):
                        self.exclude.add(ln.split("\t")[0].strip())
            print(f"[ctx] volume exclude list: {len(self.exclude)} volumes", flush=True)

        self.enable_flips = is_train and os.environ.get("RAC_ENABLE_ANATOMY_FLIPS", "0") == "1"
        self.require_mask = is_train and os.environ.get("RAC_REQUIRE_MASK", "0") == "1"
        # Stage-2 augmentation pack (train-only, defaults off).
        self.enable_random_crop = is_train and os.environ.get("RAC_RANDOM_CROP", "0") == "1"
        self.intensity_jitter_hu = float(os.environ.get("RAC_INTENSITY_JITTER_HU", "0")) if is_train else 0.0
        self.gaussian_noise_sigma = float(os.environ.get("RAC_GAUSSIAN_NOISE_SIGMA", "0")) if is_train else 0.0

        if not os.path.isdir(data_folder):
            raise FileNotFoundError(f"Missing data folder: {data_folder}")
        if not os.path.isfile(reports_csv):
            raise FileNotFoundError(f"Missing reports CSV: {reports_csv}")
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")

        df_rep = pd.read_csv(reports_csv)
        self.acc2text = {}
        self.acc2findings = {}
        self.acc2impressions = {}
        for _, row in df_rep.iterrows():
            acc = row["VolumeName"]
            findings = str(row.get("Findings_EN", "") or "")
            impressions = str(row.get("Impressions_EN", "") or "")
            # T1.10 + §3.3 option B: impression-first concat (env-gated, default OFF).
            # When ON, the diagnostic-conclusion phrase enters the first tokens of the
            # 256/384 budget instead of getting truncated in the tail. Targets the four
            # weakest v10 classes (Pulm fibrotic 0.6905, Lung nodule 0.7374, Lymph
            # 0.7875, Mosaic 0.8017) whose pathognomonic language lives in impressions.
            if os.environ.get("RAC_IMPRESSION_FIRST", "0") == "1":
                concat = (impressions + " " + findings).strip()
            else:
                concat = (findings + " " + impressions).strip()
            text = re.sub(r"[\"'()]", "", concat)
            self.acc2text[acc] = text
            self.acc2findings[acc] = re.sub(r"[\"'()]", "", findings)
            self.acc2impressions[acc] = re.sub(r"[\"'()]", "", impressions)

        df_lab = pd.read_csv(labels_csv)
        # A label column that is absent from the CSV used to become a silent 0.0
        # for every row via row.get(c, 0.0) - the same fabricated negative that
        # made "Bone lesion or fracture" score below chance when the adult half
        # of the combined set was written as all-zeros. Fail instead.
        _missing = [c for c in PATHOLOGIES if c not in df_lab.columns]
        if _missing:
            raise ValueError("%s is missing label columns: %s" % (labels_csv, _missing))
        self.acc2label = {}
        for _, row in df_lab.iterrows():
            acc = row["VolumeName"]
            # `float(x or 0.0)` is a trap: NaN is truthy, so a blank cell
            # already leaked through as NaN by accident rather than by design.
            # Make it deliberate - a blank cell means UNLABELED, which the
            # masked loss skips, and is how CT-RATE rows carry the nine classes
            # it has no published labels for. Writing 0 there instead taught the
            # model that adults never have a bone lesion; 22.8% of them do.
            vals = [(float(row[c]) if pd.notna(row[c]) else float("nan"))
                    for c in PATHOLOGIES]
            self.acc2label[acc] = np.asarray(vals, dtype=np.float32)

        skipped_report = skipped_label = skipped_nomask = skipped_excluded = 0
        self.samples = []
        stop_scan = False
        for pf in tqdm.tqdm(sorted(glob.glob(os.path.join(data_folder, "*"))),
                            desc=f"Scan {os.path.basename(data_folder)}", leave=False):
            if stop_scan:
                break
            for af in sorted(glob.glob(os.path.join(pf, "*"))):
                if stop_scan:
                    break
                for npz in sorted(glob.glob(os.path.join(af, "*.npz"))):
                    acc = os.path.basename(npz).replace(".npz", ".nii.gz")
                    if self.volume_list is not None and acc not in self.volume_list:
                        continue
                    if acc in self.exclude:
                        skipped_excluded += 1
                        continue
                    if acc not in self.acc2text:
                        skipped_report += 1
                        continue
                    if acc not in self.acc2label:
                        skipped_label += 1
                        continue
                    mask_path = None
                    if mask_root:
                        mp = _npz_to_mask_path(npz, mask_root)
                        if os.path.exists(mp):
                            mask_path = mp
                    if self.require_mask and mask_path is None:
                        skipped_nomask += 1
                        continue
                    self.samples.append((
                        npz,
                        acc,
                        self.acc2text[acc],
                        self.acc2findings.get(acc, ""),
                        self.acc2label[acc],
                        mask_path,
                    ))
                    if limit and limit > 0 and len(self.samples) >= limit:
                        stop_scan = True
                        break

        n_masked = sum(1 for s in self.samples if s[5])
        print(
            f"[RAC] {os.path.basename(data_folder)}: {len(self.samples):,} samples, "
            f"{n_masked:,} with masks, skipped_report={skipped_report:,}, "
            f"skipped_label={skipped_label:,}, skipped_nomask={skipped_nomask:,}, "
            f"skipped_excluded={skipped_excluded:,} "
            f"(require_mask={self.require_mask})"
        )
        if not self.samples:
            raise RuntimeError(f"No usable samples found in {data_folder}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_npz_hwd(npz_path: str) -> np.ndarray:
        arr = np.load(npz_path)["arr_0"]
        arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        # Pad with air, not with zero. MONAI's ResizeWithPadOrCrop fills with 0,
        # and this array is HU/1000, so a plain pad reads as 0 HU - soft tissue.
        # For an adult at ~240x240 that is a thin rim; measured on the pediatric
        # cohort the pad occupies a median 18% of the 192x192x96 input and up to
        # 77% for the smallest scans, so a small child arrives wrapped in a box
        # of fake tissue. Shifting by +1 puts air at 0 for the pad, then undoing
        # the shift leaves the padded voxels at -1.0 = -1000 HU.
        # Put both cohorts on the same air floor before anything else sees them.
        # dcm2niix writes -3024 HU for out-of-FOV voxels, which 8% of the
        # pediatric cohort carries, while CT-RATE bottoms out at -1024. With
        # RAC_HU_LUNG_LO=-1500 that difference is not cosmetic: pediatric air
        # clips to 0.000 in the lung channel while CT-RATE air sits at 0.238 -
        # the same tissue arriving as two different numbers inside one batch.
        # Nothing is denser-than-vacuum below air, so clamping loses no signal.
        # Gated on the pediatric schema. Both of these are corrections the
        # pediatric cohort needs - its out-of-FOV padding is -3024 HU and small
        # patients are mostly frame - but applying them to adults changes the
        # tensor the released checkpoints were trained on. Measured on CT-RATE
        # valid: about a third of volumes shift, one by 4.17% of voxels with a
        # max normalised delta of 0.5, which is enough that evaluate.py would no
        # longer reproduce 0.8524 bit-for-bit. The adult path is a verified
        # anchor; it does not move without a reason of its own.
        if _ACTIVE["name"] == "peds":
            AIR = 1.024                  # -1024 HU, the CT-RATE floor
            arr = np.maximum(arr, -AIR)
            arr = _pad_crop_hwd(arr + AIR, RACDatasetV4.SPATIAL_HWD) - AIR
        else:
            arr = _pad_crop_hwd(arr, RACDatasetV4.SPATIAL_HWD)
        return arr * 1000.0

    def _random_crop_192_from_240(self, ct_chwd: torch.Tensor, mask_hwd: torch.Tensor):
        """Pad to PAD_HWD then take a random SPATIAL_HWD window (paired image+mask).

        ct_chwd: (C, H, W, D) in SPATIAL_HWD geometry.
        mask_hwd: (H, W, D) uint8 in SPATIAL_HWD geometry.
        Returns the same shapes after random crop within the padded volume.
        """
        if not self.enable_random_crop:
            return ct_chwd, mask_hwd
        Ht, Wt, Dt = self.SPATIAL_HWD
        Hp, Wp, Dp = self.PAD_HWD
        # Center-pad ct and mask into PAD_HWD.
        pad_h = (Hp - Ht) // 2
        pad_w = (Wp - Wt) // 2
        pad_d = (Dp - Dt) // 2
        ct_padded = torch.zeros(ct_chwd.shape[0], Hp, Wp, Dp, dtype=ct_chwd.dtype)
        ct_padded[:, pad_h:pad_h + Ht, pad_w:pad_w + Wt, pad_d:pad_d + Dt] = ct_chwd
        mask_padded = torch.zeros(Hp, Wp, Dp, dtype=mask_hwd.dtype)
        mask_padded[pad_h:pad_h + Ht, pad_w:pad_w + Wt, pad_d:pad_d + Dt] = mask_hwd
        # Sample top-left corner uniformly inside the padded volume.
        h0 = int(np.random.randint(0, Hp - Ht + 1))
        w0 = int(np.random.randint(0, Wp - Wt + 1))
        d0 = int(np.random.randint(0, Dp - Dt + 1))
        ct_out = ct_padded[:, h0:h0 + Ht, w0:w0 + Wt, d0:d0 + Dt].contiguous()
        mask_out = mask_padded[h0:h0 + Ht, w0:w0 + Wt, d0:d0 + Dt].contiguous()
        return ct_out, mask_out

    def _intensity_jitter(self, ct_chwd: torch.Tensor, hu_delta: float) -> torch.Tensor:
        """Per-channel HU shift drawn uniform[-hu_delta, +hu_delta].

        The cached volumes are stored as HU windowed to [0, 1] per channel, so
        shifting by raw HU units would over-shift the normalized signal. By
        default (v10 behavior) we rescale hu_delta by 1400 HU globally
        (lung-window width) — which under-jitters bone (1700 HU) and over-
        jitters soft tissue (400 HU).

        When ``RAC_JITTER_PER_CHANNEL=1`` (T1.7 fix from
        ``FINAL_PIPELINE_2026-06-13.md`` §1), each channel is rescaled by its
        own window width: lung=1400, soft=400, bone=1700. Env-gated, default
        OFF so v10/v11/v12/v13/v14 scripts produce bit-identical jitter to
        before this fix.
        """
        if hu_delta <= 0.0:
            return ct_chwd
        C = ct_chwd.shape[0]
        if os.environ.get("RAC_JITTER_PER_CHANNEL", "0") == "1" and C == 3:
            channel_widths = np.array([1400.0, 400.0, 1700.0], dtype=np.float32)
            scales = hu_delta / channel_widths
        else:
            scales = np.full((C,), hu_delta / 1400.0, dtype=np.float32)
        shifts = np.random.uniform(-scales, scales, size=(C,)).astype(np.float32)
        shifts_t = torch.from_numpy(shifts).view(C, 1, 1, 1)
        return (ct_chwd + shifts_t).clamp_(0.0, 1.0)

    def _gaussian_noise(self, ct_chwd: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0.0:
            return ct_chwd
        noise = np.random.normal(0.0, sigma, size=ct_chwd.shape).astype(np.float32)
        return (ct_chwd + torch.from_numpy(noise)).clamp_(0.0, 1.0)

    def _maybe_joint_flip(self, ct_chwd: torch.Tensor, mask_hwd: torch.Tensor):
        if not self.enable_flips:
            return ct_chwd, mask_hwd
        if random.random() < 0.2:
            ct_chwd = torch.flip(ct_chwd, dims=(1,))
            mask_hwd = torch.flip(mask_hwd, dims=(0,))
        if random.random() < 0.2:
            ct_chwd = torch.flip(ct_chwd, dims=(2,))
            mask_hwd = torch.flip(mask_hwd, dims=(1,))
        return ct_chwd, mask_hwd

    def set_epoch(self, epoch: int) -> None:
        """Advance the context RNG stream.

        The training loop must call this before re-creating the DataLoader
        iterator. With ``persistent_workers=True`` the workers hold a COPY of
        this object and will never see the change, so the dropout mask would be
        frozen for the whole run -- a fixed 30% of volumes that never see their
        indication, which is a different and worse experiment. Contexts must be
        built with ``persistent_workers=False``; preflight asserts it.
        """
        self.epoch = int(epoch)

    def _ctx_rng(self, idx: int):
        import random                                          # noqa: PLC0415
        return random.Random((self.ctx_seed * 1000003 + self.epoch * 7919 + idx)
                             & 0x7FFFFFFF)

    def _context_for(self, idx: int, accession: str) -> dict:
        """Indication, counterfactual partner, age band and sex for one sample."""
        status = self.acc2ind_status.get(accession, "absent")
        ind = self.acc2ind.get(accession, self.NO_INDICATION)
        rng = self._ctx_rng(idx)

        dropped = False
        if self.ind_dropout > 0.0 and status == "present":
            # Rows that are already vacuous or absent are NOT counted as
            # dropped. Counting them would put the effective rate on the adult
            # half -- where 75.9% of CT-RATE indications are vacuous -- far above
            # the configured one, in a way no log would show.
            dropped = rng.random() < self.ind_dropout
            if dropped:
                ind = self.NO_INDICATION

        cf = ""
        # Drawn from the same stream immediately after the dropout draw, so both
        # are reproducible; and never against a dropped row, because a
        # counterfactual paired with "no indication" is not the experiment.
        if self.cf_prob > 0.0 and not dropped and status == "present" and self.ind_pool:
            if rng.random() < self.cf_prob:
                pool = self.ind_pool
                if self.cf_same_cohort:
                    mine = self.ind_cohort.get(accession, "")
                    same = [a for a in pool if self.ind_cohort.get(a, "") == mine]
                    pool = same or pool
                for _ in range(8):
                    pick = pool[rng.randrange(len(pool))]
                    if pick != accession:
                        cf = self.acc2ind.get(pick, "")
                        break

        return {
            "indication": ind,
            "indication_cf": cf,
            "age_band": int(self.acc2band.get(accession, -1)),
            "sex": int(self.acc2sex.get(accession, -1)),
            "ind_dropped": bool(dropped),
            "ind_status": status,
        }

    def __getitem__(self, idx):
        npz_path, accession, text, findings, label, mask_path = self.samples[idx]
        try:
            hu_hwd = self._load_npz_hwd(npz_path)
            ct_chwd = torch.from_numpy(_to_3ch_hwd(hu_hwd))
        except Exception:
            if self.fail_fast:
                raise
            ct_chwd = torch.zeros(3, *self.SPATIAL_HWD, dtype=torch.float32)

        mask_hwd = torch.zeros(self.SPATIAL_HWD, dtype=torch.uint8)
        has_mask = False
        if mask_path:
            m = _load_fine_mask(mask_path, self.SPATIAL_HWD, fail_fast=self.fail_fast)
            if m is not None and m.max() > 0:
                mask_hwd = torch.from_numpy(m.astype(np.uint8))
                has_mask = True

        # Stage-2 augmentation pack (train-only; helpers no-op when disabled).
        ct_chwd, mask_hwd = self._random_crop_192_from_240(ct_chwd, mask_hwd)
        ct_chwd = self._intensity_jitter(ct_chwd, self.intensity_jitter_hu)
        ct_chwd = self._gaussian_noise(ct_chwd, self.gaussian_noise_sigma)
        ct_chwd, mask_hwd = self._maybe_joint_flip(ct_chwd, mask_hwd)
        ct = ct_chwd.permute(0, 3, 1, 2).contiguous()
        mask_fine = mask_hwd.permute(2, 0, 1).contiguous().unsqueeze(0)
        # The context dict is element 8 and is returned UNCONDITIONALLY, even
        # when no context CSV was loaded -- then it carries NO_INDICATION and
        # age_band/sex = -1. An environment variable that changes the tuple
        # LENGTH is the defect class this codebase keeps writing comments about.
        return (
            ct,
            text,
            findings,
            torch.tensor(label, dtype=torch.float32),
            mask_fine,
            torch.tensor(has_mask, dtype=torch.bool),
            accession,
            self._context_for(idx, accession),
        )


def rac_collate(batch):
    cts, texts, findings, labels, masks, has_masks, accessions, ctxs = zip(*batch)
    ctx = {
        "indication": [c["indication"] for c in ctxs],
        "indication_cf": [c["indication_cf"] for c in ctxs],
        "age_band": torch.tensor([c["age_band"] for c in ctxs], dtype=torch.long),
        "sex": torch.tensor([c["sex"] for c in ctxs], dtype=torch.long),
        "ind_dropped": torch.tensor([c["ind_dropped"] for c in ctxs], dtype=torch.bool),
        "ind_status": [c["ind_status"] for c in ctxs],
    }
    return (
        torch.stack(cts),
        list(texts),
        list(findings),
        torch.stack(labels),
        torch.stack(masks),
        torch.stack(has_masks),
        list(accessions),
        ctx,
    )

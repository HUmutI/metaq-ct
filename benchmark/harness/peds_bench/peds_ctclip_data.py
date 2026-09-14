"""Drop-in CTReportDataset / CTReportDatasetinfer for the pediatric cohort.

Same constructor signatures and same __getitem__ return tuples as CT-CLIP's own
classes, so CTCLIPTrainer is used completely unmodified -- model, loss,
optimizer, schedule, grad clipping and checkpointing are all CT-CLIP's.

Two things differ from upstream, both forced by our data layout, neither
touching the model:
  * volumes come from the cached tensors written by prep_ctclip.py (verified
    equivalent to CT-CLIP's own nii_img_to_tensor to within float16), instead of
    being resampled from NIfTI inside __getitem__;
  * the cohort is defined by an explicit vollist, because our pediatric tree is
    flat rather than CT-RATE's patient/accession/volume nesting.

Deliberately replicated upstream quirk: data.py:79-82 builds `input_text_concat`
by concatenating Findings and Impressions and then immediately overwrites it
with `impression_text[0]`, so CT-CLIP is in fact trained on Findings_EN alone.
We reproduce that exactly. "Fixing" it here would train a different model from
the published one.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CACHE = "/temp_work/ch278233/PEDS_BENCH/ctclip_npz"


def _texts(reports_file):
    df = pd.read_csv(reports_file)
    return {r["VolumeName"]: (r["Findings_EN"], r["Impressions_EN"]) for _, r in df.iterrows()}


def _load(stem, cache):
    a = np.load(os.path.join(cache, stem + ".npz"))["arr"].astype(np.float32)
    return torch.from_numpy(a).unsqueeze(0)        # (1,240,480,480)


def _clean(t):
    t = str(t)
    for ch in ('"', "'", "(", ")"):
        t = t.replace(ch, "")
    return t


class PedsCTReportDataset(Dataset):
    def __init__(self, data_folder, reports_file, meta_file=None, vollist=None, cache=CACHE, **kw):
        self.cache = cache
        a2t = _texts(reports_file)
        vols = [l.strip() for l in open(vollist or data_folder) if l.strip()]
        self.samples = []
        missing_txt = missing_vol = 0
        for v in vols:
            stem = v[:-7] if v.endswith(".nii.gz") else v
            if v not in a2t:
                missing_txt += 1
                continue
            if not os.path.exists(os.path.join(cache, stem + ".npz")):
                missing_vol += 1
                continue
            pair = a2t[v]
            if pair == "Not given.":
                pair = ""
            concat = ""
            for t in pair:
                concat = concat + str(t)
            concat = pair[0]                  # upstream data.py:82 -- Findings only
            self.samples.append((stem, concat))
        print(f"[peds-ds] train n={len(self.samples)} (no-report={missing_txt}, no-volume={missing_vol})",
              flush=True)
        if not self.samples:
            raise RuntimeError("no pediatric training samples resolved")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        stem, text = self.samples[i]
        return _load(stem, self.cache), _clean(text)


class PedsCTReportDatasetInfer(Dataset):
    def __init__(self, data_folder, reports_file, meta_file=None, labels=None,
                 vollist=None, cache=CACHE, **kw):
        self.cache = cache
        a2t = _texts(reports_file)
        ldf = pd.read_csv(labels)
        self.classes = [c for c in ldf.columns if c != "VolumeName"]
        lab = {r["VolumeName"]: np.array([r[c] for c in self.classes], dtype=np.float32)
               for _, r in ldf.iterrows()}
        vols = [l.strip() for l in open(vollist or data_folder) if l.strip()]
        self.samples = []
        for v in vols:
            stem = v[:-7] if v.endswith(".nii.gz") else v
            if v not in lab or not os.path.exists(os.path.join(cache, stem + ".npz")):
                continue
            pair = a2t.get(v, ("", ""))
            self.samples.append((stem, pair[0], lab[v]))
        print(f"[peds-ds] valid n={len(self.samples)} classes={len(self.classes)}", flush=True)
        if not self.samples:
            raise RuntimeError("no pediatric validation samples resolved")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        stem, text, onehot = self.samples[i]
        return _load(stem, self.cache), _clean(text), onehot, stem

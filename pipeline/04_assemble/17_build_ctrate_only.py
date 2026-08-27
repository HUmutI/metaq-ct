#!/usr/bin/env python3
"""Assemble a CT-RATE-ONLY training set, on the published 18-class labels.

Why this exists. The number we have to beat -- ARC-CT's 0.8574 on the 18 classes
CT-RATE publishes -- was produced by a model trained on CT-RATE alone. Our best
attempt so far, C47harm, reaches 0.8442 with pediatrics mixed in, and mixing is a
plausible reason for the gap: in the combined split CT-RATE outnumbers pediatrics
six to one, but the pediatric half still pulls the shared representation toward a
different age range and a different label distribution. Removing it makes the
architecture the only thing that differs from the published model.

Three things this keeps like-for-like, because otherwise the comparison is void:

  * LABELS are the PUBLISHED ones, not ours. train_predicted_labels.csv and
    valid_predicted_labels.csv, 18 classes. Our own 27-class labeller does not
    appear anywhere in this run -- training on our labels and reporting against
    a published gate would be marking our own homework.
  * The SPLIT is the one already in use. The combined split holds out 4,589
    CT-RATE volumes from the TRAIN half for checkpoint selection; the official
    valid_* volumes are never trained on and stay the held-out set the 0.8574
    gate is measured on. Selecting on the official valid set would quietly turn
    the gate into a training signal.
  * The CONTEXT is what CT-RATE actually carries: indication where it exists
    (48.7% of volumes), age and sex from the DICOM metadata.

Writes to $RAC_WORK/CTRATE_ONLY/ and asserts before it replaces anything.
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

WORK = os.environ.get("RAC_WORK", "/temp_work/ch278233")
OUT = os.path.join(WORK, "CTRATE_ONLY")
PUB = os.path.join(WORK, "CTRATE/labels_hf/dataset/multi_abnormality_labels")

CTRATE18 = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]


def stem(x: str) -> str:
    return x.strip().replace(".nii.gz", "")


def read_list(path: str, prefix: str) -> list[str]:
    return [stem(l) for l in open(path, encoding="utf-8")
            if stem(l).startswith(prefix)]


def main() -> int:
    csv.field_size_limit(10 ** 9)
    os.makedirs(OUT, exist_ok=True)

    train = read_list(f"{WORK}/COMBINED_VOLLIST_TRAIN_clean.txt", "train_")
    valid = read_list(f"{WORK}/COMBINED_VOLLIST_VALID_clean.txt", "train_")
    assert train and valid, "bos hacim listesi"
    assert not (set(train) & set(valid)), "train ve valid kesisiyor"
    print(f"train {len(train):,} | dahili valid {len(valid):,} "
          f"(ikisi de CT-RATE'in train yarisindan)")

    # Published labels, both halves. The internal validation volumes are train_*
    # so their labels live in the TRAIN file.
    lab: dict[str, dict] = {}
    for fn in ("train_predicted_labels.csv", "valid_predicted_labels.csv"):
        p = os.path.join(PUB, fn)
        assert os.path.isfile(p), f"yayimlanan etiket dosyasi yok: {p}"
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lab[stem(r["VolumeName"])] = r
    print(f"yayimlanan etiket {len(lab):,} hacim")
    miss = [v for v in train + valid if v not in lab]
    assert not miss, f"{len(miss)} hacmin yayimlanan etiketi yok, ilki {miss[:1]}"

    lp = os.path.join(OUT, "labels18.csv")
    with open(lp + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["VolumeName"] + CTRATE18)
        for v in train + valid:
            w.writerow([v + ".nii.gz"] + [lab[v][c] for c in CTRATE18])
    os.replace(lp + ".tmp", lp)
    print(f"yazildi {lp}  ({len(train) + len(valid):,} satir, 18 sinif)")

    # Reports, filtered from the combined table. Same text the joint runs saw, so
    # a difference between this run and those cannot come from the report channel.
    keep = set(train) | set(valid)
    rp = os.path.join(OUT, "reports.csv")
    n = 0
    with open(f"{WORK}/COMBINED_reports.csv", newline="", encoding="utf-8") as fh, \
            open(rp + ".tmp", "w", newline="", encoding="utf-8") as out:
        rd = csv.DictReader(fh)
        w = csv.DictWriter(out, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            if stem(r["VolumeName"]) in keep:
                w.writerow(r); n += 1
    os.replace(rp + ".tmp", rp)
    print(f"yazildi {rp}  ({n:,} satir)")
    assert n >= len(keep) * 0.99, f"rapor kapsami dusuk: {n}/{len(keep)}"

    for name, rows in (("VOLLIST_TRAIN.txt", train), ("VOLLIST_VALID.txt", valid)):
        p = os.path.join(OUT, name)
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            fh.write("\n".join(v + ".nii.gz" for v in rows) + "\n")
        os.replace(p + ".tmp", p)
        print(f"yazildi {p}  ({len(rows):,})")

    # Context coverage, reported rather than enforced: an absent indication is a
    # legitimate input (NO_INDICATION), a missing demographics row is not.
    for tag, fn, col in (("indication", "indication.csv", "ind_status"),
                         ("demografi", "demographics.csv", "AgeBand")):
        p = f"{WORK}/CONTEXT/{fn}"
        have, present = set(), 0
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                k = stem(r["VolumeName"])
                if k in keep:
                    have.add(k)
                    if col == "ind_status":
                        present += r.get(col, "") == "present"
        cov = 100 * len(have) / len(keep)
        extra = f", ind_status=present %{100 * present / max(len(keep), 1):.1f}" \
            if col == "ind_status" else ""
        print(f"{tag}: satiri olan %{cov:.1f}{extra}")
        if tag == "demografi":
            assert cov > 99.5, f"demografi kapsami dusuk (%{cov:.1f})"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

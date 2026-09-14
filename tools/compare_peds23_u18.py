#!/usr/bin/env python3
"""Aggregate the three-seed U18 core comparison and clinical subgroups."""

from __future__ import annotations

import csv
import datetime
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze_predictions_paper import patient_groups
from audit_indication_label_mentions import LEXICON


WORK = Path("/temp_work/ch278233")
EVAL = Path(os.environ.get(
    "PEDS_U18_EVAL_ROOT",
    str(WORK / "eval_matrix/peds23_u18_r2plus1d_18_test"),
))
OUT = EVAL / "paper_summary"


def load(arm: str):
    package = np.load(EVAL / f"{arm}_ind-true/predictions.npz", allow_pickle=True)
    return package["pred"], package["true"].astype(int), package["accessions"], [str(x) for x in package["pathologies"]]


def macro(y, p, mask=None, min_pos=1, min_neg=1):
    aucs, aps = [], []
    for index in range(y.shape[1]):
        take = np.ones(len(y), dtype=bool) if mask is None else mask[:, index]
        yt, pt = y[take, index], p[take, index]
        if yt.sum() >= min_pos and (len(yt) - yt.sum()) >= min_neg:
            aucs.append(roc_auc_score(yt, pt)); aps.append(average_precision_score(yt, pt))
    return float(np.mean(aucs)), float(np.mean(aps)), len(aucs)


def main() -> int:
    full, ct = [], []
    reference = None
    for seed in range(3):
        fp, fy, fv, classes = load(f"full_seed{seed}")
        cp, cy, cv, cclasses = load(f"ct_only_seed{seed}")
        assert np.array_equal(fy, cy) and np.array_equal(fv, cv) and classes == cclasses
        if reference is None: reference = (fy, fv, classes)
        else: assert np.array_equal(reference[0], fy) and np.array_equal(reference[1], fv)
        full.append(fp); ct.append(cp)
    y, volumes, classes = reference
    groups = patient_groups(volumes)

    per_seed = []
    for seed in range(3):
        ca, cap, _ = macro(y, ct[seed]); fa, fap, _ = macro(y, full[seed])
        per_seed.append({"seed": seed, "ct_only_auroc": ca, "full_auroc": fa,
                         "delta_auroc": fa-ca, "ct_only_auprc": cap,
                         "full_auprc": fap, "delta_auprc": fap-cap})

    per_class_rows=[]
    for index,class_name in enumerate(classes):
        ct_auc=[roc_auc_score(y[:,index],p[:,index]) for p in ct]
        full_auc=[roc_auc_score(y[:,index],p[:,index]) for p in full]
        ct_ap=[average_precision_score(y[:,index],p[:,index]) for p in ct]
        full_ap=[average_precision_score(y[:,index],p[:,index]) for p in full]
        per_class_rows.append({"class":class_name,"positive_n":int(y[:,index].sum()),
            "ct_only_auroc":float(np.mean(ct_auc)),"full_auroc":float(np.mean(full_auc)),
            "delta_auroc":float(np.mean(full_auc)-np.mean(ct_auc)),
            "ct_only_auprc":float(np.mean(ct_ap)),"full_auprc":float(np.mean(full_ap)),
            "delta_auprc":float(np.mean(full_ap)-np.mean(ct_ap))})

    rng = np.random.default_rng(42); patients = np.unique(groups); deltas=[]; ap_deltas=[]
    for _ in range(2000):
        sampled = rng.choice(patients, len(patients), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == patient) for patient in sampled])
        seed_delta=[]; seed_ap=[]
        for seed in range(3):
            ca, cap, _ = macro(y[idx], ct[seed][idx]); fa, fap, _ = macro(y[idx], full[seed][idx])
            seed_delta.append(fa-ca); seed_ap.append(fap-cap)
        deltas.append(np.mean(seed_delta)); ap_deltas.append(np.mean(seed_ap))

    summary = {
        "n_studies": int(len(y)), "n_patients": int(len(patients)), "seeds": per_seed,
        "mean_ct_only_auroc": float(np.mean([row["ct_only_auroc"] for row in per_seed])),
        "mean_full_auroc": float(np.mean([row["full_auroc"] for row in per_seed])),
        "mean_delta_auroc": float(np.mean([row["delta_auroc"] for row in per_seed])),
        "delta_auroc_ci95_patient_bootstrap": [float(np.percentile(deltas,2.5)),float(np.percentile(deltas,97.5))],
        "paired_bootstrap_p_two_sided": float(min(1.0,2*min(np.mean(np.asarray(deltas)<=0),np.mean(np.asarray(deltas)>=0)))),
        "mean_ct_only_auprc": float(np.mean([row["ct_only_auprc"] for row in per_seed])),
        "mean_full_auprc": float(np.mean([row["full_auprc"] for row in per_seed])),
        "mean_delta_auprc": float(np.mean([row["delta_auprc"] for row in per_seed])),
        "delta_auprc_ci95_patient_bootstrap": [float(np.percentile(ap_deltas,2.5)),float(np.percentile(ap_deltas,97.5))],
    }

    demographics = pd.read_csv(WORK / "CONTEXT/demographics.csv").set_index("VolumeName")
    ages = pd.to_numeric(demographics.loc[list(map(str,volumes)),"AgeYears"], errors="coerce").to_numpy()
    sexes = demographics.loc[list(map(str,volumes)),"Sex"].astype(str).to_numpy()
    subgroup_rows=[]
    definitions = {
        "age_0_2": (ages>=0)&(ages<2), "age_2_5": (ages>=2)&(ages<5),
        "age_5_10": (ages>=5)&(ages<10), "age_10_15": (ages>=10)&(ages<15),
        "age_15_18": (ages>=15)&(ages<18),
    }
    for value in sorted(set(sexes)):
        definitions[f"sex_{value}"] = sexes == value

    # Temporal and acquisition-site sensitivity analyses. Site names remain
    # private; only frequency-ranked site IDs are written to result artifacts.
    with (WORK / "BCH_DATASET/LABELS/volume_map.tsv").open(newline="", encoding="utf-8-sig") as handle:
        volume_to_accession = {row["volume_name"]: row["accession"].strip()
                               for row in csv.DictReader(handle, delimiter="\t")}
    with (WORK / "BCH_DATASET/reports_8k.csv").open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        raw = {row["Accession Number"].strip(): row for row in csv.DictReader(handle)}
    records = [raw[volume_to_accession[str(volume)]] for volume in volumes]
    years = np.asarray([(datetime.datetime(1899, 12, 30) +
                         datetime.timedelta(days=float(row["Exam Started Date"]))).year
                        for row in records])
    definitions["exam_year_2011_2018"] = years <= 2018
    definitions["exam_year_2019_2026"] = years >= 2019
    sites = np.asarray([row["Point of Care"].strip() for row in records])
    site_order = sorted(set(sites), key=lambda value: int((sites == value).sum()), reverse=True)
    for rank, site in enumerate(site_order[:4], 1):
        definitions[f"acquisition_site_rank_{rank}"] = sites == site
    for name, take in definitions.items():
        for model, predictions in (("ct_only",ct),("full",full)):
            values=[macro(y[take],p[take],min_pos=10,min_neg=10) for p in predictions]
            subgroup_rows.append({"subgroup":name,"model":model,"studies":int(take.sum()),
                                  "macro_auroc":float(np.mean([x[0] for x in values])),
                                  "macro_auprc":float(np.mean([x[1] for x in values])),
                                  "eligible_classes":int(round(np.mean([x[2] for x in values])))})

    indications = pd.read_csv(WORK / "CONTEXT/indication.csv",keep_default_na=False).set_index("VolumeName")
    texts = indications.loc[list(map(str,volumes)),"Indication_EN"].astype(str)
    mention=np.zeros_like(y,dtype=bool)
    for index,class_name in enumerate(classes):
        mention[:,index]=texts.str.contains(LEXICON[class_name],case=False,regex=True).to_numpy()
    mention_rows=[]
    for stratum, mask in (("explicit_term_present",mention),("explicit_term_absent",~mention)):
        for model,predictions in (("ct_only",ct),("full",full)):
            values=[macro(y,p,mask=mask,min_pos=10,min_neg=10) for p in predictions]
            mention_rows.append({"stratum":stratum,"model":model,
                                 "macro_auroc":float(np.mean([x[0] for x in values])),
                                 "macro_auprc":float(np.mean([x[1] for x in values])),
                                 "eligible_classes":int(round(np.mean([x[2] for x in values])))})

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"core_comparison.json").write_text(json.dumps(summary,indent=2))
    for path,rows in ((OUT/"subgroups.csv",subgroup_rows),(OUT/"mention_strata.csv",mention_rows)):
        with path.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT/"per_class.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(per_class_rows[0])); writer.writeheader(); writer.writerows(per_class_rows)
    print(json.dumps(summary,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

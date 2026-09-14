#!/usr/bin/env python
"""
Linear probe on FROZEN Merlin 2048-d features (pediatric chest CT, 23 labels).

- Regularisation strength C is the ONLY tuned hyper-parameter.
- C is selected by StratifiedGroupKFold CV *on the TRAIN split only*, grouped by
  PATIENT so repeat scans of one child cannot straddle a fold boundary.
- The validation split is touched exactly once, to produce final predictions.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

FEATDIR = "/temp_work/ch278233/PEDS_BENCH/merlin/feats"
BD = "/temp_work/ch278233/BENCHMARK_DATA"

def patient(v):           # ped_00001_1.nii.gz -> 00001
    return v.split("_")[1]

def load_split(split):
    fs, vs = [], []
    files = sorted(glob.glob(os.path.join(FEATDIR, f"{split}_shard*.npz")))
    assert files, f"no shards for {split}"
    failed = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        fs.append(d["feats"]); vs.append(d["vols"])
        if "failed" in d and d["failed"].size:
            failed += list(d["failed"])
    X = np.concatenate(fs).astype(np.float64)
    V = np.concatenate(vs).astype(str)
    return X, V, failed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/temp_work/ch278233/PEDS_BENCH/preds/merlin_peds_probe_preds.npz")
    ap.add_argument("--report", default="/temp_work/ch278233/PEDS_BENCH/results/merlin_peds_probe.json")
    args = ap.parse_args()

    Xtr, Vtr, ftr = load_split("train")
    Xva, Vva, fva = load_split("valid")
    print(f"train feats {Xtr.shape} (failed {len(ftr)}), valid feats {Xva.shape} (failed {len(fva)})")
    if ftr: print("  train extraction failures:", ftr)
    if fva: print("  valid extraction failures:", fva)

    ltr = pd.read_csv(f"{BD}/peds23_labels_train.csv")
    lva = pd.read_csv(f"{BD}/peds23_labels_valid.csv")
    classes = [c for c in ltr.columns if c != "VolumeName"]
    assert len(classes) == 23, len(classes)
    assert list(lva.columns) == list(ltr.columns), "label column order differs between splits"

    ltr = ltr.set_index("VolumeName"); lva = lva.set_index("VolumeName")
    Ytr = ltr.loc[Vtr, classes].to_numpy().astype(np.int32)
    Yva = lva.loc[Vva, classes].to_numpy().astype(np.int32)

    scaler = StandardScaler().fit(Xtr)            # fit on TRAIN ONLY
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)
    groups = np.array([patient(v) for v in Vtr])
    print(f"train volumes {len(Vtr)} from {len(set(groups))} patients; "
          f"valid volumes {len(Vva)} from {len(set(patient(v) for v in Vva))} patients")

    Cs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]
    NF = 5
    cv_scores = {}
    for C in Cs:
        fold_macro = []
        sgk = StratifiedGroupKFold(n_splits=NF, shuffle=True, random_state=0)
        # stratify on the most prevalent label to keep folds balanced-ish
        strat = Ytr[:, int(Ytr.sum(0).argmax())]
        for tr, te in sgk.split(Ztr, strat, groups):
            aucs = []
            for k in range(len(classes)):
                y, yt = Ytr[tr, k], Ytr[te, k]
                if y.sum() < 2 or y.sum() == len(y) or len(np.unique(yt)) < 2:
                    continue
                clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
                clf.fit(Ztr[tr], y)
                aucs.append(roc_auc_score(yt, clf.decision_function(Ztr[te])))
            if aucs: fold_macro.append(float(np.mean(aucs)))
        cv_scores[C] = float(np.mean(fold_macro))
        print(f"  C={C:<8g} cv_macroAUC={cv_scores[C]:.4f}  (folds={len(fold_macro)})")

    bestC = max(cv_scores, key=cv_scores.get)
    print(f"selected C={bestC} by {NF}-fold patient-grouped CV on TRAIN (cv macroAUC={cv_scores[bestC]:.4f})")

    # ---- final fit on full train, predict once on valid ----
    P = np.zeros((len(Vva), len(classes)), dtype=np.float32)
    per_class = {}
    for k, c in enumerate(classes):
        y = Ytr[:, k]
        if y.sum() == 0 or y.sum() == len(y):
            P[:, k] = float(y.mean()); per_class[c] = None; continue
        clf = LogisticRegression(C=bestC, max_iter=5000, solver="lbfgs")
        clf.fit(Ztr, y)
        P[:, k] = clf.predict_proba(Zva)[:, 1]
        per_class[c] = (roc_auc_score(Yva[:, k], P[:, k])
                        if len(np.unique(Yva[:, k])) > 1 else None)

    valid_aucs = [v for v in per_class.values() if v is not None]
    macro = float(np.mean(valid_aucs))
    print(f"\nVALID macro AUC = {macro:.4f} over {len(valid_aucs)}/{len(classes)} evaluable classes, n={len(Vva)}")
    for c, v in per_class.items():
        n_pos = int(Yva[:, classes.index(c)].sum())
        print(f"  {c:<45s} AUC={('%.4f'%v) if v is not None else '  n/a ':>6s}  pos={n_pos}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, pred=P.astype(np.float32), true=Yva.astype(np.int32),
             classes=np.array(classes, dtype="U64"), volumes=np.array(Vva, dtype="U64"))
    print("wrote", args.out)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    json.dump({"model": "Merlin (StanfordMIMI) frozen image encoder + linear probe",
               "domain_caveat": "Merlin is an ABDOMINAL/pelvic CT foundation model; "
                                "applied here to PEDIATRIC CHEST CT (out-of-domain transfer).",
               "feature_dim": int(Xtr.shape[1]), "n_train": int(len(Vtr)), "n_valid": int(len(Vva)),
               "n_train_patients": len(set(groups)),
               "n_valid_patients": len(set(patient(v) for v in Vva)),
               "selected_C": bestC, "cv_scheme": f"{NF}-fold StratifiedGroupKFold grouped by patient, TRAIN only",
               "cv_macro_auc_by_C": cv_scores, "valid_macro_auc": macro,
               "n_evaluable_classes": len(valid_aucs),
               "per_class_auc": per_class,
               "extraction_failures_train": ftr, "extraction_failures_valid": fva},
              open(args.report, "w"), indent=2)
    print("wrote", args.report)

if __name__ == "__main__":
    main()

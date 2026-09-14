#!/home/ch278233/micromamba/envs/arcct/bin/python
"""Unified retrieval metrics for the pediatric benchmark.

Input: an npz in the SAME format arc-ct/tools/eval_retrieval.py writes --
  img_lat (N,D) float, txt_lat (N,D) float, accessions (N,) str, labels (N,C) int
  optional text_ok (N,) bool -- queries whose report text was empty are dropped
  from report->image R@K and the effective N is reported.

Three CT->CT readouts are produced, because the literature does not agree:
  clean      : self EXCLUDED from gallery, per-K mean Jaccard of positive label
               sets. This is what arc-ct/tools/eval_retrieval.py computes.
  origin_perk: self INCLUDED (CT-CLIP volume_to_volume_new.py never masks self).
  origin_cum : self INCLUDED and, exactly as in that script, the per-query means
               accumulate across k_list=[1,5,10,50] without reset, so the value
               reported at K is a running mean over all K' <= K. This is the
               quantity the CT-CLIP / MPS-CT / ARC-CT papers label "MAP@K".
Gallery = scans with >=1 positive label; queries = all scans (all variants).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

KS_R = (5, 10, 50, 100)
KS_I = (1, 5, 10, 50)


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _patients(acc):
    return np.array(["_".join(str(a).split("_")[:2]) for a in acc])


def recall_at_k(txt, img, text_ok=None, ks=KS_R):
    """Exact-match report->image: for query i the only relevant item is image i."""
    sim = _norm(txt) @ _norm(img).T
    n = sim.shape[0]
    q = np.arange(n) if text_ok is None else np.where(text_ok)[0]
    order = np.argsort(-sim[q], axis=1)
    gt_rank = np.argmax(order == q[:, None], axis=1)
    return {f"R@{k}": float((gt_rank < k).mean() * 100) for k in ks}, int(len(q))


def _jaccard(q, G):
    inter = (G * q[None, :]).sum(1)
    union = ((G + q[None, :]) > 0).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def image_to_image(img, labels, ks=KS_I, exclude_self=True, cumulative=False):
    imgn = _norm(img); n = imgn.shape[0]
    gal = np.where(labels.sum(1) > 0)[0]
    sim = imgn @ imgn[gal].T
    if exclude_self:
        sim[np.arange(n)[:, None] == gal[None, :]] = -1.0
    order = np.argsort(-sim, axis=1)
    maxk = max(ks)
    per = np.zeros((n, maxk))
    for i in range(n):
        per[i] = _jaccard(labels[i], labels[gal[order[i, :maxk]]])
    out, running = {}, []
    for k in ks:
        pq = per[:, :k].mean(1)                  # mean over top-K per query
        if cumulative:
            running.extend(pq.tolist())          # CT-CLIP bug: never reset
            out[f"@{k}"] = float(np.mean(running) * 100)
        else:
            out[f"@{k}"] = float(pq.mean() * 100)
    return out


def evaluate(img, txt, labels, acc, text_ok=None, n_boot=0, seed=42):
    r, n_q = recall_at_k(txt, img, text_ok)
    res = {"n": int(img.shape[0]), "n_text_queries": n_q,
           "chance_R@K": {f"R@{k}": float(k / img.shape[0] * 100) for k in KS_R},
           "report_to_image": r,
           "image_to_image": {
               "clean_self_excluded": image_to_image(img, labels, exclude_self=True),
               "origin_self_included_perK": image_to_image(img, labels, exclude_self=False),
               "origin_self_included_cumulative": image_to_image(img, labels, exclude_self=False, cumulative=True),
           }}
    if n_boot:
        rng = np.random.default_rng(seed); pats = _patients(acc); up = np.unique(pats)
        idx_by = {p: np.where(pats == p)[0] for p in up}
        R = {k: [] for k in r}; J = {k: [] for k in res["image_to_image"]["clean_self_excluded"]}
        for _ in range(n_boot):
            samp = np.concatenate([idx_by[p] for p in rng.choice(up, len(up))])
            ok = None if text_ok is None else text_ok[samp]
            rr, _ = recall_at_k(txt[samp], img[samp], ok)
            for k in R: R[k].append(rr[k])
            jj = image_to_image(img[samp], labels[samp], exclude_self=True)
            for k in J: J[k].append(jj[k])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res["ci_patient_bootstrap"] = {"n_boot": n_boot, "report_to_image": {k: ci(v) for k, v in R.items()},
                                       "image_to_image_clean": {k: ci(v) for k, v in J.items()}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--labels-csv", default="/temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv",
                    help="canonical labels; overrides labels stored in the npz so EVERY row uses one relevance set")
    a = ap.parse_args()
    z = np.load(a.latents, allow_pickle=True)
    ok = z["text_ok"].astype(bool) if "text_ok" in z.files else None
    acc = np.array([str(x) for x in z["accessions"]])
    labels = z["labels"].astype(int)
    if a.labels_csv and os.path.exists(a.labels_csv):
        import pandas as pd
        df = pd.read_csv(a.labels_csv).set_index("VolumeName")
        cols = [c for c in df.columns]
        keys = [x if x.endswith(".nii.gz") else x + ".nii.gz" for x in acc]
        miss = [k for k in keys if k not in df.index]
        if miss:
            sys.exit(f"[retr] FATAL: {len(miss)} accessions not in {a.labels_csv}, e.g. {miss[:3]}")
        labels = (df.loc[keys, cols].to_numpy(dtype=float) > 0.5).astype(int)
        print(f"[retr] labels overridden from {os.path.basename(a.labels_csv)}: {labels.shape}, classes={len(cols)}")
    res = evaluate(z["img_lat"], z["txt_lat"], labels, acc, ok, a.bootstrap)
    res["labels_source"] = os.path.abspath(a.labels_csv) if a.labels_csv else "npz"
    res["model"] = a.name; res["latents"] = os.path.abspath(a.latents)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    r = res["report_to_image"]; ii = res["image_to_image"]
    print(f"[retr] {a.name}: N={res['n']} text-queries={res['n_text_queries']}")
    print("       R->I  R@5/10/50/100 = " + "/".join(f"{r[f'R@{k}']:.2f}" for k in KS_R))
    for key, lab in [("clean_self_excluded", "clean"), ("origin_self_included_perK", "self-incl"),
                     ("origin_self_included_cumulative", "origin-cum")]:
        print(f"       I->I  {lab:11s} @5/10/50 = " + "/".join(f"{ii[key][f'@{k}']:.2f}" for k in (5, 10, 50)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

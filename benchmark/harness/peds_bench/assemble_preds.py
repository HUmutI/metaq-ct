"""Assemble fVLM zero-shot predictions into the deliverable .npz and report AUCs."""
import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_peds import PEDS23, build_test_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-glob', default='/temp_work/ch278233/PEDS_BENCH/fvlm/preds_raw/*.jsonl')
    ap.add_argument('--labels', default='/temp_work/ch278233/BENCHMARK_DATA/peds23_labels_valid.csv')
    ap.add_argument('--vollist', default='/temp_work/ch278233/BENCHMARK_DATA/peds23_vollist_valid.txt')
    ap.add_argument('--out', default='/temp_work/ch278233/PEDS_BENCH/preds/fvlm_peds_zeroshot_preds.npz')
    args = ap.parse_args()

    items, is_official = build_test_items()
    lab = pd.read_csv(args.labels).drop_duplicates('VolumeName').set_index('VolumeName')
    want = [l.strip() for l in open(args.vollist) if l.strip()]

    recs = {}
    for f in sorted(glob.glob(args.pred_glob)):
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            if d.get('status') == 'OK':
                recs[d['volume']] = d

    vols, P, status = [], [], {'ok': 0, 'missing': 0}
    nan_counts = {c: 0 for c in PEDS23}
    for v in want:
        d = recs.get(v)
        if d is None or v not in lab.index:
            status['missing'] += 1; continue
        row = []
        for c in PEDS23:
            p = d['probs'].get(c, np.nan)
            if not np.isfinite(p): nan_counts[c] += 1
            row.append(p)
        vols.append(v); P.append(row); status['ok'] += 1

    P = np.asarray(P, dtype=np.float32)
    Y = lab.loc[vols, PEDS23].to_numpy().astype(np.int32)

    # fVLM's own calc_metrics.py substitutes 0 for the (rare) NaN probabilities.
    Pf = np.where(np.isfinite(P), P, 0.0).astype(np.float32)

    aucs, prev = {}, {}
    for j, c in enumerate(PEDS23):
        y = Y[:, j]; prev[c] = int(y.sum())
        aucs[c] = float(roc_auc_score(y, Pf[:, j])) if 0 < y.sum() < len(y) else float('nan')

    off = [c for c, o in zip(PEDS23, is_official) if o]
    ext = [c for c, o in zip(PEDS23, is_official) if not o]
    def macro(cs):
        vals = [aucs[c] for c in cs if np.isfinite(aucs[c])]
        return float(np.mean(vals)) if vals else float('nan'), len(vals)

    m23, n23 = macro(PEDS23); mof, nof = macro(off); mex, nex = macro(ext)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             pred=Pf, true=Y, classes=np.array(PEDS23, dtype='U64'),
             volumes=np.array(vols, dtype='U64'),
             class_is_official=is_official,
             class_organ=np.array([it[0] for it in items], dtype='U16'),
             pred_raw_with_nan=P)

    print(f'n volumes scored : {len(vols)}   (missing/failed: {status["missing"]})')
    print(f'pred shape {Pf.shape}  true shape {Y.shape}')
    print()
    print(f'{"class":42s} {"off":4s} {"organ":10s} {"pos":6s} {"NaN":5s} AUC')
    for c, o, it in zip(PEDS23, is_official, items):
        print(f'{c:42s} {"Y" if o else "n":4s} {it[0]:10s} {prev[c]:6d} {nan_counts[c]:5d} {aucs[c]:.4f}')
    print()
    print(f'MACRO AUC, all 23 classes               : {m23:.4f}  (n={n23})')
    print(f'MACRO AUC, {nof} OFFICIAL fVLM-vocab classes: {mof:.4f}')
    print(f'MACRO AUC, {nex} EXTRAPOLATED-prompt classes : {mex:.4f}')
    print(f'\nsaved -> {args.out}')

    json.dump({'n': len(vols), 'auc_per_class': aucs, 'prevalence': prev,
               'macro_auc_all23': m23, 'macro_auc_official12': mof,
               'macro_auc_extrapolated11': mex,
               'official_classes': off, 'extrapolated_classes': ext,
               'nan_counts': nan_counts},
              open(args.out.replace('.npz', '_metrics.json'), 'w'), indent=2)

if __name__ == '__main__':
    main()

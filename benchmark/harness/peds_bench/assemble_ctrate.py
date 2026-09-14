"""Assemble the CT-RATE harness-validation control and report AUCs."""
import os, sys, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_peds import OFFICIAL_ITEMS

LABELS = '/temp_work/ch278233/CTRATE/multi_abnormality_labels/valid_predicted_labels.csv'
lab = pd.read_csv(LABELS).drop_duplicates('VolumeName').set_index('VolumeName')

recs = {}
fails = {}
for f in sorted(glob.glob('/home/ch278233/BENCHMARK/harness/peds_bench/ctrate_raw/preds_*.jsonl')):
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        if d.get('status') == 'OK': recs[d['volume']] = d
        else: fails[d['volume']] = d.get('status')

cls = list(OFFICIAL_ITEMS.keys())
vols = [v for v in recs if v in lab.index]
P = np.array([[recs[v]['probs'].get(c, np.nan) for c in cls] for v in vols], dtype=np.float32)
Y = lab.loc[vols, cls].to_numpy().astype(int)
Pf = np.where(np.isfinite(P), P, 0.0)

print(f'CT-RATE control: scored {len(vols)} volumes  (failed/skipped: {len(fails)})')
from collections import Counter
print('failure reasons:', Counter(fails.values()).most_common(6))
aucs = {}
print(f'\n{"class":38s} {"pos":6s} {"NaN":5s} AUC')
for j, c in enumerate(cls):
    y = Y[:, j]
    a = roc_auc_score(y, Pf[:, j]) if 0 < y.sum() < len(y) else float('nan')
    aucs[c] = a
    print(f'{c:38s} {y.sum():6d} {int((~np.isfinite(P[:,j])).sum()):5d} {a:.4f}')
vals = [v for v in aucs.values() if np.isfinite(v)]
print(f'\nMACRO AUC over {len(vals)} official fVLM CT-RATE items: {np.mean(vals):.4f}')
json.dump({'n': len(vols), 'auc': aucs, 'macro': float(np.mean(vals))},
          open('/home/ch278233/BENCHMARK/harness/peds_bench/ctrate_control_metrics.json', 'w'), indent=2)

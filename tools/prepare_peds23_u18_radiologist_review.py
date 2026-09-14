#!/usr/bin/env python3
"""Prepare a blinded, PHI-restricted CT/indication review packet."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd


WORK=Path('/temp_work/ch278233')
EVAL=WORK/'eval_matrix/peds23_u18_test'
OUT=WORK/'eval_matrix/peds23_u18_radiologist_review'


def volume_path(volume: str) -> Path:
    accession = volume.removesuffix('.nii.gz').removesuffix('.npz')
    patient = '_'.join(accession.split('_')[:2])
    path = WORK/'PEDS_NPZ_NESTED'/patient/accession/f'{accession}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'CT volume missing for {volume}: {path}')
    return path


def load(tag):
    z=np.load(EVAL/tag/'predictions.npz',allow_pickle=True)
    return z['pred'],z['true'].astype(int),z['accessions'],[str(x) for x in z['pathologies']]


def main():
    full,y,volumes,classes=load('full_seed0_ind-true')
    shuffled,ys,vs,cs=load('full_seed0_ind-shuffled')
    ct,yc,vc,cc=load('ct_only_seed0_ind-true')
    assert np.array_equal(y,ys) and np.array_equal(y,yc)
    assert np.array_equal(volumes,vs) and np.array_equal(volumes,vc) and classes==cs==cc
    positive=np.maximum(y.sum(1),1)
    scores={
      'indication_sensitive':((full-shuffled)*y).sum(1)/positive,
      'multimodal_gain':((full-ct)*y).sum(1)/positive,
      'challenging':-(y*np.log(np.clip(full,1e-6,1))+(1-y)*np.log(np.clip(1-full,1e-6,1))).mean(1),
    }
    chosen=[]; seen=set()
    for category,score in scores.items():
        for index in np.argsort(score)[::-1]:
            if index in seen: continue
            chosen.append((category,index)); seen.add(index)
            if sum(c==category for c,_ in chosen)>=20: break
    ctx=pd.read_csv(WORK/'CONTEXT/indication.csv',keep_default_na=False).set_index('VolumeName')
    dem=pd.read_csv(WORK/'CONTEXT/demographics.csv').set_index('VolumeName')
    OUT.mkdir(parents=True,exist_ok=True)
    blind=[]; key=[]
    for number,(category,index) in enumerate(chosen,1):
        volume=str(volumes[index]); case=f'PEDS-{number:03d}'
        blind.append({'CaseID':case,'Age':dem.loc[volume,'AgeYears'],'Sex':dem.loc[volume,'Sex'],
          'Indication':ctx.loc[volume,'Indication_EN'],'VolumePath':str(volume_path(volume)),
          'Radiologist findings':'','Indication clinically relevant?':'','Image-label agreement?':'','Comments':''})
        key.append({'CaseID':case,'selection_category':category,'VolumeName':volume,
          'positive_labels':'; '.join(classes[j] for j in np.flatnonzero(y[index])),
          'mean_positive_full':float((full[index]*y[index]).sum()/positive[index]),
          'mean_positive_ct_only':float((ct[index]*y[index]).sum()/positive[index]),
          'mean_positive_shuffled':float((shuffled[index]*y[index]).sum()/positive[index])})
    for name,rows in [('blinded_review.csv',blind),('private_analysis_key.csv',key)]:
        path=OUT/name
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        os.chmod(path,0o600)
    print(f'prepared {len(blind)} blinded cases in {OUT}')


if __name__=='__main__': main()

#!/usr/bin/env python3
"""One labeller for both cohorts: replace CT-RATE's published 18 with our LLM's.

The joint dataset previously mixed two annotators. The pediatric half was
labelled by our prompt, the adult half by CT-RATE's published RadBERT labels, and
the same class name therefore carried two thresholds. Measured over 22,974 adult
reports, 8 of 18 shared classes were systematically offset - and one-sidedly,
which random disagreement would not be: Lymphadenopathy 40x conservative on our
side, Mosaic attenuation 341x liberal. A model trained on that reads an annotator
difference as a disease difference.

Switching the adult half to the LLM labels changes 3.9% of cells. The effect on
the cross-cohort gap, averaged over the 18 shared classes, is 11.5 -> 10.3
points, improving 12 of 18. The average moves little because most of the gap was
always real epidemiology - median age 14 against 47 - and the classes that close
are exactly the ones that were annotator artefacts:

    Lymphadenopathy          18.1 -> 7.3
    Coronary calcification   25.2 -> 20.4
    Peribronchial thickening  4.5 -> 0.2

while genuinely different diseases stay apart (Medical material 27.8, Arterial
wall calcification 23.7, Emphysema 18.8). That separation is the point: it
removes the annotator component and leaves the epidemiological one.

The published labels are NOT discarded. They remain the adult evaluation target,
so the 0.8524 reproducibility gate stays a like-for-like measurement against the
literature. This file only changes what the joint model is trained on.
"""
from __future__ import annotations
import argparse, csv, json, os, sys

sys.path.insert(0, "/home/ch278233/bch-arc-ct")
os.environ.setdefault("RAC_SCHEMA", "peds")
from arcct.schema import PEDS_PATHOLOGIES as P27, CTRATE_PATHOLOGIES as P18

NEW9 = [p for p in P27 if p not in P18]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peds", default="/temp_work/ch278233/BCH_DATASET/LABELS27_V2/labels.csv")
    ap.add_argument("--ct-llm", default="/temp_work/ch278233/CTRATE/LABELS27/labels.csv")
    ap.add_argument("--fanout", default="/temp_work/ch278233/CTRATE/LABELS27/report_fanout.json")
    ap.add_argument("--vollist", default="/temp_work/ch278233/COMBINED_VOLLIST_TRAIN.txt")
    ap.add_argument("--vollist2", default="/temp_work/ch278233/COMBINED_VOLLIST_VALID.txt")
    ap.add_argument("--out", default="/temp_work/ch278233/COMBINED_labels_harmonised.csv")
    a = ap.parse_args()
    csv.field_size_limit(10 ** 7)

    keep = set()
    for p in (a.vollist, a.vollist2):
        if os.path.isfile(p):
            keep |= {l.strip() for l in open(p) if l.strip()}
    print("  hedef hacim (train+valid): %d" % len(keep))

    rows, n_ped, n_ct, blank = [], 0, 0, 0
    for r in csv.DictReader(open(a.peds)):
        if r["VolumeName"] in keep:
            rows.append([r["VolumeName"]] + [r[c] for c in P27]); n_ped += 1

    llm = {r["VolumeName"]: r for r in csv.DictReader(open(a.ct_llm))}
    fan = json.load(open(a.fanout))
    miss = 0
    for rep, vols in fan.items():
        src = llm.get(rep)
        for v in vols:
            if v not in keep:
                continue
            if src is None:
                miss += 1
                continue
            vals = [src.get(c, "") for c in P27]
            blank += sum(1 for x in vals if x == "")
            rows.append([v] + vals); n_ct += 1

    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["VolumeName"] + list(P27)); w.writerows(rows)
    os.replace(tmp, a.out)

    print("  yazilan: %d satir (pediatrik %d + yetiskin %d)" % (len(rows), n_ped, n_ct))
    print("  llm cikarimi olmayan yetiskin hacim: %d" % miss)
    print("  bos hucre: %d" % blank)
    got = {r[0] for r in rows}
    print("  listede olup etiketi olmayan: %d" % len(keep - got))
    print("  -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

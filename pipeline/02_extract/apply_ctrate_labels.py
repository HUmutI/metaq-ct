#!/usr/bin/env python3
"""Qwen's CT-RATE label shards -> one 27-column CSV covering all 47,149 volumes.

The LLM saw 22,976 distinct reports, not 47,149 volumes: the same study appears
under several reconstructions (train_1234_a_1, train_1234_a_2, ...) and those
rows carry identical report text. report_fanout.json records, for each report
the LLM processed, every volume that shares it. This puts the labels back.

Why this file exists at all: build_combined.py wrote 0 for all nine
pediatric-only classes on all 16,400 adults - 0 positives, none of them
observed. An LLM smoke over 40 adult reports found bone lesions in 22.5%. The
combined model duly learned that adult-looking scans have no bone lesions and
scored 0.373 on that class, below chance.

Only the nine new classes are meant for training. The eighteen shared ones are
extracted too, but as a DIAGNOSTIC: compared against CT-RATE's own published
labels they measure whether our prompt's definitions match the reference. They
must not replace the published labels in training, or the adult reproducibility
gate (mean AUC 0.8524) stops being a like-for-like check.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
from collections import Counter

sys.path.insert(0, "/home/ch278233/arc-ct")
os.environ.setdefault("RAC_SCHEMA", "peds")
from arcct.schema import PEDS_PATHOLOGIES

CTRATE_18 = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]
NEW_9 = [p for p in PEDS_PATHOLOGIES if p not in CTRATE_18]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="/temp_work/ch278233/CTRATE/LABELS27")
    ap.add_argument("--published", default="/temp_work/ch278233/CTRATE/labels_hf/dataset/"
                                           "multi_abnormality_labels/train_predicted_labels.csv")
    ap.add_argument("--out", default="/temp_work/ch278233/CTRATE/CTRATE_labels27.csv")
    ap.add_argument("--diag", default="/temp_work/ch278233/CTRATE/qwen_vs_published.txt")
    ap.add_argument("--max-missing", type=float, default=0.01,
                    help="tolerated fraction of reports the LLM could not parse")
    ap.add_argument("--allow-partial", action="store_true",
                    help="run against an unfinished extraction (leaves the 9 blank)")
    a = ap.parse_args()

    assert len(PEDS_PATHOLOGIES) == 27, len(PEDS_PATHOLOGIES)
    assert len(NEW_9) == 9, NEW_9

    # --- what the LLM produced, per distinct report ---
    qwen: dict[str, dict[str, int]] = {}
    bad = 0
    for f in sorted(glob.glob(os.path.join(a.labels_dir, "labels.shard*.jsonl"))):
        for ln in open(f, errors="replace"):
            try:
                r = json.loads(ln)
            except Exception:
                bad += 1
                continue
            if r.get("status") != "ok":
                bad += 1
                continue
            lab = {}
            for n, v in (r.get("labels") or {}).items():
                p = v.get("p") if isinstance(v, dict) else v
                lab[n] = int(float(p or 0) > 0.5)
            qwen[r["VolumeName"]] = lab
    print("  llm kaydi (benzersiz rapor): %d   (atlanan %d)" % (len(qwen), bad))

    fan = json.load(open(os.path.join(a.labels_dir, "report_fanout.json")))
    print("  fan-out girisi             : %d" % len(fan))

    # Refuse to run on a partial extraction. The glob above takes whatever shard
    # files happen to exist, and the extractor writes regions before labels, so
    # running this mid-job yields a CSV with almost no adult rows - and, because
    # a missing row silently drops the volume from training, that empties the
    # adult half instead of failing. Demand completeness explicitly.
    # The point of this check is to catch running against a half-finished
    # extraction, where ~50% is missing and the output would silently be a
    # near-empty adult cohort. It is NOT meant to block on a couple of parse
    # failures - 2 of 22,976 tonight - because those volumes now keep their
    # published 18 labels and only get blanks for the 9 (see below). A fraction
    # is the right test; a strict equality would have failed the whole chain.
    n_short = len(fan) - len(qwen)
    frac = n_short / max(len(fan), 1)
    if frac > a.max_missing and not a.allow_partial:
        print("  !! cikarma eksik: %d/%d rapor yok (%.2f%%, siniri %.2f%%). "
              "--allow-partial ile zorlanabilir."
              % (n_short, len(fan), 100 * frac, 100 * a.max_missing))
        return 2
    if n_short:
        print("  cikarilamayan rapor: %d (%.3f%%) - 18 yayin etiketi korunuyor, 9'u bos"
              % (n_short, 100 * frac))

    # --- published 18, the training target for the shared classes ---
    pub = {r["VolumeName"]: r for r in csv.DictReader(open(a.published))}
    print("  yayinlanmis etiket satiri  : %d" % len(pub))

    # --- emit one row per volume ---
    miss_llm = miss_pub = 0
    rows = []
    for rep_vol, vols in fan.items():
        lab = qwen.get(rep_vol)
        for v in vols:
            p = pub.get(v)
            if p is None:
                miss_pub += 1
                continue
            # The LLM failing on a report must not cost us the 18 labels we
            # already have from CT-RATE. Emit the row, fill the published 18,
            # and leave the 9 blank - an empty cell reads back as NaN, i.e.
            # "unlabeled", which the masked loss skips. Dropping the row instead
            # removes the volume from training entirely, silently, because a
            # missing label row is a silent skip in the loader.
            if lab is None:
                miss_llm += 1
            # published wins on the 18; the LLM supplies the 9 that have no
            # published equivalent. Never the other way round.
            out = {"VolumeName": v}
            for c in CTRATE_18:
                out[c] = int(float(p.get(c, 0) or 0) > 0.5)
            for c in NEW_9:
                out[c] = "" if lab is None else lab.get(c, 0)
            rows.append(out)

    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["VolumeName"] + list(PEDS_PATHOLOGIES))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, a.out)          # a killed write must not leave a short CSV
    print("  yazilan hacim              : %d  (llm yok %d, yayin yok %d)"
          % (len(rows), miss_llm, miss_pub))

    # --- prevalence of the nine, the whole point of the exercise ---
    print("\n  9 yeni sinif, yetiskin prevalansi (onceki deger: hepsi 0.0%%):")
    for c in NEW_9:
        # blanks are unlabeled, not negative: they must leave the denominator
        # too, or the prevalence of a class quietly reads low
        vals = [r[c] for r in rows if r[c] != ""]
        n = sum(int(v) for v in vals)
        print("     %-42s %6d  %5.2f%%  (etiketli %d)"
              % (c, n, 100.0 * n / max(len(vals), 1), len(vals)))

    # --- diagnostic: do our definitions agree with the reference? ---
    agree = Counter(); total = Counter(); only_q = Counter(); only_p = Counter()
    for rep_vol, vols in fan.items():
        lab = qwen.get(rep_vol)
        if lab is None:
            continue
        p = pub.get(vols[0])
        if p is None:
            continue
        for c in CTRATE_18:
            q = lab.get(c, 0)
            r = int(float(p.get(c, 0) or 0) > 0.5)
            total[c] += 1
            if q == r:
                agree[c] += 1
            elif q:
                only_q[c] += 1
            else:
                only_p[c] += 1
    with open(a.diag, "w") as fh:
        fh.write("%-42s %8s %10s %10s\n" % ("sinif", "uyum%", "yalniz_llm", "yalniz_yayin"))
        for c in CTRATE_18:
            t = max(total[c], 1)
            fh.write("%-42s %7.1f%% %10d %12d\n"
                     % (c, 100.0 * agree[c] / t, only_q[c], only_p[c]))
        t = sum(total.values()) or 1
        fh.write("\n%-42s %7.1f%%\n" % ("TOPLAM", 100.0 * sum(agree.values()) / t))
    print("\n  tani raporu: %s" % a.diag)
    return 0


if __name__ == "__main__":
    sys.exit(main())

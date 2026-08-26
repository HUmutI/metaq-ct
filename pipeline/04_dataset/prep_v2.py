#!/usr/bin/env python3
"""Gate the re-extracted pediatric labels before a 3-hour training run starts.

Every check here exists because something like it already went wrong tonight: a
single 0-byte npz killed a 12,000-step run 13 minutes in, and a class labelled
0 for all 16,400 adults scored below chance. Structural problems are cheap to
find now and expensive to find at hour three, so this exits non-zero and the
chained training never launches.

It also prints the V1 -> V2 prevalence diff, which is the actual experiment:
"Arterial wall calcification" fired on 3.3% of pediatric studies under the old
definition, and the sentences it cited were calcified catheter tracts in veins,
pulmonary artery calcification after congenital heart surgery, and tracheal
cartilage - not aortic atherosclerosis, which is what the adult model scores
0.9365 on. If the tightened definition worked, that number collapses.
"""
from __future__ import annotations
import csv, json, os, sys

V1 = "/temp_work/ch278233/BCH_DATASET/LABELS27"
V2 = "/temp_work/ch278233/BCH_DATASET/LABELS27_V2"
EXPECT_VOLS = 8817
FAIL: list[str] = []


def bad(m):
    FAIL.append(m)
    print("  !! %s" % m)


def load(d):
    rows = list(csv.DictReader(open(os.path.join(d, "labels.csv"))))
    cols = [c for c in rows[0] if c != "VolumeName"] if rows else []
    return rows, cols


def main() -> int:
    r2, c2 = load(V2)
    r1, c1 = load(V1)
    print("[gate] V2 satir %d  sutun %d   (V1 satir %d sutun %d)"
          % (len(r2), len(c2), len(r1), len(c1)))

    # A handful of parse failures is normal and must not block a training run:
    # tonight 4 of 8,817 failed (0.045%), and a strict equality here cancelled
    # three queued jobs over it. What this check is actually for is catching a
    # half-finished or empty extraction, so make it a fraction.
    short = EXPECT_VOLS - len(r2)
    if short > EXPECT_VOLS * 0.005:
        bad("V2 etiket satiri %d, beklenen %d (eksik %d, siniri %d)"
            % (len(r2), EXPECT_VOLS, short, int(EXPECT_VOLS * 0.005)))
    elif short:
        print("  cikarilamayan rapor: %d (%.3f%%) - tolerans icinde"
              % (short, 100.0 * short / EXPECT_VOLS))
    if c2 != c1:
        bad("sutunlar V1 ile ayni degil: fark %s" % (set(c2) ^ set(c1)))
    if len(c2) != 27:
        bad("sutun sayisi %d, 27 olmali" % len(c2))

    v2n = {r["VolumeName"] for r in r2}
    v1n = {r["VolumeName"] for r in r1}
    # V2 may be a subset of V1 (a failed report has no row); it must never
    # contain a volume V1 does not, which would mean the two are not the same
    # cohort and the comparison is meaningless.
    if v2n - v1n:
        bad("V2'de V1'de olmayan %d hacim var" % len(v2n - v1n))
    if len(v1n - v2n) > EXPECT_VOLS * 0.005:
        bad("V2'de eksik hacim %d, tolerans disi" % len(v1n - v2n))

    cache = json.load(open(os.path.join(V2, "region_cache.json")))
    print("[gate] V2 bolge onbellegi anahtari: %d" % len(cache))
    if EXPECT_VOLS - len(cache) > EXPECT_VOLS * 0.005:
        bad("bolge onbellegi %d anahtar, beklenen ~%d" % (len(cache), EXPECT_VOLS))
    empty = sum(1 for o in cache.values() if not any(o.values()))
    if empty > EXPECT_VOLS * 0.02:
        bad("tamamen bos bolge yonlendirmesi olan rapor: %d" % empty)

    def rate(rows, c):
        return 100.0 * sum(float(r[c] or 0) > 0.5 for r in rows) / max(len(rows), 1)

    print("\n[gate] prevalans V1 -> V2 (tum 8817 hacim)")
    print("  %-42s %8s %8s %9s" % ("sinif", "V1%", "V2%", "fark"))
    print("  " + "-" * 72)
    for c in c1:
        a, b = rate(r1, c), rate(r2, c)
        mark = ""
        if c == "Arterial wall calcification":
            mark = "   <<< daraltilan tanim"
        elif abs(b - a) > 5.0:
            mark = "   <<< buyuk kayma"
        print("  %-42s %7.2f%% %7.2f%% %+8.2f%s" % (c, a, b, b - a, mark))
        # a class that collapsed to nothing or exploded to everything is a
        # prompt bug, not a finding
        if b > 90.0:
            bad("%s V2'de %.1f%% - prompt bozulmus olmali" % (c, b))
    if all(rate(r2, c) == 0.0 for c in c2):
        bad("V2'de hicbir sinifta pozitif yok")

    # Actually build the dataset with the new labels. dataset.py and
    # train_stage2.py were edited tonight (NaN-aware labels, masked losses); the
    # loss changes were unit-tested numerically, but the loading path had only
    # been reasoned about. A training job that dies at step 1 because of an edit
    # costs the whole night, and this costs a minute.
    print("\n[gate] gercek veri yolu denemesi")
    try:
        import numpy as np
        sys.path.insert(0, "/home/ch278233/bch-arc-ct")
        os.environ.setdefault("RAC_SCHEMA", "peds")
        from arcct.dataset import RACDatasetV4
        ds = RACDatasetV4(
            data_folder="/temp_work/ch278233/PEDS_NPZ_NESTED",
            reports_csv="/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv",
            labels_csv=os.path.join(V2, "labels.csv"),
            mask_root="/temp_work/ch278233/PEDS_MASKS10_192",
            volume_list_path="/temp_work/ch278233/PEDS_VOLLIST_VALID.txt",
            region_cache_path=os.path.join(V2, "region_cache.json"),
        )
        print("  ornek: %d" % len(ds))
        if len(ds) < 1700:
            bad("dataset yalnizca %d ornek gordu, ~1773 bekleniyordu" % len(ds))
        # __getitem__ returns (ct, text, findings, label, mask, has_mask, acc);
        # the label is index 3. Getting this wrong fails the gate and blocks a
        # perfectly good training run, so it is asserted rather than guessed.
        item = ds[0]
        assert len(item) == 7, "beklenmeyen ornek yapisi: %d eleman" % len(item)
        lab = np.asarray(item[3])
        print("  etiket vektoru: shape %s dtype %s  NaN %d"
              % (lab.shape, lab.dtype, int(np.isnan(lab).sum())))
        if lab.shape[-1] != 27:
            bad("etiket vektoru %d uzunlukta, 27 olmali" % lab.shape[-1])
        if not np.isfinite(lab).all():
            print("  (NaN var - pediatrik tarafta beklenmez ama olumcul degil)")
    except Exception as exc:
        import traceback; traceback.print_exc()
        bad("dataset kurulamadi: %s: %s" % (type(exc).__name__, exc))

    print("\n[gate] SORUN: %d" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

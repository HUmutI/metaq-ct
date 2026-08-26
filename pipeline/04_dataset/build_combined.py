#!/usr/bin/env python3
"""Assemble the joint pediatric + CT-RATE training set.

The two cohorts differ in three ways the loader cares about, and each is
handled here rather than by hoping the loader copes:

* **Labels.** Pediatric reports carry all 27 classes; CT-RATE carries the
  original 18. The 9 pediatric additions are written as 0 for adult volumes -
  not because they are known absent, but because CT-RATE never annotated them.
  That is the honest encoding: a 0 the model can learn from, rather than a
  missing column that would silently drop the row.
* **Masks.** Only the pediatric half has pediatric-schema masks, so the job
  runs with RAC_REQUIRE_MASK=0 and the anatomy queries are bound on the
  pediatric half only. `masked/batch` in the log reports the real fraction.
* **Split.** Both halves split by PATIENT. Pediatric reuses PEDS_SPLIT.json so
  the two runs are comparable on the same validation set; CT-RATE splits on its
  own patient id (train_1234_a_1 -> train_1234).
"""
import sys, os
sys.path.insert(0, "/home/ch278233/pipeline/lib")
from __future__ import annotations
import csv, hashlib, json, os, random, sys

csv.field_size_limit(2 ** 31 - 1)
OUT = "/temp_work/ch278233/COMBINED_NPZ"
PEDS_NPZ = "/temp_work/ch278233/PEDS_NPZ_NESTED"
CT_NPZ = "/temp_work/ch278452/CTRATE_NPZ/train"
D27 = "/temp_work/ch278233/BCH_DATASET/LABELS27"
PATH27 = None


def main() -> int:
    sys.path.insert(0, "/home/ch278233/pipeline")
    from qwen_extract import PATHOLOGIES as P27
    ct_lab_path = None
    for c in ("/temp_work/ch278233/CTRATE/labels_hf/dataset/multi_abnormality_labels/train_predicted_labels.csv",
              "/temp_work/ch278233/CTRATE/labels_hf/train_predicted_labels.csv"):
        if os.path.exists(c):
            ct_lab_path = c
            break
    if not ct_lab_path:
        for root, _d, fn in os.walk("/temp_work/ch278233/CTRATE/labels_hf"):
            for f in fn:
                if "train" in f and f.endswith(".csv"):
                    ct_lab_path = os.path.join(root, f)
    if not ct_lab_path:
        print("HATA: CT-RATE train etiketleri bulunamadi")
        return 1
    print("[comb] ct labels: %s" % ct_lab_path)

    # ---- npz tree: hardlink both cohorts under one root -------------------
    os.makedirs(OUT, exist_ok=True)
    n_ped = n_ct = 0
    for sub in sorted(os.listdir(PEDS_NPZ)):
        sd = os.path.join(PEDS_NPZ, sub)
        if not os.path.isdir(sd):
            continue
        for st in sorted(os.listdir(sd)):
            src_d = os.path.join(sd, st)
            dst_d = os.path.join(OUT, sub, st)
            os.makedirs(dst_d, exist_ok=True)
            for f in os.listdir(src_d):
                d = os.path.join(dst_d, f)
                if not os.path.exists(d):
                    os.link(os.path.join(src_d, f), d)
                n_ped += 1
    have_ct = set()
    n_corrupt = 0
    for f in sorted(os.listdir(CT_NPZ)):
        # Skip the atomic-write staging files. The fetch job writes
        # <name>.npz.tmp.<pid>.npz and renames it into place, so a listing taken
        # while it runs contains names that will not exist a moment later - the
        # first build died on exactly that race.
        # A truncated npz is invisible until a DataLoader worker opens it, and
        # then it kills the run: train_12868_a_2.npz was 0 bytes and took down a
        # 12,000-step job 13 minutes in with EOFError. It is STILL 0 bytes in
        # the source tree, so a rebuild would hardlink it straight back in.
        # Size is enough here - validate_npz.py does the real header check - and
        # a genuine volume is ~24 MB, so 1 KB cannot be a false positive.
        try:
            if f.endswith(".npz") and ".tmp." not in f \
                    and os.path.getsize(os.path.join(CT_NPZ, f)) < 1024:
                n_corrupt += 1
                continue
        except OSError:
            continue                      # vanished between listdir and stat
        if not f.endswith(".npz") or ".tmp." in f:
            continue
        stem = f[:-4]
        pid = "_".join(stem.split("_")[:2])            # train_1234
        dst_d = os.path.join(OUT, pid, stem)
        os.makedirs(dst_d, exist_ok=True)
        d = os.path.join(dst_d, f)
        if not os.path.exists(d):
            os.link(os.path.join(CT_NPZ, f), d)
        have_ct.add(stem + ".nii.gz")
        n_ct += 1
    print("[comb] npz linked: %d pediatric, %d adult" % (n_ped, n_ct))

    if n_corrupt:
        print("[comb] bozuk/kesik npz atlandi: %d" % n_corrupt)

    # ---- reports ----------------------------------------------------------
    rows = []
    for r in csv.DictReader(open(f"{D27}/../LABELS/reports_arcct.csv")):
        rows.append((r["VolumeName"], r.get("Findings_EN", ""), r.get("Impressions_EN", "")))
    base = "/temp_work/ch278233/CTRATE/ct_reports_hf/dataset/radiology_text_reports"
    for r in csv.DictReader(open(f"{base}/train_reports.csv")):
        if r["VolumeName"] in have_ct:
            rows.append((r["VolumeName"], r.get("Findings_EN", ""), r.get("Impressions_EN", "")))
    with open("/temp_work/ch278233/COMBINED_reports.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["VolumeName", "Findings_EN", "Impressions_EN"])
        w.writerows(rows)
    print("[comb] reports: %d" % len(rows))

    # ---- labels: 27 real columns for both cohorts --------------------------
    # This used to read the published CT-RATE CSV directly and let `r.get(p,"0")`
    # fill the nine pediatric-only classes with 0. Those columns do not exist
    # there, so every adult came out negative on all nine: 0 positives across
    # 16,400 volumes. It is a fabricated negative, not an observed one - an LLM
    # pass over the reports finds bone lesions in 21.9% of adults - and the
    # combined model learned it, scoring 0.373 on that class, below chance.
    #
    # CTRATE_labels27.csv now carries the published 18 unchanged plus the nine
    # extracted from the reports. A blank cell there means the extraction failed
    # for that report (3 volumes); it reads back as NaN and the masked loss
    # skips it, which is the honest encoding for "we do not know".
    ped_labels = os.environ.get("RAC_PEDS_LABELS", f"{D27}/labels.csv")
    ct_labels27 = os.environ.get(
        "RAC_CT_LABELS27", "/temp_work/ch278233/CTRATE/CTRATE_labels27.csv")
    if not os.path.isfile(ct_labels27):
        raise FileNotFoundError(
            "%s yok - once apply_ctrate_labels.py calistir" % ct_labels27)

    out = []
    for r in csv.DictReader(open(ped_labels)):
        out.append([r["VolumeName"]] + [r[p] for p in P27])
    n_adult = 0
    rd = csv.DictReader(open(ct_labels27))
    _missing = [p for p in P27 if p not in (rd.fieldnames or [])]
    if _missing:
        raise ValueError("%s eksik sutun: %s" % (ct_labels27, _missing))
    n_blank = 0
    for r in rd:
        v = r.get("VolumeName", "")
        if v not in have_ct:
            continue
        vals = [r[p] for p in P27]
        n_blank += sum(1 for x in vals if x == "")
        out.append([v] + vals)
        n_adult += 1
    print("[comb] yetiskin etiketsiz hucre (NaN olarak okunacak): %d" % n_blank)
    with open("/temp_work/ch278233/COMBINED_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["VolumeName"] + P27)
        w.writerows(out)
    print("[comb] labels: %d rows (%d adult)" % (len(out), n_adult))

    # ---- split -------------------------------------------------------------
    sp = json.load(open("/temp_work/ch278233/PEDS_SPLIT.json"))
    tr = [v + ".nii.gz" for v in sp["train"]]
    va = [v + ".nii.gz" for v in sp["valid"]]
    bypat = {}
    for v in sorted(have_ct):
        bypat.setdefault("_".join(v.split("_")[:2]), []).append(v)
    # The adult split is a function of the PATIENT ID, not of what happens to be
    # on disk. It used to be `random.Random(0).shuffle(sorted(bypat))`: the seed
    # is fixed but the list is not, and the CT-RATE fetch is still running. At
    # the first build the tree held 16,401 volumes / 7,115 patients; it now holds
    # 26,517 / 11,323. Re-running the old logic on today's listing keeps only 78
    # of the 712 current validation patients and moves 652 trained patients into
    # validation - and since train_stage2.py auto-resumes from CTClip.best.pt,
    # that is training on the test set for ~90% of it, with nothing to show for
    # it in any log.
    #
    # Hashing the patient id fixes the assignment for good: a patient lands in
    # the same split no matter how many volumes have been downloaded, and new
    # patients are assigned without disturbing existing ones.
    def adult_split(patient: str) -> str:
        h = hashlib.md5(patient.encode()).hexdigest()
        return "valid" if int(h[:8], 16) % 10 == 0 else "train"

    for p in sorted(bypat):
        (va if adult_split(p) == "valid" else tr).extend(bypat[p])

    assert not (set(tr) & set(va)), "SIZINTI"
    # volume-level disjointness cannot see a patient in both halves, which is
    # the leak that actually matters
    pat_tr = {"_".join(v.split("_")[:2]) for v in tr}
    pat_va = {"_".join(v.split("_")[:2]) for v in va}
    assert not (pat_tr & pat_va), "HASTA SIZINTISI: %s" % sorted(pat_tr & pat_va)[:5]
    for name, lst in (("TRAIN", tr), ("VALID", va)):
        open("/temp_work/ch278233/COMBINED_VOLLIST_%s.txt" % name, "w").write("\n".join(lst) + "\n")
    print("[comb] split: train %d (peds %d + adult %d), valid %d"
          % (len(tr), len(sp["train"]), len(tr) - len(sp["train"]), len(va)))
    print("[comb] overlap check: %d" % len(set(tr) & set(va)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Recall probe for the PHI scrubber. Synthetic strings only -- safe to commit.

Two questions, and only the first one is negotiable:

  RECALL      does every planted identifier disappear? Target 100%. A missed
              name that reaches a checkpoint is unrecoverable.
  PRECISION   does clinically load-bearing text survive? Target high, but a
              false redaction is visible in the audit and costs one line in the
              vocabulary, so it loses every tie against recall.

The strings below are invented. No real report text, name, MRN, accession or
date appears in this file or in its output.

Run: python tools/test_scrub_phi.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "pipeline", "lib"))

from scrub_phi import (DEFAULT_EPONYMS, NamePool, RowPHI,   # noqa: E402
                       classify, scrub)

FAILS: list[str] = []
PLACEHOLDERS = ("[NAME]", "[DATE]", "[ID]", "[SITE]", "[CONTACT]", "[LOC]", "[AGE")

POOL = NamePool(
    patient=frozenset({"kowalczyk", "delacroix", "abernathy", "quintero", "smith"}),
    provider=frozenset({"vandenberg", "okonkwo", "fitzgerald"}),
)
# Stands in for the CT-RATE-derived vocabulary.
VOCAB = frozenset({
    "nodule", "nodules", "pneumonia", "consolidation", "effusion", "pneumothorax",
    "osteosarcoma", "metastatic", "metastases", "transplant", "stridor",
    "bronchiectasis", "thymoma", "mediastinal", "recurrence", "scoliosis",
    "cystic", "fibrosis", "aspiration", "empyema", "atelectasis", "sepsis",
    "neutropenia", "fever", "cough", "dyspnea", "wheeze", "trauma", "fracture",
    "lymphoma", "leukemia", "sarcoma", "chemotherapy", "radiation", "stem",
    "cell", "bone", "marrow", "graft", "rejection", "infection", "abscess",
})


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def S(text, row=RowPHI()):
    return scrub(text, row, POOL, VOCAB, frozenset(DEFAULT_EPONYMS))


def main() -> int:
    # ---- RECALL: the row's own identifiers ------------------------------
    row = RowPHI(first="Marek", last="Kowalczyk", mrn="4471029",
                 accession="ACC88213470", dates=("2024-03-17",))
    txt = ("Marek Kowalczyk, MRN 4471029, accession ACC88213470, scanned "
           "2024-03-17 and again on March 17, 2024 at Boston Children's Hospital. "
           "Ordered by Dr. Vandenberg, reachable at v.oko@example.org or "
           "617-555-0142. Kowalczyk's prior study from 3/17/2024 in room 4B.")
    out = S(txt, row).text
    for leaked in ("Marek", "Kowalczyk", "4471029", "ACC88213470", "2024-03-17",
                   "March 17", "3/17/2024", "Vandenberg", "example.org",
                   "617-555-0142", "Boston Children"):
        check(leaked.lower() not in out.lower(), f"RECALL: {leaked!r} survived -> {out!r}")
    check("2024" not in out, f"RECALL: a bare year survived -> {out!r}")

    # ---- RECALL: names NOT in any dictionary (the fail-closed layer) -----
    for unknown in ("Thibodeaux", "Nakamura", "Oyelaran", "Bergstrom"):
        got = S(f"History per {unknown}, evaluate for pneumonia.").text
        check(unknown not in got,
              f"RECALL: unknown capitalised token {unknown!r} survived -> {got!r}")

    # ---- RECALL: the pool, in any casing --------------------------------
    # A name typed in caps or lowercase is still a name. The dictionary layer
    # has to be case-insensitive over EVERY token, not just Titlecase ones.
    for name in ("Delacroix", "DELACROIX", "delacroix", "Abernathy", "Quintero",
                 "SMITH", "okonkwo", "FITZGERALD"):
        got = S(f"Patient {name} with fever.").text
        check(name.lower() not in got.lower(), f"RECALL: pool name {name!r} survived")

    # ---- PRECISION: clinical content must survive -----------------------
    keep_cases = [
        ("History of metastatic osteosarcoma, evaluate pulmonary nodules.",
         ("osteosarcoma", "nodules")),
        ("Follow up bronchiectasis in cystic fibrosis.", ("bronchiectasis", "fibrosis")),
        ("Rule out pneumonia. Fever and cough.", ("pneumonia", "Fever", "cough")),
        ("Down syndrome, assess for aspiration.", ("Down", "aspiration")),
        ("Post stem cell transplant, evaluate for graft rejection.",
         ("transplant", "graft", "rejection")),
        ("Known Ewing sarcoma, restaging.", ("Ewing", "sarcoma")),
        ("Hodgkin lymphoma, mediastinal recurrence.", ("Hodgkin", "lymphoma", "mediastinal")),
        ("Neutropenia, rule out Aspergillus infection.", ("Neutropenia", "Aspergillus")),
    ]
    for text, must in keep_cases:
        got = S(text).text
        for w in must:
            check(w.lower() in got.lower(),
                  f"PRECISION: {w!r} was redacted out of {text!r} -> {got!r}")

    # ---- an eponym that is ALSO this patient's surname must go -----------
    # L1 runs first, which is the whole reason the layer order is fixed.
    r2 = RowPHI(last="Down")
    got = S("Down syndrome in this patient.", r2).text
    check("down" not in got.lower(),
          f"RECALL: the patient's own surname survived because it is an eponym -> {got!r}")

    # ---- L5: age becomes a band, not nothing ----------------------------
    # NB: bands are in YEARS. A 2-month-old is under one year, so "<1y" is
    # right and the first draft of this test was wrong, not the code.
    for text, want in [("2-month-old with stridor", "[AGE <1y]"),
                       ("14 year old with chest trauma", "[AGE 13-18y]"),
                       ("3 day old, rule out aspiration", "[AGE <1y]"),
                       ("6-week-old with cough", "[AGE <1y]")]:
        got = S(text).text
        check(want in got, f"L5: {text!r} -> {got!r}, expected {want}")
        check("old" not in got.replace("[AGE", ""),
              f"L5: the age phrase was only partly replaced -> {got!r}")
    got = S("2-month-old with stridor").text
    check("stridor" in got, f"L5 must not eat the clinical half -> {got!r}")

    # ---- measurements are not dates or ids ------------------------------
    got = S("Interval growth of a 2 cm nodule; compare 20 mm lesion.").text
    check("2 cm" in got and "20 mm" in got, f"PRECISION: a measurement was redacted -> {got!r}")

    # ---- L6 cap ---------------------------------------------------------
    long_text = " ".join(["nodule"] * 200)
    res = S(long_text)
    check(len(res.text.split()) <= 64, f"L6: {len(res.text.split())} words survived the cap")
    check(res.truncated, "L6: truncation must be recorded")

    # ---- the scrubber must actually fire --------------------------------
    res = S(txt, row)
    check(sum(res.hits.values()) > 0, "a run that matched nothing is a wiring failure")

    # ---- classify -------------------------------------------------------
    for text, want in [("", "absent"), ("   ", "absent"), ("Not given.", "vacuous"),
                       ("None", "vacuous"), ("N/A", "vacuous"), ("-", "vacuous"),
                       ("chest pain", "vacuous"),           # 2 words
                       ("evaluate for pneumonia", "present"),
                       ("History of metastatic osteosarcoma, evaluate nodules", "present")]:
        got = classify(text)
        check(got == want, f"classify({text!r}) = {got}, expected {want}")

    # ---- over-redaction budget on realistic-looking clinical prose ------
    corpus = [t for t, _ in keep_cases]
    words = sum(len(t.split()) for t in corpus)
    names = sum(S(t).text.count("[NAME]") for t in corpus)
    rate = 100.0 * names / max(words, 1)
    check(rate < 5.0, f"over-redaction {rate:.1f} [NAME] per 100 words on clean "
                      "clinical prose -- the vocabulary is too thin")

    if FAILS:
        print("scrub_phi FAIL (%d):" % len(FAILS))
        for f in FAILS:
            print("   ", f)
        return 1
    print(f"scrub_phi OK · every planted identifier removed · unknown capitalised "
          f"tokens fail closed · eponyms and clinical terms survive · age becomes a "
          f"band · over-redaction {rate:.1f} per 100 words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

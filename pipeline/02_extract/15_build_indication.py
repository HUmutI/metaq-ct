#!/usr/bin/env python3
"""Extract, scrub and key the clinical indication -- the field the model is missing.

The indication has always been inside the pediatric report blob and the
sectioniser has always recognised its header, but ``findings_impression()``
keeps only findings and impressions, so the section is parsed and discarded. It
has never existed as a per-volume field, and nothing downstream can read it.
This produces that field.

    peds    reports_8k.csv col I (Report Text)  -- PHI, stays on the cluster
            -> split_sections -> indication_of -> scrub -> keyed by VolumeName
    ctrate  {train,validation}_reports.csv col ClinicalInformation_EN
            -> already de-identified upstream -> normalised only

Writes /temp_work/ch278233/CONTEXT/indication.csv

    VolumeName, Indication_EN, ind_status, ind_words, ind_source, scrub_hits, cohort

``ind_source`` is the matched header name, a categorical -- never text. The file
is written to a .tmp and only renamed after every assertion passes, so a failed
run leaves nothing behind that a training job could pick up.

CT-RATE needs no scrubbing, but the structural assertions run over it anyway.
"Already de-identified upstream" is an assumption, and the check costs nothing;
a nonzero count there is a finding to report, not to silence.

Run:
    python pipeline/02_extract/15_build_indication.py --cohort peds   [--dry-run]
    python pipeline/02_extract/15_build_indication.py --cohort ctrate [--dry-run]
    python pipeline/02_extract/15_build_indication.py --cohort both
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "lib")
sys.path.insert(0, LIB)

from prepare_reports import indication_of                       # noqa: E402
from scrub_phi import (DEFAULT_EPONYMS, NamePool, RowPHI, build_clinical_vocab,  # noqa: E402
                       build_name_pool, classify, load_wordlist, normalise,
                       safe_eponyms, scrub, scrub_structural)

PEDS_REPORTS = "/temp_work/ch278233/BCH_DATASET/reports_8k.csv"
PEDS_MAP = "/temp_work/ch278233/BCH_DATASET/LABELS/volume_map.tsv"
PEDS_ARCCT = "/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv"
CTRATE = [
    ("train", "/temp_work/ch278233/CTRATE/ct_reports_hf/dataset/radiology_text_reports/train_reports.csv"),
    ("valid", "/temp_work/ch278233/CTRATE/ct_reports_hf/dataset/radiology_text_reports/validation_reports.csv"),
]
OUT_DIR = "/temp_work/ch278233/CONTEXT"
OUT = os.path.join(OUT_DIR, "indication.csv")
EPONYM_FILE = os.path.join(LIB, "eponyms.txt")
CLINICAL_FILE = os.path.join(LIB, "clinical_terms.txt")

DATE_COLS = ("Ordered Date", "Scheduled Date", "Patient Arrived Date",
             "Exam Started Date", "Exam Completed Date", "Report Created Date",
             "Preliminary Report By", "Report Finalized Date", "Report Addendum Date")

KEY_RE = re.compile(r"^(ped_\d{5}_\d+|(?:train|valid)_\d+_[a-z]+_\d+)\.nii\.gz$")

# Measured on this cohort before anything was written: a header is present in
# 98.9% of reports and the section carries >= 5 words in 98.3%. A run that comes
# out far from that has changed the parse, not the data.
EXPECTED_PRESENT_RATE = 0.983
PRESENT_TOLERANCE = 0.02


def load_ctrate_vocab() -> frozenset:
    """Clinical vocabulary from PUBLIC text only.

    Built from CT-RATE findings rather than the pediatric reports: a vocabulary
    learned from the corpus being scrubbed would learn the patient names it is
    supposed to be filtering out.
    """
    texts = []
    for _, path in CTRATE:
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                texts.append(row.get("Findings_EN", ""))
                texts.append(row.get("Impressions_EN", ""))
    vocab = set(build_clinical_vocab(texts, min_count=5))
    # The label vocabulary is clinical by definition and must never be redacted.
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
        from arcct import schema
        sch = schema.active()
        for name in sch["PATHOLOGIES"]:
            vocab.update(w.lower() for w in re.findall(r"[A-Za-z]{3,}", name))
        for name in sch["FINE_LABEL_NAMES"].values():
            vocab.update(w.lower() for w in re.findall(r"[A-Za-z]{3,}", name))
        for words in (sch.get("REGION_KEYWORDS") or {}).values():
            for w in words:
                vocab.update(x.lower() for x in re.findall(r"[A-Za-z]{3,}", w))
    except Exception as exc:                                  # noqa: BLE001
        print(f"[vocab] WARNING could not add the label vocabulary: {exc}")
    print(f"[vocab] {len(vocab):,} words from CT-RATE (public) + the label schema")
    return frozenset(vocab)


def peds_clinical_vocab(pool: NamePool, min_count: int = 20) -> frozenset:
    """Pediatric clinical vocabulary, from the DE-IDENTIFIED report text.

    CT-RATE alone is too thin for this corpus: the audit measured restaging
    (206), evali (28) and regorafenib (21) being redacted as unknown names,
    none of which appears in adult CT-RATE findings.

    Three properties keep this from whitelisting a patient name:
      * the source is reports_arcct.csv -- findings and impressions AFTER
        de-identification, not the PHI spreadsheet;
      * every token in the cohort's name pool is subtracted;
      * every token that /usr/share/dict/words lists as a PROPER noun (a
        capitalised entry) is subtracted, which removes place and person names
        the pool does not cover.
    A word still has to occur 20+ times across 8,817 reports to qualify.
    """
    if not os.path.isfile(PEDS_ARCCT):
        print(f"[vocab] no {PEDS_ARCCT}; skipping the pediatric vocabulary")
        return frozenset()
    csv.field_size_limit(10 ** 9)
    texts = []
    with open(PEDS_ARCCT, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            texts.append(row.get("Findings_EN", ""))
            texts.append(row.get("Impressions_EN", ""))
    vocab = set(build_clinical_vocab(texts, min_count=min_count))
    before = len(vocab)
    vocab -= (pool.patient | pool.provider)
    proper = set()
    for path in ("/usr/share/dict/words",):
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                proper = {ln.strip().lower() for ln in fh if ln[:1].isupper()}
            break
    vocab -= proper
    print(f"[vocab] {len(vocab):,} pediatric words (>= {min_count} uses); "
          f"dropped {before - len(vocab):,} for colliding with a name in the "
          f"cohort or with a proper noun")
    return frozenset(vocab)


def build_peds(vocab: frozenset, eponyms: frozenset) -> tuple[list[dict], Counter]:
    csv.field_size_limit(10 ** 9)
    acc2vol = {}
    with open(PEDS_MAP, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            acc2vol[str(row["accession"]).strip()] = str(row["volume_name"]).strip()

    with open(PEDS_REPORTS, newline="", encoding="utf-8", errors="replace") as fh:
        raw_rows = list(csv.DictReader(fh))
    pool = build_name_pool(raw_rows)
    print(f"[peds] name pool: {len(pool.patient):,} patient tokens, "
          f"{len(pool.provider):,} provider tokens")

    rows, hits, srcs = [], Counter(), Counter()
    n_header = 0
    for row in raw_rows:
        acc = str(row.get("Accession Number", "")).strip()
        vol = acc2vol.get(acc)
        if vol is None:
            continue
        blob = str(row.get("Report Text", "") or "")
        body, source = indication_of(blob)
        if source:
            n_header += 1
        phi = RowPHI(
            first=str(row.get("Patient First Name", "") or ""),
            last=str(row.get("Patient Last Name", "") or ""),
            mrn=str(row.get("Patient MRN", "") or ""),
            accession=acc,
            dates=tuple(str(row.get(c, "") or "") for c in DATE_COLS),
        )
        res = scrub(body, phi, pool, vocab, eponyms) if body else None
        text = res.text if res else ""
        if res:
            hits.update(res.hits)
        status = classify(text)
        if status != "present":
            text = ""
        srcs[source or "-"] += 1
        rows.append({
            "VolumeName": vol,
            "Indication_EN": text,
            "ind_status": status,
            "ind_words": len(text.split()),
            "ind_source": source or "-",
            "scrub_hits": sum(res.hits.values()) if res else 0,
            "cohort": "peds",
        })
    print(f"[peds] {len(rows):,} volumes, header present in {n_header:,} "
          f"({100.0 * n_header / max(len(rows), 1):.1f}%)")
    print(f"[peds] source headers: {dict(srcs.most_common())}")
    print(f"[peds] scrub hits by layer: {dict(hits.most_common())}")
    return rows, hits


def build_ctrate() -> list[dict]:
    rows, ct_hits = [], Counter()
    for split, path in CTRATE:
        if not os.path.isfile(path):
            print(f"[ctrate] MISSING {path}")
            continue
        n = 0
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                vol = str(row.get("VolumeName", "") or "").strip()
                if not vol:
                    continue
                res = scrub_structural(row.get("ClinicalInformation_EN", ""))
                text = res.text
                ct_hits.update(res.hits)
                status = classify(text)
                if status != "present":
                    text = ""
                rows.append({
                    "VolumeName": vol,
                    "Indication_EN": text,
                    "ind_status": status,
                    "ind_words": len(text.split()),
                    "ind_source": "column",
                    "scrub_hits": sum(res.hits.values()),
                    "cohort": "ctrate",
                })
                n += 1
        print(f"[ctrate/{split}] {n:,} volumes")
    pres = sum(1 for r in rows if r["ind_status"] == "present")
    print(f"[ctrate] present {pres:,}/{len(rows):,} ({100.0 * pres / max(len(rows), 1):.1f}%)")
    # Nonzero here means the upstream de-identification left something behind.
    # Report it; it is a finding about the public dataset, not a defect here.
    print(f"[ctrate] structural hits: {dict(ct_hits.most_common()) or 'none'}")
    return rows


def assertions(rows: list[dict], peds_raw: list[dict], pool: NamePool,
               vocab: frozenset, hits: Counter, eponyms: frozenset) -> list[str]:
    """Every one of these is fatal. Nothing is written until they all pass."""
    bad = []
    texts = [(r["VolumeName"], r["Indication_EN"]) for r in rows if r["Indication_EN"]]

    def scan(name, pattern, limit=3):
        rx = re.compile(pattern, re.I) if isinstance(pattern, str) else pattern
        hit = [(v, t) for v, t in texts if rx.search(t)]
        if hit:
            bad.append(f"A-{name}: {len(hit)} rows match, e.g. volumes "
                       f"{[v for v, _ in hit[:limit]]}")

    accs = {str(r.get("Accession Number", "")).strip() for r in peds_raw}
    mrns = {str(r.get("Patient MRN", "")).strip() for r in peds_raw}
    accs = {a for a in accs if len(a) >= 4}
    mrns = {m for m in mrns if len(m) >= 4}
    joined = " \n ".join(t for _, t in texts).lower()
    leaked_acc = [a for a in accs if a.lower() in joined]
    if leaked_acc:
        bad.append(f"A1 accession: {len(leaked_acc)} accession strings survive")
    leaked_mrn = [m for m in mrns if m.lower() in joined]
    if leaked_mrn:
        bad.append(f"A2 MRN: {len(leaked_mrn)} MRN strings survive")

    scan("A3 digits", r"(?<!\d)\d{5,}(?!\d)")
    scan("A6 date", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:19|20)\d{2}\b"
                    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d")
    scan("A7 contact", r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")

    # The SAME tokeniser the scrubber uses. A different one splits hyphenated and
    # apostrophised words differently and reports 'leaks' that do not exist as
    # tokens in the text at all -- the first run of this check flagged 'del' and
    # 'min', neither of which appears as a word in the output.
    from scrub_phi import WORD_TOKEN                            # noqa: PLC0415
    # ONLY the pediatric rows. The name pool is built from BCH patients and
    # providers; CT-RATE is public text about different people, so a coincidence
    # between an ordinary English word and a BCH surname ("day", "blood",
    # "friend" are all real surnames) is a category error, not a leak.
    tokens = set()
    for v, t in texts:
        if v.startswith("ped_"):
            tokens.update(m.group(0).lower() for m in WORD_TOKEN.finditer(t))
    # The assertion has to mirror the scrubber's policy or it fails on the
    # deliberate residual. A pool token that is ALSO ordinary clinical
    # vocabulary survives on purpose -- see _cap() -- so it is reported as a
    # number, not as a leak. A pool token surviving for any OTHER reason is a
    # real leak and is fatal.
    from scrub_phi import ALWAYS_KEEP, _stems                   # noqa: PLC0415
    known = set(vocab) | set(ALWAYS_KEEP) | set(eponyms)
    survivors = tokens & (pool.patient | pool.provider)
    deliberate = {w for w in survivors if any(st in known for st in _stems(w))}
    leaked_names = survivors - deliberate
    if leaked_names:
        bad.append(f"A4/A5 names: {len(leaked_names)} pool tokens survive in the "
                   f"pediatric rows for no vocabulary reason, e.g. "
                   f"{sorted(leaked_names)[:5]}")
    if deliberate:
        print(f"[audit] {len(deliberate)} cohort surnames survive because they are "
              f"also clinical vocabulary (deliberate, see _cap): "
              f"{sorted(deliberate)[:8]}{' ...' if len(deliberate) > 8 else ''}")

    badkey = [r["VolumeName"] for r in rows if not KEY_RE.match(r["VolumeName"])]
    if badkey:
        bad.append(f"A8 key format: {len(badkey)} keys are not de-identified, "
                   f"e.g. {badkey[:3]}")

    if len(rows) != len({r["VolumeName"] for r in rows}):
        bad.append("A9 duplicate VolumeName keys")

    peds_rows = [r for r in rows if r["cohort"] == "peds"]
    if peds_rows:
        rate = sum(1 for r in peds_rows if r["ind_status"] == "present") / len(peds_rows)
        if abs(rate - EXPECTED_PRESENT_RATE) > PRESENT_TOLERANCE:
            bad.append(f"A10 present rate {rate:.3f} is more than "
                       f"{PRESENT_TOLERANCE} from the measured "
                       f"{EXPECTED_PRESENT_RATE:.3f}; the parse has changed")
        short = [r for r in peds_rows if r["ind_status"] == "present" and r["ind_words"] < 3]
        if short:
            bad.append(f"A10b {len(short)} 'present' rows carry fewer than 3 words")
        if sum(hits.values()) == 0:
            bad.append("A12 the scrubber matched nothing at all -- that is a wiring "
                       "failure, not a clean corpus")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("peds", "ctrate", "both"), default="both")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sample", type=int, default=200,
                    help="rows to write to review_sample.txt for human review")
    a = ap.parse_args()

    eponyms = frozenset(DEFAULT_EPONYMS) | load_wordlist(EPONYM_FILE)
    vocab = load_ctrate_vocab()

    rows, hits, peds_raw = [], Counter(), []
    pool = NamePool(frozenset(), frozenset())
    if a.cohort in ("peds", "both"):
        csv.field_size_limit(10 ** 9)
        with open(PEDS_REPORTS, newline="", encoding="utf-8", errors="replace") as fh:
            peds_raw = list(csv.DictReader(fh))
        pool = build_name_pool(peds_raw)
        # Enforced, not curated: an eponym that is also a surname in this cohort
        # is removed from the whitelist before it can protect anything.
        eponyms = safe_eponyms(eponyms, pool)
        vocab = vocab | peds_clinical_vocab(pool)
        # Curated terms go through the SAME collision guard as the eponyms: an
        # entry that is also a surname in this cohort never reaches the
        # whitelist, so extending the list cannot open a leak.
        vocab = vocab | safe_eponyms(load_wordlist(CLINICAL_FILE), pool)
        p_rows, hits = build_peds(vocab, eponyms)
        rows += p_rows
    if a.cohort in ("ctrate", "both"):
        rows += build_ctrate()

    bad = assertions(rows, peds_raw, pool, vocab, hits, eponyms)
    words = [r["ind_words"] for r in rows if r["ind_status"] == "present"]
    if words:
        w = sorted(words)
        print(f"\n[both] present {len(w):,} rows  median {w[len(w) // 2]} words  "
              f"p90 {w[int(0.9 * len(w))]}  max {w[-1]}")
    print(f"[both] status: {dict(Counter(r['ind_status'] for r in rows))}")

    if bad:
        print("\nFATAL -- nothing written:")
        for b in bad:
            print("   ", b)
        return 1
    print("\nall assertions passed")

    if a.dry_run:
        print("[dry-run] nothing written")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["VolumeName", "Indication_EN", "ind_status", "ind_words",
            "ind_source", "scrub_hits", "cohort"]
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w_.writeheader()
        for r in sorted(rows, key=lambda r: r["VolumeName"]):
            w_.writerow(r)
    os.replace(tmp, OUT)
    digest = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    with open(OUT + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    print(f"wrote {OUT}  ({len(rows):,} rows)\nsha256 {digest}")

    # The artefact the IRB side needs: scrubbed text a human can actually read
    # through. Stays on the cluster; only the path is printed.
    import random
    present = [r for r in rows if r["ind_status"] == "present" and r["cohort"] == "peds"]
    sample = random.Random(0).sample(present, min(a.sample, len(present)))
    spath = os.path.join(OUT_DIR, "review_sample.txt")
    with open(spath, "w", encoding="utf-8") as fh:
        fh.write("# Scrubbed pediatric indications, seeded sample, for human review.\n"
                 "# If ANY line still carries a name, a date or an identifier, the\n"
                 "# scrubber is not ready and no training may start.\n\n")
        for r in sample:
            fh.write(f"{r['ind_source']:<20s} | {r['Indication_EN']}\n")
    print(f"review sample: {spath}  ({len(sample)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

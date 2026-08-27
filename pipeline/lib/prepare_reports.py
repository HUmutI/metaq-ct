#!/usr/bin/env python3
"""Turn the BCH pediatric report spreadsheet into ARC-CT-shaped inputs.

Produces, under --out-dir:

  volume_map.tsv      accession <-> de-identified volume name  (STAYS ON CLUSTER)
  reports_arcct.csv   VolumeName, Findings_EN, Impressions_EN  (ARC-CT schema)
  sentences.jsonl     {"VolumeName": ..., "sentences": [...]}  (LLM stage input)

Three things this does that matter:

1. **De-identified volume names.** Volumes are renamed ``ped_<subj>_<study>.nii.gz``
   where ``subj`` is a sequential id assigned per MRN. No accession or MRN ever
   reaches the LLM stage, the label CSV, the region cache, or a checkpoint.
   The two-token name also gives ``evaluate.py``'s patient-clustered bootstrap
   (which splits VolumeName on "_") real patient clusters instead of per-scan
   ones.

2. **Section splitting.** The source has one ``Report Text`` blob; ARC-CT wants
   ``Findings_EN`` / ``Impressions_EN`` separately, and stage 2 sets
   ``RAC_IMPRESSION_FIRST=1`` so the impression must actually be its own field.

3. **Sentence indexing.** The LLM stage returns sentence *indices*, never
   sentence text, so region text is reconstructed here from the original report.
   That makes verbatim fidelity exact by construction rather than by asking the
   model to quote carefully.

This script prints aggregate statistics only - never report text, never an
accession, never an MRN.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Section headers seen in BCH pediatric chest CT reports.
SECTION_RE = re.compile(
    r"(?im)^\s*\*{0,2}\s*("
    r"CLINICAL\s+(?:HISTORY|INDICATION|INFORMATION)|HISTORY|INDICATION|REASON\s+FOR\s+EXAM"
    r"|TECHNIQUE|PROCEDURE|EXAM(?:INATION)?|PROTOCOL"
    r"|COMPARISON|COMPARISONS?|PRIOR\s+STUDIES"
    r"|FINDINGS?|OBSERVATIONS?"
    r"|IMPRESSION S?|IMPRESSIONS?|CONCLUSION|ASSESSMENT|SUMMARY"
    r"|RECOMMENDATION S?|RECOMMENDATIONS?"
    r")\s*\**\s*:?\s*"
)

FINDINGS_KEYS = {"FINDINGS", "FINDING", "OBSERVATIONS", "OBSERVATION"}
IMPRESSION_KEYS = {"IMPRESSION", "IMPRESSIONS", "CONCLUSION", "ASSESSMENT",
                   "SUMMARY"}

# Long digit runs are MRNs / accessions / phone numbers; explicit dates are not
# needed for labelling. Everything stays on the cluster either way, but keeping
# identifiers out of the text means they cannot leak into a checkpoint or log.
ID_RE = re.compile(r"\b\d{6,}\b")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
MRN_LABEL_RE = re.compile(r"(?i)\b(?:mrn|medical record(?:\s+number)?|acc(?:ession)?(?:\s+(?:no|number|#))?)\b\s*[:#]?\s*\S+")

# Sentence splitting: real sentence ends, plus list/newline structure. Avoids
# splitting on decimals ("1.5 cm") and common abbreviations.
ABBREV = r"(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bvs)(?<!\bcf)(?<!\be\.g)(?<!\bi\.e)(?<!\bapprox)"
SENT_SPLIT_RE = re.compile(r"(?:(?<=[.!?])" + ABBREV + r"(?<!\d\.)\s+|\n+|(?:^|\s)[-•*]\s+)")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_x000D_", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = MRN_LABEL_RE.sub("[ID]", s)
    s = DATE_RE.sub("[DATE]", s)
    s = ID_RE.sub("[ID]", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_sections(text: str) -> dict[str, str]:
    """Split a report blob on its section headers. Returns UPPERCASE keys."""
    out: dict[str, str] = {}
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return {}
    for i, m in enumerate(matches):
        key = re.sub(r"\s+", " ", m.group(1)).strip().upper().rstrip("S ") \
            if m.group(1).strip().upper().startswith("IMPRESSION") \
            else re.sub(r"\s+", " ", m.group(1)).strip().upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if key in out:
            out[key] = (out[key] + "\n" + body).strip()
        else:
            out[key] = body
    return out


INDICATION_KEYS = ("CLINICAL HISTORY", "CLINICAL INDICATION", "CLINICAL INFORMATION",
                   "INDICATION", "REASON FOR EXAM", "HISTORY")


def indication_of(text: str) -> tuple[str, str]:
    """-> (body, which_header) for the clinical-context section, or ("", "").

    The sectioniser has always recognised these headers and findings_impression()
    has always thrown the section away. This returns it instead, so the same
    split produces both the report text the model is trained against and the
    indication it is conditioned on -- one parse, no second definition to drift.

    The key order is significant: a report carrying both CLINICAL HISTORY and a
    bare HISTORY should yield the more specific one, so HISTORY is last.
    """
    sections = split_sections(text)
    if not sections:
        return "", ""
    for key in INDICATION_KEYS:
        body = sections.get(key, "").strip()
        if body:
            return body, key
    return "", ""


def findings_impression(text: str) -> tuple[str, str, str]:
    """-> (findings, impression, how_it_was_split)"""
    sections = split_sections(text)
    if sections:
        find = " ".join(v for k, v in sections.items()
                        if k.replace(" ", "").rstrip("S") in
                        {k2.rstrip("S") for k2 in FINDINGS_KEYS}).strip()
        imp = " ".join(v for k, v in sections.items()
                       if k.replace(" ", "").rstrip("S") in
                       {k2.rstrip("S") for k2 in IMPRESSION_KEYS}).strip()
        if find or imp:
            if find and imp:
                return find, imp, "both_sections"
            return (find, imp, "findings_only") if find else (find, imp, "impression_only")
    # No recognisable headers: treat the whole blob as findings so no clinical
    # content is silently dropped.
    return text.strip(), "", "no_sections"


def sentences_of(text: str, max_len: int = 600) -> list[str]:
    parts = [p.strip(" \t-•*") for p in SENT_SPLIT_RE.split(text or "")]
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 3:
            continue
        while len(p) > max_len:            # never let one blob dominate
            out.append(p[:max_len].strip())
            p = p[max_len:].strip()
        if p:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default="/temp_work/ch278233/BCH_DATASET/reports_8k.csv")
    ap.add_argument("--nii-dir", default="/temp_work/ch278233/BCH_DATASET/THIN_LUNG_NII",
                    help="only accessions with a volume here are emitted")
    ap.add_argument("--out-dir", default="/temp_work/ch278233/BCH_DATASET/LABELS")
    ap.add_argument("--acc-col", default="Accession Number")
    ap.add_argument("--mrn-col", default="Patient MRN")
    ap.add_argument("--date-col", default="Exam Completed Date")
    ap.add_argument("--text-col", default="Report Text")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    have = {f[:-4] for f in os.listdir(args.nii_dir) if f.endswith(".nii")}
    print("volumes on disk: %d" % len(have))

    rows = []
    stats = Counter()
    with open(args.reports, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (args.acc_col, args.mrn_col, args.text_col)
                   if c not in (reader.fieldnames or [])]
        if missing:
            print("ERROR: reports CSV lacks column(s): %s" % missing)
            print("       columns present: %s" % (reader.fieldnames or []))
            return 1
        for r in reader:
            stats["csv_rows"] += 1
            acc = (r.get(args.acc_col) or "").strip()
            if not acc:
                stats["no_accession"] += 1
                continue
            if acc not in have:
                stats["no_volume"] += 1
                continue
            text = clean_text(r.get(args.text_col) or "")
            if not text:
                stats["empty_report"] += 1
                continue
            rows.append((acc, (r.get(args.mrn_col) or "").strip(),
                         (r.get(args.date_col) or "").strip(), text))

    # Subject ids by MRN, study index by date order within subject. Assigned
    # over MRN-sorted order so the mapping is deterministic across reruns.
    by_mrn = defaultdict(list)
    for acc, mrn, date, text in rows:
        by_mrn[mrn or ("__noMRN__" + acc)].append((date, acc, text))

    mapping = {}
    for si, mrn in enumerate(sorted(by_mrn), start=1):
        for sj, (_date, acc, _t) in enumerate(sorted(by_mrn[mrn]), start=1):
            mapping[acc] = "ped_%05d_%d" % (si, sj)
    stats["subjects"] = len(by_mrn)
    stats["volumes_emitted"] = len(mapping)

    map_path = os.path.join(args.out_dir, "volume_map.tsv")
    with open(map_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["accession", "volume_name", "nii_path"])
        for acc in sorted(mapping):
            w.writerow([acc, mapping[acc] + ".nii.gz",
                        os.path.join(args.nii_dir, acc + ".nii")])

    rep_path = os.path.join(args.out_dir, "reports_arcct.csv")
    sent_path = os.path.join(args.out_dir, "sentences.jsonl")
    nsent = Counter()
    with open(rep_path, "w", newline="") as rf, open(sent_path, "w") as sf:
        w = csv.writer(rf)
        w.writerow(["VolumeName", "Findings_EN", "Impressions_EN"])
        for acc, _mrn, _date, text in sorted(rows, key=lambda x: mapping[x[0]]):
            vol = mapping[acc]
            find, imp, how = findings_impression(text)
            stats["split_" + how] += 1
            w.writerow([vol + ".nii.gz", find, imp])
            sents = sentences_of(find) + sentences_of(imp)
            nsent[min(len(sents), 60)] += 1
            sf.write(json.dumps({
                "VolumeName": vol + ".nii.gz",
                "n_findings_sentences": len(sentences_of(find)),
                "sentences": sents,
            }) + "\n")

    print("\n--- counts ---")
    for k in ("csv_rows", "no_accession", "no_volume", "empty_report",
              "subjects", "volumes_emitted"):
        print("  %-22s %6d" % (k, stats[k]))
    print("\n--- how the report split ---")
    for k in sorted(k for k in stats if k.startswith("split_")):
        print("  %-22s %6d" % (k[6:], stats[k]))
    tot = sum(nsent.values()) or 1
    vals = sorted(nsent.elements())
    print("\n--- sentences per report ---")
    print("  median %d   p90 %d   max(capped 60) %d   reports %d"
          % (vals[len(vals) // 2], vals[int(len(vals) * 0.9)], max(vals), tot))
    print("\nwrote:\n  %s\n  %s\n  %s" % (map_path, rep_path, sent_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

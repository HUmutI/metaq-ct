#!/usr/bin/env python3
"""Stratified 200-report sample for building a gold standard.

Why stratified and not random: a random draw would be dominated by the classes
that are easy and common, and would carry almost no signal about the ones we
have measured to be in doubt. The strata are chosen from measurements, not
intuition:

  * the 8 classes whose thresholds diverge one-sidedly from CT-RATE's published
    labels (Lymphadenopathy 40x conservative, Mosaic attenuation 341x liberal,
    ...) - these are where our prompt and the reference disagree systematically
  * the weakest classes by pediatric AUC (Bone lesion 0.61, Lung nodule 0.67,
    Pleural thickening 0.68, Pneumothorax) - where label noise would hurt most
  * Arterial wall calcification, the one drift we found and corrected, as a
    positive control: the gold standard should confirm the correction
  * a random stratum, so the sample also measures the ordinary case

Both label versions (V1 pre-correction, V2 post-correction) travel with each
report, so the adjudication can say which one was right rather than only
scoring one.

De-identification: the source export already replaced dates with [TARIH] and
carries no MRN-like numbers, but three leaks survive it and are scrubbed here -
month+year strings (3.4% of reports), physician names in Dr./M.D. form (4.5%),
and the age expressions that reach findings text through the no-section fallback
(0.1%). The indication section never enters this file at all; it is not part of
what the labeller reads.
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, sys

sys.path.insert(0, "/home/ch278233/bch-arc-ct")
os.environ.setdefault("RAC_SCHEMA", "peds")
from arcct.schema import PEDS_PATHOLOGIES as P27

# measured trouble spots, in the order they were established
DIVERGENT = ["Lymphadenopathy", "Mosaic attenuation pattern",
             "Coronary artery wall calcification", "Peribronchial thickening",
             "Atelectasis", "Interlobular septal thickening",
             "Bronchiectasis", "Consolidation"]
WEAK = ["Bone lesion or fracture", "Lung nodule",
        "Pleural thickening or nodule", "Pneumothorax"]
CONTROL = ["Arterial wall calcification"]

# Dates survive in more shapes than "March 2010". The source export replaced
# most of them with [TARIH], but "July 6, 2014", a bare "November 29" and the
# times that accompany a phone handoff ("7:12 PM") all came through. A month
# name is the reliable anchor: take it with whatever day and year follow.
# Alternation order matters and the obvious spelling is wrong: with the day and
# year as two independent optional groups, "March 2010" lets the day group eat
# " 20" and leaves "10" behind, redacting to "[DATE]10". Try the longest form
# first - day-and-year, then year alone, then a bare day.
MONTH = (r"\b(?:January|February|March|April|May|June|July|August|September|"
         r"October|November|December)\b"
         r"(?:\s+\d{1,2}\s*,?\s*\d{4}|\s+\d{4}|\s+\d{1,2}(?!\d))?")
TIME = r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?"
# Physician names appear in three forms and the obvious pattern catches only
# two. "ALLISON WISE MD" is written in full caps, so a rule that expects a
# lowercase tail after the initial capital walks straight past it - it was
# still in the clear after the first pass, in a sentence whose other name
# the same rule had already redacted. Match names case-insensitively, and
# allow one or two name tokens before the credential.
PHYS = (r"\bDr\.?\s+[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)?"
        r"|\b[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)?,?\s+M\.?\s?D\.?\b"
        r"|\b[A-Z][A-Za-z'-]+\s+[A-Z][A-Za-z'-]+,?\s+(?:MD|DO|RN|NP|PA)\b")
AGE = r"\b\d{1,3}[\s-]*(?:years?|yrs?|months?|mos?|weeks?|wks?|days?)[\s-]*old\b"


# Names also arrive without their title. The sentence splitter treats "Dr." as
# an end of sentence, so "discussed with Dr. Andy Place" becomes two sentences
# and the second opens "Andy Place ..." with no title to anchor on. Findings
# came out clean while the sentence list did not - the same name redacted in one
# field and in the clear in the next.
#
# The obvious fix, "redact a capitalised pair at the start of a fragment", is
# far worse than the leak: it turns "Mild Atelectasis at the base" into
# "[PHYSICIAN] at the base" and destroys the evidence the label rests on. So the
# orphaned-name rule fires only where the split actually happened - when the
# PREVIOUS sentence ends in a title - and never on a sentence in isolation.
HANDOFF = (r"(?:discussed|reviewed|communicated|reported|performed|interpreted|"
           r"conveyed|relayed|called)\s+(?:with|to|by|at)\s+")
CTX_NAME = re.compile(HANDOFF + r"([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})")
BY_NAME = re.compile(r"\bby\s+([A-Z][A-Za-z'-]+\s+[A-Z][A-Za-z'-]+)\b")
TITLE_END = re.compile(r"\b(?:Dr|Drs|Prof|Mr|Ms|Mrs)\.?\s*$")
LEAD_NAME = re.compile(r"^([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})\b")


def scrub(t: str) -> str:
    """Redact a standalone string. Cross-sentence cases go through scrub_seq."""
    t = re.sub(MONTH, "[DATE]", t)
    t = re.sub(TIME, "[TIME]", t)
    t = re.sub(PHYS, "[PHYSICIAN]", t)
    t = CTX_NAME.sub(lambda m: m.group(0).replace(m.group(1), "[PHYSICIAN]"), t)
    t = BY_NAME.sub("by [PHYSICIAN]", t)
    t = re.sub(r"\b(?:MD|M\.D\.|DO|RN|NP|PA)\b(?!\w)", "[CREDENTIAL]", t)
    t = re.sub(AGE, "[AGE]", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def scrub_seq(sents: list[str]) -> list[str]:
    """Scrub a sentence list, repairing names the splitter orphaned."""
    out = []
    for k, s in enumerate(sents):
        s = scrub(s)
        if k and TITLE_END.search(sents[k - 1] or ""):
            s = LEAD_NAME.sub("[PHYSICIAN]", s, count=1)
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/temp_work/ch278233/GOLD_SAMPLE")
    a = ap.parse_args()
    csv.field_size_limit(10 ** 7)
    os.makedirs(a.out, exist_ok=True)

    rep = {r["VolumeName"]: r for r in csv.DictReader(
        open("/temp_work/ch278233/BCH_DATASET/LABELS/reports_arcct.csv"))}
    v1 = {r["VolumeName"]: r for r in csv.DictReader(
        open("/temp_work/ch278233/BCH_DATASET/LABELS27/labels.csv"))}
    v2 = {r["VolumeName"]: r for r in csv.DictReader(
        open("/temp_work/ch278233/BCH_DATASET/LABELS27_V2/labels.csv"))}
    reg = json.load(open("/temp_work/ch278233/BCH_DATASET/LABELS27_V2/region_cache.json"))
    sent = {}
    for ln in open("/temp_work/ch278233/BCH_DATASET/LABELS27_V2/sentences.jsonl"):
        r = json.loads(ln)
        sent[r["VolumeName"]] = r["sentences"]

    # the residual-leak list stays out: those reports carry indication text
    excl = set()
    p = "/temp_work/ch278233/LEAKAGE_EXCLUDE.txt"
    if os.path.isfile(p):
        excl = {l.split("\t")[0] for l in open(p) if not l.startswith("#")}

    pool = [v for v in v2 if v in rep and v in v1 and v in sent and v not in excl]
    rng = random.Random(a.seed)
    picked, why = [], {}

    def take(vols, tag, k):
        rng.shuffle(vols)
        n = 0
        for v in vols:
            if v in why or n >= k:
                continue
            why[v] = tag; picked.append(v); n += 1
        return n

    # strata where V1 and V2 DISAGREE are the most informative of all: the gold
    # standard settles which correction was right
    for c in CONTROL + DIVERGENT + WEAK:
        dis = [v for v in pool
               if (float(v1[v][c] or 0) > .5) != (float(v2[v][c] or 0) > .5)]
        take(dis, "v1v2_disagree:" + c, max(2, a.per_class // 3))
    for c in CONTROL + DIVERGENT + WEAK:
        pos = [v for v in pool if float(v2[v][c] or 0) > .5]
        take(pos, "positive:" + c, a.per_class)
    take(list(pool), "random", a.n - len(picked))
    picked = picked[:a.n]

    scrubbed = 0
    with open(os.path.join(a.out, "gold_sample.jsonl"), "w") as fh:
        for v in picked:
            f = scrub(rep[v]["Findings_EN"] or "")
            i = scrub(rep[v]["Impressions_EN"] or "")
            if f != re.sub(r"\s+", " ", (rep[v]["Findings_EN"] or "")).strip():
                scrubbed += 1
            fh.write(json.dumps({
                "VolumeName": v,
                "stratum": why[v],
                "findings": f,
                "impressions": i,
                "sentences": scrub_seq(sent.get(v, [])),
                "labels_v1": {c: int(float(v1[v][c] or 0) > .5) for c in P27},
                "labels_v2": {c: int(float(v2[v][c] or 0) > .5) for c in P27},
                # the region cache holds the same sentences again; scrubbing
                # findings and sentences but not this would leave a clean copy
                # of the text beside a redacted one in the same record
                "regions_v2": {k: scrub_seq(val or [])
                               for k, val in (reg.get(v) or {}).items()},
            }, ensure_ascii=False) + "\n")

    from collections import Counter
    cnt = Counter(t.split(":")[0] for t in why.values())
    print("  ornek: %d rapor" % len(picked))
    print("  tabaka: %s" % dict(cnt))
    print("  temizleme uygulanan rapor: %d" % scrubbed)
    print("  yazildi: %s/gold_sample.jsonl" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

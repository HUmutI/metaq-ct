"""De-identify a clinical indication before it can reach a model.

This text has never left the report column before. It is about to become a model
input, a checkpoint, and eventually a figure, so the scrubber is written to fail
CLOSED: an unrecognised capitalised token is redacted rather than kept. A missed
name that reaches a checkpoint is unrecoverable; an over-redacted rare clinical
term is measurable, visible in the audit, and fixed by adding one line to the
vocabulary.

Six layers, in order, earlier wins:

  L0  normalise      _x000D_, CRLF, whitespace, the header remnant
  L1  row-targeted   THIS row's own name / MRN / accession / dates -- highest
                     precision, and it runs first so that a patient whose
                     surname happens to be an eponym is still redacted
  L2  structural     long digit runs, dates in every rendering, phone, SSN,
                     email, URL, "Dr. X", room/bed, and the site literals
  L3  dictionary     every patient and provider name in the cohort, minus the
                     clinical vocabulary and the eponym whitelist
  L4  fail-closed    an unknown capitalised token -> [NAME]
  L5  age -> band    "2-month-old" becomes "[AGE 1-2y]", not nothing
  L6  cap            64 words / 512 characters

L5 is a judgement, not an oversight. Deleting the age destroys the clinical
sense of a pediatric indication -- "2-month-old with stridor" is a different
question from "with stridor" -- while leaving the number creates a second,
unbanded age channel that contradicts the band the model is given, which is
exactly the `age` reason already recorded in LEAKAGE_EXCLUDE.txt.

The clinical vocabulary is built from CT-RATE only: it is public, so it carries
no BCH PHI, and a vocabulary derived from the pediatric reports themselves would
learn the patient names it is supposed to be filtering.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, NamedTuple

# ---------------------------------------------------------------- L0
_CTRL = re.compile(r"_x000[dD]_|\r\n|\r")
_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{2,}")

# ---------------------------------------------------------------- L2
MONTHS = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
          r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|"
          r"Nov(?:ember)?|Dec(?:ember)?")
STRUCTURAL = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[CONTACT]"),
    (re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I), "[CONTACT]"),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "[CONTACT]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ID]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    (rf"(?i)\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", "[DATE]"),
    (rf"(?i)\b(?:{MONTHS})\.?\s+\d{{4}}\b", "[DATE]"),
    (rf"(?i)\b\d{{1,2}}\s+(?:{MONTHS})\.?\s+\d{{4}}\b", "[DATE]"),
    # Month + day with NO year ("aug 2"), and month/year or month-year with no
    # day ("1/2020"). Both appear in real requisitions and neither is covered by
    # the three patterns above, which all require a four-digit year in place.
    (rf"(?i)\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b", "[DATE]"),
    (rf"(?i)\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\b", "[DATE]"),
    (re.compile(r"\b\d{1,2}[/-](?:19|20)\d{2}\b"), "[DATE]"),
    # A bare year. Deliberately not (18|19|20)\d\d: "2 cm" and "20 mm" must
    # survive, so this requires a four-digit token that is not a measurement.
    (re.compile(r"(?<![\d.])\b(?:19|20)\d{2}\b(?!\s*(?:cm|mm|ml|mg|hu|kg))", re.I), "[DATE]"),
    (re.compile(r"\bMRN\s*:?\s*\d+", re.I), "[ID]"),
    (re.compile(r"\b(?:acc(?:ession)?|study|exam)\s*(?:no\.?|number|#)?\s*:?\s*\d{4,}", re.I), "[ID]"),
    (re.compile(r"(?<!\d)\d{5,}(?!\d)"), "[ID]"),
    (re.compile(r"\b(?:Dr|Doctor|Prof|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][A-Za-z'\-]+"
                r"(?:\s+[A-Z][A-Za-z'\-]+)?"), "[NAME]"),
    (re.compile(r"\b(?:room|bed|ward|suite|floor)\s*#?\s*[A-Z]?\d+[A-Z]?\b", re.I), "[LOC]"),
    # A bare month name. It is a date fragment, not a name, and tagging it
    # [NAME] both mislabels it and inflates the over-redaction count.
    (rf"(?i)\b(?:{MONTHS})\b", "[DATE]"),
    # Last sweep for a stray four-digit year. The structured rules above cover
    # the renderings that carry a day or a month name; this catches the ones
    # that reached the text some other way. The unit lookahead keeps "2020 HU"
    # and similar measurements intact.
    (re.compile(r"\b(?:19|20)\d{2}\b(?!\s*(?:cm|mm|ml|mg|hu|kg|cc|mcg))", re.I), "[DATE]"),
]
# The site strip is not optional. It also removes the cohort flag: the design
# forbids the model ever seeing which hospital a scan came from, and without
# this the indication channel would carry it in plain text.
SITE_LITERALS = [
    "boston children's hospital", "boston children", "children's hospital boston",
    "children's hospital", "childrens hospital", "bch", "harvard medical school",
    "dana-farber", "dana farber", "brigham and women", "massachusetts general",
    "mass general", "beth israel",
]

# ---------------------------------------------------------------- L5
AGE_PHRASE = re.compile(
    r"\b(\d{1,3})[\s-]*(?:and[\s-]*a[\s-]*half[\s-]*)?"
    r"(year|yr|y/?o|yo|month|mo|week|wk|day)s?[\s-]*(?:old)?\b", re.I)
# Spelled-out ages. "six-year-old" was being redacted as an unknown hyphenated
# Titlecase token, which lost the age AND left a [NAME] in its place.
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
                "twenty": 20}
AGE_WORD_PHRASE = re.compile(
    r"\b(" + "|".join(WORD_NUMBERS) + r")[\s-]+"
    r"(year|yr|month|mo|week|wk|day)s?[\s-]*(?:old)?\b", re.I)
AGE_UNIT = {"year": 1.0, "yr": 1.0, "y/o": 1.0, "yo": 1.0,
            "month": 1 / 12, "mo": 1 / 12, "week": 1 / 52, "wk": 1 / 52, "day": 1 / 365}

# ---------------------------------------------------------------- L3/L4
# Every alphabetic token of length >= 3, in ANY casing. Matching only Titlecase
# would let "DELACROIX" and "delacroix" walk straight through the dictionary
# layer, and a name typed in caps is still a name.
WORD_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z]{2,}(?:[-'][A-Za-z]+)*\b")


def _stems(lw: str) -> tuple[str, ...]:
    """The token plus its possessive and plural forms, for vocabulary lookup."""
    out = [lw]
    for suf in ("'s", "\u2019s", "s'", "s"):
        if lw.endswith(suf) and len(lw) > len(suf) + 1:
            out.append(lw[: -len(suf)])
    if lw.endswith("es") and len(lw) > 4:
        out.append(lw[:-2])
    return tuple(out)


def _known(lw: str, *vocabularies) -> bool:
    return any(st in v for st in _stems(lw) for v in vocabularies)


def _looks_like_a_name(w: str) -> bool:
    """Should an UNKNOWN token be redacted on suspicion alone?

    Titlecase, yes -- that is what a surname looks like in prose. ALL-CAPS only
    from five characters up: clinical abbreviations cluster at two to four (CT,
    GGO, RLL, COPD, PICU), and redacting those would gut the text. A lowercase
    unknown word is left alone, because at that point the rule would be
    "redact every word not in the vocabulary", which is not a scrubber.
    """
    if w[:1].isupper() and w[1:].islower():
        return True
    return w.isupper() and len(w) >= 5

# Common English and clinical words that are legitimately capitalised at the
# start of a sentence or in an abbreviation, and would otherwise be redacted.
ALWAYS_KEEP = {
    "the", "and", "for", "with", "without", "history", "patient", "rule", "out",
    "evaluate", "evaluation", "assess", "assessment", "follow", "followup",
    "status", "post", "known", "possible", "suspected", "concern", "concerns",
    "please", "prior", "compare", "comparison", "clinical", "indication",
    "reason", "exam", "examination", "chest", "abdomen", "pelvis", "lung",
    "lungs", "left", "right", "upper", "lower", "middle", "bilateral", "new",
    "recent", "acute", "chronic", "persistent", "recurrent", "initial",
    "further", "additional", "repeat", "routine", "screening", "surveillance",
    "preoperative", "postoperative", "pre", "intra", "male", "female", "boy",
    "girl", "child", "infant", "neonate", "adolescent", "teen", "yesterday",
    "today", "week", "weeks", "month", "months", "year", "years", "day", "days",
    "hour", "hours", "since", "after", "before", "during", "presenting",
    "presents", "admitted", "transferred", "referred", "underwent", "underlying",
    "there", "this", "that", "these", "those", "was", "were", "has", "had",
    "not", "now", "also", "seen", "noted", "found", "given", "per", "due",
    "secondary", "primary", "unknown", "none", "other", "same", "than", "then",
    # Function words. The audit measured "she" redacted 132 times: a pronoun at
    # the start of a sentence is Titlecase, is not radiology vocabulary, and
    # therefore looked exactly like an unknown surname.
    "she", "her", "hers", "herself", "his", "him", "himself", "its", "our",
    "their", "them", "they", "who", "whom", "whose", "which", "what", "when",
    "where", "why", "how", "both", "each", "every", "some", "any", "all",
    "off", "into", "onto", "over", "under", "above", "below", "again",
    "completed", "complete", "ongoing", "pending", "requested", "requisition",
    "research", "protocol", "study", "studies", "scan", "scans", "imaging",
    "image", "images", "report", "reported", "note", "notes", "please",
    "consider", "consult", "consultation", "referral", "workup", "eval",
    "evaluate", "evaluated", "evaluating", "reassess", "restage", "restaging",
    "staging", "stage", "interval", "change", "changes", "stable", "unchanged",
    "improved", "improving", "worse", "worsening", "resolved", "resolving",
}

# Diseases and syndromes named after people. Redacting these would remove the
# clinically load-bearing half of a pediatric indication, and L1 has already
# taken care of the case where a patient's own surname is one of them.
DEFAULT_EPONYMS = {
    "down", "hodgkin", "ewing", "wilms", "kartagener", "marfan", "digeorge",
    "hirschsprung", "crohn", "burkitt", "langerhans", "kawasaki", "noonan",
    "turner", "klinefelter", "prader", "willi", "angelman", "duchenne",
    "becker", "pompe", "gaucher", "niemann", "pick", "hurler", "morquio",
    "ehlers", "danlos", "osler", "weber", "rendu", "sturge", "vater",
    "wiskott", "aldrich", "chediak", "higashi", "bruton", "swyer", "james",
    "macleod", "pierre", "robin", "treacher", "collins", "goldenhar",
    "poland", "jeune", "scimitar", "bochdalek", "morgagni", "meckel",
    "ladd", "nissen", "fontan", "glenn", "blalock", "taussig", "norwood",
    "ross", "ebstein", "tetralogy", "fallot", "eisenmenger", "takayasu",
    "wegener", "churg", "strauss", "goodpasture", "sjogren", "raynaud",
    "hodgkins", "nonhodgkin", "epstein", "barr", "pneumocystis", "aspergillus",
    "candida", "nocardia", "mycobacterium", "staphylococcus", "streptococcus",
    "klebsiella", "pseudomonas", "haemophilus", "bordetella", "legionella",
    "chlamydia", "mycoplasma", "cryptococcus", "histoplasma", "coccidioides",
    "blastomyces", "toxoplasma", "cytomegalovirus", "varicella", "influenza",
    "covid", "sars", "rsv",
}


# What the fail-closed layer decided to redact, accumulated across a run. Used
# only by the over-redaction audit: a word that is being redacted hundreds of
# times is a vocabulary gap, not a name.
REDACTED: Counter = Counter()
REDACTED_L3: Counter = Counter()


class ScrubResult(NamedTuple):
    text: str
    hits: Counter
    truncated: bool


@dataclass(frozen=True)
class RowPHI:
    """Everything this particular row is allowed to look for by name."""

    first: str = ""
    last: str = ""
    mrn: str = ""
    accession: str = ""
    dates: tuple[str, ...] = ()


class NamePool(NamedTuple):
    patient: frozenset      # lowercase surname/first-name tokens
    provider: frozenset


def _tokens(value: str) -> Iterable[str]:
    for t in re.split(r"[^A-Za-z'\-]+", str(value or "")):
        t = t.strip("'-")
        if len(t) >= 3:
            yield t.lower()


def build_name_pool(rows: Iterable[dict],
                    patient_cols=("Patient First Name", "Patient Last Name"),
                    provider_cols=("Ordered By", "Scheduled By", "Exam Started By",
                                   "Exam Completed By", "Report Created By",
                                   "Preliminary Report By", "Report Finalized By",
                                   "Report Addendum By")) -> NamePool:
    pat, prov = set(), set()
    for row in rows:
        for c in patient_cols:
            pat.update(_tokens(row.get(c, "")))
        for c in provider_cols:
            prov.update(_tokens(row.get(c, "")))
    return NamePool(frozenset(pat), frozenset(prov))


def build_clinical_vocab(texts: Iterable[str], min_count: int = 5) -> frozenset:
    """Words that occur often in PUBLIC report text, so they are not names."""
    c = Counter()
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z\-']{2,}", str(t or "")):
            c[w.lower()] += 1
    return frozenset(w for w, n in c.items() if n >= min_count)


def load_wordlist(path: str) -> frozenset:
    if not path or not os.path.isfile(path):
        return frozenset()
    out = set()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.split("#", 1)[0].strip().lower()
            if ln:
                out.add(ln)
    return frozenset(out)


def safe_eponyms(eponyms: Iterable[str], pool: NamePool,
                 verbose: bool = True) -> frozenset:
    """Drop every eponym that is also a real surname IN THIS COHORT.

    Curating an eponym list by intuition does not work: Miller-Fisher, Brown-
    Sequard, Dubin-Johnson and Still disease are all real, and Miller, Fisher,
    Brown, Johnson and Still are all ordinary surnames. Rather than guess which
    ones are safe, intersect the list with the cohort's own name pool and remove
    the overlap. What survives is, by construction, not a patient or provider
    name in this dataset -- so an entry can never turn the whitelist into a leak.

    The row-targeted layer already covers the case where the eponym is THIS
    patient's surname; this covers the case where it is some other patient's.
    """
    names = pool.patient | pool.provider
    keep = {e for e in eponyms if e not in names}
    dropped = sorted(set(eponyms) - keep)
    if verbose and dropped:
        print(f"[eponyms] {len(dropped)} dropped for colliding with a real name "
              f"in this cohort: {dropped[:10]}{' ...' if len(dropped) > 10 else ''}")
    return frozenset(keep)


def normalise(text: str) -> str:
    t = _CTRL.sub("\n", str(text or ""))
    t = _WS.sub(" ", t)
    t = _NL.sub("\n", t)
    return t.strip()


def _flatten(t: str) -> str:
    """One indication, one line.

    normalise() keeps single newlines, which is right for a report blob and
    wrong for this field: a multi-line indication becomes several lines in the
    review sample, and a reviewer scanning for a surviving name reads the
    continuation lines as separate un-tagged records. 200 of 20,832 rows were
    affected. Collapsing here rather than in normalise() leaves the report path
    untouched.
    """
    return re.sub(r"\s+", " ", t).strip()


VACUOUS = re.compile(r"^\s*(not\s+given|none|n/?a|nil|unknown|-{1,3}|\.)\s*\.?\s*$", re.I)


def classify(text: str, min_words: int = 3) -> str:
    """present | vacuous | absent -- one definition, used by every consumer."""
    t = (text or "").strip()
    if not t:
        return "absent"
    if VACUOUS.match(t):
        return "vacuous"
    return "present" if len(t.split()) >= min_words else "vacuous"


def _age_replacement(n: float, unit: str) -> str:
    from_years = n * AGE_UNIT.get(unit.lower(), 1.0)
    if from_years > 89:
        return "[AGE 90+]"
    edges = [(1, "<1y"), (2, "1-2y"), (5, "2-5y"), (10, "5-10y"), (13, "10-13y"),
             (18, "13-18y"), (25, "18-25y"), (40, "25-40y"), (60, "40-60y")]
    for hi, name in edges:
        if from_years < hi:
            return f"[AGE {name}]"
    return "[AGE 60+y]"


def scrub(text: str, row: RowPHI, pool: NamePool, vocab: frozenset,
          eponyms: frozenset = frozenset(DEFAULT_EPONYMS),
          max_words: int = 64, max_chars: int = 512) -> ScrubResult:
    hits: Counter = Counter()
    t = normalise(text)

    def sub(pattern, repl, key, s):
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        s2, n = pattern.subn(repl, s)
        if n:
            hits[key] += n
        return s2

    # -- L1 row-targeted --------------------------------------------------
    for value in (row.first, row.last):
        for tok in _tokens(value):
            t = sub(re.compile(rf"\b{re.escape(tok)}(?:'s)?\b", re.I),
                    "[NAME]", "L1_name", t)
    for value, tag, key in ((row.mrn, "[ID]", "L1_mrn"),
                            (row.accession, "[ID]", "L1_acc")):
        v = str(value or "").strip()
        if len(v) >= 4:
            t = sub(re.compile(rf"\b{re.escape(v)}\b", re.I), tag, key, t)
    for d in row.dates:
        for rendering in _date_renderings(d):
            t = sub(re.compile(re.escape(rendering), re.I), "[DATE]", "L1_date", t)

    # -- L5 age BEFORE the structural digit rules, or [ID] eats the number -
    def _age_sub(m):
        # Padded: the pattern eats the trailing separator of "2-month-old with",
        # which would otherwise produce "[AGE <1y]with". _flatten() collapses the
        # extra spaces at the end.
        try:
            return " " + _age_replacement(float(m.group(1)), m.group(2)) + " "
        except (TypeError, ValueError):
            return " [AGE] "
    t, n_age2 = AGE_WORD_PHRASE.subn(
        lambda m: " " + _age_replacement(WORD_NUMBERS[m.group(1).lower()],
                                         m.group(2)) + " ", t)
    if n_age2:
        hits["L5_age"] += n_age2
    t, n_age = AGE_PHRASE.subn(_age_sub, t)
    if n_age:
        hits["L5_age"] += n_age

    # -- L2 structural ----------------------------------------------------
    for pat, repl in STRUCTURAL:
        t = sub(pat, repl, f"L2_{repl.strip('[]')}", t)
    low = t.lower()
    for lit in SITE_LITERALS:
        if lit in low:
            t = sub(re.compile(re.escape(lit), re.I), "[SITE]", "L2_SITE", t)
            low = t.lower()

    # -- L3 dictionary + L4 fail-closed -----------------------------------
    def _cap(m):
        w = m.group(0)
        lw = w.lower()
        # ORDINARY CLINICAL LANGUAGE WINS OVER THE NAME POOL, and the ordering
        # matters more than anything else in this function.
        #
        # The pool holds every surname in the cohort, and a surprising number of
        # them are ordinary words: the audit measured "prior" redacted 866
        # times, "medical" 185, "not" 150, "iii" 60 (stage III), "ray" 67
        # (chest X-ray), "wall" 66 (chest wall), "lam" 12 (the disease). With
        # the pool checked first, "Prior abnormal chest CT" became "[NAME]
        # abnormal chest CT" and "stage III" became "stage [NAME]".
        #
        # Vocabulary membership is a frequency threshold -- 5+ uses in public
        # CT-RATE findings or 20+ in the de-identified pediatric findings -- so
        # a hit here means the token is real clinical language, not a name.
        #
        # The residual this accepts: ANOTHER patient's or provider's surname
        # that is also common clinical vocabulary, appearing in this patient's
        # indication. Two things still cover it -- L1 already removed THIS
        # patient's own name unconditionally, and L2's "Dr. X" rule removes the
        # form a provider is actually named in. Trading those 1,076 measured
        # false redactions for that residual is the right direction.
        #
        # "Ewing's" and "Ewings" reach the eponym list through _stems().
        if _known(lw, ALWAYS_KEEP, eponyms, vocab):
            return w
        if lw in pool.patient or lw in pool.provider:
            hits["L3_dict"] += 1
            REDACTED_L3[lw] += 1       # audit only
            return "[NAME]"
        if _looks_like_a_name(w):
            hits["L4_unknown"] += 1
            REDACTED[lw] += 1          # audit only; see tools/audit_redactions.py
            return "[NAME]"
        return w

    t = WORD_TOKEN.sub(_cap, t)

    # -- L6 cap -----------------------------------------------------------
    truncated = False
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
        truncated = True
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0]
        truncated = True
    if truncated:
        hits["L6_truncated"] += 1
    # collapse runs of identical placeholders left behind by overlapping rules
    t = re.sub(r"(\[(?:NAME|DATE|ID|SITE|CONTACT|LOC)\])(?:[\s,]*\1)+", r"\1", t)
    return ScrubResult(_flatten(t), hits, truncated)


def scrub_structural(text: str, max_words: int = 64,
                     max_chars: int = 512) -> ScrubResult:
    """L0 + L2 + L5 + L6 only -- for text that is already de-identified upstream.

    CT-RATE is public and carries no BCH patient, so the row-targeted and
    dictionary layers have nothing to match and the fail-closed layer would just
    shred ordinary prose. The structural rules still run: "already
    de-identified" is an assumption, it costs nothing to check, and a hit there
    is a finding worth reporting rather than one worth silencing.
    """
    hits: Counter = Counter()
    t = normalise(text)

    def _age_sub(m):
        try:
            return " " + _age_replacement(float(m.group(1)), m.group(2)) + " "
        except (TypeError, ValueError):
            return " [AGE] "
    t, n_age2 = AGE_WORD_PHRASE.subn(
        lambda m: " " + _age_replacement(WORD_NUMBERS[m.group(1).lower()],
                                         m.group(2)) + " ", t)
    if n_age2:
        hits["L5_age"] += n_age2
    t, n_age = AGE_PHRASE.subn(_age_sub, t)
    if n_age:
        hits["L5_age"] += n_age

    for pat, repl in STRUCTURAL:
        rx = re.compile(pat) if isinstance(pat, str) else pat
        t, n = rx.subn(repl, t)
        if n:
            hits[f"L2_{repl.strip('[]')}"] += n
    low = t.lower()
    for lit in SITE_LITERALS:
        if lit in low:
            t, n = re.compile(re.escape(lit), re.I).subn("[SITE]", t)
            if n:
                hits["L2_SITE"] += n
            low = t.lower()

    truncated = False
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
        truncated = True
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0]
        truncated = True
    if truncated:
        hits["L6_truncated"] += 1
    t = re.sub(r"(\[(?:NAME|DATE|ID|SITE|CONTACT|LOC)\])(?:[\s,]*\1)+", r"\1", t)
    return ScrubResult(_flatten(t), hits, truncated)


def _date_renderings(raw: str) -> list[str]:
    """Every way one spreadsheet timestamp might appear inside prose."""
    s = str(raw or "").strip()
    if not s:
        return []
    out = {s}
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if not m:
        m2 = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
        if m2:
            mo, d, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        else:
            return [s.split(" ")[0]] if " " in s else [s]
    else:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    names = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
    nm = names[mo - 1] if 1 <= mo <= 12 else ""
    out.update({
        f"{mo}/{d}/{y}", f"{mo:02d}/{d:02d}/{y}", f"{mo}-{d}-{y}",
        f"{y}-{mo:02d}-{d:02d}", f"{d}/{mo}/{y}",
        f"{nm} {d}, {y}", f"{nm} {d} {y}", f"{nm[:3]} {d}, {y}",
    })
    return [x for x in out if len(x) >= 6]

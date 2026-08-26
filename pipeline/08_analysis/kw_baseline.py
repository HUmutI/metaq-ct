#!/usr/bin/env python3
"""Keyword/regex baseline for the 18 CT-RATE labels on pediatric reports.

Single source of truth. `validate.py`'s inline LABEL_KEYWORDS was substring
matching with no word boundaries (`if kw in s.lower()`), so `stent` matched
*consistent* 2644 times, `line` matched *linear* 2212 times and `port` matched
*portions* 1720 times. That inflated Medical material to 74.8%. The lexicon here
is ported from pipeline/lib/ctrate_map/ctrate_map.py, which is already \\b-anchored and
worded for pediatric reports, and gives 31.5% on the same corpus.

Two design choices that make the comparison against the LLM fair rather than
rigged:

1. **Evidence in the same schema as qwen_extract.py** — every positive carries
   the sentence index that justified it plus the matched surface form. Without
   this only the LLM is auditable and the citation audit can only be run on one
   side.
2. **Every policy is a toggle**, so the 74.8 -> 31.5 swing can be attributed to
   named knobs (word boundaries, negation, history suppression, section scope)
   instead of being asserted.

Aggregates and matched surface forms are safe to print; sentences are not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# label set (must stay identical to arcct/dataset.py PATHOLOGIES)
# --------------------------------------------------------------------------- #
PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

# Ported verbatim from pipeline/lib/ctrate_map/ctrate_map.py CTRATE. Kept as source
# strings so the ablation can re-compile them with boundaries stripped.
LEXICON_SRC: dict[str, str] = {
    "Medical material":
        r"port-?a-?cath|portacath|central (?:venous )?(?:line|catheter)|\bpicc\b|"
        r"tracheostomy|chest tube|thoracostomy|pacemaker|\bstent\b|surgical clip|"
        r"sternal wire|\bcoil\b|g-?tube|feeding tube|endotracheal tube|\bicd\b|"
        r"catheter|prosthe|\bmesh\b|drain\b",
    "Arterial wall calcification":
        r"(?:aortic|arterial|aorta)[^.\n]{0,40}calcifi|"
        r"calcifi[^.\n]{0,40}(?:aorta|aortic|artery|arterial)",
    "Cardiomegaly":
        r"cardiomegaly|enlarged (?:cardiac silhouette|heart)|"
        r"heart size is (?:mildly |moderately |markedly )?(?:enlarged|increased)|"
        r"cardiac (?:enlargement|silhouette is enlarged)|ventricular (?:enlargement|dilat)|"
        r"mild cardiac enlargement",
    "Pericardial effusion": r"pericardial (?:effusion|fluid)",
    "Coronary artery wall calcification":
        r"coronary[^.\n]{0,30}calcifi|calcifi[^.\n]{0,30}coronary",
    "Hiatal hernia": r"hiatal hernia|hiatus hernia",
    "Lymphadenopathy":
        r"lymphadenopathy|adenopathy|enlarged (?:mediastinal |hilar |axillary )?(?:lymph )?nodes?|"
        r"prominent (?:lymph )?nodes?|borderline (?:enlarged )?(?:lymph )?node|"
        r"pathologic(?:ally enlarged)? node",
    "Emphysema": r"emphysema|bulla\b|bullae|bullous|pneumatocele",
    "Atelectasis": r"atelecta|(?:lobar|segmental|subsegmental) collapse|collapsed (?:lobe|lung)",
    "Lung nodule": r"\bnodule|nodular (?:opacit|densit|focus|foci)|micronodul",
    "Lung opacity":
        r"opacit|ground[- ]?glass|\bGGO\b|infiltrat|airspace disease|"
        r"density in the (?:right|left) (?:upper|middle|lower) lobe",
    "Pulmonary fibrotic sequela":
        r"fibrosi|fibrotic|\bscar(?:ring)?\b|architectural distortion|traction bronchiect|"
        r"honeycomb|reticulation|post[- ]?(?:inflammatory|infectious) (?:change|scar)",
    "Peribronchial thickening":
        r"peribronchial thickening|bronchial wall thickening|airway wall thickening|"
        r"bronchial thickening|peribronchovascular thickening",
    "Consolidation": r"consolidat",
    "Bronchiectasis": r"bronchiecta",
    "Interlobular septal thickening":
        r"(?:interlobular )?septal thickening|interstitial thickening|smooth interlobular|"
        r"crazy[- ]paving",
    "Mosaic attenuation pattern":
        r"mosaic (?:attenuation|perfusion|pattern)|air trapping|"
        r"heterogeneous (?:lung )?attenuation",
    "Pleural effusion":
        r"pleural effusion|pleural fluid|hydrothorax|hemothorax|"
        r"(?:small|moderate|large) effusion",
}

# Split ctrate_map's single NEG into two, so the ablation can separate "the
# finding is absent" from "the finding is only mentioned as history or as the
# reason for the scan". Those are different label definitions, and the LLM
# prompt implements only the first.
NEG_SRC = (r"\b(no|not|nor|neither|without|negative for|free of|absence of|absent|"
           r"resolved|resolution of|rather than|unlikely|denies)\b")
HISTORY_SRC = (r"\b(rule out|r/o|evaluate for|assess for|screen for|surveillance for|"
               r"history of|h/o|prior|previously|status post|s/p)\b")
# Negation that follows the finding. ctrate_map only looks backwards, which is
# why boilerplate like "... lines and tubes are not identified" scored positive.
POST_NEG_SRC = (r"^\s*(?:\w+\s+){0,4}?(?:is|are|was|were|has|have)\s+(?:not\s+|no\s+)"
                r"(?:been\s+)?(?:seen|identified|present|visualized|noted|demonstrated)|"
                r"^\s*(?:\w+\s+){0,4}?(?:has|have)\s+resolved|"
                r"^\s*(?:\w+\s+){0,3}?(?:removed|absent)\b")
CLAUSE_SPLIT = r"[.;:*\n]|\bbut\b|\bhowever\b|\balthough\b|\bexcept\b"


@dataclass(frozen=True)
class Policy:
    """Every knob the ablation grid toggles."""
    word_boundaries: bool = True
    negation: bool = True
    post_negation: bool = True
    history_as_negation: bool = True
    scope: str = "all"          # "all" | "findings" | "impression"
    window: int = 60

    def tag(self) -> str:
        return "wb%d_neg%d_post%d_hist%d_%s" % (
            self.word_boundaries, self.negation, self.post_negation,
            self.history_as_negation, self.scope)


DEFAULT = Policy()


def _strip_boundaries(pattern: str) -> str:
    """Reproduce the broken behaviour on purpose, for the ablation only."""
    return pattern.replace(r"\b", "")


_CACHE: dict[tuple[str, bool], re.Pattern] = {}


def compiled(label: str, word_boundaries: bool = True) -> re.Pattern:
    key = (label, word_boundaries)
    if key not in _CACHE:
        src = LEXICON_SRC[label]
        if not word_boundaries:
            src = _strip_boundaries(src)
        _CACHE[key] = re.compile(src, re.I)
    return _CACHE[key]


NEG_RE = re.compile(NEG_SRC, re.I)
HISTORY_RE = re.compile(HISTORY_SRC, re.I)
POST_NEG_RE = re.compile(POST_NEG_SRC, re.I)
CLAUSE_RE = re.compile(CLAUSE_SPLIT, re.I)


def _suppressed(sentence: str, start: int, end: int, pol: Policy) -> str | None:
    """Return the suppression reason for a hit, or None if it stands."""
    if pol.negation or pol.history_as_negation:
        before = sentence[max(0, start - pol.window):start]
        clause = CLAUSE_RE.split(before)[-1]
        if pol.negation and NEG_RE.search(clause):
            return "negated"
        if pol.history_as_negation and HISTORY_RE.search(clause):
            return "historical"
    if pol.post_negation:
        after = sentence[end:end + pol.window]
        clause = CLAUSE_RE.split(after)[0]
        if POST_NEG_RE.search(clause):
            return "post_negated"
    return None


def score_report(sentences: list[str], n_findings: int = 0,
                 pol: Policy = DEFAULT) -> dict[str, dict]:
    """Label a report.

    Returns {label: {"p": 0|1, "e": sentence_index_or_-1, "form": matched_text}},
    deliberately the same shape qwen_extract.py writes, so both methods can go
    through the identical citation audit.
    """
    if pol.scope == "findings":
        idxs = range(0, n_findings or len(sentences))
    elif pol.scope == "impression":
        idxs = range(n_findings, len(sentences))
    else:
        idxs = range(len(sentences))

    out: dict[str, dict] = {}
    for label in PATHOLOGIES:
        rx = compiled(label, pol.word_boundaries)
        hit = {"p": 0, "e": -1, "form": ""}
        for i in idxs:
            s = sentences[i]
            for m in rx.finditer(s):
                if _suppressed(s, m.start(), m.end(), pol) is None:
                    hit = {"p": 1, "e": i, "form": m.group(0).lower()}
                    break
            if hit["p"]:
                break
        out[label] = hit
    return out


def matched_forms(sentences: list[str], pol: Policy = DEFAULT) -> dict[str, list[str]]:
    """Every surface form a label's regex fires on, suppression ignored.

    This is the audit that makes a broken lexicon visible in one glance: if
    `consistent` shows up under Medical material, the pattern is wrong. Single
    tokens only - safe to print, unlike the sentences they came from.
    """
    out: dict[str, list[str]] = {l: [] for l in PATHOLOGIES}
    for label in PATHOLOGIES:
        rx = compiled(label, pol.word_boundaries)
        for s in sentences:
            out[label].extend(m.group(0).lower() for m in rx.finditer(s))
    return out

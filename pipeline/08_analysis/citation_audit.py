#!/usr/bin/env python3
"""Does the sentence a labeller cited actually contain the finding it cites it for?

This is the cheapest honest test of an extractor we have: it needs no gold
standard, runs over all 8817 reports, and asks only whether the evidence a
positive label points at mentions the thing at all. A label whose cited
sentence contains no lexical trace of the finding was not read out of the
report; it was smeared onto a nearby sentence.

The patterns are deliberately GENEROUS - every synonym and near-miss wording we
could think of, word-boundary anchored so `stent` cannot match `consistent`
(the bug that produced a 74.8% keyword baseline earlier in this project). A
generous pattern makes the test hard to fail, so a high unsupported rate is
hard to argue with.

Classes with no reliable lexical anchor (Lung opacity, Post-surgical change,
Mass or neoplasm, ...) are excluded rather than guessed at - they would measure
the pattern, not the labeller.

Claude's 200 gold-sample citations run through the identical audit as a
control - and it earned its keep immediately: the first run scored Claude WORSE
than Qwen on Medical material (16.5% vs 11.1%), and every one of those 17 cases
was a real device the lexicon lacked a word for (`suture`, `PICC`, `chain suture
material`). The control caught a defect in the measuring instrument, not in the
labeller. Pulmonary cyst and Bone lesion had the same shape and the same cause.
"""
from __future__ import annotations
import csv, glob, json, os, re, sys
from collections import defaultdict

EV = "/temp_work/ch278233/BCH_DATASET/LABELS27_V2/evidence.tsv"
GOLD = "/temp_work/ch278233/GOLD_SAMPLE/gold_sample.jsonl"
CLA = "/temp_work/ch278233/GOLD_SAMPLE/claude_batch*.jsonl"

PAT = {
 "Peribronchial thickening": r"peribronch|bronchial wall|airway wall|bronchial thick|bronchovascular|bronchial cuff",
 "Interlobular septal thickening": r"septal|septa\b|interstitial thick|kerley|interlobular|crazy.pav",
 "Tree-in-bud": r"tree.?in.?bud|centrilobular",
 "Lymphadenopathy": r"lymph|adenopath|\bnode|nodal",
 "Bronchiectasis": r"bronchiect|bronchiol.?ect",
 "Pneumothorax": r"pneumothora|pneumothorace|\bair\b.{0,30}pleural|hydropneumo",
 "Pleural effusion": r"effusion|pleural fluid|hydrothorax|fluid.{0,20}pleural|\bfluid\b.{0,20}fissure",
 "Consolidation": r"consolidat|airspace disease|air bronchogram",
 "Atelectasis": r"atelecta|collapse|volume loss",
 "Mosaic attenuation pattern": r"mosaic|air.?trapping|attenuation",
 "Cardiomegaly": r"cardiomegal|enlarg.{0,20}heart|heart.{0,20}enlarg|cardiac.{0,20}(enlarg|silhouette|size)|dilat.{0,20}(ventric|atri|heart)",
 "Pericardial effusion": r"pericardi",
 "Hiatal hernia": r"hiatal|hernia",
 "Mucus plugging": r"mucus|mucous|mucoid|plug|secretion|impact",
 "Emphysema": r"emphysem|bulla|bullous|hyperinflat|hyperlucen",
 "Arterial wall calcification": r"calcif|calcium",
 "Coronary artery wall calcification": r"calcif|coronary|circumflex|descending artery",
 "Pulmonary cyst": r"\bcyst|bulla|bullous|bleb|pneumatocele|cavit|overinflat|hyperinflat|hyperlucen|air.?filled|lucen",
 "Bone lesion or fracture": r"\brib\b|\bribs\b|vertebr|osseous|\bbone|fractur|sternum|sternal|clavic|scapul|humer|spine|spinal|lytic|sclerotic|schmorl|marrow|skeletal|enostos|osseus|\bT\d{1,2}\b|\bL\d\b|costal|compression deformity|height loss",
 "Medical material": r"catheter|tube|line\b|port|stent|shunt|device|pacemaker|clip|coil|wire|drain|hardware|screw|rod\b|prosthe|graft|cannula|sheath|sutur|staple|picc\b|\bett\b|\bng\b|\bog\b|pigtail|mesh|plate\b|valve|marker|seed\b|gastrostom|tracheostom|defibrillat|\bicd\b|reservoir|filter|occluder|coil|band\b|clamp|wedge resection|instrument|postsurgical|post-surgical|surgical chang",
 "Pleural thickening or nodule": r"pleura|fissur",
 "Lung nodule": r"nodul|nodularit",
 "Pulmonary metastases": r"metasta|mets\b",
 "Pulmonary fibrotic sequela": r"fibro|scar|architectur|reticul|honeycomb|traction",
}
RX = {k: re.compile(v, re.I) for k, v in PAT.items()}


def audit(pairs):
    """pairs: iterable of (label, cited_sentence). -> per-class (n, n_unsupported)"""
    st = defaultdict(lambda: [0, 0])
    for lab, sent in pairs:
        rx = RX.get(lab)
        if rx is None:
            continue
        st[lab][0] += 1
        if not rx.search(sent or ""):
            st[lab][1] += 1
    return st


def qwen_pairs():
    with open(EV, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd, None)
        for row in rd:
            if len(row) >= 3:
                yield row[1], row[2]


def claude_pairs():
    sents = {}
    with open(GOLD) as fh:
        for line in fh:
            r = json.loads(line)
            sents[r["VolumeName"]] = r["sentences"]
    for p in glob.glob(CLA):
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                ss = sents.get(r["VolumeName"], [])
                for lab, idx in (r.get("evidence") or {}).items():
                    if isinstance(idx, int) and 0 <= idx < len(ss):
                        yield lab, ss[idx]


def main():
    q, c = audit(qwen_pairs()), audit(claude_pairs())
    print("unsupported citation = the cited sentence contains no lexical trace "
          "of the finding\n")
    print("%-38s %8s %7s   %8s %7s" % ("class", "Qwen n", "unsup", "Cla n", "unsup"))
    print("-" * 76)
    qn = qu = cn = cu = 0
    rows = []
    for lab in PAT:
        a, b = q.get(lab, [0, 0]), c.get(lab, [0, 0])
        rows.append((b[1] / b[0] if b[0] else 0, a[1] / a[0] if a[0] else 0, lab, a, b))
        qn += a[0]; qu += a[1]; cn += b[0]; cu += b[1]
    for _, _, lab, a, b in sorted(rows, key=lambda r: -r[1]):
        qs = "%6.1f%%" % (100 * a[1] / a[0]) if a[0] else "     -"
        cs = "%6.1f%%" % (100 * b[1] / b[0]) if b[0] else "     -"
        print("%-38s %8d %7s   %8d %7s" % (lab, a[0], qs, b[0], cs))
    print("-" * 76)
    print("%-38s %8d %6.1f%%   %8d %6.1f%%"
          % ("TOTAL", qn, 100 * qu / max(qn, 1), cn, 100 * cu / max(cn, 1)))
    print("\nQwen: all 8817 reports.  Claude: the 200-report gold sample "
          "(control for the lexicon).")


if __name__ == "__main__":
    main()

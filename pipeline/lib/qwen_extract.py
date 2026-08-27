#!/usr/bin/env python3
"""Local-LLM extraction of 27 pathology labels + 10 anatomical regions.

Runs Qwen through vLLM **on this cluster**. No report text leaves the machine -
that is the point: arc-ct's tools/extract_labels.py defaults to an external
endpoint (``https://ark.cn-beijing.volces.com/api/v3``), which pediatric PHI
must never reach.

Two things make this more reliable than the scripts it replaces:

1. **Structured decoding.** The output grammar is pinned to a JSON schema, so a
   schema-valid object is the only thing the model can emit. The originals
   free-generated JSON, regex-scraped it, and on failure silently wrote an
   all-zero label row / an all-empty region set - indistinguishable from a
   genuinely normal study and never retried on resume. Here a failure is
   recorded as ``status != "ok"`` and is retried.

2. **Indices, not text.** The model receives numbered sentences and returns
   sentence *indices*. Region text is reconstructed from the original report, so
   quotes are verbatim by construction, output is short, and any index outside
   range is a detectable error rather than a plausible-looking paraphrase.

Outputs one JSONL per task, resumable, keyed by de-identified VolumeName.

  RAC_SHARD_ID / RAC_SHARD_N   shard across an sbatch array (one GPU each)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PATHOLOGIES = [
    # --- the 18 CT-RATE classes, all kept ---
    # Dropping the four that are near-absent in children (coronary and arterial
    # calcification, hiatal hernia, emphysema) was the earlier plan, but the
    # model has to keep working on adults, where those are among its strongest
    # classes. A head that is deleted cannot be forgotten - it is simply gone.
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
    # --- 9 pediatric additions, by measured prevalence ---
    "Post-surgical or post-treatment change",
    "Pulmonary metastases",
    "Tree-in-bud",
    "Pulmonary cyst",
    "Mass or neoplasm",
    "Mucus plugging",
    "Pleural thickening or nodule",
    "Bone lesion or fracture",
    "Pneumothorax",
]

# Same 10 groups the AnatomyQFormer routes over, described in report language.
REGIONS = {
    1: "left upper lobe (including lingula)",
    2: "left lower lobe",
    3: "right upper lobe",
    4: "right middle lobe",
    5: "right lower lobe",
    6: "trachea, main bronchi and esophagus",
    7: "heart, pericardium, aorta and mediastinal vessels",
    8: "pleura and pleural space",
    9: "chest wall, ribs, spine and soft tissues",
    10: "upper abdomen (liver, spleen, kidneys, stomach)",
}

LABEL_DEFS = {
    "Medical material": "tubes, catheters, lines, stents, surgical clips, hardware, pacemaker",
    # Measured failure, not a hypothetical. Under the old wording - "calcification
    # in a systemic artery wall (not coronary)" - this label fired on 59 of 1773
    # pediatric validation studies (3.3%), and the cited sentences were mostly
    # not this finding at all: calcified thrombus in the brachiocephalic VEIN,
    # the tract of a Port-A-Cath, pulmonary artery calcification after congenital
    # heart surgery, tracheal and bronchial CARTILAGE calcification, and normal
    # variant ligamentum arteriosum. Adult prevalence is 28.4% and means aortic
    # atherosclerosis; the model scored 0.9365 on it there and 0.6381 here. It
    # had not forgotten - it was being graded against a different finding.
    # So name the vessels, and name the confounders out loud: the old wording was
    # already technically correct and that was not enough.
    "Arterial wall calcification": (
        "atherosclerotic calcification in the wall of the aorta or a systemic "
        "arterial branch (brachiocephalic, subclavian, carotid). "
        "NOT coronary arteries. "
        "NOT veins of any kind, including calcified venous thrombus or "
        "calcification along a central catheter or Port-A-Cath tract. "
        "NOT pulmonary arteries or pulmonary veins. "
        "NOT cardiac valve, annulus or pericardial calcification. "
        "NOT tracheal or bronchial cartilage calcification. "
        "NOT ligamentum arteriosum or ductus arteriosus calcification, which is "
        "a normal variant. "
        "NOT calcification at a surgical cannulation, anastomosis or graft site. "
        "Calcification in an abdominal systemic branch (hepatic, splenic, renal, "
        "mesenteric) DOES count"
    ),
    # 17.1% unanchored on 240 positives. The literal reading is deliberate even
    # though it costs a clinically obvious case: an 8.5 cm left atrium with the
    # heart never called enlarged scores 0 here. CT-RATE's adult labels come from
    # RadBERT reading the word, and the whole point of one prompt across both
    # cohorts is that this class means the same thing on both sides.
    "Cardiomegaly": (
        "the report calls the heart or cardiac silhouette enlarged, or says "
        "cardiomegaly. "
        "NOT dilatation of a single chamber (left atrium, right ventricle) "
        "unless the heart overall is also called enlarged. "
        "NOT main pulmonary artery or aortic dilatation. "
        "NOT pericardial effusion"),
    "Pericardial effusion": "fluid in the pericardial space",
    "Coronary artery wall calcification": "calcification specifically in coronary arteries",
    "Hiatal hernia": "stomach herniating through the diaphragmatic hiatus",
    # Second correction to this class, and the first one failing is the lesson.
    # Removing "prominent" was right but not sufficient: on all 8817 reports,
    # 36.2% of the 693 positives still cite a sentence that calls the nodes
    # sub-threshold or benign (22.5% "subcentimeter"/"reactive"/"fatty hila",
    # a further 13.7% whose only measurement is under 10 mm). Against the gold
    # standard the error is perfectly one-sided - 17 false positives, 0 misses,
    # kappa 0.676. A prose carve-out ("not pathologic") is evidently something
    # the model can read past; a number is not. So state the threshold.
    "Lymphadenopathy": (
        "lymph nodes the report calls enlarged, pathologic, or abnormal by size "
        "criteria, or that measure 10 mm or more in short axis. "
        "NOT nodes described as subcentimeter, sub-centimeter, or under 10 mm. "
        "NOT nodes called prominent, scattered, nonspecific, reactive, normal in "
        "size, or benign in morphology (fatty hila). "
        "An increase in the size or number of nodes that remain subcentimeter is "
        "NOT lymphadenopathy"),
    # Bullae and pneumatoceles moved to Pulmonary cyst: in this cohort they are
    # cystic fibrosis and post-infectious change, not smoking-related alveolar
    # destruction. Leaving them here made a CF scan read as emphysema.
    "Emphysema": "emphysema from alveolar destruction (adult, smoking-related)",
    "Atelectasis": "atelectasis, collapse, volume loss",
    "Lung nodule": "nodule, micronodule, nodularity, granuloma",
    # "not better named elsewhere" asked the reader to do a comparison it was
    # never given the terms for, and all five gold-standard annotators named this
    # the hardest class. The operational rule below is the one they converged on
    # independently: score the head noun, not the modifier.
    "Lung opacity": (
        "ground-glass, airspace or parenchymal opacity where opacity, "
        "opacification or ground-glass is the finding itself. "
        "NOT when the word only modifies another finding: score "
        "\"consolidative opacity\" as consolidation and \"nodular opacity\" as a "
        "nodule. NOT ground-glass that is only a halo around a nodule or "
        "cavity"),
    # Claude called this 14 times where Qwen did not and missed only 2, with 0%
    # unanchored citations against Qwen's 6.5% - the gap is the surgical-scarring
    # overlap, so say that both can be true at once rather than making the reader
    # choose.
    "Pulmonary fibrotic sequela": (
        "fibrosis, scarring, reticulation, architectural distortion or "
        "honeycombing. Scarring counts whatever its cause, so scarring "
        "attributed to prior surgery scores BOTH this and post-surgical "
        "change"),
    "Pleural effusion": "fluid in the pleural space",
    "Mosaic attenuation pattern": "mosaic attenuation, air trapping, mosaic perfusion",
    # Worst citation support of all 27 classes: 20.3% of 1327 positives cite a
    # sentence with no bronchial wall in it at all. The mechanism is visible in
    # evidence.tsv - ped_03305_13 cites "Similar thickening of the left major
    # fissure" for this label. One unqualified word, "thickening", is enough for
    # the model to attach the class to the nearest sentence containing it, so
    # the fix is to name the other thickenings and rule them out.
    "Peribronchial thickening": (
        "thickening of the bronchial or peribronchial wall, bronchial cuffing, "
        "bronchial wall thickening. The airway wall itself must be thickened. "
        "NOT fissural thickening. NOT pleural thickening. NOT interlobular "
        "septal or interstitial thickening. NOT chest-wall or soft-tissue "
        "thickening. NOT bronchiectasis on its own"),
    "Consolidation": "consolidation, dense airspace disease, pneumonia",
    "Bronchiectasis": "bronchiectasis, bronchial dilatation",
    # Same failure, same cause: 15.7% unanchored, and ped_03305_13 cites "New
    # subtle tree in bud groundglass nodularity" for it - the tree-in-bud
    # sentence, reused verbatim for a second class.
    "Interlobular septal thickening": (
        "thickening of the interlobular septa, interstitial septal thickening, "
        "Kerley lines, crazy-paving. "
        "NOT peribronchial or bronchial wall thickening. NOT fissural "
        "thickening. NOT pleural thickening. NOT tree-in-bud or centrilobular "
        "nodularity. NOT ground-glass opacity on its own"),
    "Post-surgical or post-treatment change": "post-surgical or post-treatment change: "
        "resection, lobectomy, thoracotomy, sternotomy, suture or staple line, "
        "surgical scarring, post-radiation change",
    # 9.0% unanchored. The overlap rule is stated because our own annotators
    # split on it: one scored an explicitly metastatic nodule as metastasis only,
    # the others as both.
    "Pulmonary metastases": (
        "metastasis or metastatic deposit that the report names as metastatic "
        "(keep separate from a plain nodule). "
        "NOT nodules merely being followed in a cancer patient. "
        "NOT \"recurrent disease\" or \"residual disease\" unless metastasis is "
        "named"),
    # This one is our own wording, not a model failure. "or centrilobular
    # branching opacities" was meant to catch the pattern described without its
    # name; what it actually did was license every "centrilobular groundglass
    # nodule" in the cohort - 8 one-sided false positives against the gold
    # standard, 0 misses. Centrilobular nodules are only tree-in-bud when they
    # branch, so the branching has to carry the weight.
    "Tree-in-bud": (
        "tree-in-bud nodularity, or centrilobular nodules the report describes "
        "as branching or linear-and-nodular. "
        "NOT centrilobular ground-glass or solid nodules without branching. "
        "NOT ground-glass nodularity on its own"),
    "Pulmonary cyst": "pulmonary cyst or cystic change, bulla, bleb, pneumatocele, "
        "congenital lobar overinflation",
    "Mass or neoplasm": "thoracic mass, tumour or neoplasm - mediastinal, thymic, "
        "chest-wall or pulmonary - not described as a discrete nodule or metastasis",
    "Mucus plugging": "mucus or mucous plugging, mucoid impaction, mucus-filled airways",
    "Pleural thickening or nodule": "pleural thickening, pleural nodule, pleural mass "
        "or deposit",
    # The only class where the disagreement is two-sided (10 vs 15) - and the
    # five gold-standard annotators split on it among themselves, which means the
    # definition, not the reader, is underspecified. Schmorl's nodes went 1 for
    # one annotator and 0 for two others. Both choices are defensible; leaving it
    # unstated is not. Degenerative and congenital findings are ruled out because
    # this class exists to carry clinically actionable osseous disease.
    "Bone lesion or fracture": (
        "fracture, lytic or sclerotic focal bone lesion, vertebral compression "
        "or height loss, rib or vertebral lesion, marrow replacement. "
        "NOT Schmorl's nodes. NOT degenerative endplate or disc change. "
        "NOT scoliosis or kyphosis. NOT congenital rib fusion or a rib anomaly. "
        "NOT osteopenia on its own"),
    "Pneumothorax": "pneumothorax or hydropneumothorax",
}

# The cohort word is descriptive of the INPUT, not of the label definitions.
# The definitions must stay byte-identical across cohorts - that is the whole
# point of re-extracting both sides with one prompt, after "Arterial wall
# calcification" drifted from aortic atherosclerosis (adult, 28.4%) to calcified
# catheter tracts in veins (pediatric, 3.3%) and cost that class 0.30 AUC.
#
# But telling the model "pediatric" while handing it a 65-year-old's scan is a
# false premise, not a harmless constant, and it pushes the wrong way: a model
# told the patient is a child has every reason to suppress atherosclerosis. The
# 40-report smoke already showed that shape - CT-RATE called Arterial wall
# calcification on 3 studies where we did not, the single largest disagreement
# in that direction. So the word tracks the data; the definitions do not move.
COHORT = os.environ.get("RAC_COHORT", "pediatric")
assert COHORT in ("pediatric", "adult"), COHORT
COHORT_A = ("an " if COHORT[0] in "aeiou" else "a ") + COHORT

SYSTEM = (
    f"You are a careful {COHORT} chest-CT radiology report analyst. "
    "You answer only with JSON that matches the requested schema. "
    "You never infer a finding that the report does not state."
)


def numbered(sentences: list[str]) -> str:
    return "\n".join("[%d] %s" % (i, s) for i, s in enumerate(sentences))


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
def region_schema(n_sentences: int = 0) -> dict:
    """Bound every array to the report it belongs to.

    Without maxItems the grammar lets the model keep emitting indices forever;
    it then hits max_tokens with the JSON still open and the record is lost as a
    parse failure. The label task never showed this because its schema is fixed
    size. Bounding `maximum` as well makes an out-of-range index ungrammatical
    instead of merely detectable.
    """
    item: dict = {"type": "integer", "minimum": 0}
    arr: dict = {"type": "array", "items": item}
    if n_sentences > 0:
        item["maximum"] = n_sentences - 1
        arr["maxItems"] = n_sentences
    return {
        "type": "object",
        "properties": {str(k): dict(arr, items=dict(item)) for k in REGIONS},
        "required": [str(k) for k in REGIONS],
        "additionalProperties": False,
    }


def label_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            p: {
                "type": "object",
                "properties": {
                    "p": {"type": "integer", "enum": [0, 1]},
                    "e": {"type": "integer", "minimum": -1},
                },
                "required": ["p", "e"],
                "additionalProperties": False,
            }
            for p in PATHOLOGIES
        },
        "required": list(PATHOLOGIES),
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def region_prompt(sentences: list[str]) -> str:
    lines = "\n".join("  %d = %s" % (k, v) for k, v in REGIONS.items())
    return f"""Below are the numbered sentences of {COHORT_A} chest CT report.

{numbered(sentences)}

Anatomical regions:
{lines}

For each region, list the indices of the sentences that describe a finding in
that region. Return JSON only.

Rules:
- Use sentence indices only. Never write sentence text.
- A sentence may belong to several regions; list it under each.
- A sentence describing both lungs generally belongs to all five lobe regions
  (1-5).
- A sentence stating a region is normal DOES count as describing that region.
- If no sentence describes a region, return an empty list for it.
- Only use indices between 0 and {len(sentences) - 1}."""


def label_prompt(sentences: list[str]) -> str:
    # The count is derived, never typed. It read "18 findings" above 27
    # definitions for the whole first extraction - the schema still forced 27
    # keys so nothing looked broken, but the model was told one thing and shown
    # another, and the nine new classes have no other supervision to fall back on.
    n_findings = len(PATHOLOGIES)
    defs = "\n".join('  "%s": %s' % (p, LABEL_DEFS[p]) for p in PATHOLOGIES)
    return f"""Below are the numbered sentences of {COHORT_A} chest CT report.

{numbered(sentences)}

Decide, for each of the {n_findings} findings below, whether the report states it is PRESENT:

{defs}

For each finding return an object with:
  "p": 1 if the report states the finding is present or strongly suspected,
       otherwise 0.
  "e": the index of the single sentence that STATES the finding, or -1 when p=0.

Rules:
- The evidence sentence is a GATE, not a justification. If no single sentence
  states the finding itself, p is 0. A sentence that mentions a different but
  similar-sounding finding is not evidence: "thickening of the major fissure" is
  not evidence for bronchial wall thickening, and "tree-in-bud nodularity" is not
  evidence for interlobular septal thickening.
- One sentence often states two findings, and then it is evidence for both.
  "Surgical sutures in the left lower lobe with adjacent linear scarring" is
  evidence for post-surgical change AND for scarring; do not make them compete.
- Explicitly negated findings ("no pleural effusion", "without consolidation")
  are 0.
- A finding described as resolved or improved to absent is 0: "interval
  resolution of the pneumothorax" is 0. But an unchanged finding is PRESENT:
  "no significant change in the pulmonary nodules" and "stable bronchiectasis"
  are both 1.
- Findings mentioned only as clinical history, indication, or as something to
  exclude are 0.
- A finding described in a comparison to a prior study but not present now is 0.
- Do not infer findings from other findings. Only what the report states.
- "e" must be a valid sentence index between 0 and {len(sentences) - 1} whenever
  p is 1.
Return JSON only."""


# --------------------------------------------------------------------------- #
# vLLM structured-output plumbing (the API was renamed across versions)
# --------------------------------------------------------------------------- #
def make_sampling(schema: dict, max_tokens: int):
    from vllm import SamplingParams
    common = dict(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    try:                                    # vLLM >= 0.27
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(
            structured_outputs=StructuredOutputsParams(json=schema), **common)
    except Exception:
        pass
    try:                                    # vLLM ~0.8 - 0.26
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(
            guided_decoding=GuidedDecodingParams(json=schema), **common)
    except Exception:
        pass
    return SamplingParams(guided_json=schema, **common)   # older still


def chat_texts(tokenizer, prompts: list[str]) -> list[str]:
    out = []
    for p in prompts:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": p}]
        try:
            out.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        except TypeError:                   # non-Qwen3 templates
            out.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
    return out


# --------------------------------------------------------------------------- #
def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok":
                    done.add(r["VolumeName"])
    return done


def validate_regions(obj: dict, n: int) -> tuple[dict, list]:
    out, problems = {}, []
    for k in REGIONS:
        idxs = obj.get(str(k), []) or []
        good = sorted({i for i in idxs if isinstance(i, int) and 0 <= i < n})
        if len(good) != len(set(idxs)):
            problems.append("region%d_bad_index" % k)
        out[str(k)] = good
    return out, problems


def validate_labels(obj: dict, n: int) -> tuple[dict, list]:
    out, problems = {}, []
    for p in PATHOLOGIES:
        v = obj.get(p) or {}
        present = 1 if v.get("p") == 1 else 0
        ev = v.get("e", -1)
        if not isinstance(ev, int):
            ev = -1
        if present and not (0 <= ev < n):
            problems.append("%s_missing_evidence" % p)
            ev = -1
        out[p] = {"p": present, "e": ev}
    return out, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="/temp_work/ch278233/BCH_DATASET/LABELS/sentences.jsonl")
    ap.add_argument("--out-dir", default="/temp_work/ch278233/BCH_DATASET/LABELS")
    ap.add_argument("--model", default=os.environ.get("PED_QWEN_MODEL", "Qwen/Qwen3-8B"))
    ap.add_argument("--task", choices=["regions", "labels", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=512, help="reports per flush")
    args = ap.parse_args()

    shard_id = int(os.environ.get("RAC_SHARD_ID", "0"))
    shard_n = int(os.environ.get("RAC_SHARD_N", "1"))
    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    with open(args.sentences) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("sentences"):
                records.append(r)
    if shard_n > 1:
        records = records[shard_id::shard_n]
    if args.limit:
        records = records[:args.limit]
    print("[ped] %d reports (shard %d/%d)" % (len(records), shard_id, shard_n))

    tasks = ["regions", "labels"] if args.task == "both" else [args.task]
    suffix = "" if shard_n == 1 else ".shard%d" % shard_id
    todo = {}
    for t in tasks:
        out_path = os.path.join(args.out_dir, "%s%s.jsonl" % (t, suffix))
        done = load_done(out_path)
        pend = [r for r in records if r["VolumeName"] not in done]
        todo[t] = (out_path, pend)
        print("[ped] %-8s done=%d pending=%d" % (t, len(done), len(pend)))
    if not any(len(v[1]) for v in todo.values()):
        print("[ped] nothing to do")
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_frac,
              trust_remote_code=True)

    # A region object can carry up to 10 index arrays over a 60-sentence report;
    # at 1024 tokens the JSON gets truncated mid-array, which reads as a parse
    # failure even though constrained decoding kept it schema-valid. The smoke
    # run lost 1 of 3 regions exactly this way.
    builders = {"regions": (region_prompt, region_schema(), 3072, validate_regions),
                "labels": (label_prompt, label_schema(), 1536, validate_labels)}

    for t in tasks:
        out_path, pend = todo[t]
        if not pend:
            continue
        build, schema, max_tok, validate = builders[t]
        sp = make_sampling(schema, max_tok)
        t0, n_ok, n_bad, n_prob = time.time(), 0, 0, 0
        with open(out_path, "a") as fh:
            for start in range(0, len(pend), args.chunk):
                batch = pend[start:start + args.chunk]
                texts = chat_texts(tokenizer, [build(r["sentences"]) for r in batch])
                if t == "regions":
                    # one schema per report: the index bound depends on its length
                    sps = [make_sampling(region_schema(len(r["sentences"])), max_tok)
                           for r in batch]
                    outs = llm.generate(texts, sps)
                else:
                    outs = llm.generate(texts, sp)
                for r, o in zip(batch, outs):
                    raw = o.outputs[0].text
                    n = len(r["sentences"])
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        n_bad += 1
                        fh.write(json.dumps({
                            "VolumeName": r["VolumeName"], "status": "parse_fail",
                            "n_sentences": n, "raw_len": len(raw),
                            "raw_tail": raw[-160:]}) + "\n")
                        continue
                    clean, problems = validate(obj, n)
                    if problems:
                        n_prob += 1
                    n_ok += 1
                    fh.write(json.dumps({
                        "VolumeName": r["VolumeName"], "status": "ok",
                        "n_sentences": n, t: clean,
                        "problems": problems}) + "\n")
                fh.flush()
                el = time.time() - t0
                seen = min(start + args.chunk, len(pend))
                print("[ped] %-8s %d/%d  ok=%d parse_fail=%d with_problems=%d "
                      "rate=%.1f/s eta=%.1fmin"
                      % (t, seen, len(pend), n_ok, n_bad, n_prob,
                         seen / max(el, 1e-6), (len(pend) - seen) / max(seen / max(el, 1e-6), 1e-6) / 60),
                      flush=True)
        print("[ped] %s DONE ok=%d parse_fail=%d with_problems=%d -> %s"
              % (t, n_ok, n_bad, n_prob, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

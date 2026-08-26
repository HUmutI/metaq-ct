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
        "NOT ligamentum arteriosum calcification, which is a normal variant"
    ),
    "Cardiomegaly": "enlarged cardiac silhouette",
    "Pericardial effusion": "fluid in the pericardial space",
    "Coronary artery wall calcification": "calcification specifically in coronary arteries",
    "Hiatal hernia": "stomach herniating through the diaphragmatic hiatus",
    # "prominent" removed on purpose. Pediatric reports use it to mean the
    # opposite of pathologic - "prominent nodes ... none are considered
    # pathologic by CT size criteria" - and including it drove this class to the
    # lowest agreement (kappa 0.32) and lowest citation support (22.4%) of all 18.
    "Lymphadenopathy": "lymph nodes the report calls enlarged, pathologic, or "
                       "abnormal by size criteria; nodes described as prominent "
                       "but not pathologic are NOT positive",
    # Bullae and pneumatoceles moved to Pulmonary cyst: in this cohort they are
    # cystic fibrosis and post-infectious change, not smoking-related alveolar
    # destruction. Leaving them here made a CF scan read as emphysema.
    "Emphysema": "emphysema from alveolar destruction (adult, smoking-related)",
    "Atelectasis": "atelectasis, collapse, volume loss",
    "Lung nodule": "nodule, micronodule, nodularity, granuloma",
    "Lung opacity": "ground-glass, airspace or parenchymal opacity not better named elsewhere",
    "Pulmonary fibrotic sequela": "fibrosis, scarring, reticulation, architectural distortion",
    "Pleural effusion": "fluid in the pleural space",
    "Mosaic attenuation pattern": "mosaic attenuation, air trapping, mosaic perfusion",
    "Peribronchial thickening": "bronchial or peribronchial wall thickening, cuffing",
    "Consolidation": "consolidation, dense airspace disease, pneumonia",
    "Bronchiectasis": "bronchiectasis, bronchial dilatation",
    "Interlobular septal thickening": "interlobular septal or interstitial septal thickening",
    "Post-surgical or post-treatment change": "post-surgical or post-treatment change: "
        "resection, lobectomy, thoracotomy, sternotomy, suture or staple line, "
        "surgical scarring, post-radiation change",
    "Pulmonary metastases": "metastasis or metastatic deposit named as such "
        "(keep separate from a plain nodule)",
    "Tree-in-bud": "tree-in-bud nodularity or centrilobular branching opacities",
    "Pulmonary cyst": "pulmonary cyst or cystic change, bulla, bleb, pneumatocele, "
        "congenital lobar overinflation",
    "Mass or neoplasm": "thoracic mass, tumour or neoplasm - mediastinal, thymic, "
        "chest-wall or pulmonary - not described as a discrete nodule or metastasis",
    "Mucus plugging": "mucus or mucous plugging, mucoid impaction, mucus-filled airways",
    "Pleural thickening or nodule": "pleural thickening, pleural nodule, pleural mass "
        "or deposit",
    "Bone lesion or fracture": "fracture, lytic or blastic bone lesion, rib or "
        "vertebral lesion, marrow replacement",
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
  "e": the index of the single sentence that best supports p=1, or -1 when p=0.

Rules:
- Explicitly negated findings ("no pleural effusion", "without consolidation",
  "resolved") are 0.
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

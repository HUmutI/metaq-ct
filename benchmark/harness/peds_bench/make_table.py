#!/home/ch278233/micromamba/envs/arcct/bin/python
"""Assemble the pediatric benchmark table from whatever metric bundles exist.

Reads /temp_work/ch278233/PEDS_BENCH/results/*.json (written by 77_metric_bundle.py)
plus the ARC-CT aggregate, and emits markdown + LaTeX.

Models with no result file are printed with an explicit reason rather than being
silently dropped -- a reader must be able to tell "we measured this and it scored
X" from "this could not be obtained".
"""
from __future__ import annotations
import json, os, glob

RES = "/temp_work/ch278233/PEDS_BENCH/results"

# Baselines that cannot be produced, with the reason. Kept in the output on purpose.
UNAVAILABLE = {
    "BIUD":        "no code or weights ever released; trained on private ChestCT-16K",
    "ViSD-Boost":  "inference-only release: no training code, no preprocessing, no license",
    "BrgSA":       "Apache-2.0 with full training code, but weights are Baidu-Pan only",
    "fVLM":        "needs per-organ report decomposition + organ masks; no supervised fine-tune",
    "CT-CLIP (pediatric fine-tune)": "blocked: needs 314 GB of preprocessed volumes, quota-limited",
}

# Rows split by whether the model saw pediatric data. Ranking a pediatric-trained
# model against an untrained zero-shot transfer in one list would be misleading:
# they answer different questions.
TRAINED = ["ARC-CT (full, +indication/age/sex)", "ARC-CT (ct-only, no metadata)",
           "GreenRFM", "MPS-CT", "BiomedCLIP (probe)", "Merlin (probe)"]
TRANSFER = ["HLIP", "BiomedCLIP", "CT-CLIP", "CT-CLIP VocabFine"]
ORDER = TRAINED + TRANSFER + ["CT-CLIP (pediatric fine-tune)", "fVLM",
                              "BrgSA", "ViSD-Boost", "BIUD"]


def load():
    rows = {}
    arc = os.path.join(RES, "arcct_peds.json")
    if os.path.exists(arc):
        for name, d in json.load(open(arc)).items():
            rows[name] = dict(auc=d["auc"], sd=d.get("auc_sd"), prec=d["prec"],
                              acc=d["acc"], f1=d["f1"], n=d.get("n_seeds", 1),
                              readout="prompt", src="arcct_peds.json")
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        if f.endswith("arcct_peds.json"):
            continue
        b = json.load(open(f))
        if "macro" not in b:
            continue
        m = b["macro"]
        name = b.get("model", os.path.basename(f))
        readout = b.get("readout", "")
        if "defective" in readout:
            name += " (defective readout)"
        rows[name] = dict(auc=m["auc"], sd=None, prec=m["precision"], acc=m["acc"],
                          f1=m["f1"], n=1, readout=readout, src=os.path.basename(f))
    return rows


def main():
    rows = load()
    seen = set()
    print("\n## Pediatric benchmark — BCH cohort, 23 classes, n=1507 validation\n")
    print("| Method | readout | AUC | Prec | Acc | F1 | source |")
    print("|---|---|---:|---:|---:|---:|---|")
    # BUGFIX: rows that are measured but not in TRAINED/TRANSFER (ablations, the
    # documented-defective GreenRFM readout) previously fell through into the
    # "Not reproducible" block, which asserted the opposite of the truth about
    # them. They now get their own clearly-labelled section.
    unavail = [n for n in ORDER if n not in TRAINED and n not in TRANSFER]
    extra = sorted(k for k in rows if k not in ORDER)
    seq = ([("**Adapted to pediatric data**", None)] + [(n, 1) for n in TRAINED]
           + [("**Zero-shot transfer (no pediatric training)**", None)] + [(n, 1) for n in TRANSFER])
    if extra:
        seq += [("**Supplementary (measured; ablations and diagnostics, not ranked)**", None)]
        seq += [(n, 1) for n in extra]
    seq += [("**Not reproducible — no measurement attempted or possible**", None)]
    seq += [(n, 1) for n in unavail]
    for name, _ in seq:
        if _ is None:
            print(f"| {name} | | | | | | |")
            continue
        if name in seen:
            continue
        seen.add(name)
        if name in rows:
            r = rows[name]
            auc = f"{r['auc']:.4f}" + (f" ± {r['sd']:.4f}" if r.get("sd") else "")
            n = f" (n={r['n']})" if r["n"] > 1 else ""
            print(f"| {name}{n} | {r['readout']} | **{auc}** | {r['prec']:.4f} | "
                  f"{r['acc']:.4f} | {r['f1']:.4f} | `{r['src']}` |")
        elif name in UNAVAILABLE:
            print(f"| {name} | — | *not available* | — | — | — | {UNAVAILABLE[name]} |")
    # LaTeX mirrors the markdown grouping exactly. Previously this emitted one flat
    # list, which silently promoted a supplementary ablation into the main ranking
    # and dropped the trained-vs-transfer distinction that makes the table honest.
    print("\n\n% ---- LaTeX ----")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Method & AUC & Prec & Acc & F1 \\\\")
    blocks = [("\\textit{Adapted to pediatric data}", TRAINED),
              ("\\textit{Zero-shot transfer (no pediatric training)}", TRANSFER)]
    if extra:
        blocks.append(("\\textit{Supplementary (not ranked)}", extra))
    for title, names in blocks:
        print("\\midrule")
        print(f"\\multicolumn{{5}}{{l}}{{{title}}} \\\\")
        for name in names:
            if name not in rows:
                continue
            r = rows[name]
            auc = f"{r['auc']:.3f}" + (f"$\\pm${r['sd']:.3f}" if r.get("sd") else "")
            print(f"{name.replace('&', chr(92)+chr(92)+'&')} & {auc} & {r['prec']:.3f} & "
                  f"{r['acc']:.3f} & {r['f1']:.3f} \\\\")
    miss = [n for n in unavail if n not in rows]
    if miss:
        print("\\midrule")
        print("\\multicolumn{5}{l}{\\textit{Not reproducible (see caption)}} \\\\")
        for name in miss:
            print(f"{name} & \\multicolumn{{4}}{{c}}{{\\textit{{not reproducible}}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    print("\nAUC is the primary column: it is threshold-free and calibration-independent.")
    print("Prec/Acc/F1 are at a fixed 0.5 threshold and therefore also measure how well")
    print("each model happens to be calibrated for a pediatric cohort it never saw.")
    print("All rows scored by ARC-CT's own compute_metric_bundle on the identical split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

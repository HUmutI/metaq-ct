#!/home/ch278233/micromamba/envs/arcct/bin/python
"""Assemble the pediatric retrieval table (ARC-CT Table 4 layout) from
retrieval/results/*.json. Our arms are averaged over seeds (mean +- sd);
single-model baselines carry patient-bootstrap 95% CIs. Both CT->CT
conventions are shown: clean (self excluded, per-K) and the origin/Table-4
convention (self included, cumulative), each labelled as such."""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

R = "/temp_work/ch278233/PEDS_BENCH/retrieval/results"
# Refined C1 freezes the inherited base, so its metadata-free z_gen is the CT-only
# parent's z_gen (verified: img max|diff| 6e-8, txt bit-identical). One z_gen row,
# seed-averaged; the duplicate refined_c1 'general' runs are suppressed. The two
# conditioned readouts use each GALLERY study's indication/age/sex and are therefore
# not an equal-input comparison with the CT-only baselines -- labelled as such.
SKIP = {"ours_refined_c1"}
# Present only in the readout-ablation table, never as an extra row in the
# cross-model table (where it would read as a third arm of ours).
ABLATION_ONLY = {"ours_refined_c1_conditioned"}
# The cross-model table carries exactly two of our arms, the full model and its
# matched CT-only control, so it compares methods rather than our own readouts.
# The readout variants live in their own ablation table below.
DISPLAY = {  # stem -> (name, group)
    "ours_refined_c1_mixed":      (r"\ours{} (full, metadata-conditioned)$^{\mathrm{m}}$", "adapted"),
    "ours_ct_only":               (r"\ours{} / CT-only control (metadata-free)", "adapted"),
    "greenrfm":        ("GreenRFM, local adaptation (epoch 9)", "adapted"),
    "mpsct":           ("MPS-CT, local reproduction (update 3,000)", "adapted"),
    "hlip":            ("H-LIP", "transfer"),
    "biomedclip":      ("BiomedCLIP (2-D, slice-wise)", "transfer"),
    "merlin":          ("Merlin (abdominal)", "transfer"),
    "ctclip":          ("CT-CLIP", "transfer"),
    "ctclip_vocabfine":("CT-CLIP VocabFine", "transfer"),
}

# Context length and truncation count come from each latent file's own meta block,
# written by the extractor that ran the model -- never retyped here, so the note
# cannot drift from what was actually encoded.
TRUNC_ORDER = [("ours", None), ("ctclip", "CT-CLIP"), ("ctclip_vocabfine", "VocabFine"),
               ("hlip", "H-LIP"), ("greenrfm", "GreenRFM"), ("mpsct", "MPS-CT"),
               ("biomedclip", "BiomedCLIP"), ("merlin", "Merlin")]


def trunc_note():
    import ast
    L = "/temp_work/ch278233/PEDS_BENCH/retrieval"
    parts = ["ours 256 tokens"]
    n_total = None
    for stem, label in TRUNC_ORDER:
        if label is None:
            continue
        f = f"{L}/{stem}_valid.npz"
        if not os.path.exists(f):
            continue
        m = ast.literal_eval(str(np.load(f, allow_pickle=True)["meta"]))
        ctx = m.get("text_ctx") or m.get("context_length") or m.get("context_tokens")
        nt = m.get("n_truncated", m.get("n_truncated_reports"))
        # Extractors that did not record it: measured afterwards by replaying the
        # model's own tokenizer (retr_measure_trunc.py).
        side = f"{L}/{stem}_trunc.json"
        if nt is None and os.path.exists(side):
            sm = json.load(open(side)); nt = sm["n_truncated"]; ctx = ctx or sm["context_length"]
        if n_total is None:
            n_total = len(np.load(f, allow_pickle=True)["accessions"])
        if nt is None:
            parts.append(f"{label} {ctx:,} (count not recorded)")
        elif nt == 0:
            parts.append(f"{label} {ctx:,} (none truncated)")
        else:
            pct = 100.0 * nt / n_total
            extra = f", {pct:.0f}\\%" if pct >= 10 else ""
            parts.append(f"{label} {ctx:,} ({nt:,} of {n_total:,} truncated{extra})")
    return "; ".join(parts)

READOUTS = [  # stem -> label, for the readout-ablation table only
    ("ours_ct_only",                r"$z_{\mathrm{gen}}$, metadata-free (shared with CT-only)"),
    ("ours_refined_c1_mixed",       r"$\mathrm{norm}(z_{\mathrm{gen}}{+}z_{\mathrm{ind}})$, full model$^{\mathrm{m}}$"),
    ("ours_refined_c1_conditioned", r"$z_{\mathrm{ind}}$, conditioned only$^{\mathrm{m}}$"),
]
KR = ["R@5", "R@10", "R@50", "R@100"]; KI = ["@5", "@10", "@50"]


def load():
    rows = {}
    for f in sorted(glob.glob(f"{R}/*.json")):
        d = json.load(open(f)); n = d["model"]
        base = n.rsplit("_seed", 1)[0] if "_seed" in n else n
        rows.setdefault(base, []).append(d)
    return rows


def agg(ds):
    r = np.array([[d["report_to_image"][k] for k in KR] for d in ds])
    c = np.array([[d["image_to_image"]["clean_self_excluded"][k] for k in KI] for d in ds])
    o = np.array([[d["image_to_image"]["origin_self_included_cumulative"][k] for k in KI] for d in ds])
    out = {"n_seeds": len(ds), "n": ds[0]["n"], "r": r.mean(0), "c": c.mean(0), "o": o.mean(0),
           "r_sd": r.std(0, ddof=1) if len(ds) > 1 else None, "c_sd": c.std(0, ddof=1) if len(ds) > 1 else None}
    if len(ds) == 1 and "ci_patient_bootstrap" in ds[0]:
        ci = ds[0]["ci_patient_bootstrap"]
        out["r_ci"] = [ci["report_to_image"][k] for k in KR]
        out["c_ci"] = [ci["image_to_image_clean"][k] for k in KI]
    return out


def fmt(v, sd=None, ci=None):
    s = f"{v:.1f}"
    if sd is not None: s += f"$\\pm${sd:.1f}"
    return s


FRAG = "/home/ch278233/paper/generated/peds23_retrieval.tex"


def _latex_readouts(rows, n):
    """Our three readouts on the same latents, kept out of the cross-model table."""
    ks = [(k, lab) for k, lab in READOUTS if k in rows]
    if len(ks) < 2:
        return
    A = {k: agg(rows[k]) for k, _ in ks}
    bc = {i: max(A[k]["c"][i] for k, _ in ks) for i in range(3)}
    br = {i: max(A[k]["r"][i] for k, _ in ks) for i in range(4)}
    B = lambda t, v, b: (f"\\best{{{t}}}" if abs(v - b) < 1e-9 else t)
    print(r"\begin{table}[t]\centering")
    print(r"\caption{Readout ablation for \ours{} on the same pediatric retrieval task and the same stored latents as \Cref{tab:pedsretrieval}; three-seed means. The two tasks have opposite winners: the metadata-free readout retrieves the exact report best, the conditioned readout retrieves the most label-similar studies best.}")
    print(r"\label{tab:pedsretrieval-readout}\small")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{@{}lrrrr@{}}\toprule")
    print(r"Readout & R@5 & R@100 & Jaccard@5 & Jaccard@50 \\ \midrule")
    for k, lab in ks:
        a = A[k]
        print(" & ".join([lab,
                          B(fmt(a["r"][0], a["r_sd"][0] if a["r_sd"] is not None else None), a["r"][0], br[0]),
                          B(fmt(a["r"][3], a["r_sd"][3] if a["r_sd"] is not None else None), a["r"][3], br[3]),
                          B(fmt(a["c"][0], a["c_sd"][0] if a["c_sd"] is not None else None), a["c"][0], bc[0]),
                          B(fmt(a["c"][2], a["c_sd"][2] if a["c_sd"] is not None else None), a["c"][2], bc[2])]) + r" \\")
    print(r"\bottomrule\end{tabular}}")
    print(r"\vspace{1mm}\parbox{\columnwidth}{\footnotesize $^{\mathrm{m}}$Uses each gallery study's own indication, age, and sex; not an equal-input comparison with the metadata-free row.}")
    print(r"\end{table}")


def main():
    import io, contextlib
    rows = load()
    if not rows: sys.exit("no results yet")
    n = next(iter(rows.values()))[0]["n"]
    chance = {k: 100 * int(k[2:]) / n for k in KR}
    rows = {k: v for k, v in rows.items() if k not in SKIP}
    order = ([k for k in DISPLAY if k in rows]
             + sorted(k for k in rows if k not in DISPLAY and k not in ABLATION_ONLY))
    A = {k: agg(rows[k]) for k in order}
    # bold best per column within all measured rows
    best_r = {i: max(A[k]["r"][i] for k in order) for i in range(4)}
    best_c = {i: max(A[k]["c"][i] for k in order) for i in range(3)}
    best_o = {i: max(A[k]["o"][i] for k in order) for i in range(3)}
    B = lambda s, v, b: (f"\\best{{{s}}}" if abs(v - b) < 1e-9 else s)

    print(f"\n## Pediatric retrieval — {n} studies, 648 patients\n")
    print("| Method | R@5 | R@10 | R@50 | R@100 | I→I clean @5 | @10 | @50 | I→I Table-4 conv. @5 | @10 | @50 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for grp in ("adapted", "transfer"):
        print(f"| **{'Adapted to pediatric cohort' if grp=='adapted' else 'Released weights, no pediatric training'}** |||||||||||")
        for k in order:
            if DISPLAY.get(k, ("", "transfer"))[1] != grp: continue
            import re as _re
            a = A[k]; nm = _re.sub(r"\$\^\{\\mathrm\{m\}\}\$|\\tnote\{\w\}", "", DISPLAY.get(k, (k, ""))[0])
            nm = (nm.replace("\\ours{}", "MetaQ-CT").replace("\\mathrm{", "").replace("{+}", "+")
                    .replace("$", "").replace("{", "").replace("}", ""))
            if a["n_seeds"] > 1: nm += f" (n={a['n_seeds']})"
            cells = [f"{a['r'][i]:.2f}" for i in range(4)] + [f"{a['c'][i]:.2f}" for i in range(3)] + [f"{a['o'][i]:.2f}" for i in range(3)]
            print(f"| {nm} | " + " | ".join(cells) + " |")
    print(f"| chance (K/N) | " + " | ".join(f"{chance[k]:.2f}" for k in KR) + " | — | — | — | — | — | — |")

    print("\n% ---- LaTeX ----")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _latex(rows, order, A, n, chance, best_r, best_c, best_o)
        _latex_readouts(rows, n)
    tex = buf.getvalue()
    print(tex)
    os.makedirs(os.path.dirname(FRAG), exist_ok=True)
    open(FRAG, "w").write("% Auto-generated by retr_make_table.py from PEDS_BENCH/retrieval/results; do not edit by hand.\n" + tex)
    print(f"% fragment -> {FRAG}")
    return 0


def _latex(rows, order, A, n, chance, best_r, best_c, best_o):
    B = lambda s, v, b: (f"\\best{{{s}}}" if abs(v - b) < 1e-9 else s)
    print(r"\begin{table*}[t]\centering")
    print(r"\caption{Pediatric cross-modal retrieval on the historical cohort (" + f"{n:,}" + r" studies, 648 patients), no retrieval-specific fitting. Report$\rightarrow$image is exact-match Recall@K over all " + f"{n:,}" + r" studies. Image$\rightarrow$image is mean label-set Jaccard of the top-$K$ retrieved studies (gallery = studies with $\geq$1 positive label): \emph{clean} excludes the query from its own gallery; \emph{Table-4 convention} reproduces the CT-CLIP evaluation script used by prior work, which includes the query and accumulates across $K$, and is shown only for comparability with published adult tables. Our arms are three-seed means; chance Recall@K is $K/N$.}")
    print(r"\label{tab:pedsretrieval}\small")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{@{}lrrrrrrrrrr@{}}\toprule")
    print(r" & \multicolumn{4}{c}{Report$\rightarrow$image R@K} & \multicolumn{3}{c}{Image$\rightarrow$image, clean} & \multicolumn{3}{c}{Image$\rightarrow$image, Table-4 conv.}\\")
    print(r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}")
    print(r"Method & @5 & @10 & @50 & @100 & @5 & @10 & @50 & @5 & @10 & @50 \\ \midrule")
    for grp, title in (("adapted", "Adapted to the pediatric cohort"), ("transfer", "Released weights, no pediatric training")):
        print(r"\multicolumn{11}{@{}l}{\emph{" + title + r"}}\\")
        for k in order:
            if DISPLAY.get(k, ("", "transfer"))[1] != grp: continue
            a = A[k]; nm = DISPLAY.get(k, (k, ""))[0]
            rc = [B(fmt(a["r"][i], a["r_sd"][i] if a["r_sd"] is not None else None), a["r"][i], best_r[i]) for i in range(4)]
            cc = [B(fmt(a["c"][i], a["c_sd"][i] if a["c_sd"] is not None else None), a["c"][i], best_c[i]) for i in range(3)]
            oc = [B(f"{a['o'][i]:.1f}", a["o"][i], best_o[i]) for i in range(3)]
            print(f"{nm} & " + " & ".join(rc + cc + oc) + r" \\")
    print(r"\midrule Chance ($K/N$) & " + " & ".join(f"{chance[k]:.1f}" for k in KR) + r" & -- & -- & -- & -- & -- & -- \\")
    print(r"\bottomrule\end{tabular}}")
    print(r"\vspace{1mm}\parbox{\textwidth}{\footnotesize $^{\mathrm{m}}$Uses each gallery study's own indication, age, and sex through the conditioned bank; not an equal-input comparison with the metadata-free rows, shown to test whether conditioning changes retrieval. Every model encodes the identical impression-first report string; each uses its own tokenizer at its own trained context length, which truncates long reports: " + trunc_note() + r". Single-model rows carry 1,000-draw patient-bootstrap intervals in the accompanying JSON.}")
    print(r"\end{table*}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

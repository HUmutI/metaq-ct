#!/usr/bin/env python3
"""Render completed U18 artifacts into a LaTeX fragment imported by main.tex."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


WORK = Path("/temp_work/ch278233")
EVAL = Path(os.environ.get(
    "PEDS_U18_EVAL_ROOT",
    str(WORK / "eval_matrix/peds23_u18_r2plus1d_18_test"),
))
R3D_EVAL = Path(os.environ.get(
    "PEDS_U18_R3D_EVAL_ROOT",
    str(WORK / "eval_matrix/peds23_u18_r3d_18_test"),
))
IMBALANCE_EVAL = Path(os.environ.get(
    "PEDS_U18_IMBALANCE_EVAL_ROOT",
    str(WORK / "eval_matrix/peds23_u18_imbalance_test"),
))
OUTPUT = Path(os.environ.get(
    "PEDS_U18_TEX_OUTPUT",
    "/home/ch278233/paper/generated/peds23_u18_results.tex",
))
SNAPSHOT = os.environ.get("PEDS_U18_SNAPSHOT", "0") == "1"
LABEL_SUFFIX = "-snapshot" if SNAPSHOT else ""
ARMS = [
    ("full_seed0", "Full model, seed 0"), ("full_seed1", "Full model, seed 1"),
    ("full_seed2", "Full model, seed 2"), ("ct_only_seed0", "CT-only, seed 0"),
    ("ct_only_seed1", "CT-only, seed 1"), ("ct_only_seed2", "CT-only, seed 2"),
    ("c2_off", "C2 off"), ("c1_off", "C1 off"),
    ("indication_only", "Indication conditioning only"),
    ("demographics_only", "Demographic conditioning only"),
    ("metadata_none", "No metadata"), ("anatomy_queries_off", "Anatomy queries off"),
    ("routing_off", "Routing off"), ("region_align_off", "Region alignment off"),
    ("maskfree", "Mask-free"), ("age_scalar", "Scalar age"),
    ("xattn_only", "Cross-attention only"), ("film_only", "FiLM only"),
    ("drop00", "Indication dropout 0.0"), ("drop10", "Indication dropout 0.1"),
    ("drop40", "Indication dropout 0.4"), ("no_augmentation", "Augmentation off"),
    ("ctxcls_aux", "Auxiliary ctx\\_cls"),
    ("counterfactual005", "Counterfactual loss 0.05"),
    ("qwen_region_text", "Qwen region text"),
    ("naive_metadata_fusion", "Naive pooled-metadata fusion"),
    ("l2_on", "L2 normalization on"),
    ("standard_negative", "Standard negative prompts"),
    ("ind_weight1", "Indication-loss weight 1"),
    ("ind_weight4", "Indication-loss weight 4"),
    ("one_global_query", "One global query"),
    ("six_global_queries", "Six global queries"),
    ("counterfactual015", "Counterfactual loss 0.15"),
    ("general_selection", "General-AUROC checkpoint selection"),
    ("r3d18_full_seed0", "R3D-18 backbone, full model, seed 0"),
    ("pathology_class_logit", "Pathology-specific class-logit prompt readout"),
    ("imbalance_asl", "Asymmetric loss (ASL), seed 0"),
    ("imbalance_bnm001", "BCE + BNM ($\\lambda=0.001$), seed 0"),
    ("imbalance_bnm003", "BCE + BNM ($\\lambda=0.003$), seed 0"),
    ("imbalance_bnm010", "BCE + BNM ($\\lambda=0.010$), seed 0"),
    ("imbalance_asl_bnm001", "ASL + BNM ($\\lambda=0.001$), seed 0"),
    ("imbalance_asl_bnm003", "ASL + BNM ($\\lambda=0.003$), seed 0"),
    ("imbalance_asl_bnm010", "ASL + BNM ($\\lambda=0.010$), seed 0"),
]


def metric(arm: str):
    if arm == "r3d18_full_seed0":
        path = R3D_EVAL / "full_seed0_ind-true/paper_metrics.json"
    elif arm.startswith("imbalance_"):
        path = IMBALANCE_EVAL / f"{arm.removeprefix('imbalance_')}/paper_metrics.json"
    else:
        path = EVAL / f"{arm}_ind-true/paper_metrics.json"
    return json.loads(path.read_text())["macro"] if path.exists() else None


def main() -> int:
    comparison = json.loads((EVAL / "paper_summary/core_comparison.json").read_text())
    lines = [
        "% Auto-generated from audited prediction artifacts; do not edit by hand.",
        ("\\subsection{Frozen true-pediatric ($<18$) results}"
         if SNAPSHOT else "\\subsection{Frozen true-pediatric ($<18$) results}"),
        "\\begin{table}[t]", "\\centering",
        ("\\caption{Three-seed primary comparison on the MRN-disjoint $<18$ test split.}"
         if SNAPSHOT else "\\caption{Three-seed primary comparison on the MRN-disjoint $<18$ test split.}"),
        f"\\label{{tab:u18-primary{LABEL_SUFFIX}}}", "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
        "Seed & CT-only AUROC & Full AUROC & $\\Delta$ \\\\", "\\midrule",
    ]
    for row in comparison["seeds"]:
        lines.append(f'{row["seed"]} & {row["ct_only_auroc"]:.4f} & {row["full_auroc"]:.4f} & {row["delta_auroc"]:+.4f} \\\\')
    lo, hi = comparison["delta_auroc_ci95_patient_bootstrap"]
    p_value = comparison["paired_bootstrap_p_two_sided"]
    p_text = "p<0.001" if p_value == 0 else f"p={p_value:.4g}"
    lines += [
        "\\midrule",
        f'Mean & {comparison["mean_ct_only_auroc"]:.4f} & {comparison["mean_full_auroc"]:.4f} & {comparison["mean_delta_auroc"]:+.4f} \\\\',
        "\\bottomrule", "\\end{tabular}",
        f'\\parbox{{\\columnwidth}}{{\\footnotesize Patient-bootstrap 95\\% CI for the mean paired AUROC difference: [{lo:.4f}, {hi:.4f}]; two-sided paired bootstrap ${p_text}$.}}',
        "\\end{table}",
        "\\begin{table*}[t]", "\\centering",
        ("\\caption{Audited $<18$ ablations. Arms without a completed frozen-test evaluation are omitted.}"
         if SNAPSHOT else "\\caption{Complete frozen $<18$ architecture ablation. Every row uses the same train/development/test split.}"),
        f"\\label{{tab:u18-ablation{LABEL_SUFFIX}}}", "\\begin{tabular}{@{}lrrlll@{}}", "\\toprule",
        "Arm & AUROC & AUPRC & Brier & ECE & Sens.@95\\% spec. \\\\", "\\midrule",
    ]
    for arm, label in ARMS:
        value = metric(arm)
        if value is None:
            continue
        else:
            lines.append(f'{label} & {value["auroc"]:.4f} & {value["auprc"]:.4f} & {value["brier"]:.4f} & {value["ece_10bin"]:.4f} & {value["sensitivity_at_95_specificity"]:.4f} \\\\')
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

    subgroup = EVAL / "paper_summary/subgroups.csv"
    if subgroup.exists():
        rows = list(csv.DictReader(subgroup.open()))
        lines += ["\\begin{table}[t]", "\\centering", "\\caption{Age and sex subgroup consistency on the $<18$ test split.}",
                  f"\\label{{tab:u18-subgroups{LABEL_SUFFIX}}}", "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
                  "Subgroup & Model & $N$ & AUROC \\\\", "\\midrule"]
        for row in rows:
            name = row["subgroup"].replace("_", "\\_")
            model = row["model"].replace("_", "\\_")
            lines.append(f'{name} & {model} & {row["studies"]} & {float(row["macro_auroc"]):.4f} \\\\')
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    mention = EVAL / "paper_summary/mention_strata.csv"
    if mention.exists():
        rows = list(csv.DictReader(mention.open()))
        lines += ["\\begin{table}[t]", "\\centering",
                  "\\caption{Performance stratified by explicit target terminology in the indication.}",
                  f"\\label{{tab:u18-mention-strata{LABEL_SUFFIX}}}", "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
                  "Terminology & Model & AUROC & AUPRC \\\\", "\\midrule"]
        for row in rows:
            name = row["stratum"].replace("_", "\\_")
            model = row["model"].replace("_", "\\_")
            lines.append(f'{name} & {model} & {float(row["macro_auroc"]):.4f} & {float(row["macro_auprc"]):.4f} \\\\')
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    per_class = EVAL / "paper_summary/per_class.csv"
    if per_class.exists():
        rows = list(csv.DictReader(per_class.open()))
        lines += ["\\begin{table*}[t]", "\\centering",
                  "\\caption{Complete three-seed per-class results on the frozen $<18$ test split.}",
                  f"\\label{{tab:u18-per-class{LABEL_SUFFIX}}}", "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
                  "Class & Pos. & CT AUC & Full AUC & $\\Delta$ AUC & CT AUPRC & Full AUPRC & $\\Delta$ AUPRC \\\\", "\\midrule"]
        for row in rows:
            name = row["class"].replace("&", "\\&")
            lines.append(f'{name} & {row["positive_n"]} & {float(row["ct_only_auroc"]):.4f} & {float(row["full_auroc"]):.4f} & {float(row["delta_auroc"]):+.4f} & {float(row["ct_only_auprc"]):.4f} & {float(row["full_auprc"]):.4f} & {float(row["delta_auprc"]):+.4f} \\\\')
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

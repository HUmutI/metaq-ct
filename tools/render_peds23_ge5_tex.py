#!/usr/bin/env python3
"""Render every completed historical age>=5 pediatric ablation artifact."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path("/temp_work/ch278233/eval_matrix/peds23_biltir_final")
OUT = Path("/home/ch278233/paper/generated/peds23_ge5_ablation.tex")
ARMS = [
    ("final_drop03_band", "Full current model, seed 0"),
    ("final_drop03_band_seed1", "Full current model, seed 1"),
    ("final_drop03_band_seed2", "Full current model, seed 2"),
    ("ct_only_seed0", "CT-only, seed 0"), ("ct_only_seed1", "CT-only, seed 1"),
    ("ct_only_seed2", "CT-only, seed 2"), ("drop00", "Indication dropout 0.0"),
    ("drop10", "Indication dropout 0.1"), ("drop40", "Indication dropout 0.4"),
    ("age_scalar", "Scalar age"), ("c2_off", "C2 off"), ("c1_off", "C1 off"),
    ("indication_only", "Indication conditioning only"),
    ("demographics_only", "Demographic conditioning only"),
    ("metadata_none", "No metadata"), ("anatomy_queries_off", "Anatomy queries off"),
    ("routing_off", "Routing off"), ("region_align_off", "Region alignment off"),
    ("fully_maskfree", "Fully mask-free"), ("l2_on", "L2 normalization on"),
    ("standard_negative", "Standard negative prompts"),
    ("ctxcls_aux", "Auxiliary ctx\\_cls"),
    ("counterfactual005", "Counterfactual loss 0.05"),
    ("ind_weight1", "Indication-loss weight 1"),
    ("ind_weight4", "Indication-loss weight 4"),
    ("qwen_region_text", "Qwen region text"), ("one_global_query", "One global query"),
    ("six_global_queries", "Six global queries"), ("no_augmentation", "Augmentation off"),
    ("indication_present_train", "Indication-present-only training"),
    ("general_selection", "General-AUROC checkpoint selection"),
]


def bundle(arm: str, kind: str, ind: str = "true", meta: str = "checkpoint"):
    path = ROOT / f"{arm}_ind-{ind}_meta-{meta}" / f"metric_bundle_{kind}.json"
    return json.loads(path.read_text())["macro"] if path.exists() else None


def value(item, key):
    return "n/a" if item is None else f'{item[key]:.4f}'


def main() -> int:
    newline = r" \\"
    lines = [
        "% Auto-generated historical pediatric ablation table.",
        r"\subsection{Completed historical age-$\geq5$ current-recipe ablations}",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Completed current-recipe age-$\geq5$ 23-target ablations on the same historical validation split. Global denotes the conditioned/fused prompt readout; arms without a completed evaluation are omitted.}",
        r"\label{tab:peds-ge5-complete}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        "Arm & Global AUC & General AUC & Global F1 & Global accuracy & Global precision" + newline,
        r"\midrule",
    ]
    full_arms = ("final_drop03_band", "final_drop03_band_seed1", "final_drop03_band_seed2")
    ct_arms = ("ct_only_seed0", "ct_only_seed1", "ct_only_seed2")
    for label, arms in (("Full model, three-seed mean", full_arms),
                        ("CT-only, three-seed mean", ct_arms)):
        global_metrics = [bundle(arm, "global") for arm in arms]
        general_metrics = [
            bundle(arm, "general_prompt") or bundle(arm, "global")
            for arm in arms
        ]
        if all(item is not None for item in global_metrics + general_metrics):
            lines.append(
                f'{label} & {mean(x["auc"] for x in global_metrics):.4f} & '
                f'{mean(x["auc"] for x in general_metrics):.4f} & '
                f'{mean(x["f1"] for x in global_metrics):.4f} & '
                f'{mean(x["acc"] for x in global_metrics):.4f} & '
                f'{mean(x["precision"] for x in global_metrics):.4f}' + newline
            )
    lines.append(r"\midrule")
    for arm, label in ARMS:
        global_metric = bundle(arm, "global")
        general_metric = bundle(arm, "general_prompt") or (
            global_metric if arm.startswith("ct_only") else None
        )
        if global_metric is None:
            continue
        lines.append(f'{label} & {value(global_metric,"auc")} & {value(general_metric,"auc")} & {value(global_metric,"f1")} & {value(global_metric,"acc")} & {value(global_metric,"precision")}' + newline)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", r"\begin{table}[t]", r"\centering",
              r"\caption{Available current-recipe indication interventions on the age-$\geq5$ cohort. Only completed evaluations are shown.}",
              r"\label{tab:peds-ge5-intervention-final}", r"\begin{tabular}{@{}llr@{}}", r"\toprule",
              "Seed & Indication & AUROC" + newline, r"\midrule"]
    for seed, arm in enumerate(("final_drop03_band", "final_drop03_band_seed1", "final_drop03_band_seed2")):
        for name, indication in (("Correct", "true"), ("Blank", "none"),
                                 ("Shuffled", "shuffled")):
            result = bundle(arm, "global", indication)
            if result is not None:
                lines.append(f'{seed} & {name} & {value(result,"auc")}' + newline)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

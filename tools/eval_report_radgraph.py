"""Add official F1-RadGraph scores to report-generation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--model-type", default="modern-radgraph-xl")
    args = p.parse_args()
    from radgraph import F1RadGraph
    path = Path(args.predictions)
    rows = json.loads(path.read_text())
    metric = F1RadGraph(reward_level="all", model_type=args.model_type)
    mean, per_sample, _, _ = metric(
        hyps=[x["prediction"] for x in rows], refs=[x["reference"] for x in rows])
    names = ("radgraph_entity_f1", "radgraph_entity_relation_f1", "radgraph_complete_f1")
    result = {name: float(value) for name, value in zip(names, mean)}
    result["n_samples"] = len(rows)
    result["model_type"] = args.model_type
    out = path.parent / "radgraph_metrics.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

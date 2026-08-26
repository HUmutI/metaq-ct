#!/usr/bin/env python3
"""Fail a job before it burns a GPU on the wrong tree.

The home directory was reorganised: ``~/arc-ct`` is the published adult tree and
``~/bch-arc-ct`` is the working copy. ``~/arc-ct`` has no ``arcct/schema.py`` and
no ``configs/stage2_peds.env``. Every sbatch used to point at ``~/arc-ct``, and
because the scripts run under ``set -uo pipefail`` (no ``-e``), a missing env
file printed one line and the run continued on code defaults -- RAC_SCHEMA=ctrate
(18 classes), the Q-Former switched off, the wrong HU windows. Silent, and wrong
in a way that only shows up as a bad AUC days later.

This script is the gate: it asserts that the ``arcct`` package actually imported
from ARCCT_ROOT and that the schema in force has the expected class count.

    python pipeline/lib/check_env.py --expect-classes 27 --root "$ARCCT_ROOT"
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ARCCT_ROOT", ""))
    ap.add_argument("--expect-classes", type=int, default=0,
                    help="0 = only check the import path, not the schema size")
    ap.add_argument("--expect-schema", default="")
    a = ap.parse_args()

    problems: list[str] = []

    try:
        import arcct.schema as schema
    except Exception as exc:                                  # noqa: BLE001
        print(f"FATAL: cannot import arcct.schema: {exc}")
        return 78

    pkg = os.path.realpath(os.path.dirname(os.path.dirname(schema.__file__)))
    if a.root:
        want = os.path.realpath(a.root)
        if pkg != want:
            problems.append(f"arcct imported from {pkg}, expected {want}")

    # schema.active() is the resolver -- the module-level names are the two raw
    # tables, not the one in force. Reading PEDS_* directly would report 27 even
    # when RAC_SCHEMA is unset and the run is really on the 18-class adult path,
    # which is the exact failure this gate exists to catch.
    active = schema.active()
    got_schema = os.environ.get("RAC_SCHEMA", "<unset>")
    if a.expect_schema and active["name"] != a.expect_schema:
        problems.append(f"schema in force is {active['name']!r} "
                        f"(RAC_SCHEMA={got_schema}), expected {a.expect_schema!r}")

    n = len(active["PATHOLOGIES"])
    if a.expect_classes and n != a.expect_classes:
        problems.append(f"{n} pathology classes in force, expected {a.expect_classes}")

    print(f"[check_env] arcct={pkg} RAC_SCHEMA={got_schema} "
          f"active={active['name']} classes={n} "
          f"regions={len(active['FINE_LABEL_NAMES']) - 1} "
          f"qformer_queries={os.environ.get('RAC_QFORMER_QUERIES', '<unset>')} "
          f"anatomy_qformer={os.environ.get('RAC_USE_ANATOMY_QFORMER', '<unset>')}")

    if problems:
        for p in problems:
            print(f"FATAL: {p}")
        return 78
    return 0


if __name__ == "__main__":
    sys.exit(main())

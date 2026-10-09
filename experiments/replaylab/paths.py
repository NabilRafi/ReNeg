"""Where the replay experiments read their inputs and write their outputs.

RENEG_EXPORTS   folder written by scripts/export_features.py          (default: exports/vit-b-16)
RENEG_EXP_OUT   results of the experiments: .jsonl rows, tables, logs (default: runs/experiments)
RENEG_TRACES    TANL traces built by experiments/01_build_traces.py   (default: <RENEG_EXP_OUT>/traces, ~3.5 GB)
"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORTS = os.path.abspath(os.environ.get("RENEG_EXPORTS", os.path.join(REPO, "exports", "vit-b-16")))
OUT = os.path.abspath(os.environ.get("RENEG_EXP_OUT", os.path.join(REPO, "runs", "experiments")))
TRACES = os.path.abspath(os.environ.get("RENEG_TRACES", os.path.join(OUT, "traces")))
RESULTS = os.path.join(REPO, "results")


def out(name: str) -> str:
    """Path of an output file in RENEG_EXP_OUT (the folder is created)."""
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)

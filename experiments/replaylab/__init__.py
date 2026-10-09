"""replaylab: CPU replays of ReNeg on exported CLIP features (the design tests, ablations and diagnostics).

TANL runs once per stream (``traces.build_traces``); ReNeg's variants are then replayed on the stored trace in
seconds to minutes. ``experiments/00_check_equivalence.py`` shows the replay equals ``reneg.ReNegTANL``.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:                                   # a clone without `pip install -e .`
    sys.path.insert(0, _SRC)

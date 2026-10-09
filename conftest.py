"""Make `src/` importable when the tests run from a clone without `pip install -e .`."""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

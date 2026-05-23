"""Thin wrapper over `src.ensemble.main` — computes pairwise + leave-one-out
ensemble across every model with cached scores in results/per_model/*."""
from __future__ import annotations
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ensemble import main  # noqa: E402

if __name__ == "__main__":
    main()

"""Shared verification utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def make_results_dir(*parts: str) -> Path:
    """Create a results directory under results/ and return the path."""
    base = Path("results")
    if parts:
        base = base.joinpath(*parts)
    base.mkdir(parents=True, exist_ok=True)
    return base


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with deterministic formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def seed_rng(seed: int) -> np.random.Generator:
    """Create deterministic NumPy random generator."""
    return np.random.default_rng(seed)

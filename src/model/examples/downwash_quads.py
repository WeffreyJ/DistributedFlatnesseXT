"""Extensibility hook: placeholder second model with same interfaces."""

from __future__ import annotations

import numpy as np

from src.model.coupling import active_edges, delta_accel
from src.model.dynamics import f


def output_y(x: np.ndarray, params) -> np.ndarray:
    """Example output map for alternative models."""
    return x[: params.N]


def pivot_jacobians(x: np.ndarray, params) -> dict[str, np.ndarray]:
    """Example pivot interface mirror."""
    n = params.N
    return {"Dx2_f1": np.eye(n), "Du_f2": np.eye(n)}


__all__ = ["f", "delta_accel", "active_edges", "output_y", "pivot_jacobians"]

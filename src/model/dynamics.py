"""Toy model dynamics for N coupled second-order agents."""

from __future__ import annotations

import numpy as np

from src.model.coupling import plant_delta_accel


def split_state(x: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    x1 = x[:N].copy()
    x2 = x[N : 2 * N].copy()
    return x1, x2


def pack_state(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.concatenate([x1, x2], axis=0)


def f(x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
    """Continuous-time plant dynamics xdot = f(x, u)."""
    plant_family = str(getattr(params, "plant_family", "tier0")).strip().lower()
    if plant_family not in {"tier0", "tier1", "tier2"}:
        raise ValueError(f"Unsupported system.plant_family={plant_family!r}")
    N = params.N
    x1, x2 = split_state(x, N)
    x1dot = x2
    x2dot = u + plant_delta_accel(x, params)
    return pack_state(x1dot, x2dot)

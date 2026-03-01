"""Toy model dynamics for N coupled second-order agents."""

from __future__ import annotations

import numpy as np

from src.model.coupling import delta_accel


def split_state(x: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    x1 = x[:N].copy()
    x2 = x[N : 2 * N].copy()
    return x1, x2


def pack_state(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.concatenate([x1, x2], axis=0)


def f(x: np.ndarray, u: np.ndarray, params) -> np.ndarray:
    """Continuous-time plant dynamics xdot = f(x, u)."""
    N = params.N
    x1, x2 = split_state(x, N)
    x1dot = x2
    x2dot = u + delta_accel(x, params)
    return pack_state(x1dot, x2dot)

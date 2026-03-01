"""Flatness recursion maps Phi and reconstruction Psi."""

from __future__ import annotations

import numpy as np

from src.model.coupling import delta_accel


# Implicit inverse map hook h_k^i (toy direct form, for extensibility)
def h_inverse_layer(i: int, k: int, zeta: dict[str, np.ndarray], x: np.ndarray, params) -> float:
    """Implicit inverse placeholder for future higher-order models."""
    if k == 1:
        return float(zeta["y"][i])
    if k == 2:
        return float(zeta["ydot"][i])
    if k == 3:
        d = delta_accel(x, params)
        return float(zeta["v"][i] - d[i])
    raise ValueError(f"Unsupported layer {k}")


def compute_phi(zeta: dict[str, np.ndarray], x: np.ndarray, pi: list[int], params) -> dict[tuple[int, int], float]:
    """Compute recursion entries Phi_k^i for k=1..3 (r=2 => r+1=3)."""
    n = params.N
    phi: dict[tuple[int, int], float] = {}

    for i in range(n):
        phi[(i, 1)] = h_inverse_layer(i, 1, zeta, x, params)
        phi[(i, 2)] = h_inverse_layer(i, 2, zeta, x, params)

    # Triangular evaluation order for extensibility (top to bottom in pi)
    d = delta_accel(x, params)
    for i in pi:
        phi[(i, 3)] = float(zeta["v"][i] - d[i])

    return phi


def psi(phi: dict[tuple[int, int], float], params) -> np.ndarray:
    """Reconstruction map Psi = Phi_{r+1}; returns stacked control input u."""
    return np.array([phi[(i, 3)] for i in range(params.N)], dtype=float)

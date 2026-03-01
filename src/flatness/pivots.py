"""Pivot Jacobian interfaces for Gate 3."""

from __future__ import annotations

import numpy as np


def pivot_jacobians(x: np.ndarray, params) -> dict[str, np.ndarray]:
    """Return toy pivot Jacobians D_{x2}f1 and D_u f2."""
    n = params.N
    return {
        "Dx2_f1": np.eye(n),
        "Du_f2": np.eye(n),
    }


def min_singular_pivot(x: np.ndarray, params) -> float:
    """Minimum singular value across pivot blocks."""
    piv = pivot_jacobians(x, params)
    mins = []
    for mat in piv.values():
        svals = np.linalg.svd(mat, compute_uv=False)
        mins.append(float(np.min(svals)))
    return float(min(mins)) if mins else 0.0

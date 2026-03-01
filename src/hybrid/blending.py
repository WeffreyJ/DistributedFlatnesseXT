"""Switch blending utilities."""

from __future__ import annotations

import numpy as np


def _smoothstep(z: float) -> float:
    zc = float(np.clip(z, 0.0, 1.0))
    return zc * zc * (3.0 - 2.0 * zc)


def blend(u_a: np.ndarray, u_b: np.ndarray, rho: float, eps: float, eta: float) -> np.ndarray:
    """Blend controls from old/new mode across band [eps, eps + eta]."""
    if eta <= 0.0:
        return u_b.copy()
    w = _smoothstep((rho - eps) / eta)
    return (1.0 - w) * u_a + w * u_b

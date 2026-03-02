"""Switch blending utilities."""

from __future__ import annotations

import numpy as np


def _smoothstep(z: float) -> float:
    zc = float(np.clip(z, 0.0, 1.0))
    return zc * zc * (3.0 - 2.0 * zc)


def blend(u_old: np.ndarray, u_new: np.ndarray, rho: float, eps: float, eta: float) -> np.ndarray:
    """Blend controls from old/new mode across band [eps, eps + eta]."""
    if rho <= eps:
        return u_old.copy()
    if eta <= 0.0 or rho >= eps + eta:
        return u_new.copy()
    w = _smoothstep((rho - eps) / eta)
    return (1.0 - w) * u_old + w * u_new


def blend_progress(u_old: np.ndarray, u_new: np.ndarray, progress: float) -> np.ndarray:
    """Blend controls with smooth time-progress weight in [0, 1]."""
    w = _smoothstep(progress)
    return (1.0 - w) * u_old + w * u_new

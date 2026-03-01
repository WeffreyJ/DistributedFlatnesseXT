"""Flat-space tracking controller."""

from __future__ import annotations

import numpy as np


def virtual_input(
    y: np.ndarray,
    ydot: np.ndarray,
    y_ref: np.ndarray,
    ydot_ref: np.ndarray,
    yddot_ref: np.ndarray,
    kp: float,
    kd: float,
) -> np.ndarray:
    """Compute desired flat acceleration v for PD tracking."""
    return yddot_ref - kp * (y - y_ref) - kd * (ydot - ydot_ref)

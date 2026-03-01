"""Reference trajectory scenarios."""

from __future__ import annotations

import numpy as np


def sinusoidal_crossing_reference(t: float, params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate y_ref, ydot_ref, yddot_ref with intentional altitude crossings."""
    base = np.array(params.base, dtype=float)
    amp = np.array(params.amp, dtype=float)
    freq = np.array(params.freq, dtype=float)
    phase = np.array(params.phase, dtype=float)

    omega_t = freq * t + phase
    y = base + amp * np.sin(omega_t)
    ydot = amp * freq * np.cos(omega_t)
    yddot = -amp * (freq**2) * np.sin(omega_t)
    return y, ydot, yddot

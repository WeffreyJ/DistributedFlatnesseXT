"""Hybrid ordering, hysteresis, and lockout logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.coupling import s_from_x


@dataclass
class OrderingState:
    """Hybrid mode memory for permutation switching."""

    current_pi: list[int] | None = None
    last_switch_t: float = -1.0e9
    last_switch_k: int = -10**9
    transition_active: bool = False
    transition_target_pi: list[int] | None = None
    transition_start_k: int = -1
    transition_M: int = 0


def s_metric(x: np.ndarray, N: int) -> np.ndarray:
    """Ordering metric s_i(x)."""
    return s_from_x(x, N)


def compute_pi(s: np.ndarray) -> list[int]:
    """Sort indices by descending s with index tie-break."""
    return sorted(range(len(s)), key=lambda i: (-float(s[i]), int(i)))


def rho_margin(s: np.ndarray, pi: list[int]) -> float:
    """Adjacent margin in current ordering."""
    if len(pi) < 2:
        return np.inf
    vals = [abs(float(s[pi[k]]) - float(s[pi[k + 1]])) for k in range(len(pi) - 1)]
    return float(min(vals)) if vals else np.inf


def rho_global(s: np.ndarray) -> float:
    """Minimum pairwise separation across all agent metrics."""
    if len(s) < 2:
        return np.inf
    m = np.inf
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            m = min(m, abs(float(s[i]) - float(s[j])))
    return float(m)


def step_mode(
    x: np.ndarray,
    t: float,
    k: int,
    state: OrderingState,
    epsilon: float,
    lockout_sec: float,
    lockout_samples: int,
    s_override: np.ndarray | None = None,
) -> tuple[list[int], bool]:
    """Update permutation mode with hysteresis + lockout.

    TODO: formalize mixed time/sample lockout semantics for variable-step solvers.
    TODO: ensure ordering remains computed from one globally-consistent snapshot in distributed setting.
    """
    s = s_override if s_override is not None else s_metric(x, N=len(x) // 2)
    new_pi = compute_pi(s)

    if state.current_pi is None:
        state.current_pi = new_pi
        state.last_switch_t = t
        state.last_switch_k = k
        return state.current_pi, True

    lockout_time_active = (t - state.last_switch_t) < lockout_sec
    lockout_sample_active = (k - state.last_switch_k) < lockout_samples
    if lockout_time_active or lockout_sample_active:
        return state.current_pi, False

    if new_pi != state.current_pi and rho_margin(s, new_pi) >= epsilon:
        state.current_pi = new_pi
        state.last_switch_t = t
        state.last_switch_k = k
        return state.current_pi, True
    return state.current_pi, False

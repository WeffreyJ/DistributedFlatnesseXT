"""Order-dependent coupling terms Delta and active edge computation."""

from __future__ import annotations

import numpy as np


def s_from_x(x: np.ndarray, N: int) -> np.ndarray:
    """Extract ordering metric s_i(x) = x1_i."""
    return x[:N].copy()


def s_metric(x: np.ndarray, params) -> np.ndarray:
    """Ordering metric adapter for coupling interfaces."""
    return s_from_x(x, params.N)


def _resolve_gamma_edge(params, mode: dict | None = None) -> float:
    gamma_edge = getattr(params, "gamma_edge", None)
    if gamma_edge is not None:
        return float(gamma_edge)
    if mode is not None and "epsilon" in mode:
        return max(float(params.gamma), float(mode["epsilon"]) + 1.0e-6)
    return float(params.gamma)


def active_edges(
    x: np.ndarray,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return active edges (j, i) meaning j -> i based on altitude separation."""
    s_vec = s_from_x(x, params.N) if s is None else np.asarray(s, dtype=float)
    coupling_mode = getattr(params, "coupling_mode", "upstream_u")
    gamma_edge = _resolve_gamma_edge(params, mode=mode)
    edges: list[tuple[int, int]] = []

    if coupling_mode == "wake_surrogate":
        wake_Rx = float(getattr(params, "wake_Rx", 6.0))
        for j in range(params.N):
            for i in range(params.N):
                if j == i:
                    continue
                dx = float(s_vec[j] - s_vec[i])
                if dx > gamma_edge and dx < wake_Rx:
                    edges.append((j, i))
        return edges

    for j in range(params.N):
        for i in range(params.N):
            if j == i:
                continue
            ds = s_vec[j] - s_vec[i]
            if ds > gamma_edge and ds < params.R_coup:
                edges.append((j, i))
    return edges


def delta_accel(
    x: np.ndarray,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> np.ndarray:
    """Compute acceleration-layer coupling Delta_2^i for each agent."""
    s_vec = s_from_x(x, params.N) if s is None else np.asarray(s, dtype=float)
    coupling_mode = getattr(params, "coupling_mode", "upstream_u")

    if coupling_mode == "wake_surrogate":
        k_wake = float(getattr(params, "k_wake", 0.35))
        L = float(getattr(params, "wake_decay_L", 2.0))
        wake_Rx = float(getattr(params, "wake_Rx", 6.0))
        gamma_edge = _resolve_gamma_edge(params, mode=mode)
        d = np.zeros(params.N, dtype=float)
        for (j, i) in active_edges(x, params, mode=mode, s=s_vec):
            dx = float(s_vec[j] - s_vec[i])
            if dx <= gamma_edge or dx >= wake_Rx:
                continue
            d[i] += -k_wake * float(np.exp(-dx / max(L, 1.0e-9)))
        return d

    d = np.zeros(params.N, dtype=float)
    for (j, i) in active_edges(x, params, mode=mode, s=s_vec):
        d[i] += params.a_downwash * (s_vec[j] - s_vec[i])
    return d


def delta(i: int, k_minus_1: int, x: np.ndarray, params, mode: dict | None = None) -> float:
    """Generic Delta_k^i interface. Only acceleration layer (k=2) is nonzero in toy model."""
    if k_minus_1 != 1:
        return 0.0
    return float(delta_accel(x, params, mode=mode)[i])

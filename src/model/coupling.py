"""Order-dependent coupling terms Delta and active edge computation."""

from __future__ import annotations

import numpy as np

from src.model.coupling_tier1 import (
    active_edges_tier1,
    delta_accel_tier1,
    is_active_edge_pair_tier1,
    pairwise_coupling_weight_tier1,
)
from src.model.coupling_tier2 import (
    active_edges_tier2_nominal,
    delta_accel_tier2_nominal,
    delta_accel_tier2_plant,
    is_active_edge_pair_tier2_nominal,
    pairwise_coupling_weight_tier2_nominal,
)
from src.model.residual_tier2 import residual_delta_accel_tier2


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


def _resolve_plant_family(params) -> str:
    return str(getattr(params, "plant_family", "tier0")).strip().lower()


def is_active_edge_pair(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> bool:
    """Return whether the directed pair leader->follower is active."""
    plant_family = _resolve_plant_family(params)
    if plant_family == "tier1":
        return is_active_edge_pair_tier1(x, leader, follower, params, s=s)
    if plant_family == "tier2":
        return is_active_edge_pair_tier2_nominal(x, leader, follower, params, s=s)
    if leader == follower:
        return False

    s_vec = s_from_x(x, params.N) if s is None else np.asarray(s, dtype=float)
    coupling_mode = getattr(params, "coupling_mode", "upstream_u")
    gamma_edge = _resolve_gamma_edge(params, mode=mode)
    ds = float(s_vec[leader] - s_vec[follower])

    if coupling_mode == "wake_surrogate":
        wake_Rx = float(getattr(params, "wake_Rx", 6.0))
        return bool(ds > gamma_edge and ds < wake_Rx)
    return bool(ds > gamma_edge and ds < float(getattr(params, "R_coup", np.inf)))


def pairwise_coupling_weight(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> float:
    """Return the directed pairwise coupling contribution leader->follower."""
    plant_family = _resolve_plant_family(params)
    if plant_family == "tier1":
        return pairwise_coupling_weight_tier1(x, leader, follower, params, s=s)
    if plant_family == "tier2":
        return pairwise_coupling_weight_tier2_nominal(x, leader, follower, params, s=s)
    s_vec = s_from_x(x, params.N) if s is None else np.asarray(s, dtype=float)
    if not is_active_edge_pair(x, leader, follower, params, mode=mode, s=s_vec):
        return 0.0

    coupling_mode = getattr(params, "coupling_mode", "upstream_u")
    ds = float(s_vec[leader] - s_vec[follower])

    if coupling_mode == "wake_surrogate":
        k_wake = float(getattr(params, "k_wake", 0.35))
        L = float(getattr(params, "wake_decay_L", 2.0))
        return -k_wake * float(np.exp(-ds / max(L, 1.0e-9)))

    return float(params.a_downwash) * ds


def active_edges(
    x: np.ndarray,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return active edges (j, i) meaning j -> i based on altitude separation."""
    plant_family = _resolve_plant_family(params)
    if plant_family == "tier1":
        return active_edges_tier1(x, params, s=s)
    if plant_family == "tier2":
        return active_edges_tier2_nominal(x, params, s=s)
    s_vec = s_from_x(x, params.N) if s is None else np.asarray(s, dtype=float)
    coupling_mode = getattr(params, "coupling_mode", "upstream_u")
    gamma_edge = _resolve_gamma_edge(params, mode=mode)
    edges: list[tuple[int, int]] = []

    if coupling_mode == "wake_surrogate":
        wake_Rx = float(getattr(params, "wake_Rx", 6.0))
        for j in range(params.N):
            for i in range(params.N):
                if is_active_edge_pair(x, j, i, params, mode=mode, s=s_vec):
                    edges.append((j, i))
        return edges

    for j in range(params.N):
        for i in range(params.N):
            if is_active_edge_pair(x, j, i, params, mode=mode, s=s_vec):
                edges.append((j, i))
    return edges


def delta_accel(
    x: np.ndarray,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> np.ndarray:
    """Compute acceleration-layer coupling Delta_2^i for each agent."""
    plant_family = _resolve_plant_family(params)
    if plant_family == "tier1":
        return delta_accel_tier1(x, params, s=s)
    if plant_family == "tier2":
        return delta_accel_tier2_nominal(x, params, s=s)
    s_vec = s_from_x(x, params.N) if s is None else np.asarray(s, dtype=float)
    coupling_mode = getattr(params, "coupling_mode", "upstream_u")

    if coupling_mode == "wake_surrogate":
        d = np.zeros(params.N, dtype=float)
        for (j, i) in active_edges(x, params, mode=mode, s=s_vec):
            d[i] += pairwise_coupling_weight(x, j, i, params, mode=mode, s=s_vec)
        return d

    d = np.zeros(params.N, dtype=float)
    for (j, i) in active_edges(x, params, mode=mode, s=s_vec):
        d[i] += pairwise_coupling_weight(x, j, i, params, mode=mode, s=s_vec)
    return d


def delta(i: int, k_minus_1: int, x: np.ndarray, params, mode: dict | None = None) -> float:
    """Generic Delta_k^i interface. Only acceleration layer (k=2) is nonzero in toy model."""
    if k_minus_1 != 1:
        return 0.0
    return float(delta_accel(x, params, mode=mode)[i])


def residual_delta_accel(
    x: np.ndarray,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> np.ndarray:
    """Return plant-side residual interaction only."""
    del mode
    if _resolve_plant_family(params) == "tier2":
        return residual_delta_accel_tier2(x, params, s=s)
    return np.zeros(int(params.N), dtype=float)


def plant_delta_accel(
    x: np.ndarray,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> np.ndarray:
    """Return the interaction term actually applied in plant dynamics."""
    plant_family = _resolve_plant_family(params)
    if plant_family == "tier2":
        return delta_accel_tier2_plant(x, params, s=s)
    return delta_accel(x, params, mode=mode, s=s)

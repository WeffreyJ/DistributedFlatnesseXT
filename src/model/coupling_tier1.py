"""Tier-1 geometry-aware coupling kernel and edge logic."""

from __future__ import annotations

import numpy as np

from src.model.wake_geometry import pairwise_geometry, transverse_metric_squared


def _tier1_cfg(params):
    tier1 = getattr(params, "tier1", None)
    if tier1 is None:
        raise ValueError("system.tier1 configuration is required for plant_family='tier1'")
    return tier1


def _gamma_edge(params) -> float:
    return float(getattr(params, "gamma_edge", getattr(params, "gamma", 0.0)))


def is_active_edge_pair_tier1(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    *,
    s: np.ndarray | None = None,
) -> bool:
    """Return whether a directed Tier-1 interaction is active."""
    if int(leader) == int(follower):
        return False
    cfg = _tier1_cfg(params)
    geom = pairwise_geometry(x, leader, follower, params, s=s)
    ds = float(geom["ds"])
    if ds <= _gamma_edge(params) or ds >= float(getattr(params, "wake_Rx", np.inf)):
        return False
    max_transverse = float(getattr(cfg.edge, "transverse_radius", np.inf))
    return bool(float(geom["transverse_radius"]) <= max_transverse)


def pairwise_coupling_weight_tier1(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    *,
    s: np.ndarray | None = None,
) -> float:
    """Return one directed Tier-1 pairwise contribution."""
    if not is_active_edge_pair_tier1(x, leader, follower, params, s=s):
        return 0.0
    cfg = _tier1_cfg(params)
    geom = pairwise_geometry(x, leader, follower, params, s=s)
    ds_eff = max(float(geom["ds"]) - _gamma_edge(params), 0.0)
    gain = float(getattr(cfg.kernel, "gain", 0.35))
    longitudinal_decay = float(getattr(cfg.kernel, "longitudinal_decay", 2.0))
    r_core = float(getattr(cfg.kernel, "r_core", 0.5))
    power_p = float(getattr(cfg.kernel, "power_p", 1.0))
    transverse_sq = transverse_metric_squared(float(geom["dl"]), float(geom["dh"]), params)
    regularized = max(r_core * r_core + transverse_sq, 1.0e-9)
    return -gain * float(np.exp(-ds_eff / max(longitudinal_decay, 1.0e-9))) / float(regularized ** (0.5 * power_p))


def active_edges_tier1(
    x: np.ndarray,
    params,
    *,
    s: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for leader in range(int(params.N)):
        for follower in range(int(params.N)):
            if is_active_edge_pair_tier1(x, leader, follower, params, s=s):
                edges.append((int(leader), int(follower)))
    return edges


def delta_accel_tier1(
    x: np.ndarray,
    params,
    *,
    s: np.ndarray | None = None,
) -> np.ndarray:
    d = np.zeros(int(params.N), dtype=float)
    for leader, follower in active_edges_tier1(x, params, s=s):
        d[int(follower)] += pairwise_coupling_weight_tier1(x, leader, follower, params, s=s)
    return d

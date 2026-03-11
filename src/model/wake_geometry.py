"""Tier-1 geometry helpers for pairwise wake/interference kernels."""

from __future__ import annotations

import numpy as np


def _tier1_cfg(params):
    tier1 = getattr(params, "tier1", None)
    if tier1 is None:
        raise ValueError("system.tier1 configuration is required for plant_family='tier1'")
    return tier1


def lateral_offsets(params) -> np.ndarray:
    cfg = _tier1_cfg(params)
    vals = np.asarray(getattr(cfg.geometry, "lateral_offsets", []), dtype=float)
    if vals.shape != (int(params.N),):
        raise ValueError(f"tier1.geometry.lateral_offsets must have shape {(int(params.N),)}, got {vals.shape}")
    return vals


def vertical_offsets(params) -> np.ndarray:
    cfg = _tier1_cfg(params)
    vals = np.asarray(getattr(cfg.geometry, "vertical_offsets", []), dtype=float)
    if vals.shape != (int(params.N),):
        raise ValueError(f"tier1.geometry.vertical_offsets must have shape {(int(params.N),)}, got {vals.shape}")
    return vals


def longitudinal_positions(x: np.ndarray, params) -> np.ndarray:
    return np.asarray(x[: int(params.N)], dtype=float).copy()


def pairwise_geometry(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    *,
    s: np.ndarray | None = None,
) -> dict[str, float]:
    """Return pairwise longitudinal/lateral/vertical separations for one directed pair."""
    s_vec = longitudinal_positions(x, params) if s is None else np.asarray(s, dtype=float)
    lat = lateral_offsets(params)
    vert = vertical_offsets(params)
    ds = float(s_vec[int(leader)] - s_vec[int(follower)])
    dl = float(lat[int(leader)] - lat[int(follower)])
    dh = float(vert[int(leader)] - vert[int(follower)])
    return {
        "ds": ds,
        "dl": dl,
        "dh": dh,
        "transverse_radius": float(np.sqrt(dl * dl + dh * dh)),
    }


def transverse_metric_squared(dl: float, dh: float, params) -> float:
    """Return the regularized transverse metric used by the Tier-1 kernel."""
    cfg = _tier1_cfg(params)
    alpha_h = float(getattr(cfg.kernel, "alpha_h", 1.0))
    return float(dl * dl + alpha_h * dh * dh)

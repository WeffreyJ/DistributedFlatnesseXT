"""Tier-2 nominal-plus-residual coupling scaffold."""

from __future__ import annotations

import numpy as np

from src.model.coupling_tier1 import (
    active_edges_tier1,
    delta_accel_tier1,
    is_active_edge_pair_tier1,
    pairwise_coupling_weight_tier1,
)
from src.model.residual_tier2 import residual_delta_accel_tier2


def is_active_edge_pair_tier2_nominal(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    *,
    s: np.ndarray | None = None,
) -> bool:
    """Tier-2 nominal directed edge rule.

    The reduction witness phase defines Tier-2 nominal interaction as an exact
    reduction to the Tier-1 nominal branch.
    """
    return is_active_edge_pair_tier1(x, leader, follower, params, s=s)


def pairwise_coupling_weight_tier2_nominal(
    x: np.ndarray,
    leader: int,
    follower: int,
    params,
    *,
    s: np.ndarray | None = None,
) -> float:
    """Tier-2 nominal pairwise contribution."""
    return pairwise_coupling_weight_tier1(x, leader, follower, params, s=s)


def active_edges_tier2_nominal(
    x: np.ndarray,
    params,
    *,
    s: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Tier-2 nominal active-edge pattern."""
    return active_edges_tier1(x, params, s=s)


def delta_accel_tier2_nominal(
    x: np.ndarray,
    params,
    *,
    s: np.ndarray | None = None,
) -> np.ndarray:
    """Tier-2 nominal interaction used by the controller-side evaluator."""
    return delta_accel_tier1(x, params, s=s)


def delta_accel_tier2_plant(
    x: np.ndarray,
    params,
    *,
    s: np.ndarray | None = None,
) -> np.ndarray:
    """Tier-2 plant interaction: nominal branch plus structured residual."""
    nominal = delta_accel_tier2_nominal(x, params, s=s)
    residual = residual_delta_accel_tier2(x, params, s=s)
    return np.asarray(nominal, dtype=float) + np.asarray(residual, dtype=float)

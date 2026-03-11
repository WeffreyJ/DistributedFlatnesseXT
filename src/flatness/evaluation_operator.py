"""Explicit evaluation-operator helpers for mode-dependent reconstruction."""

from __future__ import annotations

import numpy as np

from src.model.coupling import delta_accel, pairwise_coupling_weight


def _resolve_params(cfg):
    return getattr(cfg, "system", cfg)


def _resolve_evaluation_cfg(cfg):
    return getattr(cfg, "evaluation", None)


def supports_explicit_evaluator(cfg) -> bool:
    """Return whether evaluator output is a state-only per-agent vector."""
    params = _resolve_params(cfg)
    return str(getattr(params, "coupling_mode", "upstream_u")) != "upstream_u"


def get_evaluator_mode(cfg) -> str:
    """Return configured evaluator mode, with backward-compatible fallback."""
    eval_cfg = _resolve_evaluation_cfg(cfg)
    if eval_cfg is not None and getattr(eval_cfg, "mode", None) is not None:
        return str(eval_cfg.mode)

    params = _resolve_params(cfg)
    wake_mode = str(getattr(params, "wake_surrogate_mode", "pi_upstream"))
    if wake_mode == "physical_all_edges":
        return "full"
    return "upstream_truncated"


def get_local_window_agents(cfg) -> int:
    eval_cfg = _resolve_evaluation_cfg(cfg)
    if eval_cfg is not None and getattr(eval_cfg, "local_window_agents", None) is not None:
        return int(eval_cfg.local_window_agents)
    return 1


def inverse_permutation(pi: list[int]) -> list[int]:
    inv = np.zeros(len(pi), dtype=int)
    for rank, agent in enumerate(pi):
        inv[int(agent)] = int(rank)
    return [int(v) for v in inv]


def agent_upstream_set(agent: int, pi: list[int]) -> list[int]:
    """Return the agents ranked ahead of the given agent in pi."""
    try:
        rank = [int(v) for v in pi].index(int(agent))
    except ValueError as exc:
        raise ValueError(f"Agent {agent} is not present in permutation {pi}") from exc
    return [int(v) for v in pi[:rank]]


def pairwise_wake_weight(x: np.ndarray, leader: int, follower: int, cfg) -> float:
    """Return the directed pairwise contribution for the selected plant family."""
    params = _resolve_params(cfg)
    return float(pairwise_coupling_weight(x, leader, follower, params))


def _compute_truncated_evaluator(
    x: np.ndarray,
    pi: list[int],
    cfg,
    *,
    local_window_agents: int | None = None,
) -> np.ndarray:
    params = _resolve_params(cfg)
    if not supports_explicit_evaluator(cfg):
        raise ValueError("Explicit evaluator is unsupported for coupling_mode='upstream_u'.")

    n = int(params.N)
    window = None if local_window_agents is None else max(int(local_window_agents), 0)
    rank = inverse_permutation(pi)
    out = np.zeros(n, dtype=float)

    for agent in pi:
        allowed = agent_upstream_set(int(agent), pi)
        if window is not None:
            allowed = [j for j in allowed if (rank[int(agent)] - rank[int(j)]) <= window]
        for leader in allowed:
            out[int(agent)] += pairwise_wake_weight(x, int(leader), int(agent), params)
    return out


def compute_evaluator_upstream_truncated(x: np.ndarray, pi: list[int], cfg) -> np.ndarray:
    """Return E_pi(x) using the current upstream-truncated inclusion rule."""
    return _compute_truncated_evaluator(x, pi, cfg, local_window_agents=None)


def compute_evaluator_full(x: np.ndarray, pi: list[int], cfg) -> np.ndarray:
    """Return permutation-invariant all-active-edge evaluation."""
    del pi
    params = _resolve_params(cfg)
    if not supports_explicit_evaluator(cfg):
        raise ValueError("Explicit evaluator is unsupported for coupling_mode='upstream_u'.")
    return np.asarray(delta_accel(x, params), dtype=float)


def compute_evaluator_local_window(
    x: np.ndarray,
    pi: list[int],
    cfg,
    local_window_agents: int = 1,
) -> np.ndarray:
    """Return an upstream evaluator restricted to nearby upstream ranks."""
    return _compute_truncated_evaluator(x, pi, cfg, local_window_agents=local_window_agents)


def compute_evaluator(
    x: np.ndarray,
    pi: list[int],
    cfg,
    *,
    mode: str | None = None,
) -> np.ndarray:
    """Return evaluator output E_pi(x) as a per-agent correction vector."""
    resolved_mode = str(mode if mode is not None else get_evaluator_mode(cfg)).strip().lower()

    if resolved_mode == "upstream_truncated":
        return compute_evaluator_upstream_truncated(x, pi, cfg)
    if resolved_mode == "full":
        return compute_evaluator_full(x, pi, cfg)
    if resolved_mode == "local_window":
        return compute_evaluator_local_window(x, pi, cfg, local_window_agents=get_local_window_agents(cfg))
    raise ValueError(f"Unsupported evaluator mode: {resolved_mode!r}")

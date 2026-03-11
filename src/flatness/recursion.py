"""Flatness recursion maps Phi and reconstruction Psi."""

from __future__ import annotations

import numpy as np

from src.flatness.evaluation_operator import compute_evaluator
from src.model.coupling import delta_accel

Phi = dict[tuple[int, int], float]


# Implicit inverse map hook h_k^i (toy direct form, for extensibility)
def h_inverse_layer(i: int, k: int, zeta: dict[str, np.ndarray], x: np.ndarray, params) -> float:
    """Implicit inverse placeholder for future higher-order models."""
    if k == 1:
        return float(zeta["y"][i])
    if k == 2:
        return float(zeta["ydot"][i])
    if k == 3:
        d = delta_accel(x, params)
        return float(zeta["v"][i] - d[i])
    raise ValueError(f"Unsupported layer {k}")


def build_phi(
    x: np.ndarray,
    zeta: dict[str, np.ndarray],
    pi: list[int],
    sys,
    params=None,
    *,
    evaluator_output: np.ndarray | None = None,
) -> Phi:
    """Build recursion entries Phi_k^i for k=1..3 (r=2 => r+1=3).

    This is the snapshot-based entrypoint used by simulation and sensitivity gates.
    `evaluator_output` is the per-agent correction E_pi(x) used by explicit
    evaluator-based reconstruction modes.
    """
    p = params if params is not None else sys
    n = int(p.N)
    phi: Phi = {}

    for i in range(n):
        phi[(i, 1)] = h_inverse_layer(i, 1, zeta, x, p)
        phi[(i, 2)] = h_inverse_layer(i, 2, zeta, x, p)

    # Triangular evaluation order for extensibility (top to bottom in pi).
    coupling_mode = getattr(p, "coupling_mode", "upstream_u")
    if coupling_mode == "upstream_u":
        u_partial = np.zeros(n, dtype=float)
        k_u = float(getattr(p, "k_u", 0.3))
        for idx_in_order, agent in enumerate(pi):
            upstream = pi[:idx_in_order]
            delta_i = k_u * float(np.sum(u_partial[upstream])) if upstream else 0.0
            u_partial[agent] = float(zeta["v"][agent] - delta_i)
            phi[(agent, 3)] = float(u_partial[agent])
    elif coupling_mode in {"downwash", "wake_surrogate"}:
        d = np.asarray(
            evaluator_output if evaluator_output is not None else compute_evaluator(x=x, pi=pi, cfg=sys),
            dtype=float,
        )
        if d.shape != (n,):
            raise ValueError(f"evaluator_output must have shape {(n,)}, got {d.shape}")
        for agent in pi:
            phi[(agent, 3)] = float(zeta["v"][agent] - d[agent])
    else:
        raise ValueError(f"Unsupported coupling_mode={coupling_mode!r}")

    return phi


def compute_phi(zeta: dict[str, np.ndarray], x: np.ndarray, pi: list[int], params) -> Phi:
    """Backward-compatible wrapper for existing callers."""
    return build_phi(x=x, zeta=zeta, pi=pi, sys=params, params=params)


def psi(phi: Phi, params) -> np.ndarray:
    """Reconstruction map Psi = Phi_{r+1}; returns stacked control input u."""
    return np.array([phi[(i, 3)] for i in range(params.N)], dtype=float)

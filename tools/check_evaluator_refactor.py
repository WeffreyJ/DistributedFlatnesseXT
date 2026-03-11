"""Lightweight smoke checks for the explicit evaluator refactor."""

from __future__ import annotations

import argparse

import numpy as np

from src.config import load_config
from src.flatness.evaluation_operator import compute_evaluator
from src.flatness.recursion import build_phi, psi
from src.hybrid.ordering import compute_pi


def _legacy_upstream_control(
    x: np.ndarray,
    zeta: dict[str, np.ndarray],
    pi: list[int],
    params,
) -> np.ndarray:
    """Reference the pre-refactor wake-surrogate pi-upstream reconstruction."""
    n = int(params.N)
    s = np.asarray(x[:n], dtype=float)
    out = np.zeros(n, dtype=float)
    k_wake = float(getattr(params, "k_wake", 0.35))
    L = float(getattr(params, "wake_decay_L", 2.0))
    wake_Rx = float(getattr(params, "wake_Rx", 6.0))
    gamma_edge = float(getattr(params, "gamma_edge", getattr(params, "gamma", 0.0)))

    for idx_in_order, agent in enumerate(pi):
        delta_i = 0.0
        for leader in pi[:idx_in_order]:
            dx = float(s[int(leader)] - s[int(agent)])
            if dx > gamma_edge and dx < wake_Rx:
                delta_i += -k_wake * float(np.exp(-dx / max(L, 1.0e-9)))
        out[int(agent)] = float(zeta["v"][int(agent)] - delta_i)
    return out


def run_check(config_path: str) -> None:
    cfg = load_config(config_path)
    n = int(cfg.system.N)

    x1 = np.asarray([0.55, 0.18, -0.27], dtype=float)
    x2 = np.asarray([0.04, -0.02, 0.01], dtype=float)
    x = np.concatenate([x1, x2], axis=0)
    zeta = {
        "y": x1.copy(),
        "ydot": x2.copy(),
        "v": np.asarray([0.3, -0.1, 0.2], dtype=float),
    }
    pi = compute_pi(x1)

    E_up = compute_evaluator(x=x, pi=pi, cfg=cfg, mode="upstream_truncated")
    E_up_again = compute_evaluator(x=x, pi=pi, cfg=cfg, mode="upstream_truncated")
    E_full = compute_evaluator(x=x, pi=pi, cfg=cfg, mode="full")
    E_full_again = compute_evaluator(x=x, pi=pi, cfg=cfg, mode="full")

    assert E_up.shape == (n,), f"upstream_truncated shape mismatch: {E_up.shape}"
    assert E_full.shape == (n,), f"full shape mismatch: {E_full.shape}"
    assert np.allclose(E_up, E_up_again), "upstream_truncated evaluator is not deterministic"
    assert np.allclose(E_full, E_full_again), "full evaluator is not deterministic"

    phi = build_phi(x=x, zeta=zeta, pi=pi, sys=cfg, params=cfg.system, evaluator_output=E_up)
    u_refactor = psi(phi, cfg.system)
    u_legacy = _legacy_upstream_control(x=x, zeta=zeta, pi=pi, params=cfg.system)
    if not np.allclose(u_refactor, u_legacy, atol=1.0e-12, rtol=1.0e-12):
        raise AssertionError(f"legacy mismatch: refactor={u_refactor}, legacy={u_legacy}")

    # Witness state: agent 0 is physically ahead of agent 1, but pi ranks agent 1 first.
    # Full evaluation keeps the active 0->1 contribution; upstream truncation excludes it.
    x1_witness = np.asarray([0.9, 0.1, -0.2], dtype=float)
    x2_witness = np.asarray([0.0, 0.0, 0.0], dtype=float)
    x_witness = np.concatenate([x1_witness, x2_witness], axis=0)
    zeta_witness = {
        "y": x1_witness.copy(),
        "ydot": x2_witness.copy(),
        "v": np.asarray([0.15, -0.05, 0.1], dtype=float),
    }
    pi_witness = [1, 0, 2]

    E_up_witness = compute_evaluator(x=x_witness, pi=pi_witness, cfg=cfg, mode="upstream_truncated")
    E_full_witness = compute_evaluator(x=x_witness, pi=pi_witness, cfg=cfg, mode="full")
    u_up_witness = psi(
        build_phi(
            x=x_witness,
            zeta=zeta_witness,
            pi=pi_witness,
            sys=cfg,
            params=cfg.system,
            evaluator_output=E_up_witness,
        ),
        cfg.system,
    )
    u_full_witness = psi(
        build_phi(
            x=x_witness,
            zeta=zeta_witness,
            pi=pi_witness,
            sys=cfg,
            params=cfg.system,
            evaluator_output=E_full_witness,
        ),
        cfg.system,
    )
    if np.allclose(E_up_witness, E_full_witness, atol=1.0e-12, rtol=1.0e-12) and np.allclose(
        u_up_witness,
        u_full_witness,
        atol=1.0e-12,
        rtol=1.0e-12,
    ):
        raise AssertionError("Witness state did not distinguish full and upstream_truncated evaluators")

    print("Evaluator refactor smoke check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluator refactor smoke check")
    parser.add_argument("--config", default="configs/system.yaml")
    args = parser.parse_args()
    run_check(args.config)


if __name__ == "__main__":
    main()

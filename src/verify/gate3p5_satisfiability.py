"""Gate 3.5: Satisfiability inequality for dwell-time stability condition."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from src.config import load_config
from src.verify.utils import dump_json, make_results_dir


def run_gate35(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gate3p5")

    kp = float(cfg.controller.kp)
    kd = float(cfg.controller.kd)
    A = np.array([[0.0, 1.0], [-kp, -kd]], dtype=float)

    # scipy solves A X + X A^T = Q, so use Q = -I for A^T P + P A = -I.
    P = solve_continuous_lyapunov(A.T, -np.eye(2))
    eig = np.linalg.eigvalsh(P)
    c1 = float(np.min(eig))
    c2 = float(np.max(eig))
    alpha = 1.0
    mu = 1.0

    tau_d = float(cfg.ordering.lockout_sec)
    lhs = float(mu * np.exp(-(alpha / c2) * tau_d))
    passed = lhs < 1.0

    tau_grid = np.linspace(0.0, 2.0, 200)
    lhs_grid = mu * np.exp(-(alpha / c2) * tau_grid)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tau_grid, lhs_grid, label=r"$\mu e^{-\alpha\tau_d/c_2}$")
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1.0, label="threshold=1")
    ax.axvline(tau_d, color="k", linestyle=":", linewidth=1.0, label=f"tau_d={tau_d:.2f}")
    ax.set_xlabel("tau_d [s]")
    ax.set_ylabel("LHS")
    ax.set_title("Gate 3.5 Satisfiability")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "satisfiability_vs_tau_d.png", dpi=150)
    plt.close(fig)

    payload = {
        "gate": "Gate 3.5",
        "A": A.tolist(),
        "P": P.tolist(),
        "c1": c1,
        "c2": c2,
        "alpha": alpha,
        "mu": mu,
        "tau_d": tau_d,
        "lhs": lhs,
        "status": "PASS" if passed else "FAIL",
    }
    out_path = out_dir / "stability_margin_report.json"
    dump_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3.5 satisfiability")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gate35(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

"""Gate 3: Compute constants table (d_underline, c_overline, L_Psi, J)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.flatness.pivots import min_singular_pivot
from src.model.coupling import delta_accel
from src.verify.utils import dump_json, make_results_dir


def _jump_stats(u: np.ndarray, switch_steps: list[int]) -> list[float]:
    jumps: list[float] = []
    for k in switch_steps:
        if 1 <= k < len(u):
            jumps.append(float(np.linalg.norm(u[k] - u[k - 1])))
    return jumps


def run_gate3(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gate3")

    n = cfg.system.N
    x0 = np.concatenate(
        [
            np.array(cfg.reference.base, dtype=float),
            np.zeros(n, dtype=float),
        ]
    )

    sim_off = simulate_closed_loop(
        cfg,
        x0=x0,
        horizon=float(cfg.gate3.run_horizon),
        options=SimOptions(blending_on=False, noise_delta=0.0, seed=int(cfg.seed) + 300),
    )
    sim_on = simulate_closed_loop(
        cfg,
        x0=x0,
        horizon=float(cfg.gate3.run_horizon),
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(cfg.seed) + 301),
    )

    d_underline = min_singular_pivot(sim_off["x"][0], cfg.system)
    c_overline = float(2.0 * cfg.system.a_downwash * (cfg.system.N - 1))
    L_psi = float(1.0 + c_overline)

    jumps_off = _jump_stats(np.asarray(sim_off["u"]), list(sim_off["switch_steps"]))
    jumps_on = _jump_stats(np.asarray(sim_on["u"]), list(sim_on["switch_steps"]))
    J_off = float(max(jumps_off)) if jumps_off else 0.0
    J_on = float(max(jumps_on)) if jumps_on else 0.0

    payload = {
        "gate": "Gate 3",
        "d_underline": float(d_underline),
        "c_overline": c_overline,
        "L_Psi": L_psi,
        "J_without_blending": J_off,
        "J_with_blending": J_on,
        "switch_count_without_blending": len(sim_off["switch_steps"]),
        "switch_count_with_blending": len(sim_on["switch_steps"]),
    }
    dump_json(out_dir / "constants_table.json", payload)

    x_hist = np.asarray(sim_off["x"])
    rho = np.asarray(sim_off["rho"])
    mism = np.array([np.linalg.norm(delta_accel(x_hist[k], cfg.system)) for k in range(len(rho))])
    m = (rho > 1e-4) & (mism > 1e-8)
    if np.any(m):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(rho[m], mism[m], ".", alpha=0.6)
        ax.set_xlabel("rho")
        ax.set_ylabel("||mismatch||")
        ax.set_title("Mismatch vs rho")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "mismatch_vs_rho_loglog.png", dpi=150)
        plt.close(fig)

    all_jumps = np.array(jumps_off + jumps_on, dtype=float)
    if all_jumps.size > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(all_jumps, bins=min(25, all_jumps.size))
        ax.set_title("Control Jump Histogram")
        ax.set_xlabel("||Delta u|| at switch")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "jump_histogram.png", dpi=150)
        plt.close(fig)

    u = np.asarray(sim_off["u"])
    t = np.asarray(sim_off["t"])[:-1]
    udot = np.gradient(u, axis=0) / float(cfg.system.dt)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for i in range(n):
        axes[0].plot(t, u[:, i], label=f"u{i}")
        axes[1].plot(t, udot[:, i], label=f"udot{i}")
    axes[0].set_ylabel("u")
    axes[1].set_ylabel("udot")
    axes[1].set_xlabel("t [s]")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "u_udot_timeseries.png", dpi=150)
    plt.close(fig)

    return out_dir / "constants_table.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3 constants")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gate3(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

"""Gate 3: Compute constants table (d_underline, c_overline, L_Psi, J)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.flatness.pivots import min_singular_pivot
from src.model.coupling import delta_accel
from src.verify.utils import dump_json, make_results_dir


def _jump_stats(u: np.ndarray, switch_steps: list[int]) -> tuple[list[float], list[int]]:
    jumps: list[float] = []
    valid_steps: list[int] = []
    for k in switch_steps:
        if 1 <= k < len(u):
            jumps.append(float(np.linalg.norm(u[k] - u[k - 1])))
            valid_steps.append(int(k))
    return jumps, valid_steps


def _window_rows(sim: dict, center_k: int, width: int = 10) -> list[dict[str, float | int]]:
    u_old = np.asarray(sim["u_old"])
    u_new = np.asarray(sim["u_new"])
    u_applied = np.asarray(sim["u_applied"])
    rho = np.asarray(sim["rho"])
    k0 = max(0, center_k - width)
    k1 = min(len(u_applied), center_k + width + 1)
    rows: list[dict[str, float | int]] = []
    for k in range(k0, k1):
        rows.append(
            {
                "k": int(k),
                "rho": float(rho[k]),
                "u_old_norm": float(np.linalg.norm(u_old[k])),
                "u_new_norm": float(np.linalg.norm(u_new[k])),
                "u_applied_norm": float(np.linalg.norm(u_applied[k])),
            }
        )
    return rows


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

    u_off = np.asarray(sim_off["u_applied"])
    u_on = np.asarray(sim_on["u_applied"])
    u_old_on = np.asarray(sim_on["u_old"])
    u_new_on = np.asarray(sim_on["u_new"])

    jumps_off, valid_steps_off = _jump_stats(u_off, list(sim_off["switch_steps"]))
    jumps_on, valid_steps_on = _jump_stats(u_on, list(sim_on["switch_steps"]))
    raw_on = [float(np.linalg.norm(u_new_on[k] - u_old_on[k])) for k in valid_steps_on]
    J_off = float(max(jumps_off)) if jumps_off else 0.0
    J_on_applied = float(max(jumps_on)) if jumps_on else 0.0
    J_on_raw = float(max(raw_on)) if raw_on else 0.0

    no_mode_mismatch = len(valid_steps_on) > 0 and J_on_raw <= 1.0e-12
    if no_mode_mismatch:
        warnings.warn(
            "No mode mismatch: u_old == u_new at switches (toy may not exercise Case B inversion mismatch).",
            stacklevel=1,
        )

    improve_target = 0.8 * J_off if J_off > 0.0 else 1.0e-3
    assert_ok = J_on_applied <= improve_target + 1.0e-6
    if not assert_ok:
        # Dump a local diagnostic window around the worst blended jump.
        worst_idx = int(np.argmax(np.asarray(jumps_on)))
        center_k = valid_steps_on[worst_idx]
        rows = _window_rows(sim_on, center_k=center_k, width=10)
        with (out_dir / "blending_assert_window.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["k", "rho", "u_old_norm", "u_new_norm", "u_applied_norm"])
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "gate": "Gate 3",
        "d_underline": float(d_underline),
        "c_overline": c_overline,
        "L_Psi": L_psi,
        "J_without_blending": J_off,
        "J_with_blending": J_on_applied,
        "J_applied_with_blending": J_on_applied,
        "J_raw_with_blending": J_on_raw,
        "J_improvement_target": improve_target,
        "no_mode_mismatch_warning": bool(no_mode_mismatch),
        "blending_reduces_J_assertion": bool(assert_ok),
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

    u = np.asarray(sim_off["u_applied"])
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

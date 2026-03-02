"""Gate 4: Monte Carlo trend tests over dwell, blending, and noise.

Theorem-style switched bounds are driven by baseline tracking and switch-induced
jump effects. This gate separates those via:
- error_outside_switch: mean tracking error away from switch windows
- error_spike_windowed: peak tracking error inside switch windows
- error_spike_instant: instantaneous spike at/just after switch steps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import make_results_dir


def _sample_x0(cfg, rng: np.random.Generator) -> np.ndarray:
    n = cfg.system.N
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def _error_series(sim: dict, cfg) -> np.ndarray:
    """Return per-step tracking error using the existing Gate 4 semantics."""
    x_hist = np.asarray(sim["x"])
    y_ref = np.asarray(sim["y_ref"])
    ydot_ref = np.asarray(sim["ydot_ref"])

    n = int(cfg.system.N)
    y = x_hist[:-1, :n]
    ydot = x_hist[:-1, n : 2 * n]
    if y_ref.shape != y.shape:
        raise ValueError(f"y_ref shape mismatch: y_ref={y_ref.shape}, y={y.shape}")
    if ydot_ref.shape != ydot.shape:
        raise ValueError(f"ydot_ref shape mismatch: ydot_ref={ydot_ref.shape}, ydot={ydot.shape}")

    e = np.linalg.norm(np.concatenate([y - y_ref, ydot - ydot_ref], axis=1), axis=1)
    return e


def _compute_error_envelope(sim: dict, cfg) -> float:
    e = _error_series(sim, cfg)
    return float(np.max(e)) if e.size > 0 else 0.0


def _split_switch_windows(e: np.ndarray, switch_steps: list[int], W: int) -> tuple[float, float]:
    """Return (error_outside_switch, error_spike_windowed)."""
    T = len(e)
    if T == 0:
        return 0.0, 0.0

    if not switch_steps:
        return float(np.mean(e)), float(np.max(e))

    mask = np.zeros(T, dtype=bool)
    for ks in switch_steps:
        k0 = max(0, int(ks) - W)
        k1 = min(T, int(ks) + W + 1)
        mask[k0:k1] = True

    spike_vals = e[mask]
    outside_vals = e[~mask]

    error_spike = float(np.max(spike_vals)) if spike_vals.size > 0 else float(np.max(e))
    error_outside = float(np.mean(outside_vals)) if outside_vals.size > 0 else float(np.mean(e))
    return error_outside, error_spike


def _error_spike_instant(e: np.ndarray, switch_steps: list[int]) -> float:
    """Return instantaneous spike metric around switching: max(e[k_s], e[k_s+1])."""
    T = len(e)
    if T == 0:
        return 0.0
    if not switch_steps:
        return float(np.max(e))
    vals: list[float] = []
    for ks in switch_steps:
        if 0 <= ks < T:
            vals.append(float(e[ks]))
        if 0 <= ks + 1 < T:
            vals.append(float(e[ks + 1]))
    return float(max(vals)) if vals else float(np.max(e))


def _plot_by_tau(
    out_path: Path,
    rows: list[dict[str, float | int | bool]],
    tau_values: list[float],
    field: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for blending in [False, True]:
        xs = []
        ys = []
        for tau_d in tau_values:
            vals = [
                float(r[field])
                for r in rows
                if bool(r["blending"]) == blending and float(r["tau_d"]) == float(tau_d)
            ]
            if vals:
                xs.append(float(tau_d))
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker="o", label=f"blending={blending}")
    ax.set_xlabel("tau_d [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_gate4(cfg_path: str) -> Path:
    exp_cfg = load_config(cfg_path)
    sys_cfg = load_config(exp_cfg.base_system_config)

    if hasattr(exp_cfg, "gate4"):
        g4 = exp_cfg.gate4
    else:
        g4 = sys_cfg.gate4

    out_dir = make_results_dir("gate4")
    dt = float(sys_cfg.system.dt)
    spike_window_sec = float(getattr(g4, "spike_window_sec", 0.2))
    W = max(1, int(spike_window_sec / max(dt, 1.0e-9)))

    rows: list[dict[str, float | int | bool]] = []

    for tau_idx, tau_d in enumerate(g4.tau_d_values):
        for noise_idx, noise_delta in enumerate(g4.noise_delta_values):
            for run in range(int(g4.mc_runs)):
                # Pair trajectories across blending modes by sharing x0 for same (tau, noise, run).
                x0_seed = int(sys_cfg.seed) + 404 + 100000 * tau_idx + 1000 * noise_idx + run
                rng_x0 = np.random.default_rng(x0_seed)
                x0 = _sample_x0(sys_cfg, rng_x0)
                for blending in g4.blending:
                    sim = simulate_closed_loop(
                        sys_cfg,
                        x0=x0,
                        options=SimOptions(
                            blending_on=bool(blending),
                            noise_delta=float(noise_delta),
                            seed=int(sys_cfg.seed) + 10000 + run,
                            lockout_sec_override=float(tau_d),
                        ),
                    )

                    e = _error_series(sim, sys_cfg)
                    err_env = _compute_error_envelope(sim, sys_cfg)
                    err_outside, err_spike = _split_switch_windows(
                        e=e,
                        switch_steps=[int(v) for v in sim["switch_steps"]],
                        W=W,
                    )
                    err_spike_inst = _error_spike_instant(
                        e=e,
                        switch_steps=[int(v) for v in sim["switch_steps"]],
                    )
                    assert err_spike >= err_outside - 1.0e-12

                    num_switches = int(len(sim["switch_steps"]))
                    sw_rate = float(num_switches / max(float(sim["horizon"]), 1e-9))
                    if num_switches >= 2:
                        avg_dt_switch = float(np.mean(np.diff(np.asarray(sim["switch_times"], dtype=float))))
                    else:
                        avg_dt_switch = float(sim["horizon"])

                    u_applied = np.asarray(sim["u_applied"])
                    u_old = np.asarray(sim["u_old"])
                    u_new = np.asarray(sim["u_new"])
                    applied_jumps = []
                    raw_jumps = []
                    for k in sim["switch_steps"]:
                        if 1 <= k < len(u_applied):
                            applied_jumps.append(float(np.linalg.norm(u_applied[k] - u_applied[k - 1])))
                            raw_jumps.append(float(np.linalg.norm(u_new[k] - u_old[k])))
                    J = float(max(applied_jumps)) if applied_jumps else 0.0
                    J_raw = float(max(raw_jumps)) if raw_jumps else 0.0

                    rows.append(
                        {
                            "tau_d": float(tau_d),
                            "blending": bool(blending),
                            "noise_delta": float(noise_delta),
                            "run": int(run),
                            "error_envelope": err_env,
                            "error_outside_switch": err_outside,
                            "error_spike_windowed": err_spike,
                            "error_spike_instant": err_spike_inst,
                            "num_switches": num_switches,
                            "avg_time_between_switches": avg_dt_switch,
                            "switch_rate": sw_rate,
                            "J": J,
                            "J_raw": J_raw,
                        }
                    )

    csv_path = out_dir / "gate4_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tau_d",
                "blending",
                "noise_delta",
                "run",
                "error_envelope",
                "error_outside_switch",
                "error_spike_windowed",
                "error_spike_instant",
                "num_switches",
                "avg_time_between_switches",
                "switch_rate",
                "J",
                "J_raw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # For dwell-time plots, isolate noise_delta=0 to highlight tau_d trend cleanly.
    rows_tau = [r for r in rows if float(r["noise_delta"]) == 0.0]

    _plot_by_tau(
        out_path=out_dir / "error_envelope_by_tau.png",
        rows=rows_tau,
        tau_values=[float(v) for v in g4.tau_d_values],
        field="error_envelope",
        title="Gate 4: Error Envelope vs Dwell",
        ylabel="mean error envelope",
    )

    _plot_by_tau(
        out_path=out_dir / "error_spike_by_tau.png",
        rows=rows_tau,
        tau_values=[float(v) for v in g4.tau_d_values],
        field="error_spike_windowed",
        title="Gate 4: Switch-window spike vs dwell",
        ylabel="mean spike error",
    )

    _plot_by_tau(
        out_path=out_dir / "error_spike_instant_by_tau.png",
        rows=rows_tau,
        tau_values=[float(v) for v in g4.tau_d_values],
        field="error_spike_instant",
        title="Gate 4: Instant spike vs dwell",
        ylabel="mean instant spike error",
    )

    _plot_by_tau(
        out_path=out_dir / "J_by_tau_and_blending.png",
        rows=rows_tau,
        tau_values=[float(v) for v in g4.tau_d_values],
        field="J",
        title="Gate 4: J vs Dwell Time",
        ylabel="mean J",
    )

    _plot_by_tau(
        out_path=out_dir / "switch_rate_by_tau.png",
        rows=rows_tau,
        tau_values=[float(v) for v in g4.tau_d_values],
        field="switch_rate",
        title="Gate 4: Switching rate vs dwell",
        ylabel="mean switch rate [1/s]",
    )

    # Keep existing noise trend plot across all tau values.
    fig, ax = plt.subplots(figsize=(7, 4))
    for blending in [False, True]:
        xs = []
        ys = []
        for nd in g4.noise_delta_values:
            vals = [
                float(r["switch_rate"])
                for r in rows
                if bool(r["blending"]) == blending and float(r["noise_delta"]) == float(nd)
            ]
            if vals:
                xs.append(float(nd))
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker="s", label=f"blending={blending}")
    ax.set_xlabel("noise delta")
    ax.set_ylabel("mean switch rate [1/s]")
    ax.set_title("Gate 4: Switching Rate vs Noise")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "switch_rate_by_noise.png", dpi=150)
    plt.close(fig)

    # Optional additional view: instant spike versus noise at tau_d = 0.
    fig, ax = plt.subplots(figsize=(7, 4))
    rows_noise = [r for r in rows if float(r["tau_d"]) == 0.0]
    for blending in [False, True]:
        xs = []
        ys = []
        for nd in g4.noise_delta_values:
            vals = [
                float(r["error_spike_instant"])
                for r in rows_noise
                if bool(r["blending"]) == blending and float(r["noise_delta"]) == float(nd)
            ]
            if vals:
                xs.append(float(nd))
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker="d", label=f"blending={blending}")
    ax.set_xlabel("noise delta")
    ax.set_ylabel("mean instant spike error")
    ax.set_title("Gate 4: Instant spike vs noise (tau_d = 0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "error_spike_instant_by_noise.png", dpi=150)
    plt.close(fig)

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 4 Monte Carlo trends")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gate4(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

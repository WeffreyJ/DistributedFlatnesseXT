"""Gate 4: Monte Carlo trend tests over dwell, blending, and noise."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.experiments.scenario_crossing import sinusoidal_crossing_reference
from src.model.dynamics import split_state
from src.verify.utils import make_results_dir


def _sample_x0(cfg, rng: np.random.Generator) -> np.ndarray:
    n = cfg.system.N
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def _compute_error_envelope(sim: dict, cfg) -> float:
    x_hist = np.asarray(sim["x"])
    t = np.asarray(sim["t"])
    errs = []
    for k in range(len(t)):
        y_ref, ydot_ref, _ = sinusoidal_crossing_reference(float(t[k]), cfg.reference)
        x1, x2 = split_state(x_hist[k], cfg.system.N)
        e = np.concatenate([x1 - y_ref, x2 - ydot_ref])
        errs.append(np.linalg.norm(e))
    return float(np.max(errs))


def run_gate4(cfg_path: str) -> Path:
    exp_cfg = load_config(cfg_path)
    sys_cfg = load_config(exp_cfg.base_system_config)

    if hasattr(exp_cfg, "gate4"):
        g4 = exp_cfg.gate4
    else:
        g4 = sys_cfg.gate4

    out_dir = make_results_dir("gate4")
    rng = np.random.default_rng(int(sys_cfg.seed) + 404)

    rows: list[dict[str, float | int | bool]] = []

    for tau_d in g4.tau_d_values:
        for blending in g4.blending:
            for noise_delta in g4.noise_delta_values:
                for run in range(int(g4.mc_runs)):
                    x0 = _sample_x0(sys_cfg, rng)
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

                    err_env = _compute_error_envelope(sim, sys_cfg)
                    sw_rate = float(len(sim["switch_times"]) / max(float(sim["horizon"]), 1e-9))
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
                            "switch_rate": sw_rate,
                            "J": J,
                            "J_raw": J_raw,
                        }
                    )

    csv_path = out_dir / "gate4_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tau_d", "blending", "noise_delta", "run", "error_envelope", "switch_rate", "J", "J_raw"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Plot 1: error envelope by tau_d (aggregated over noise)
    fig, ax = plt.subplots(figsize=(7, 4))
    arr = rows
    for blending in [False, True]:
        xs = []
        ys = []
        for tau_d in g4.tau_d_values:
            vals = [r["error_envelope"] for r in arr if r["blending"] == blending and r["tau_d"] == float(tau_d)]
            if vals:
                xs.append(float(tau_d))
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker="o", label=f"blending={blending}")
    ax.set_xlabel("tau_d [s]")
    ax.set_ylabel("mean error envelope")
    ax.set_title("Gate 4: Error Envelope vs Dwell")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "error_envelope_by_tau.png", dpi=150)
    plt.close(fig)

    # Plot 2: switch rate by noise
    fig, ax = plt.subplots(figsize=(7, 4))
    for blending in [False, True]:
        xs = []
        ys = []
        for nd in g4.noise_delta_values:
            vals = [r["switch_rate"] for r in arr if r["blending"] == blending and r["noise_delta"] == float(nd)]
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

    # Plot 3: J by tau and blending
    fig, ax = plt.subplots(figsize=(7, 4))
    for blending in [False, True]:
        xs = []
        ys = []
        for tau_d in g4.tau_d_values:
            vals = [r["J"] for r in arr if r["blending"] == blending and r["tau_d"] == float(tau_d)]
            if vals:
                xs.append(float(tau_d))
                ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, marker="^", label=f"blending={blending}")
    ax.set_xlabel("tau_d [s]")
    ax.set_ylabel("mean J")
    ax.set_title("Gate 4: J vs Dwell Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "J_by_tau_and_blending.png", dpi=150)
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

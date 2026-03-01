"""Operational reachable trimmed set R via Monte Carlo envelope."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.hybrid.ordering import rho_global
from src.verify.utils import dump_json, make_results_dir, seed_rng


def _sample_x0(cfg, rng: np.random.Generator) -> np.ndarray:
    n = cfg.system.N
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def run_reachset(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("reachset")
    rng = seed_rng(int(cfg.seed))

    mc_runs = int(cfg.reachset.mc_runs)
    horizon = float(cfg.reachset.horizon)
    shrink = float(cfg.reachset.shrink_margin)

    traj_mins = []
    traj_maxs = []
    outside_eps = 0
    total_samples = 0

    for run in range(mc_runs):
        x0 = _sample_x0(cfg, rng)
        sim = simulate_closed_loop(
            cfg,
            x0=x0,
            horizon=horizon,
            options=SimOptions(blending_on=False, noise_delta=0.0, seed=int(cfg.seed) + run),
        )
        x_hist = sim["x"]
        traj_mins.append(np.min(x_hist, axis=0))
        traj_maxs.append(np.max(x_hist, axis=0))

        n = cfg.system.N
        for row in x_hist:
            rho = rho_global(row[:n])
            if rho < float(cfg.ordering.epsilon):
                outside_eps += 1
            total_samples += 1

    min_env = np.min(np.vstack(traj_mins), axis=0)
    max_env = np.max(np.vstack(traj_maxs), axis=0)

    span = max_env - min_env
    R_min = min_env + shrink * span
    R_max = max_env - shrink * span

    payload = {
        "mc_runs": mc_runs,
        "horizon": horizon,
        "state_dim": int(2 * cfg.system.N),
        "envelope_min": min_env.tolist(),
        "envelope_max": max_env.tolist(),
        "R_min": R_min.tolist(),
        "R_max": R_max.tolist(),
        "fraction_outside_X_epsilon": float(outside_eps / max(total_samples, 1)),
    }

    out_path = out_dir / "R_bounds.json"
    dump_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute reachable trimmed set R")
    parser.add_argument("--config", required=True, help="Path to system YAML config")
    args = parser.parse_args()
    path = run_reachset(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

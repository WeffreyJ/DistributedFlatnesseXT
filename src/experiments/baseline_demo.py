"""Baseline vs hybrid demo for success-evaluation bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json


def _sample_x0(cfg, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(cfg.system.N)
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def _traj_metrics(sim: dict) -> dict[str, np.ndarray]:
    t = np.asarray(sim["t"])[:-1]
    u = np.asarray(sim["u_applied"])
    u_old = np.asarray(sim["u_old"])
    u_new = np.asarray(sim["u_new"])
    rho = np.asarray(sim["rho"])

    u_norm = np.linalg.norm(u, axis=1)
    du = np.zeros_like(u_norm)
    if len(u_norm) > 1:
        du[1:] = np.linalg.norm(np.diff(u, axis=0), axis=1)

    jump_vals = []
    raw_vals = []
    for k in sim["switch_steps"]:
        if 1 <= k < len(u):
            jump_vals.append(float(np.linalg.norm(u[k] - u[k - 1])))
            raw_vals.append(float(np.linalg.norm(u_new[k] - u_old[k])))

    J = float(max(jump_vals)) if jump_vals else 0.0
    J_raw = float(max(raw_vals)) if raw_vals else 0.0
    eps = 1.0e-9
    jump_ratio = float(J / (J_raw + eps)) if J_raw > 1.0e-8 else 0.0

    return {
        "t": t,
        "u_norm": u_norm,
        "du_norm": du,
        "rho": rho,
        "J": np.array([J]),
        "J_raw": np.array([J_raw]),
        "jump_ratio": np.array([jump_ratio]),
    }


def _write_traj_csv(path: Path, sim: dict, metrics: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "t",
                "u_applied_norm",
                "du_norm",
                "rho",
                "num_edges",
                "switch_event",
            ],
        )
        writer.writeheader()
        switch_steps = set(int(k) for k in sim["switch_steps"])
        num_edges = np.asarray(sim.get("num_edges", np.zeros_like(metrics["t"], dtype=int)))
        for k, t in enumerate(metrics["t"]):
            writer.writerow(
                {
                    "k": int(k),
                    "t": float(t),
                    "u_applied_norm": float(metrics["u_norm"][k]),
                    "du_norm": float(metrics["du_norm"][k]),
                    "rho": float(metrics["rho"][k]),
                    "num_edges": int(num_edges[k]) if k < len(num_edges) else 0,
                    "switch_event": int(k in switch_steps),
                }
            )


def run_baseline_demo(cfg_path: str, out_dir: str | Path = "results/baseline_demo") -> Path:
    cfg = load_config(cfg_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = _sample_x0(cfg, seed=int(cfg.seed) + 2026)

    sim_baseline = simulate_closed_loop(
        cfg,
        x0=x0,
        options=SimOptions(
            blending_on=False,
            noise_delta=0.0,
            seed=int(cfg.seed) + 3001,
            disable_switching=True,
            force_fixed_pi=True,
        ),
    )
    sim_hybrid = simulate_closed_loop(
        cfg,
        x0=x0,
        options=SimOptions(
            blending_on=True,
            noise_delta=0.0,
            seed=int(cfg.seed) + 3001,
            disable_switching=False,
        ),
    )

    m_base = _traj_metrics(sim_baseline)
    m_hyb = _traj_metrics(sim_hybrid)

    _write_traj_csv(out / "baseline_traj.csv", sim_baseline, m_base)
    _write_traj_csv(out / "hybrid_traj.csv", sim_hybrid, m_hyb)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m_base["t"], m_base["u_norm"], label="baseline (fixed order, no blending)")
    ax.plot(m_hyb["t"], m_hyb["u_norm"], label="hybrid (switching + blending)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("||u_applied||")
    ax.set_title("Applied Control Norm: Baseline vs Hybrid")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "u_applied_timeseries.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m_base["t"], m_base["du_norm"], label="baseline")
    ax.plot(m_hyb["t"], m_hyb["du_norm"], label="hybrid")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("||u[k]-u[k-1]||")
    ax.set_title("Applied Jump Proxy: Baseline vs Hybrid")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "du_norm_timeseries.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(m_base["t"], m_base["rho"], label="baseline")
    ax.plot(m_hyb["t"], m_hyb["rho"], label="hybrid")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("rho")
    ax.set_title("Ordering Margin rho: Baseline vs Hybrid")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rho_timeseries.png", dpi=150)
    plt.close(fig)

    J_base = float(m_base["J"][0])
    J_hyb = float(m_hyb["J"][0])
    J_raw_hyb = float(m_hyb["J_raw"][0])
    jump_ratio_hyb = float(m_hyb["jump_ratio"][0])

    warning = None
    if J_raw_hyb <= 1.0e-8:
        warning = (
            "No raw mismatch observed (J_raw ~ 0). Increase wake gain or enforce a less compatible fixed order baseline."
        )

    summary = {
        "J_max_baseline": J_base,
        "J_max_hybrid": J_hyb,
        "J_raw_max_hybrid": J_raw_hyb,
        "jump_ratio_hybrid": jump_ratio_hyb,
        "switch_count_baseline": int(len(sim_baseline["switch_steps"])),
        "switch_count_hybrid": int(len(sim_hybrid["switch_steps"])),
        "warning": warning,
    }
    dump_json(out / "baseline_demo_summary.json", summary)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs hybrid demo")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/baseline_demo")
    args = parser.parse_args()
    out = run_baseline_demo(args.config, out_dir=args.out)
    print(f"Wrote baseline demo to {out}")


if __name__ == "__main__":
    main()

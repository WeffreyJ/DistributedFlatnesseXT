"""Gate B: blending transient evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir, seed_rng


def _sample_x0(cfg, rng: np.random.Generator) -> np.ndarray:
    n = int(cfg.system.N)
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def run_gateB(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gateB")

    gate_cfg = getattr(getattr(cfg, "verify", object()), "gateB", None)
    mc_runs = int(getattr(gate_cfg, "mc_runs", 10))
    noise_delta = float(getattr(gate_cfg, "noise_delta", 0.0))
    tau_vals = list(getattr(gate_cfg, "tau_d_values", [float(cfg.ordering.lockout_sec)]))

    rng = seed_rng(int(cfg.seed) + 8300)

    rows: list[dict[str, float | int]] = []
    for tau_d in tau_vals:
        for run in range(mc_runs):
            x0 = _sample_x0(cfg, rng)
            sim = simulate_closed_loop(
                cfg,
                x0=x0,
                options=SimOptions(
                    blending_on=True,
                    noise_delta=noise_delta,
                    seed=int(cfg.seed) + 8400 + run,
                    lockout_sec_override=float(tau_d),
                ),
            )

            blend_active = np.asarray(sim["blend_active"], dtype=bool)
            blend_id = np.asarray(sim["blend_id"], dtype=int)
            e_norm = np.asarray(sim["e_norm"], dtype=float)
            j_raw = np.asarray(sim["J_raw"], dtype=float)

            ids = sorted({int(v) for v in blend_id[blend_active] if int(v) >= 0})
            for bid in ids:
                idx = np.where(blend_id == bid)[0]
                if idx.size == 0:
                    continue
                k0 = int(np.min(idx))
                k1 = int(np.max(idx))
                rows.append(
                    {
                        "tau_d": float(tau_d),
                        "run": int(run),
                        "blend_id": int(bid),
                        "k_start": k0,
                        "k_end": k1,
                        "duration_steps": int(k1 - k0 + 1),
                        "max_e_during_blend": float(np.max(e_norm[idx])),
                        "p90_e_during_blend": float(np.quantile(e_norm[idx], 0.9)),
                        "J_raw_at_start": float(j_raw[k0]) if 0 <= k0 < len(j_raw) else 0.0,
                    }
                )

    csv_path = out_dir / "blend_windows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tau_d",
                "run",
                "blend_id",
                "k_start",
                "k_end",
                "duration_steps",
                "max_e_during_blend",
                "p90_e_during_blend",
                "J_raw_at_start",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    vals = np.asarray([float(r["max_e_during_blend"]) for r in rows], dtype=float)
    jrs = np.asarray([float(r["J_raw_at_start"]) for r in rows], dtype=float)
    durs = np.asarray([float(r["duration_steps"]) for r in rows], dtype=float)

    payload = {
        "gate": "Gate B",
        "num_blend_windows": int(len(rows)),
        "max_e_median": float(np.median(vals)) if vals.size else 0.0,
        "max_e_p90": float(np.quantile(vals, 0.9)) if vals.size else 0.0,
        "max_e_max": float(np.max(vals)) if vals.size else 0.0,
        "duration_steps_median": float(np.median(durs)) if durs.size else 0.0,
        "duration_steps_max": float(np.max(durs)) if durs.size else 0.0,
        "J_raw_start_median": float(np.median(jrs)) if jrs.size else 0.0,
    }
    dump_json(out_dir / "gateB.json", payload)

    if vals.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=min(30, max(8, vals.size // 3)))
        ax.set_xlabel("max e_norm during blend")
        ax.set_ylabel("count")
        ax.set_title("Gate B: Blend-Window Peak Error")
        fig.tight_layout()
        fig.savefig(out_dir / "max_e_hist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(jrs, vals, ".", alpha=0.7)
        ax.set_xlabel("J_raw at blend start")
        ax.set_ylabel("max e_norm during blend")
        ax.set_title("Gate B: Blend Error vs Raw Mismatch")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "max_e_vs_Jraw.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(durs, bins=min(20, max(5, len(np.unique(durs)))))
        ax.set_xlabel("blend duration [steps]")
        ax.set_ylabel("count")
        ax.set_title("Gate B: Blend Duration Histogram")
        fig.tight_layout()
        fig.savefig(out_dir / "duration_hist.png", dpi=150)
        plt.close(fig)

    return out_dir / "gateB.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B blend transient evaluation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gateB(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

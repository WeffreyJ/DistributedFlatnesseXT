"""Gate R: empirical envelope invariance check via boundary sampling."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir, seed_rng


def _load_reachset_bounds() -> tuple[np.ndarray, np.ndarray, dict]:
    p = Path("results/reachset/R_bounds.json")
    if not p.exists():
        raise FileNotFoundError("Missing results/reachset/R_bounds.json. Run reachset first.")
    import json

    payload = json.loads(p.read_text(encoding="utf-8"))
    if "R_min" in payload and "R_max" in payload:
        lo = np.asarray(payload["R_min"], dtype=float)
        hi = np.asarray(payload["R_max"], dtype=float)
    elif "bbox_low" in payload and "bbox_high" in payload:
        lo = np.asarray(payload["bbox_low"], dtype=float)
        hi = np.asarray(payload["bbox_high"], dtype=float)
    else:
        raise KeyError("Reachset file missing R_min/R_max or bbox_low/bbox_high")
    return lo, hi, payload


def _sample_boundary_points(
    low: np.ndarray,
    high: np.ndarray,
    samples: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = len(low)
    points = np.zeros((samples, dim), dtype=float)
    face_dim = np.zeros(samples, dtype=int)
    face_side = np.zeros(samples, dtype=int)

    for idx in range(samples):
        x = rng.uniform(low, high)
        if method == "faces":
            d = idx % dim
            side = int(rng.integers(0, 2))
        else:  # random_shell fallback
            d = int(rng.integers(0, dim))
            side = int(rng.integers(0, 2))
        x[d] = low[d] if side == 0 else high[d]
        points[idx] = x
        face_dim[idx] = d
        face_side[idx] = side
    return points, face_dim, face_side


def run_gateR(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gateR")

    gate_cfg = getattr(getattr(cfg, "verify", object()), "gateR", None)
    samples = int(getattr(gate_cfg, "samples", 200))
    horizon_steps = int(getattr(gate_cfg, "horizon_steps", 300))
    method = str(getattr(gate_cfg, "boundary_sampling", "faces"))
    escape_tol = float(getattr(gate_cfg, "escape_tol", 0.0))
    blending_on = bool(getattr(gate_cfg, "blending_on", True))
    noise_delta = float(getattr(gate_cfg, "noise_delta", 0.0))

    low, high, reach_payload = _load_reachset_bounds()
    rng = seed_rng(int(cfg.seed) + 8100)
    pts, face_dim, face_side = _sample_boundary_points(low, high, samples=samples, method=method, rng=rng)

    dt = float(cfg.system.dt)
    horizon = float(horizon_steps * dt)

    escaped = np.zeros(samples, dtype=bool)
    escape_step = np.full(samples, -1, dtype=int)
    worst_dim = np.full(samples, -1, dtype=int)
    face_escape_counts: dict[str, int] = defaultdict(int)

    escaped_examples_x: list[np.ndarray] = []
    escaped_examples_t: list[np.ndarray] = []

    for idx in range(samples):
        sim = simulate_closed_loop(
            cfg,
            x0=pts[idx],
            horizon=horizon,
            options=SimOptions(
                blending_on=blending_on,
                noise_delta=noise_delta,
                seed=int(cfg.seed) + 8200 + idx,
            ),
        )

        x_hist = np.asarray(sim["x"], dtype=float)
        below = x_hist < (low[None, :] - escape_tol)
        above = x_hist > (high[None, :] + escape_tol)
        outside = np.any(below | above, axis=1)
        if np.any(outside):
            escaped[idx] = True
            k = int(np.argmax(outside))
            escape_step[idx] = k
            if np.any(below[k]):
                d = int(np.argmax(below[k]))
            else:
                d = int(np.argmax(above[k]))
            worst_dim[idx] = d
            face_escape_counts[f"d{int(face_dim[idx])}_s{int(face_side[idx])}"] += 1
            if len(escaped_examples_x) < 10:
                escaped_examples_x.append(x_hist)
                escaped_examples_t.append(np.asarray(sim["t"], dtype=float))

    escaped_frac = float(np.mean(escaped)) if samples > 0 else 0.0
    esc_steps = escape_step[escape_step >= 0]
    esc_times = esc_steps.astype(float) * dt

    payload = {
        "gate": "Gate R",
        "num_boundary_samples": int(samples),
        "boundary_method": method,
        "horizon_steps": int(horizon_steps),
        "horizon_sec": horizon,
        "escape_tol": escape_tol,
        "escape_fraction": escaped_frac,
        "num_escapes": int(np.sum(escaped)),
        "escape_time_median_sec": float(np.median(esc_times)) if esc_times.size else None,
        "escape_time_p90_sec": float(np.quantile(esc_times, 0.9)) if esc_times.size else None,
        "face_escape_counts": dict(face_escape_counts),
        "reachset_source": "results/reachset/R_bounds.json",
        "reachset_meta": reach_payload,
    }
    dump_json(out_dir / "gateR.json", payload)

    if esc_times.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(esc_times, bins=min(25, max(5, esc_times.size // 2)))
        ax.set_xlabel("escape time [s]")
        ax.set_ylabel("count")
        ax.set_title("Gate R: Escape Time Histogram")
        fig.tight_layout()
        fig.savefig(out_dir / "escape_time_hist.png", dpi=150)
        plt.close(fig)

    keys = sorted(face_escape_counts.keys())
    if keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = [face_escape_counts[k] for k in keys]
        ax.bar(np.arange(len(keys)), vals)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("escape count")
        ax.set_title("Gate R: Escape Count by Boundary Face")
        fig.tight_layout()
        fig.savefig(out_dir / "escape_fraction_by_face.png", dpi=150)
        plt.close(fig)

    if escaped_examples_x:
        # Save ragged trajectories as object arrays for debugging reproducibility.
        np.savez(
            out_dir / "escape_examples.npz",
            x=np.array(escaped_examples_x, dtype=object),
            t=np.array(escaped_examples_t, dtype=object),
        )

    return out_dir / "gateR.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate R envelope invariance")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gateR(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

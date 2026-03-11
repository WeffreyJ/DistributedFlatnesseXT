"""Gate S: permutation sensitivity near ties."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.flatness.recursion import build_phi, psi
from src.verify.utils import dump_json, make_results_dir, seed_rng


def _sample_x0(cfg, rng: np.random.Generator) -> np.ndarray:
    n = int(cfg.system.N)
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def _swap_pair_in_pi(pi: list[int], i: int, j: int, require_adjacent: bool) -> list[int] | None:
    if i < 0 or j < 0:
        return None
    out = [int(v) for v in pi]
    try:
        a = out.index(int(i))
        b = out.index(int(j))
    except ValueError:
        return None
    if require_adjacent and abs(a - b) != 1:
        return None
    out[a], out[b] = out[b], out[a]
    return out


def _zeta_from_snapshot(
    zeta_r: np.ndarray,
    n: int,
    y_slice: tuple[int, int],
    ydot_slice: tuple[int, int],
    v_slice: tuple[int, int],
) -> dict[str, np.ndarray]:
    return {
        "y": np.asarray(zeta_r[y_slice[0] : y_slice[1]], dtype=float),
        "ydot": np.asarray(zeta_r[ydot_slice[0] : ydot_slice[1]], dtype=float),
        "v": np.asarray(zeta_r[v_slice[0] : v_slice[1]], dtype=float),
    }


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_gateS(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gateS")
    n = int(cfg.system.N)

    gate_cfg = getattr(getattr(cfg, "verify", object()), "gateS", None)
    mc_runs = int(getattr(gate_cfg, "mc_runs", getattr(cfg.gate1, "mc_runs", 8)))
    tie_gap_delta = float(getattr(gate_cfg, "tie_gap_delta", getattr(cfg.ordering, "tie_gap_delta", 0.05)))
    max_samples = int(getattr(gate_cfg, "max_samples", 2000))
    require_adjacent = bool(getattr(gate_cfg, "require_adjacent", True))
    blending_on = bool(getattr(gate_cfg, "blending_on", False))
    flat_cfg = getattr(cfg, "flat", None)
    zeta_layout = getattr(flat_cfg, "zeta_layout", None) if flat_cfg is not None else None
    y_slice = tuple(getattr(zeta_layout, "y", [0, n])) if zeta_layout is not None else (0, n)
    ydot_slice = tuple(getattr(zeta_layout, "ydot", [n, 2 * n])) if zeta_layout is not None else (n, 2 * n)
    v_slice = tuple(getattr(zeta_layout, "v", [2 * n, 3 * n])) if zeta_layout is not None else (2 * n, 3 * n)

    rng = seed_rng(int(cfg.seed) + 7100)
    rows: list[dict[str, float | int]] = []

    for run in range(mc_runs):
        if len(rows) >= max_samples:
            break
        x0 = _sample_x0(cfg, rng)
        sim = simulate_closed_loop(
            cfg,
            x0=x0,
            options=SimOptions(blending_on=blending_on, noise_delta=0.0, seed=int(cfg.seed) + 7200 + run),
        )

        # Prefer dedicated near-tie snapshots when available.
        snapshots = sim.get("phi_snapshots", [])
        if snapshots:
            iterable = []
            for snap in snapshots:
                k = int(snap["k"])
                iterable.append(
                    {
                        "k": k,
                        "tie_gap_min": float(snap.get("tie_gap_min", sim["tie_gap_min"][k])),
                        "tie_pair": tuple(snap.get("tie_pair", (-1, -1))),
                        "x": np.asarray(snap["x"], dtype=float),
                        "zeta_r": np.asarray(snap["zeta_r"], dtype=float),
                        "pi": [int(v) for v in snap["pi"]],
                        "pi_candidate": [int(v) for v in snap.get("pi_candidate", snap["pi"])],
                    }
                )
        else:
            iterable = []
            for k in range(len(sim["k"])):
                if float(sim["tie_gap_min"][k]) > tie_gap_delta:
                    continue
                iterable.append(
                    {
                        "k": int(k),
                        "tie_gap_min": float(sim["tie_gap_min"][k]),
                        "tie_pair": (int(sim["tie_i"][k]), int(sim["tie_j"][k])),
                        "x": np.asarray(sim["x"][k], dtype=float),
                        "zeta_r": np.asarray(sim["zeta_r"][k], dtype=float),
                        "pi": [int(v) for v in sim["pi"][k]],
                        "pi_candidate": [int(v) for v in sim["pi_candidate"][k]],
                    }
                )

        for item in iterable:
            if len(rows) >= max_samples:
                break
            tie_gap = float(item["tie_gap_min"])
            if tie_gap > tie_gap_delta:
                continue

            i_tie, j_tie = item["tie_pair"]
            pi = item["pi"]
            pi_swap = _swap_pair_in_pi(pi, int(i_tie), int(j_tie), require_adjacent=require_adjacent)
            if pi_swap is None:
                continue

            x = np.asarray(item["x"], dtype=float)
            zeta = _zeta_from_snapshot(
                np.asarray(item["zeta_r"], dtype=float),
                n=n,
                y_slice=(int(y_slice[0]), int(y_slice[1])),
                ydot_slice=(int(ydot_slice[0]), int(ydot_slice[1])),
                v_slice=(int(v_slice[0]), int(v_slice[1])),
            )

            u_base = psi(build_phi(x=x, zeta=zeta, pi=pi, sys=cfg, params=cfg.system), cfg.system)
            u_swap = psi(build_phi(x=x, zeta=zeta, pi=pi_swap, sys=cfg, params=cfg.system), cfg.system)
            s_val = float(np.linalg.norm(u_swap - u_base))

            # Optional narrative metric: compare actual candidate ordering mismatch at same snapshot.
            pi_candidate = [int(v) for v in item["pi_candidate"]]
            u_candidate = psi(
                build_phi(x=x, zeta=zeta, pi=pi_candidate, sys=cfg, params=cfg.system),
                cfg.system,
            )
            s_candidate = float(np.linalg.norm(u_candidate - u_base))

            rows.append(
                {
                    "episode": int(run),
                    "k": int(item["k"]),
                    "tie_gap_min": tie_gap,
                    "tie_i": int(i_tie),
                    "tie_j": int(j_tie),
                    "S": s_val,
                    "S_candidate": s_candidate,
                    "u_base_norm": float(np.linalg.norm(u_base)),
                    "u_swap_norm": float(np.linalg.norm(u_swap)),
                    "u_candidate_norm": float(np.linalg.norm(u_candidate)),
                }
            )

    csv_path = out_dir / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "k",
                "tie_gap_min",
                "tie_i",
                "tie_j",
                "S",
                "S_candidate",
                "u_base_norm",
                "u_swap_norm",
                "u_candidate_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    s_vals = np.asarray([float(r["S"]) for r in rows], dtype=float)
    s_cand = np.asarray([float(r["S_candidate"]) for r in rows], dtype=float)
    tie_vals = np.asarray([float(r["tie_gap_min"]) for r in rows], dtype=float)
    nonzero_eps = 1.0e-12
    s_nonzero = s_vals[s_vals > nonzero_eps]
    s_cand_nonzero = s_cand[s_cand > nonzero_eps]
    gamma_edge = float(getattr(cfg.system, "gamma_edge", getattr(cfg.system, "gamma", 0.0)))
    structural_zero_risk = bool(require_adjacent and tie_gap_delta <= gamma_edge)

    payload = {
        "gate": "Gate S",
        "num_samples": int(len(rows)),
        "num_episodes": int(mc_runs),
        "require_adjacent": bool(require_adjacent),
        "tie_gap_delta": tie_gap_delta,
        "gamma_edge": gamma_edge,
        "structural_zero_risk": structural_zero_risk,
        "S_min": float(np.min(s_vals)) if s_vals.size else 0.0,
        "S_median": float(np.median(s_vals)) if s_vals.size else 0.0,
        "S_p90": float(np.quantile(s_vals, 0.9)) if s_vals.size else 0.0,
        "S_max": float(np.max(s_vals)) if s_vals.size else 0.0,
        "S_nonzero_count": int(s_nonzero.size),
        "S_nonzero_fraction": float(s_nonzero.size / max(len(rows), 1)),
        "S_p90_nonzero": float(np.quantile(s_nonzero, 0.9)) if s_nonzero.size else 0.0,
        "S_candidate_median": float(np.median(s_cand)) if s_cand.size else 0.0,
        "S_candidate_nonzero_count": int(s_cand_nonzero.size),
        "S_candidate_nonzero_fraction": float(s_cand_nonzero.size / max(len(rows), 1)),
        "S_candidate_p90_nonzero": float(np.quantile(s_cand_nonzero, 0.9)) if s_cand_nonzero.size else 0.0,
        "corr_S_vs_tie_gap": _corr(s_vals.tolist(), tie_vals.tolist()),
        "corr_S_candidate_vs_S": _corr(s_vals.tolist(), s_cand.tolist()),
    }
    dump_json(out_dir / "gateS_sensitivity.json", payload)

    if s_vals.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s_vals, bins=min(30, max(8, s_vals.size // 5)))
        ax.set_xlabel("S = ||u(pi_swap)-u(pi)||")
        ax.set_ylabel("count")
        ax.set_title("Gate S: Sensitivity Histogram")
        fig.tight_layout()
        fig.savefig(out_dir / "S_hist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tie_vals, s_vals, ".", alpha=0.6, label="adjacent swap")
        if s_cand.size:
            ax.plot(tie_vals, s_cand, ".", alpha=0.45, label="candidate mismatch")
        ax.set_xlabel("tie_gap_min")
        ax.set_ylabel("sensitivity norm")
        ax.set_title("Gate S: Sensitivity vs Tie Gap")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "S_vs_tie_gap.png", dpi=150)
        plt.close(fig)

    return out_dir / "gateS_sensitivity.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate S sensitivity near ties")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gateS(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

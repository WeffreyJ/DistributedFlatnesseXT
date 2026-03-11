"""Tier-1 order-sensitivity diagnostic against the matched full evaluator."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.flatness.evaluation_operator import (
    compute_evaluator,
    get_evaluator_mode,
    get_local_window_agents,
)
from src.flatness.recursion import build_phi, psi
from src.verify.utils import dump_json, make_results_dir


NONZERO_TOL = 1.0e-12


def _tier1_cfg_for_rollout(config_path: str):
    cfg = load_config(config_path)
    cfg_t1 = copy.deepcopy(cfg)
    cfg_t1.system.plant_family = "tier1"
    return cfg_t1


def _base_x0(cfg) -> np.ndarray:
    n = int(cfg.system.N)
    return np.concatenate(
        [
            np.array(cfg.reference.base, dtype=float),
            np.zeros(n, dtype=float),
        ],
        axis=0,
    )


def _as_pi_list(entry) -> list[int]:
    return [int(v) for v in entry]


def _zeta_from_log(sim: dict, k: int, cfg) -> dict[str, np.ndarray]:
    layout = getattr(getattr(cfg, "flat", object()), "zeta_layout", None)
    zeta_r = np.asarray(sim["zeta_r"][k], dtype=float)
    if layout is None:
        n = int(cfg.system.N)
        return {
            "y": zeta_r[:n].copy(),
            "ydot": zeta_r[n : 2 * n].copy(),
            "v": zeta_r[2 * n : 3 * n].copy(),
        }
    return {
        "y": zeta_r[int(layout.y[0]) : int(layout.y[1])].copy(),
        "ydot": zeta_r[int(layout.ydot[0]) : int(layout.ydot[1])].copy(),
        "v": zeta_r[int(layout.v[0]) : int(layout.v[1])].copy(),
    }


def _approx_modes(cfg) -> list[str]:
    modes = ["upstream_truncated"]
    local_agents = int(get_local_window_agents(cfg))
    if local_agents > 0:
        modes.append("local_window")
    return modes


def _reconstruct_same_snapshot(
    x: np.ndarray,
    zeta: dict[str, np.ndarray],
    pi: list[int],
    cfg,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    E = compute_evaluator(x=x, pi=pi, cfg=cfg, mode=mode)
    phi = build_phi(x=x, zeta=zeta, pi=pi, sys=cfg, params=cfg.system, evaluator_output=E)
    u = psi(phi, params=cfg.system)
    return u, E


def _collect_event_pairs(sim: dict, cfg) -> list[dict[str, object]]:
    steps = int(len(sim["k"]))
    t_control = np.asarray(sim["t_control"], dtype=float)
    x_hist = np.asarray(sim["x"], dtype=float)
    blend_active = np.asarray(sim["blend_active"], dtype=bool)
    switch_reason = [str(v) for v in sim["switch_reason"]]
    pi_hist = [_as_pi_list(v) for v in sim["pi"]]
    pi_candidate_hist = [_as_pi_list(v) for v in sim["pi_candidate"]]

    rows: list[dict[str, object]] = []
    mode_before = pi_hist[0].copy()
    transition_pair: tuple[list[int], list[int]] | None = None

    for k in range(steps):
        candidate = pi_candidate_hist[k].copy()
        if switch_reason[k] == "transition_start":
            transition_pair = (mode_before.copy(), candidate.copy())

        if transition_pair is not None and blend_active[k]:
            pi_old = transition_pair[0].copy()
            pi_new = transition_pair[1].copy()
        else:
            pi_old = mode_before.copy()
            pi_new = candidate.copy()

        event_type: str | None = None
        if switch_reason[k] == "transition_start":
            event_type = "transition_start"
        elif switch_reason[k] == "immediate_switch":
            event_type = "immediate_switch"
        elif transition_pair is not None and blend_active[k]:
            event_type = "transition_active"
        elif candidate != mode_before:
            event_type = "candidate_change"

        if event_type is not None:
            rows.append(
                {
                    "step": int(k),
                    "time_sec": float(t_control[k]),
                    "event_type": str(event_type),
                    "pi_old": pi_old,
                    "pi_new": pi_new,
                    "x_snapshot": np.asarray(x_hist[k], dtype=float),
                    "zeta": _zeta_from_log(sim, k, cfg),
                    "tie_gap_min": float(sim["tie_gap_min"][k]) if "tie_gap_min" in sim else float("inf"),
                    "blend_active": bool(blend_active[k]),
                    "switching_context": str(switch_reason[k]),
                }
            )

        if transition_pair is not None and (k == steps - 1 or not bool(blend_active[k + 1])):
            transition_pair = None
        mode_before = pi_hist[k].copy()

    return rows


def _materialize_rows(event_rows: list[dict[str, object]], cfg, approx_mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in event_rows:
        x_snapshot = np.asarray(event["x_snapshot"], dtype=float)
        zeta = event["zeta"]
        pi_old = [int(v) for v in event["pi_old"]]
        pi_new = [int(v) for v in event["pi_new"]]

        u_old_full, E_old_full = _reconstruct_same_snapshot(x_snapshot, zeta, pi_old, cfg, mode="full")
        u_new_full, E_new_full = _reconstruct_same_snapshot(x_snapshot, zeta, pi_new, cfg, mode="full")
        u_old_approx, E_old_approx = _reconstruct_same_snapshot(x_snapshot, zeta, pi_old, cfg, mode=approx_mode)
        u_new_approx, E_new_approx = _reconstruct_same_snapshot(x_snapshot, zeta, pi_new, cfg, mode=approx_mode)

        E_dev_old = float(np.linalg.norm(E_old_approx - E_old_full))
        E_dev_new = float(np.linalg.norm(E_new_approx - E_new_full))
        u_dev_old = float(np.linalg.norm(u_old_approx - u_old_full))
        u_dev_new = float(np.linalg.norm(u_new_approx - u_new_full))
        E_gap_full = float(np.linalg.norm(E_new_full - E_old_full))
        u_gap_full = float(np.linalg.norm(u_new_full - u_old_full))
        E_gap_approx = float(np.linalg.norm(E_new_approx - E_old_approx))
        u_gap_approx = float(np.linalg.norm(u_new_approx - u_old_approx))

        rows.append(
            {
                "approx_mode": str(approx_mode),
                "step": int(event["step"]),
                "time_sec": float(event["time_sec"]),
                "event_type": str(event["event_type"]),
                "pi_old": json.dumps([int(v) for v in pi_old]),
                "pi_new": json.dumps([int(v) for v in pi_new]),
                "E_dev_old": E_dev_old,
                "E_dev_new": E_dev_new,
                "u_dev_old": u_dev_old,
                "u_dev_new": u_dev_new,
                "E_gap_full": E_gap_full,
                "u_gap_full": u_gap_full,
                "E_gap_approx": E_gap_approx,
                "u_gap_approx": u_gap_approx,
                "tie_gap_min": float(event["tie_gap_min"]),
                "blend_active": bool(event["blend_active"]),
                "switching_context": str(event["switching_context"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "approx_mode",
        "step",
        "time_sec",
        "event_type",
        "pi_old",
        "pi_new",
        "E_dev_old",
        "E_dev_new",
        "u_dev_old",
        "u_dev_new",
        "E_gap_full",
        "u_gap_full",
        "E_gap_approx",
        "u_gap_approx",
        "tie_gap_min",
        "blend_active",
        "switching_context",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_empty_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mode_colors(modes: list[str]) -> dict[str, str]:
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    return {mode: palette[idx % len(palette)] for idx, mode in enumerate(modes)}


def _scatter_dev_plot(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        _save_empty_plot(out_path, "Tier-1 deviation", "No rows available")
        return
    modes = sorted({str(r["approx_mode"]) for r in rows})
    colors = _mode_colors(modes)
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in modes:
        mode_rows = [r for r in rows if str(r["approx_mode"]) == mode]
        xvals = np.asarray(
            [float(r["E_dev_old"]) for r in mode_rows] + [float(r["E_dev_new"]) for r in mode_rows],
            dtype=float,
        )
        yvals = np.asarray(
            [float(r["u_dev_old"]) for r in mode_rows] + [float(r["u_dev_new"]) for r in mode_rows],
            dtype=float,
        )
        ax.plot(xvals, yvals, ".", alpha=0.7, color=colors[mode], label=mode)
    lo = 0.0
    hi = float(
        max(
            max(float(r["E_dev_old"]) for r in rows),
            max(float(r["E_dev_new"]) for r in rows),
            max(float(r["u_dev_old"]) for r in rows),
            max(float(r["u_dev_new"]) for r in rows),
            1.0e-12,
        )
    )
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("E deviation from full")
    ax.set_ylabel("u deviation from full")
    ax.set_title("Tier-1 approx deviation vs control deviation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hist_by_mode(rows: list[dict[str, object]], *, key: str, xlabel: str, title: str, out_path: Path) -> None:
    if not rows:
        _save_empty_plot(out_path, title, "No rows available")
        return
    modes = sorted({str(r["approx_mode"]) for r in rows})
    colors = _mode_colors(modes)
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in modes:
        values = np.asarray([float(r[key]) for r in rows if str(r["approx_mode"]) == mode], dtype=float)
        if values.size == 0:
            continue
        ax.hist(values, bins=min(24, max(8, values.size)), alpha=0.45, label=mode, color=colors[mode])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _mode_summary(mode: str, rows: list[dict[str, object]]) -> dict[str, object]:
    E_dev_old = np.asarray([float(r["E_dev_old"]) for r in rows], dtype=float)
    E_dev_new = np.asarray([float(r["E_dev_new"]) for r in rows], dtype=float)
    u_dev_old = np.asarray([float(r["u_dev_old"]) for r in rows], dtype=float)
    u_dev_new = np.asarray([float(r["u_dev_new"]) for r in rows], dtype=float)
    E_gap_approx = np.asarray([float(r["E_gap_approx"]) for r in rows], dtype=float)
    u_gap_approx = np.asarray([float(r["u_gap_approx"]) for r in rows], dtype=float)
    E_gap_full = np.asarray([float(r["E_gap_full"]) for r in rows], dtype=float)
    u_gap_full = np.asarray([float(r["u_gap_full"]) for r in rows], dtype=float)
    max_E_dev = float(max(np.max(E_dev_old), np.max(E_dev_new))) if rows else 0.0
    max_u_dev = float(max(np.max(u_dev_old), np.max(u_dev_new))) if rows else 0.0
    return {
        "approx_mode": str(mode),
        "num_rows": int(len(rows)),
        "mean_E_dev_old": float(np.mean(E_dev_old)) if E_dev_old.size else 0.0,
        "mean_E_dev_new": float(np.mean(E_dev_new)) if E_dev_new.size else 0.0,
        "mean_u_dev_old": float(np.mean(u_dev_old)) if u_dev_old.size else 0.0,
        "mean_u_dev_new": float(np.mean(u_dev_new)) if u_dev_new.size else 0.0,
        "nonzero_E_gap_approx_fraction": float(np.mean(E_gap_approx > NONZERO_TOL)) if E_gap_approx.size else 0.0,
        "nonzero_u_gap_approx_fraction": float(np.mean(u_gap_approx > NONZERO_TOL)) if u_gap_approx.size else 0.0,
        "max_E_gap_approx": float(np.max(E_gap_approx)) if E_gap_approx.size else 0.0,
        "max_u_gap_approx": float(np.max(u_gap_approx)) if u_gap_approx.size else 0.0,
        "max_E_dev": max_E_dev,
        "max_u_dev": max_u_dev,
        "max_E_gap_full": float(np.max(E_gap_full)) if E_gap_full.size else 0.0,
        "max_u_gap_full": float(np.max(u_gap_full)) if u_gap_full.size else 0.0,
        "mean_E_gap_full": float(np.mean(E_gap_full)) if E_gap_full.size else 0.0,
        "mean_u_gap_full": float(np.mean(u_gap_full)) if u_gap_full.size else 0.0,
    }


def _summary_payload(rows: list[dict[str, object]], cfg, *, rollout_mode: str) -> dict[str, object]:
    modes = sorted({str(r["approx_mode"]) for r in rows})
    per_mode = {
        mode: _mode_summary(mode, [r for r in rows if str(r["approx_mode"]) == mode])
        for mode in modes
    }
    return {
        "gate": "Gate 2 Tier-1 order sensitivity",
        "plant_family": "tier1",
        "reference_mode": "full",
        "rollout_evaluator_mode": str(rollout_mode),
        "num_rows": int(len(rows)),
        "approx_modes": modes,
        "local_window_agents": int(get_local_window_agents(cfg)),
        "modes": per_mode,
        "nonzero_tol": float(NONZERO_TOL),
    }


def run_gate2_tier1_order_sensitivity(config_path: str = "configs/system.yaml") -> Path:
    cfg = _tier1_cfg_for_rollout(config_path)
    rollout_mode = str(get_evaluator_mode(cfg))
    out_dir = make_results_dir("gate2_tier1_order_sensitivity")

    sim = simulate_closed_loop(
        cfg,
        x0=_base_x0(cfg),
        horizon=float(getattr(cfg.gate3, "run_horizon", cfg.system.horizon)),
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(cfg.seed) + 19200),
    )
    event_rows = _collect_event_pairs(sim, cfg)

    rows: list[dict[str, object]] = []
    for approx_mode in _approx_modes(cfg):
        rows.extend(_materialize_rows(event_rows, cfg, approx_mode))

    table_path = out_dir / "tier1_order_sensitivity_table.csv"
    summary_path = out_dir / "tier1_order_sensitivity_summary.json"
    _write_csv(table_path, rows)
    dump_json(summary_path, _summary_payload(rows, cfg, rollout_mode=rollout_mode))

    _scatter_dev_plot(rows, out_dir / "E_dev_vs_u_dev_tier1.png")
    _hist_by_mode(
        rows,
        key="E_gap_approx",
        xlabel="E_gap_approx",
        title="Tier-1 approximate evaluator-gap histogram",
        out_path=out_dir / "E_gap_approx_hist_tier1.png",
    )
    _hist_by_mode(
        rows,
        key="u_gap_approx",
        xlabel="u_gap_approx",
        title="Tier-1 approximate raw-gap histogram",
        out_path=out_dir / "u_gap_approx_hist_tier1.png",
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-1 order-sensitivity diagnostic against full evaluator.")
    parser.add_argument("--config", default="configs/system.yaml", help="Path to config YAML.")
    args = parser.parse_args()
    summary_path = run_gate2_tier1_order_sensitivity(args.config)
    print(summary_path)


if __name__ == "__main__":
    main()

"""Tier-1 matched operator-mismatch diagnostics for same-snapshot raw-gap identity."""

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
from src.flatness.evaluation_operator import compute_evaluator
from src.flatness.recursion import build_phi, psi
from src.model.coupling import delta_accel
from src.verify.utils import dump_json, make_results_dir


ABS_TOL = 1.0e-9
REL_TOL = 1.0e-7


def _matched_tier1_cfg(config_path: str):
    cfg = load_config(config_path)
    cfg_t1 = copy.deepcopy(cfg)
    cfg_t1.system.plant_family = "tier1"
    cfg_t1.evaluation.mode = "full"
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
    layout = cfg.flat.zeta_layout
    zeta_r = np.asarray(sim["zeta_r"][k], dtype=float)
    return {
        "y": zeta_r[int(layout.y[0]) : int(layout.y[1])].copy(),
        "ydot": zeta_r[int(layout.ydot[0]) : int(layout.ydot[1])].copy(),
        "v": zeta_r[int(layout.v[0]) : int(layout.v[1])].copy(),
    }


def _consistency_ok(u_gap: float, E_gap: float) -> bool:
    scale = max(abs(u_gap), abs(E_gap), 1.0)
    return abs(u_gap - E_gap) <= ABS_TOL + REL_TOL * scale


def _reconstruct_same_snapshot_pair(
    x: np.ndarray,
    zeta: dict[str, np.ndarray],
    pi_old: list[int],
    pi_new: list[int],
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    E_old = compute_evaluator(x=x, pi=pi_old, cfg=cfg, mode="full")
    E_new = compute_evaluator(x=x, pi=pi_new, cfg=cfg, mode="full")
    phi_old = build_phi(x=x, zeta=zeta, pi=pi_old, sys=cfg, params=cfg.system, evaluator_output=E_old)
    phi_new = build_phi(x=x, zeta=zeta, pi=pi_new, sys=cfg, params=cfg.system, evaluator_output=E_new)
    u_old = psi(phi_old, params=cfg.system)
    u_new = psi(phi_new, params=cfg.system)
    return u_old, u_new, E_old, E_new


def _collect_rows(sim: dict, cfg) -> list[dict[str, object]]:
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
            x_snapshot = np.asarray(x_hist[k], dtype=float)
            zeta = _zeta_from_log(sim, k, cfg)
            u_old, u_new, E_old, E_new = _reconstruct_same_snapshot_pair(
                x=x_snapshot,
                zeta=zeta,
                pi_old=pi_old,
                pi_new=pi_new,
                cfg=cfg,
            )
            plant_delta = np.asarray(delta_accel(x_snapshot, cfg.system), dtype=float)
            tier1_full_matches_delta = bool(
                np.allclose(E_old, plant_delta, atol=ABS_TOL, rtol=REL_TOL)
                and np.allclose(E_new, plant_delta, atol=ABS_TOL, rtol=REL_TOL)
            )
            E_gap = float(np.linalg.norm(E_new - E_old))
            u_gap = float(np.linalg.norm(u_new - u_old))
            abs_error = float(abs(u_gap - E_gap))

            rows.append(
                {
                    "step": int(k),
                    "time_sec": float(t_control[k]),
                    "event_type": str(event_type),
                    "pi_old": json.dumps([int(v) for v in pi_old]),
                    "pi_new": json.dumps([int(v) for v in pi_new]),
                    "E_gap": E_gap,
                    "u_gap": u_gap,
                    "gap_identity_error": abs_error,
                    "tier1_full_matches_delta": bool(tier1_full_matches_delta),
                    "blend_active": bool(blend_active[k]),
                    "switching_context": str(switch_reason[k]),
                }
            )

        if transition_pair is not None and (k == steps - 1 or not bool(blend_active[k + 1])):
            transition_pair = None
        mode_before = pi_hist[k].copy()

    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "time_sec",
        "event_type",
        "pi_old",
        "pi_new",
        "E_gap",
        "u_gap",
        "gap_identity_error",
        "tier1_full_matches_delta",
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


def _scatter_with_diagonal(xvals: np.ndarray, yvals: np.ndarray, *, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    if xvals.size == 0 or yvals.size == 0:
        _save_empty_plot(out_path, title, "No rows available")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xvals, yvals, ".", alpha=0.75)
    lo = float(min(np.min(xvals), np.min(yvals)))
    hi = float(max(np.max(xvals), np.max(yvals)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hist_plot(values: np.ndarray, *, xlabel: str, title: str, out_path: Path) -> None:
    if values.size == 0:
        _save_empty_plot(out_path, title, "No rows available")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(30, max(8, values.size)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    E_gap = np.asarray([float(r["E_gap"]) for r in rows], dtype=float)
    u_gap = np.asarray([float(r["u_gap"]) for r in rows], dtype=float)
    abs_error = np.asarray([float(r["gap_identity_error"]) for r in rows], dtype=float)
    consistent = np.asarray([_consistency_ok(float(r["u_gap"]), float(r["E_gap"])) for r in rows], dtype=bool)
    matches = np.asarray([bool(r["tier1_full_matches_delta"]) for r in rows], dtype=bool)
    return {
        "gate": "Gate 3 operator mismatch Tier-1",
        "plant_family": "tier1",
        "evaluator_mode": "full",
        "num_rows": int(len(rows)),
        "gap_identity_fraction": float(np.mean(consistent)) if consistent.size else 1.0,
        "max_abs_gap_identity_error": float(np.max(abs_error)) if abs_error.size else 0.0,
        "mean_abs_gap_identity_error": float(np.mean(abs_error)) if abs_error.size else 0.0,
        "tier1_full_matches_delta_fraction": float(np.mean(matches)) if matches.size else 1.0,
        "max_E_gap": float(np.max(E_gap)) if E_gap.size else 0.0,
        "max_u_gap": float(np.max(u_gap)) if u_gap.size else 0.0,
        "consistency_abs_tol": float(ABS_TOL),
        "consistency_rel_tol": float(REL_TOL),
    }


def run_gate3_operator_mismatch_tier1(config_path: str = "configs/system.yaml") -> Path:
    cfg = _matched_tier1_cfg(config_path)
    out_dir = make_results_dir("gate3_operator_mismatch_tier1")
    sim = simulate_closed_loop(
        cfg,
        x0=_base_x0(cfg),
        horizon=float(getattr(cfg.gate3, "run_horizon", cfg.system.horizon)),
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(cfg.seed) + 19100),
    )

    rows = _collect_rows(sim, cfg)
    table_path = out_dir / "operator_gap_table_tier1.csv"
    summary_path = out_dir / "operator_gap_summary_tier1.json"
    _write_csv(table_path, rows)
    dump_json(summary_path, _summary_payload(rows))

    E_gap = np.asarray([float(r["E_gap"]) for r in rows], dtype=float)
    u_gap = np.asarray([float(r["u_gap"]) for r in rows], dtype=float)
    _scatter_with_diagonal(
        E_gap,
        u_gap,
        xlabel="E_gap",
        ylabel="u_gap",
        title="Tier-1 matched raw gap vs evaluator gap",
        out_path=out_dir / "u_gap_vs_E_gap_tier1.png",
    )
    _hist_plot(E_gap, xlabel="E_gap", title="Tier-1 evaluator gap histogram", out_path=out_dir / "E_gap_hist_tier1.png")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-1 matched operator-mismatch diagnostic")
    parser.add_argument("--config", default="configs/system.yaml")
    args = parser.parse_args()
    path = run_gate3_operator_mismatch_tier1(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

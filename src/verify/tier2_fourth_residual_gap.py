"""Tier-2 nominal-vs-plant diagnostic for the fourth residual family."""

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
from src.model.coupling import plant_delta_accel
from src.model.residual_tier2 import support_transition_pair_terms_tier2
from src.verify.utils import dump_json, make_results_dir


NONZERO_TOL = 1.0e-12


def _tier2_support_transition_cfg(config_path: str):
    cfg = load_config(config_path)
    cfg_t2 = copy.deepcopy(cfg)
    cfg_t2.system.plant_family = "tier2"
    cfg_t2.monitor.enabled = False
    cfg_t2.monitor.mode = "shadow"
    cfg_t2.system.tier2.residual.enabled = True
    cfg_t2.system.tier2.residual.mode = "support_transition_bias"
    if float(getattr(cfg_t2.system.tier2.residual, "amplitude", 0.0)) == 0.0:
        cfg_t2.system.tier2.residual.amplitude = 0.05
    return cfg_t2


def _base_x0(cfg) -> np.ndarray:
    n = int(cfg.system.N)
    return np.concatenate([np.array(cfg.reference.base, dtype=float), np.zeros(n, dtype=float)], axis=0)


def _as_pi_list(entry) -> list[int]:
    return [int(v) for v in entry]


def _pairwise_diagnostics(x: np.ndarray, cfg) -> dict[str, object]:
    n = int(cfg.system.N)
    shell_tol = float(getattr(cfg.system.tier2.residual, "support_transition_shell_tol", 0.05))
    nonzero_abs_d_ell: list[float] = []
    nonzero_abs_d_rho: list[float] = []
    shell_localized = 0
    shells = 0
    for leader in range(n):
        for follower in range(n):
            if leader == follower:
                continue
            terms = support_transition_pair_terms_tier2(np.asarray(x, dtype=float), leader, follower, cfg.system)
            pair_residual = float(terms["pair_residual"])
            if abs(pair_residual) > NONZERO_TOL:
                nonzero_abs_d_ell.append(abs(float(terms["d_ell"])))
                nonzero_abs_d_rho.append(abs(float(terms["d_rho"])))
                shells += 1
                if float(terms["shell_strength"]) >= shell_tol:
                    shell_localized += 1
    return {
        "mean_abs_d_ell_nonzero": float(np.mean(nonzero_abs_d_ell)) if nonzero_abs_d_ell else 0.0,
        "mean_abs_d_rho_nonzero": float(np.mean(nonzero_abs_d_rho)) if nonzero_abs_d_rho else 0.0,
        "fraction_nonzero_within_shell": float(shell_localized / shells) if shells else 0.0,
        "nonzero_pair_count": int(shells),
    }


def _collect_rows(sim: dict, cfg) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    steps = int(len(sim["k"]))
    t_control = np.asarray(sim["t_control"], dtype=float)
    x_hist = np.asarray(sim["x"], dtype=float)
    blend_active = np.asarray(sim["blend_active"], dtype=bool)
    switch_reason = [str(v) for v in sim["switch_reason"]]
    pi_hist = [_as_pi_list(v) for v in sim["pi"]]
    pi_candidate_hist = [_as_pi_list(v) for v in sim["pi_candidate"]]
    eval_mode = str(getattr(getattr(cfg, "evaluation", None), "mode", "upstream_truncated"))

    rows: list[dict[str, object]] = []
    pair_diag_rows: list[dict[str, object]] = []
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
            plant_interaction = np.asarray(plant_delta_accel(x_snapshot, cfg.system), dtype=float)
            e_nom_old = np.asarray(compute_evaluator(x_snapshot, pi_old, cfg, mode=eval_mode), dtype=float)
            e_nom_new = np.asarray(compute_evaluator(x_snapshot, pi_new, cfg, mode=eval_mode), dtype=float)
            residual_old = plant_interaction - e_nom_old
            residual_new = plant_interaction - e_nom_new
            pair_diag = _pairwise_diagnostics(x_snapshot, cfg)
            rows.append(
                {
                    "step": int(k),
                    "time_sec": float(t_control[k]),
                    "event_type": str(event_type),
                    "pi_old": json.dumps([int(v) for v in pi_old]),
                    "pi_new": json.dumps([int(v) for v in pi_new]),
                    "plant_interaction_norm": float(np.linalg.norm(plant_interaction)),
                    "nominal_interaction_norm_old": float(np.linalg.norm(e_nom_old)),
                    "nominal_interaction_norm_new": float(np.linalg.norm(e_nom_new)),
                    "nominal_vs_plant_gap_old": float(np.linalg.norm(residual_old)),
                    "nominal_vs_plant_gap_new": float(np.linalg.norm(residual_new)),
                    "residual_norm_old": float(np.linalg.norm(residual_old)),
                    "residual_norm_new": float(np.linalg.norm(residual_new)),
                    "nominal_order_gap": float(np.linalg.norm(e_nom_new - e_nom_old)),
                    "plant_order_gap": 0.0,
                    "residual_order_gap": float(np.linalg.norm(residual_new - residual_old)),
                    "switching_context": str(switch_reason[k]),
                    "mean_abs_d_ell_nonzero": float(pair_diag["mean_abs_d_ell_nonzero"]),
                    "mean_abs_d_rho_nonzero": float(pair_diag["mean_abs_d_rho_nonzero"]),
                    "fraction_nonzero_within_shell": float(pair_diag["fraction_nonzero_within_shell"]),
                    "nonzero_pair_count": int(pair_diag["nonzero_pair_count"]),
                }
            )
            pair_diag_rows.append(pair_diag)

        if transition_pair is not None and (k == steps - 1 or not bool(blend_active[k + 1])):
            transition_pair = None
        mode_before = pi_hist[k].copy()
    return rows, pair_diag_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "time_sec",
        "event_type",
        "pi_old",
        "pi_new",
        "plant_interaction_norm",
        "nominal_interaction_norm_old",
        "nominal_interaction_norm_new",
        "nominal_vs_plant_gap_old",
        "nominal_vs_plant_gap_new",
        "residual_norm_old",
        "residual_norm_new",
        "nominal_order_gap",
        "plant_order_gap",
        "residual_order_gap",
        "switching_context",
        "mean_abs_d_ell_nonzero",
        "mean_abs_d_rho_nonzero",
        "fraction_nonzero_within_shell",
        "nonzero_pair_count",
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


def _scatter_plot(
    xvals: np.ndarray,
    yvals: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    diagonal: bool = False,
) -> None:
    if xvals.size == 0 or yvals.size == 0:
        _save_empty_plot(out_path, title, "No rows available")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xvals, yvals, ".", alpha=0.75)
    if diagonal:
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


def _summary_payload(rows: list[dict[str, object]], pair_diag_rows: list[dict[str, object]], cfg) -> dict[str, object]:
    residual_norms = (
        np.asarray([float(r["residual_norm_old"]) for r in rows] + [float(r["residual_norm_new"]) for r in rows], dtype=float)
        if rows
        else np.array([], dtype=float)
    )
    nominal_vs_plant = (
        np.asarray([float(r["nominal_vs_plant_gap_old"]) for r in rows] + [float(r["nominal_vs_plant_gap_new"]) for r in rows], dtype=float)
        if rows
        else np.array([], dtype=float)
    )
    residual_order_gap = np.asarray([float(r["residual_order_gap"]) for r in rows], dtype=float) if rows else np.array([], dtype=float)
    mean_abs_d_ell = np.asarray([float(r["mean_abs_d_ell_nonzero"]) for r in rows if int(r["nonzero_pair_count"]) > 0], dtype=float)
    mean_abs_d_rho = np.asarray([float(r["mean_abs_d_rho_nonzero"]) for r in rows if int(r["nonzero_pair_count"]) > 0], dtype=float)
    frac_within_shell = np.asarray([float(r["fraction_nonzero_within_shell"]) for r in rows if int(r["nonzero_pair_count"]) > 0], dtype=float)
    shell_tol = float(getattr(cfg.system.tier2.residual, "support_transition_shell_tol", 0.05))
    residual_cfg = cfg.system.tier2.residual
    return {
        "plant_family": "tier2",
        "evaluation_mode": str(getattr(getattr(cfg, "evaluation", None), "mode", "upstream_truncated")),
        "residual_enabled": bool(getattr(residual_cfg, "enabled", False)),
        "residual_mode": str(getattr(residual_cfg, "mode", "zero")),
        "residual_amplitude": float(getattr(residual_cfg, "amplitude", 0.0)),
        "num_rows": int(len(rows)),
        "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms.size else 0.0,
        "max_residual_norm": float(np.max(residual_norms)) if residual_norms.size else 0.0,
        "mean_nominal_vs_plant_gap": float(np.mean(nominal_vs_plant)) if nominal_vs_plant.size else 0.0,
        "max_nominal_vs_plant_gap": float(np.max(nominal_vs_plant)) if nominal_vs_plant.size else 0.0,
        "mean_residual_order_gap": float(np.mean(residual_order_gap)) if residual_order_gap.size else 0.0,
        "max_residual_order_gap": float(np.max(residual_order_gap)) if residual_order_gap.size else 0.0,
        "mean_abs_d_ell_at_nonzero_residual": float(np.mean(mean_abs_d_ell)) if mean_abs_d_ell.size else 0.0,
        "mean_abs_d_rho_at_nonzero_residual": float(np.mean(mean_abs_d_rho)) if mean_abs_d_rho.size else 0.0,
        "fraction_nonzero_within_shell": float(np.mean(frac_within_shell)) if frac_within_shell.size else 0.0,
        "shell_tol": shell_tol,
        "total_nonzero_pair_diagnostics": int(sum(int(d["nonzero_pair_count"]) for d in pair_diag_rows)),
    }


def run_tier2_fourth_residual_gap(config_path: str = "configs/system.yaml") -> Path:
    cfg = _tier2_support_transition_cfg(config_path)
    out_dir = make_results_dir("tier2_fourth_residual_gap")
    sim = simulate_closed_loop(
        cfg,
        x0=_base_x0(cfg),
        horizon=float(getattr(cfg.gate3, "run_horizon", cfg.system.horizon)),
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(cfg.seed) + 20820),
    )

    rows, pair_diag_rows = _collect_rows(sim, cfg)
    _write_csv(out_dir / "tier2_fourth_residual_gap_table.csv", rows)
    dump_json(out_dir / "tier2_fourth_residual_gap_summary.json", _summary_payload(rows, pair_diag_rows, cfg))

    residual_norms = (
        np.asarray([float(r["residual_norm_old"]) for r in rows] + [float(r["residual_norm_new"]) for r in rows], dtype=float)
        if rows
        else np.array([], dtype=float)
    )
    nominal_vs_plant_old = np.asarray([float(r["nominal_vs_plant_gap_old"]) for r in rows], dtype=float) if rows else np.array([], dtype=float)
    plant_norm_old = np.asarray([float(r["plant_interaction_norm"]) for r in rows], dtype=float) if rows else np.array([], dtype=float)
    nominal_order_gap = np.asarray([float(r["nominal_order_gap"]) for r in rows], dtype=float) if rows else np.array([], dtype=float)
    residual_order_gap = np.asarray([float(r["residual_order_gap"]) for r in rows], dtype=float) if rows else np.array([], dtype=float)

    _hist_plot(
        residual_norms,
        xlabel="||Delta_plant - E_nom||",
        title="Tier-2 fourth residual norm histogram",
        out_path=out_dir / "residual_norm_hist.png",
    )
    _scatter_plot(
        plant_norm_old,
        nominal_vs_plant_old,
        xlabel="||Delta_plant||",
        ylabel="||Delta_plant - E_nom||",
        title="Tier-2 fourth residual plant-vs-nominal mismatch",
        out_path=out_dir / "plant_vs_nominal_gap_scatter.png",
    )
    _scatter_plot(
        nominal_order_gap,
        residual_order_gap,
        xlabel="nominal order gap",
        ylabel="residual order gap",
        title="Tier-2 fourth residual order-gap comparison",
        out_path=out_dir / "order_gap_nominal_vs_plant.png",
        diagonal=True,
    )
    return out_dir / "tier2_fourth_residual_gap_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-2 nominal-vs-plant diagnostic for the fourth residual family.")
    parser.add_argument("--config", default="configs/system.yaml")
    args = parser.parse_args()
    out = run_tier2_fourth_residual_gap(config_path=args.config)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

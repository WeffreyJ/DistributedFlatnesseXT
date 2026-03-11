"""Frozen-controller comparison for Tier-1 nominal vs Tier-2 support-transition-bias residual."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.flatness.evaluation_operator import compute_evaluator
from src.model.coupling import plant_delta_accel
from src.verify.utils import dump_json, make_results_dir


MISMATCH_TOL = 1.0e-12
BLOCK_ACTIONS = {"hold_current", "extend_lockout", "fallback_fixed_order"}


def _base_x0(cfg) -> np.ndarray:
    n = int(cfg.system.N)
    return np.concatenate([np.array(cfg.reference.base, dtype=float), np.zeros(n, dtype=float)], axis=0)


def _clone_cfg_variant(cfg, variant: str):
    cfg_mode = copy.deepcopy(cfg)
    if variant == "tier1_nominal":
        cfg_mode.system.plant_family = "tier1"
    elif variant == "tier2_support_transition_bias_enabled":
        cfg_mode.system.plant_family = "tier2"
        cfg_mode.system.tier2.residual.enabled = True
        cfg_mode.system.tier2.residual.mode = "support_transition_bias"
        if float(getattr(cfg_mode.system.tier2.residual, "amplitude", 0.0)) == 0.0:
            cfg_mode.system.tier2.residual.amplitude = 0.05
    else:
        raise ValueError(f"Unsupported variant: {variant!r}")
    return cfg_mode


def _mismatch_metrics(sim: dict[str, object], cfg) -> dict[str, float]:
    x_hist = np.asarray(sim.get("x", []), dtype=float)
    pi_hist = sim.get("pi", [])
    switch_reason = [str(v) for v in sim.get("switch_reason", [])]
    eval_mode = str(getattr(getattr(cfg, "evaluation", object()), "mode", "upstream_truncated"))

    mismatch = []
    switch_mismatch = []
    for k in range(min(len(pi_hist), max(len(x_hist) - 1, 0))):
        x_now = np.asarray(x_hist[k], dtype=float)
        pi_now = [int(v) for v in pi_hist[k]]
        nominal = np.asarray(compute_evaluator(x_now, pi_now, cfg, mode=eval_mode), dtype=float)
        plant = np.asarray(plant_delta_accel(x_now, cfg.system), dtype=float)
        gap = float(np.linalg.norm(plant - nominal))
        mismatch.append(gap)
        if switch_reason[k] in {"transition_start", "immediate_switch"}:
            switch_mismatch.append(gap)

    mismatch_arr = np.asarray(mismatch, dtype=float)
    switch_arr = np.asarray(switch_mismatch, dtype=float)
    frac = float(np.mean(mismatch_arr > MISMATCH_TOL)) if mismatch_arr.size else 0.0
    return {
        "mean_nominal_vs_plant_gap_over_rollout": float(np.mean(mismatch_arr)) if mismatch_arr.size else 0.0,
        "max_nominal_vs_plant_gap_over_rollout": float(np.max(mismatch_arr)) if mismatch_arr.size else 0.0,
        "fraction_steps_with_nonzero_mismatch": frac,
        "fraction_above_mismatch_tol": frac,
        "mismatch_at_switch_steps_mean": float(np.mean(switch_arr)) if switch_arr.size else 0.0,
    }


def _blocked_steps_total(sim: dict[str, object]) -> int:
    selector_switch_eligible = np.asarray(sim.get("selector_switch_eligible", []), dtype=bool)
    monitor_actions = [str(v) for v in sim.get("monitor_action", [])]
    blocked = [
        bool(selector_switch_eligible[idx]) and action in BLOCK_ACTIONS
        for idx, action in enumerate(monitor_actions)
    ]
    return int(np.count_nonzero(np.asarray(blocked, dtype=bool)))


def _metrics_for_run(sim: dict[str, object], cfg, variant: str) -> dict[str, float | int | str | bool]:
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    out = {
        "variant": str(variant),
        "plant_family": str(getattr(cfg.system, "plant_family", "unknown")),
        "selection_mode": str(getattr(getattr(cfg, "selection", object()), "mode", "unknown")),
        "monitor_mode": str(getattr(getattr(cfg, "monitor", object()), "mode", "disabled"))
        if bool(getattr(getattr(cfg, "monitor", object()), "enabled", False))
        else "disabled",
        "evaluator_mode": str(getattr(getattr(cfg, "evaluation", object()), "mode", "unknown")),
        "residual_enabled": bool(
            getattr(getattr(getattr(cfg.system, "tier2", object()), "residual", object()), "enabled", False)
        )
        if str(getattr(cfg.system, "plant_family", "")) == "tier2"
        else False,
        "residual_mode": str(getattr(getattr(getattr(cfg.system, "tier2", object()), "residual", object()), "mode", "n/a"))
        if str(getattr(cfg.system, "plant_family", "")) == "tier2"
        else "n/a",
        "residual_amplitude": float(
            getattr(getattr(getattr(cfg.system, "tier2", object()), "residual", object()), "amplitude", 0.0)
        )
        if str(getattr(cfg.system, "plant_family", "")) == "tier2"
        else 0.0,
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in sim.get("switch_reason", []) if str(reason) == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "blocked_steps_total": _blocked_steps_total(sim),
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
    }
    out.update(_mismatch_metrics(sim, cfg))
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "variant",
        "plant_family",
        "selection_mode",
        "monitor_mode",
        "evaluator_mode",
        "residual_enabled",
        "residual_mode",
        "residual_amplitude",
        "switch_count",
        "transition_start_count",
        "blend_active_steps",
        "blocked_steps_total",
        "tracking_error_mean",
        "tracking_error_max",
        "max_raw_jump",
        "max_applied_jump",
        "mean_nominal_vs_plant_gap_over_rollout",
        "max_nominal_vs_plant_gap_over_rollout",
        "fraction_steps_with_nonzero_mismatch",
        "fraction_above_mismatch_tol",
        "mismatch_at_switch_steps_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_grouped_bars(rows: list[dict[str, object]], metrics: list[str], title: str, ylabel: str, out_path: Path) -> None:
    labels = [str(row["variant"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        ax.bar(x + idx * width - 0.4 + width / 2.0, values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metrics:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_tier2_fourth_residual_compare(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier2_fourth_residual_compare",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("tier2_fourth_residual_compare") if out_dir == "results/tier2_fourth_residual_compare" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = _base_x0(cfg)
    sim_seed = int(cfg.seed) + 20851
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    variants = ["tier1_nominal", "tier2_support_transition_bias_enabled"]
    rows: list[dict[str, object]] = []
    for variant in variants:
        cfg_mode = _clone_cfg_variant(cfg, variant)
        sim = simulate_closed_loop(
            cfg_mode,
            x0=x0,
            horizon=float(cfg_mode.system.horizon),
            options=options,
        )
        rows.append(_metrics_for_run(sim, cfg_mode, variant))

    baseline = rows[0]
    delta_keys = [
        "switch_count",
        "transition_start_count",
        "blend_active_steps",
        "blocked_steps_total",
        "tracking_error_mean",
        "tracking_error_max",
        "max_raw_jump",
        "max_applied_jump",
        "mean_nominal_vs_plant_gap_over_rollout",
        "fraction_above_mismatch_tol",
    ]
    summary = {
        "shared_x0": [float(v) for v in x0],
        "shared_seed": int(sim_seed),
        "runs": {str(row["variant"]): row for row in rows},
        "delta_tier2_support_transition_bias_minus_tier1": {key: float(rows[1][key]) - float(baseline[key]) for key in delta_keys},
    }
    dump_json(out / "tier2_fourth_residual_compare.json", summary)
    _write_csv(out / "tier2_fourth_residual_compare.csv", rows)

    _plot_grouped_bars(
        rows,
        ["switch_count", "transition_start_count", "blend_active_steps", "blocked_steps_total"],
        "Tier-2 fourth residual comparison: switching burden",
        "count",
        out / "switching_compare.png",
    )
    _plot_grouped_bars(
        rows,
        ["tracking_error_mean", "tracking_error_max"],
        "Tier-2 fourth residual comparison: tracking error",
        "tracking error",
        out / "tracking_compare.png",
    )
    _plot_grouped_bars(
        rows,
        ["max_raw_jump", "max_applied_jump"],
        "Tier-2 fourth residual comparison: jump magnitudes",
        "jump norm",
        out / "jumps_compare.png",
    )
    _plot_grouped_bars(
        rows,
        [
            "mean_nominal_vs_plant_gap_over_rollout",
            "max_nominal_vs_plant_gap_over_rollout",
            "fraction_above_mismatch_tol",
        ],
        "Tier-2 fourth residual comparison: nominal-vs-plant mismatch",
        "mismatch metric",
        out / "mismatch_compare.png",
    )
    return out / "tier2_fourth_residual_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Tier-1 nominal and Tier-2 support-transition-bias residual.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier2_fourth_residual_compare")
    args = parser.parse_args()
    out = run_tier2_fourth_residual_compare(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

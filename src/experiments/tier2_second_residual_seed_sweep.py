"""Paired seed sweep for Tier-1 nominal vs Tier-2 longitudinal-bias residual."""

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
from src.experiments.runtime_monitor_audit import _sample_x0
from src.experiments.tier2_second_residual_compare import _clone_cfg_variant, _mismatch_metrics
from src.verify.utils import dump_json, make_results_dir


DEFAULT_SEED_IDS = list(range(12))


def _metrics_for_run(sim: dict[str, object], cfg, variant: str, seed_id: int) -> dict[str, float | int | str]:
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    out = {
        "seed_id": int(seed_id),
        "variant": str(variant),
        "plant_family": str(getattr(cfg.system, "plant_family", "unknown")),
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in sim.get("switch_reason", []) if str(reason) == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
    }
    out.update(_mismatch_metrics(sim, cfg))
    return out


def _paired_delta(seed_id: int, tier1: dict[str, object], tier2: dict[str, object]) -> dict[str, float | int]:
    return {
        "seed_id": int(seed_id),
        "delta_switch_count": int(tier2["switch_count"]) - int(tier1["switch_count"]),
        "delta_transition_start_count": int(tier2["transition_start_count"]) - int(tier1["transition_start_count"]),
        "delta_blend_active_steps": int(tier2["blend_active_steps"]) - int(tier1["blend_active_steps"]),
        "delta_tracking_error_mean": float(tier2["tracking_error_mean"]) - float(tier1["tracking_error_mean"]),
        "delta_tracking_error_max": float(tier2["tracking_error_max"]) - float(tier1["tracking_error_max"]),
        "delta_max_raw_jump": float(tier2["max_raw_jump"]) - float(tier1["max_raw_jump"]),
        "delta_max_applied_jump": float(tier2["max_applied_jump"]) - float(tier1["max_applied_jump"]),
        "delta_mean_nominal_vs_plant_gap": float(tier2["mean_nominal_vs_plant_gap_over_rollout"])
        - float(tier1["mean_nominal_vs_plant_gap_over_rollout"]),
        "delta_max_nominal_vs_plant_gap": float(tier2["max_nominal_vs_plant_gap_over_rollout"])
        - float(tier1["max_nominal_vs_plant_gap_over_rollout"]),
        "delta_fraction_steps_with_nonzero_mismatch": float(tier2["fraction_steps_with_nonzero_mismatch"])
        - float(tier1["fraction_steps_with_nonzero_mismatch"]),
        "delta_mismatch_at_switch_steps_mean": float(tier2["mismatch_at_switch_steps_mean"])
        - float(tier1["mismatch_at_switch_steps_mean"]),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "seed_id",
        "tier1_switch_count",
        "tier2_switch_count",
        "delta_switch_count",
        "tier1_transition_start_count",
        "tier2_transition_start_count",
        "delta_transition_start_count",
        "tier1_blend_active_steps",
        "tier2_blend_active_steps",
        "delta_blend_active_steps",
        "tier1_tracking_error_mean",
        "tier2_tracking_error_mean",
        "delta_tracking_error_mean",
        "tier1_tracking_error_max",
        "tier2_tracking_error_max",
        "delta_tracking_error_max",
        "tier1_max_raw_jump",
        "tier2_max_raw_jump",
        "delta_max_raw_jump",
        "tier1_max_applied_jump",
        "tier2_max_applied_jump",
        "delta_max_applied_jump",
        "tier1_mean_nominal_vs_plant_gap",
        "tier2_mean_nominal_vs_plant_gap",
        "delta_mean_nominal_vs_plant_gap",
        "tier1_max_nominal_vs_plant_gap",
        "tier2_max_nominal_vs_plant_gap",
        "delta_max_nominal_vs_plant_gap",
        "tier1_fraction_steps_with_nonzero_mismatch",
        "tier2_fraction_steps_with_nonzero_mismatch",
        "delta_fraction_steps_with_nonzero_mismatch",
        "tier1_mismatch_at_switch_steps_mean",
        "tier2_mismatch_at_switch_steps_mean",
        "delta_mismatch_at_switch_steps_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hist_plot(values: np.ndarray, *, xlabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if values.size:
        ax.hist(values, bins=min(20, max(5, values.size)))
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_plot(xvals: np.ndarray, yvals: np.ndarray, *, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if xvals.size and yvals.size:
        ax.plot(xvals, yvals, ".")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _delta_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "delta_switch_count",
        "delta_transition_start_count",
        "delta_blend_active_steps",
        "delta_tracking_error_mean",
        "delta_tracking_error_max",
        "delta_max_raw_jump",
        "delta_max_applied_jump",
        "delta_mean_nominal_vs_plant_gap",
        "delta_max_nominal_vs_plant_gap",
        "delta_fraction_steps_with_nonzero_mismatch",
        "delta_mismatch_at_switch_steps_mean",
    ]
    out: dict[str, object] = {}
    for key in metrics:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        out[key] = {
            "mean": float(np.mean(values)) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "min": float(np.min(values)) if values.size else 0.0,
            "max": float(np.max(values)) if values.size else 0.0,
            "positive_fraction": float(np.mean(values > 0.0)) if values.size else 0.0,
            "zero_fraction": float(np.mean(values == 0.0)) if values.size else 0.0,
            "negative_fraction": float(np.mean(values < 0.0)) if values.size else 0.0,
        }
    return out


def run_tier2_second_residual_seed_sweep(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier2_second_residual_seed_sweep",
    seed_ids: list[int] | None = None,
) -> Path:
    cfg = load_config(config_path)
    out = (
        make_results_dir("tier2_second_residual_seed_sweep")
        if out_dir == "results/tier2_second_residual_seed_sweep"
        else Path(out_dir)
    )
    out.mkdir(parents=True, exist_ok=True)

    seed_list = [int(v) for v in (seed_ids if seed_ids is not None else DEFAULT_SEED_IDS)]
    paired_rows: list[dict[str, object]] = []
    runs_by_seed: dict[str, dict[str, object]] = {}

    for seed_id in seed_list:
        x0_seed = int(cfg.seed) + 20931 + 10 * seed_id
        sim_seed = int(cfg.seed) + 20951 + 10 * seed_id
        x0 = _sample_x0(cfg, seed=x0_seed)
        options = SimOptions(blending_on=True, noise_delta=0.0, seed=sim_seed, disable_switching=False)

        seed_runs: dict[str, dict[str, object]] = {}
        for variant in ["tier1_nominal", "tier2_longitudinal_bias_enabled"]:
            cfg_mode = _clone_cfg_variant(cfg, variant)
            sim = simulate_closed_loop(
                cfg_mode,
                x0=x0,
                horizon=float(cfg_mode.system.horizon),
                options=options,
            )
            seed_runs[variant] = _metrics_for_run(sim, cfg_mode, variant, seed_id)

        delta = _paired_delta(seed_id, seed_runs["tier1_nominal"], seed_runs["tier2_longitudinal_bias_enabled"])
        paired_rows.append(
            {
                "seed_id": int(seed_id),
                "tier1_switch_count": int(seed_runs["tier1_nominal"]["switch_count"]),
                "tier2_switch_count": int(seed_runs["tier2_longitudinal_bias_enabled"]["switch_count"]),
                "delta_switch_count": int(delta["delta_switch_count"]),
                "tier1_transition_start_count": int(seed_runs["tier1_nominal"]["transition_start_count"]),
                "tier2_transition_start_count": int(seed_runs["tier2_longitudinal_bias_enabled"]["transition_start_count"]),
                "delta_transition_start_count": int(delta["delta_transition_start_count"]),
                "tier1_blend_active_steps": int(seed_runs["tier1_nominal"]["blend_active_steps"]),
                "tier2_blend_active_steps": int(seed_runs["tier2_longitudinal_bias_enabled"]["blend_active_steps"]),
                "delta_blend_active_steps": int(delta["delta_blend_active_steps"]),
                "tier1_tracking_error_mean": float(seed_runs["tier1_nominal"]["tracking_error_mean"]),
                "tier2_tracking_error_mean": float(seed_runs["tier2_longitudinal_bias_enabled"]["tracking_error_mean"]),
                "delta_tracking_error_mean": float(delta["delta_tracking_error_mean"]),
                "tier1_tracking_error_max": float(seed_runs["tier1_nominal"]["tracking_error_max"]),
                "tier2_tracking_error_max": float(seed_runs["tier2_longitudinal_bias_enabled"]["tracking_error_max"]),
                "delta_tracking_error_max": float(delta["delta_tracking_error_max"]),
                "tier1_max_raw_jump": float(seed_runs["tier1_nominal"]["max_raw_jump"]),
                "tier2_max_raw_jump": float(seed_runs["tier2_longitudinal_bias_enabled"]["max_raw_jump"]),
                "delta_max_raw_jump": float(delta["delta_max_raw_jump"]),
                "tier1_max_applied_jump": float(seed_runs["tier1_nominal"]["max_applied_jump"]),
                "tier2_max_applied_jump": float(seed_runs["tier2_longitudinal_bias_enabled"]["max_applied_jump"]),
                "delta_max_applied_jump": float(delta["delta_max_applied_jump"]),
                "tier1_mean_nominal_vs_plant_gap": float(seed_runs["tier1_nominal"]["mean_nominal_vs_plant_gap_over_rollout"]),
                "tier2_mean_nominal_vs_plant_gap": float(seed_runs["tier2_longitudinal_bias_enabled"]["mean_nominal_vs_plant_gap_over_rollout"]),
                "delta_mean_nominal_vs_plant_gap": float(delta["delta_mean_nominal_vs_plant_gap"]),
                "tier1_max_nominal_vs_plant_gap": float(seed_runs["tier1_nominal"]["max_nominal_vs_plant_gap_over_rollout"]),
                "tier2_max_nominal_vs_plant_gap": float(seed_runs["tier2_longitudinal_bias_enabled"]["max_nominal_vs_plant_gap_over_rollout"]),
                "delta_max_nominal_vs_plant_gap": float(delta["delta_max_nominal_vs_plant_gap"]),
                "tier1_fraction_steps_with_nonzero_mismatch": float(seed_runs["tier1_nominal"]["fraction_steps_with_nonzero_mismatch"]),
                "tier2_fraction_steps_with_nonzero_mismatch": float(seed_runs["tier2_longitudinal_bias_enabled"]["fraction_steps_with_nonzero_mismatch"]),
                "delta_fraction_steps_with_nonzero_mismatch": float(delta["delta_fraction_steps_with_nonzero_mismatch"]),
                "tier1_mismatch_at_switch_steps_mean": float(seed_runs["tier1_nominal"]["mismatch_at_switch_steps_mean"]),
                "tier2_mismatch_at_switch_steps_mean": float(seed_runs["tier2_longitudinal_bias_enabled"]["mismatch_at_switch_steps_mean"]),
                "delta_mismatch_at_switch_steps_mean": float(delta["delta_mismatch_at_switch_steps_mean"]),
            }
        )
        runs_by_seed[str(seed_id)] = {
            "x0_seed": int(x0_seed),
            "sim_seed": int(sim_seed),
            "x0": [float(v) for v in x0],
            "tier1_nominal": seed_runs["tier1_nominal"],
            "tier2_longitudinal_bias_enabled": seed_runs["tier2_longitudinal_bias_enabled"],
            "delta_tier2_longitudinal_bias_minus_tier1": delta,
        }

    summary = {
        "seed_ids": seed_list,
        "selection_mode": str(getattr(getattr(cfg, "selection", object()), "mode", "unknown")),
        "monitor_mode": str(getattr(getattr(cfg, "monitor", object()), "mode", "disabled"))
        if bool(getattr(getattr(cfg, "monitor", object()), "enabled", False))
        else "disabled",
        "evaluator_mode": str(getattr(getattr(cfg, "evaluation", object()), "mode", "unknown")),
        "per_seed": runs_by_seed,
        "delta_summary": _delta_summary(paired_rows),
    }
    dump_json(out / "tier2_second_residual_seed_sweep.json", summary)
    _write_csv(out / "tier2_second_residual_seed_sweep.csv", paired_rows)

    _hist_plot(
        np.asarray([float(row["delta_switch_count"]) for row in paired_rows], dtype=float),
        xlabel="delta_switch_count",
        title="Tier-2 longitudinal-bias minus Tier-1 switch-count deltas",
        out_path=out / "delta_switches_hist.png",
    )
    _hist_plot(
        np.asarray([float(row["delta_tracking_error_mean"]) for row in paired_rows], dtype=float),
        xlabel="delta_tracking_error_mean",
        title="Tier-2 longitudinal-bias minus Tier-1 tracking-error-mean deltas",
        out_path=out / "delta_tracking_hist.png",
    )
    _hist_plot(
        np.asarray([float(row["delta_max_applied_jump"]) for row in paired_rows], dtype=float),
        xlabel="delta_max_applied_jump",
        title="Tier-2 longitudinal-bias minus Tier-1 applied-jump deltas",
        out_path=out / "delta_jumps_hist.png",
    )
    _hist_plot(
        np.asarray([float(row["delta_mean_nominal_vs_plant_gap"]) for row in paired_rows], dtype=float),
        xlabel="delta_mean_nominal_vs_plant_gap",
        title="Tier-2 longitudinal-bias minus Tier-1 mean mismatch deltas",
        out_path=out / "delta_mismatch_hist.png",
    )
    _scatter_plot(
        np.asarray([float(row["delta_mismatch_at_switch_steps_mean"]) for row in paired_rows], dtype=float),
        np.asarray([float(row["delta_switch_count"]) for row in paired_rows], dtype=float),
        xlabel="delta_mismatch_at_switch_steps_mean",
        ylabel="delta_switch_count",
        title="Tier-2 longitudinal-bias mismatch-vs-switch deltas",
        out_path=out / "delta_mismatch_vs_switch_scatter.png",
    )

    return out / "tier2_second_residual_seed_sweep.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired seed sweep for Tier-1 nominal vs Tier-2 longitudinal-bias residual.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier2_second_residual_seed_sweep")
    args = parser.parse_args()
    out = run_tier2_second_residual_seed_sweep(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

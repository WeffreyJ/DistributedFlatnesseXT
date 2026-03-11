"""Paired Tier-0 vs Tier-1 seed sweep under frozen controller logic."""

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
from src.experiments.runtime_monitor_audit import _collect_blocked_intervals, _sample_x0
from src.verify.utils import dump_json, make_results_dir


DEFAULT_SEED_IDS = list(range(12))


def _clone_cfg_for_family(cfg, plant_family: str):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = str(plant_family)
    return cfg_mode


def _metrics_for_run(sim: dict[str, object], plant_family: str, seed_id: int) -> dict[str, float | int | str]:
    intervals = _collect_blocked_intervals(sim)
    blocked_steps_total = int(sum(int(interval["duration_steps"]) for interval in intervals))
    block_durations = [int(interval["duration_steps"]) for interval in intervals]
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    return {
        "seed_id": int(seed_id),
        "plant_family": str(plant_family),
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in sim.get("switch_reason", []) if str(reason) == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
        "blocked_steps_total": blocked_steps_total,
        "unique_blocked_transition_starts": int(len(intervals)),
        "mean_block_duration_steps": float(np.mean(block_durations)) if block_durations else 0.0,
    }


def _paired_delta(seed_id: int, tier0: dict[str, object], tier1: dict[str, object]) -> dict[str, float | int]:
    return {
        "seed_id": int(seed_id),
        "delta_switch_count": int(tier1["switch_count"]) - int(tier0["switch_count"]),
        "delta_transition_start_count": int(tier1["transition_start_count"]) - int(tier0["transition_start_count"]),
        "delta_blend_active_steps": int(tier1["blend_active_steps"]) - int(tier0["blend_active_steps"]),
        "delta_tracking_error_mean": float(tier1["tracking_error_mean"]) - float(tier0["tracking_error_mean"]),
        "delta_tracking_error_max": float(tier1["tracking_error_max"]) - float(tier0["tracking_error_max"]),
        "delta_max_raw_jump": float(tier1["max_raw_jump"]) - float(tier0["max_raw_jump"]),
        "delta_max_applied_jump": float(tier1["max_applied_jump"]) - float(tier0["max_applied_jump"]),
        "delta_blocked_steps_total": int(tier1["blocked_steps_total"]) - int(tier0["blocked_steps_total"]),
        "delta_unique_blocked_transition_starts": int(tier1["unique_blocked_transition_starts"])
        - int(tier0["unique_blocked_transition_starts"]),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "seed_id",
        "tier0_switch_count",
        "tier1_switch_count",
        "delta_switch_count",
        "tier0_transition_start_count",
        "tier1_transition_start_count",
        "delta_transition_start_count",
        "tier0_blend_active_steps",
        "tier1_blend_active_steps",
        "delta_blend_active_steps",
        "tier0_tracking_error_mean",
        "tier1_tracking_error_mean",
        "delta_tracking_error_mean",
        "tier0_tracking_error_max",
        "tier1_tracking_error_max",
        "delta_tracking_error_max",
        "tier0_max_raw_jump",
        "tier1_max_raw_jump",
        "delta_max_raw_jump",
        "tier0_max_applied_jump",
        "tier1_max_applied_jump",
        "delta_max_applied_jump",
        "tier0_blocked_steps_total",
        "tier1_blocked_steps_total",
        "delta_blocked_steps_total",
        "tier0_unique_blocked_transition_starts",
        "tier1_unique_blocked_transition_starts",
        "delta_unique_blocked_transition_starts",
        "tier0_mean_block_duration_steps",
        "tier1_mean_block_duration_steps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hist_plot(values: np.ndarray, *, xlabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if values.size:
        bins = min(20, max(5, values.size))
        ax.hist(values, bins=bins)
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
        "delta_blocked_steps_total",
        "delta_unique_blocked_transition_starts",
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


def run_tier0_tier1_seed_sweep(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier0_tier1_seed_sweep",
    seed_ids: list[int] | None = None,
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("tier0_tier1_seed_sweep") if out_dir == "results/tier0_tier1_seed_sweep" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seed_list = [int(v) for v in (seed_ids if seed_ids is not None else DEFAULT_SEED_IDS)]
    paired_rows: list[dict[str, object]] = []
    runs_by_seed: dict[str, dict[str, object]] = {}

    for seed_id in seed_list:
        x0_seed = int(cfg.seed) + 19401 + 10 * seed_id
        sim_seed = int(cfg.seed) + 19421 + 10 * seed_id
        x0 = _sample_x0(cfg, seed=x0_seed)
        options = SimOptions(
            blending_on=True,
            noise_delta=0.0,
            seed=sim_seed,
            disable_switching=False,
        )

        seed_runs: dict[str, dict[str, object]] = {}
        for plant_family in ["tier0", "tier1"]:
            cfg_mode = _clone_cfg_for_family(cfg, plant_family)
            sim = simulate_closed_loop(
                cfg_mode,
                x0=x0,
                horizon=float(cfg_mode.system.horizon),
                options=options,
            )
            seed_runs[plant_family] = _metrics_for_run(sim, plant_family, seed_id)

        delta = _paired_delta(seed_id, seed_runs["tier0"], seed_runs["tier1"])
        paired_rows.append(
            {
                "seed_id": int(seed_id),
                "tier0_switch_count": int(seed_runs["tier0"]["switch_count"]),
                "tier1_switch_count": int(seed_runs["tier1"]["switch_count"]),
                "delta_switch_count": int(delta["delta_switch_count"]),
                "tier0_transition_start_count": int(seed_runs["tier0"]["transition_start_count"]),
                "tier1_transition_start_count": int(seed_runs["tier1"]["transition_start_count"]),
                "delta_transition_start_count": int(delta["delta_transition_start_count"]),
                "tier0_blend_active_steps": int(seed_runs["tier0"]["blend_active_steps"]),
                "tier1_blend_active_steps": int(seed_runs["tier1"]["blend_active_steps"]),
                "delta_blend_active_steps": int(delta["delta_blend_active_steps"]),
                "tier0_tracking_error_mean": float(seed_runs["tier0"]["tracking_error_mean"]),
                "tier1_tracking_error_mean": float(seed_runs["tier1"]["tracking_error_mean"]),
                "delta_tracking_error_mean": float(delta["delta_tracking_error_mean"]),
                "tier0_tracking_error_max": float(seed_runs["tier0"]["tracking_error_max"]),
                "tier1_tracking_error_max": float(seed_runs["tier1"]["tracking_error_max"]),
                "delta_tracking_error_max": float(delta["delta_tracking_error_max"]),
                "tier0_max_raw_jump": float(seed_runs["tier0"]["max_raw_jump"]),
                "tier1_max_raw_jump": float(seed_runs["tier1"]["max_raw_jump"]),
                "delta_max_raw_jump": float(delta["delta_max_raw_jump"]),
                "tier0_max_applied_jump": float(seed_runs["tier0"]["max_applied_jump"]),
                "tier1_max_applied_jump": float(seed_runs["tier1"]["max_applied_jump"]),
                "delta_max_applied_jump": float(delta["delta_max_applied_jump"]),
                "tier0_blocked_steps_total": int(seed_runs["tier0"]["blocked_steps_total"]),
                "tier1_blocked_steps_total": int(seed_runs["tier1"]["blocked_steps_total"]),
                "delta_blocked_steps_total": int(delta["delta_blocked_steps_total"]),
                "tier0_unique_blocked_transition_starts": int(seed_runs["tier0"]["unique_blocked_transition_starts"]),
                "tier1_unique_blocked_transition_starts": int(seed_runs["tier1"]["unique_blocked_transition_starts"]),
                "delta_unique_blocked_transition_starts": int(delta["delta_unique_blocked_transition_starts"]),
                "tier0_mean_block_duration_steps": float(seed_runs["tier0"]["mean_block_duration_steps"]),
                "tier1_mean_block_duration_steps": float(seed_runs["tier1"]["mean_block_duration_steps"]),
            }
        )
        runs_by_seed[str(seed_id)] = {
            "x0_seed": int(x0_seed),
            "sim_seed": int(sim_seed),
            "x0": [float(v) for v in x0],
            "tier0": seed_runs["tier0"],
            "tier1": seed_runs["tier1"],
            "delta_tier1_minus_tier0": delta,
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
    dump_json(out / "tier0_tier1_seed_sweep.json", summary)
    _write_csv(out / "tier0_tier1_seed_sweep.csv", paired_rows)

    _hist_plot(
        np.asarray([float(row["delta_switch_count"]) for row in paired_rows], dtype=float),
        xlabel="delta_switch_count",
        title="Tier-1 minus Tier-0 switch-count deltas",
        out_path=out / "delta_switches_hist.png",
    )
    _hist_plot(
        np.asarray([float(row["delta_tracking_error_mean"]) for row in paired_rows], dtype=float),
        xlabel="delta_tracking_error_mean",
        title="Tier-1 minus Tier-0 tracking-error-mean deltas",
        out_path=out / "delta_tracking_hist.png",
    )
    _hist_plot(
        np.asarray([float(row["delta_max_applied_jump"]) for row in paired_rows], dtype=float),
        xlabel="delta_max_applied_jump",
        title="Tier-1 minus Tier-0 applied-jump deltas",
        out_path=out / "delta_jumps_hist.png",
    )
    _hist_plot(
        np.asarray([float(row["delta_blocked_steps_total"]) for row in paired_rows], dtype=float),
        xlabel="delta_blocked_steps_total",
        title="Tier-1 minus Tier-0 blocked-step deltas",
        out_path=out / "delta_blocking_hist.png",
    )
    _scatter_plot(
        np.asarray([float(row["delta_max_applied_jump"]) for row in paired_rows], dtype=float),
        np.asarray([float(row["delta_switch_count"]) for row in paired_rows], dtype=float),
        xlabel="delta_max_applied_jump",
        ylabel="delta_switch_count",
        title="Applied-jump delta vs switch-count delta",
        out_path=out / "delta_jump_vs_switch_scatter.png",
    )

    return out / "tier0_tier1_seed_sweep.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired Tier-0 vs Tier-1 seed sweep")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier0_tier1_seed_sweep")
    parser.add_argument("--seed-ids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    out = run_tier0_tier1_seed_sweep(
        config_path=args.config,
        out_dir=args.out,
        seed_ids=args.seed_ids,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

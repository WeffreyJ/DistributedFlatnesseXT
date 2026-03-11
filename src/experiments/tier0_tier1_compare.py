"""Frozen-controller comparison between Tier-0 and Tier-1 plant families."""

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


def _clone_cfg_for_family(cfg, plant_family: str):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = str(plant_family)
    return cfg_mode


def _metrics_for_run(sim: dict[str, object], cfg, plant_family: str) -> dict[str, float | int | str]:
    intervals = _collect_blocked_intervals(sim)
    blocked_steps_total = int(sum(int(interval["duration_steps"]) for interval in intervals))
    block_durations = [int(interval["duration_steps"]) for interval in intervals]
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    predicted_gap = np.asarray(sim.get("monitor_predicted_gap", []), dtype=float)
    return {
        "plant_family": str(plant_family),
        "selection_mode": str(getattr(getattr(cfg, "selection", object()), "mode", "unknown")),
        "monitor_mode": str(getattr(getattr(cfg, "monitor", object()), "mode", "disabled"))
        if bool(getattr(getattr(cfg, "monitor", object()), "enabled", False))
        else "disabled",
        "evaluator_mode": str(getattr(getattr(cfg, "evaluation", object()), "mode", "unknown")),
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
        "mean_monitor_predicted_gap": float(np.mean(predicted_gap)) if predicted_gap.size else 0.0,
        "max_monitor_predicted_gap": float(np.max(predicted_gap)) if predicted_gap.size else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "plant_family",
        "selection_mode",
        "monitor_mode",
        "evaluator_mode",
        "switch_count",
        "transition_start_count",
        "blend_active_steps",
        "tracking_error_mean",
        "tracking_error_max",
        "max_raw_jump",
        "max_applied_jump",
        "blocked_steps_total",
        "unique_blocked_transition_starts",
        "mean_block_duration_steps",
        "mean_monitor_predicted_gap",
        "max_monitor_predicted_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_grouped_bars(rows: list[dict[str, object]], metrics: list[str], title: str, ylabel: str, out_path: Path) -> None:
    labels = [str(row["plant_family"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        ax.bar(x + idx * width - 0.4 + width / 2.0, values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if metrics:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_tier0_tier1_compare(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier0_tier1_compare",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("tier0_tier1_compare") if out_dir == "results/tier0_tier1_compare" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = _sample_x0(cfg, seed=int(cfg.seed) + 19301)
    sim_seed = int(cfg.seed) + 19321
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    rows: list[dict[str, object]] = []
    for plant_family in ["tier0", "tier1"]:
        cfg_mode = _clone_cfg_for_family(cfg, plant_family)
        sim = simulate_closed_loop(
            cfg_mode,
            x0=x0,
            horizon=float(cfg_mode.system.horizon),
            options=options,
        )
        rows.append(_metrics_for_run(sim, cfg_mode, plant_family))

    summary = {
        "shared_x0": [float(v) for v in x0],
        "shared_seed": int(sim_seed),
        "runs": {str(row["plant_family"]): row for row in rows},
        "delta_tier1_minus_tier0": {
            key: float(rows[1][key]) - float(rows[0][key])
            for key in [
                "switch_count",
                "transition_start_count",
                "blend_active_steps",
                "tracking_error_mean",
                "tracking_error_max",
                "max_raw_jump",
                "max_applied_jump",
                "blocked_steps_total",
                "unique_blocked_transition_starts",
                "mean_block_duration_steps",
                "mean_monitor_predicted_gap",
                "max_monitor_predicted_gap",
            ]
        },
    }
    dump_json(out / "tier0_tier1_compare.json", summary)
    _write_csv(out / "tier0_tier1_compare.csv", rows)

    _plot_grouped_bars(
        rows,
        ["switch_count", "transition_start_count", "blend_active_steps"],
        "Tier-0 vs Tier-1 switching burden",
        "count",
        out / "switching_compare.png",
    )
    _plot_grouped_bars(
        rows,
        ["tracking_error_mean", "tracking_error_max"],
        "Tier-0 vs Tier-1 tracking error",
        "tracking error",
        out / "tracking_compare.png",
    )
    _plot_grouped_bars(
        rows,
        ["max_raw_jump", "max_applied_jump"],
        "Tier-0 vs Tier-1 jump magnitudes",
        "jump norm",
        out / "jumps_compare.png",
    )
    _plot_grouped_bars(
        rows,
        ["blocked_steps_total", "unique_blocked_transition_starts", "mean_block_duration_steps"],
        "Tier-0 vs Tier-1 monitor blocking burden",
        "count / steps",
        out / "monitor_blocking_compare.png",
    )

    return out / "tier0_tier1_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen Tier-0 and Tier-1 controller rollouts.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier0_tier1_compare")
    args = parser.parse_args()
    out = run_tier0_tier1_compare(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

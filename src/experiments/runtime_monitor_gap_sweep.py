"""Sweep monitor.predicted_gap_warn under fixed active-monitor settings."""

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


DEFAULT_THRESHOLDS = [0.25, 0.35, 0.45, 0.60, 0.80, 1.00]


def _clone_cfg(cfg, *, monitor_enabled: bool, monitor_mode: str, predicted_gap_warn: float | None = None):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.monitor.enabled = bool(monitor_enabled)
    cfg_mode.monitor.mode = str(monitor_mode)
    if predicted_gap_warn is not None:
        cfg_mode.monitor.predicted_gap_warn = float(predicted_gap_warn)
    return cfg_mode


def _metrics_for_run(sim: dict[str, object], predicted_gap_warn: float | None) -> dict[str, float | int | str | None]:
    intervals = _collect_blocked_intervals(sim)
    blocked_steps_total = int(sum(int(interval["duration_steps"]) for interval in intervals))
    block_durations = [int(interval["duration_steps"]) for interval in intervals]
    return {
        "predicted_gap_warn": None if predicted_gap_warn is None else float(predicted_gap_warn),
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in sim.get("switch_reason", []) if str(reason) == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "blocked_steps_total": blocked_steps_total,
        "unique_blocked_transition_starts": int(len(intervals)),
        "mean_block_duration_steps": float(np.mean(block_durations)) if block_durations else 0.0,
        "tracking_error_mean": float(np.mean(np.asarray(sim.get("e_norm", [0.0]), dtype=float))),
        "tracking_error_max": float(np.max(np.asarray(sim.get("e_norm", [0.0]), dtype=float))),
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
    }


def _plot_single_series(rows: list[dict[str, object]], xkey: str, ykeys: list[str], ylabel: str, title: str, out_path: Path) -> None:
    x = np.asarray([float(row[xkey]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    for ykey in ykeys:
        ax.plot(x, [float(row[ykey]) for row in rows], marker="o", label=ykey)
    ax.set_xlabel(xkey)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(ykeys) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _choose_best_threshold(baseline: dict[str, object], rows: list[dict[str, object]]) -> float | None:
    baseline_tracking = float(baseline["tracking_error_mean"])
    baseline_switches = int(baseline["switch_count"])
    baseline_jump = float(baseline["max_applied_jump"])

    qualified: list[dict[str, object]] = []
    for row in rows:
        if float(row["tracking_error_mean"]) > 1.15 * baseline_tracking:
            continue
        if int(row["switch_count"]) >= baseline_switches:
            continue
        if float(row["max_applied_jump"]) >= baseline_jump:
            continue
        qualified.append(row)

    if not qualified:
        return None
    best = min(
        qualified,
        key=lambda row: (
            float(row["tracking_error_mean"]),
            int(row["switch_count"]),
            float(row["max_applied_jump"]),
        ),
    )
    return float(best["predicted_gap_warn"])


def run_runtime_monitor_gap_sweep(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/runtime_monitor_gap_sweep",
    thresholds: list[float] | None = None,
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("runtime_monitor_gap_sweep") if out_dir == "results/runtime_monitor_gap_sweep" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    threshold_values = [float(v) for v in (thresholds if thresholds is not None else DEFAULT_THRESHOLDS)]
    x0 = _sample_x0(cfg, seed=int(cfg.seed) + 15201)
    sim_seed = int(cfg.seed) + 15221
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    baseline_cfg = _clone_cfg(cfg, monitor_enabled=False, monitor_mode="shadow")
    baseline_sim = simulate_closed_loop(
        baseline_cfg,
        x0=x0,
        horizon=float(baseline_cfg.system.horizon),
        options=options,
    )
    baseline_metrics = _metrics_for_run(baseline_sim, predicted_gap_warn=None)

    rows: list[dict[str, object]] = []
    for threshold in threshold_values:
        cfg_mode = _clone_cfg(
            cfg,
            monitor_enabled=True,
            monitor_mode="active",
            predicted_gap_warn=threshold,
        )
        sim = simulate_closed_loop(
            cfg_mode,
            x0=x0,
            horizon=float(cfg_mode.system.horizon),
            options=options,
        )
        rows.append(_metrics_for_run(sim, predicted_gap_warn=threshold))

    best_candidate_threshold = _choose_best_threshold(baseline_metrics, rows)
    summary = {
        "shared_x0": [float(v) for v in x0],
        "shared_seed": int(sim_seed),
        "baseline_reference_metrics": baseline_metrics,
        "active_monitor_runs": rows,
        "best_candidate_threshold": best_candidate_threshold,
    }
    dump_json(out / "runtime_monitor_gap_sweep.json", summary)

    with (out / "runtime_monitor_gap_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "predicted_gap_warn",
                "switch_count",
                "transition_start_count",
                "blend_active_steps",
                "blocked_steps_total",
                "unique_blocked_transition_starts",
                "mean_block_duration_steps",
                "tracking_error_mean",
                "tracking_error_max",
                "max_raw_jump",
                "max_applied_jump",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _plot_single_series(
        rows,
        "predicted_gap_warn",
        ["tracking_error_mean", "tracking_error_max"],
        "tracking error",
        "Tracking vs predicted-gap threshold",
        out / "tracking_vs_gap_threshold.png",
    )
    _plot_single_series(
        rows,
        "predicted_gap_warn",
        ["switch_count", "transition_start_count", "blend_active_steps"],
        "count",
        "Switching burden vs predicted-gap threshold",
        out / "switches_vs_gap_threshold.png",
    )
    _plot_single_series(
        rows,
        "predicted_gap_warn",
        ["blocked_steps_total", "unique_blocked_transition_starts"],
        "count",
        "Blocked-interval burden vs predicted-gap threshold",
        out / "blocked_steps_vs_gap_threshold.png",
    )
    _plot_single_series(
        rows,
        "predicted_gap_warn",
        ["max_raw_jump", "max_applied_jump"],
        "jump norm",
        "Jump magnitudes vs predicted-gap threshold",
        out / "jumps_vs_gap_threshold.png",
    )

    return out / "runtime_monitor_gap_sweep.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep monitor.predicted_gap_warn under active monitoring")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/runtime_monitor_gap_sweep")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    args = parser.parse_args()
    out = run_runtime_monitor_gap_sweep(
        config_path=args.config,
        out_dir=args.out,
        thresholds=args.thresholds,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

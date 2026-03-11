"""Monitor-consequence study on edge-boundary scenarios with active selector fixed."""

from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.experiments.runtime_monitor_audit import _collect_blocked_intervals
from src.verify.utils import dump_json, make_results_dir


MONITOR_MODES = ["disabled", "shadow", "active"]
BLOCK_ACTIONS = {"hold_current", "extend_lockout", "fallback_fixed_order"}


def _edge_cases(cfg) -> list[dict[str, object]]:
    gamma_edge = float(cfg.system.gamma_edge)
    return [
        {
            "case_name": "edge_radius_just_inside",
            "x0": [gamma_edge + 0.015, 0.0, -0.20, 0.0, 0.0, 0.0],
            "tier1_lateral_offsets": [0.0, 2.45, -2.45],
        },
        {
            "case_name": "edge_radius_just_outside",
            "x0": [gamma_edge + 0.015, 0.0, -0.20, 0.0, 0.0, 0.0],
            "tier1_lateral_offsets": [0.0, 2.55, -2.55],
        },
    ]


def _clone_cfg(cfg, *, plant_family: str, monitor_mode: str, lateral_offsets: list[float] | None = None):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = str(plant_family)
    cfg_mode.selection.mode = "active_lexicographic"
    cfg_mode.monitor.enabled = monitor_mode != "disabled"
    cfg_mode.monitor.mode = "shadow" if monitor_mode == "disabled" else str(monitor_mode)
    cfg_mode.evaluation.mode = "upstream_truncated"
    if lateral_offsets is not None:
        cfg_mode.system.tier1.geometry.lateral_offsets = [float(v) for v in lateral_offsets]
    return cfg_mode


def _reason_counts_stepwise(sim: dict[str, object], blocked_mask: np.ndarray) -> dict[str, int]:
    counts: Counter[str] = Counter()
    reason_lists = [[str(r) for r in reasons] for reasons in sim.get("monitor_risk_reasons", [])]
    for idx, is_blocked in enumerate(blocked_mask):
        if not is_blocked:
            continue
        if idx < len(reason_lists) and reason_lists[idx]:
            counts.update(reason_lists[idx])
        else:
            counts.update(["unknown"])
    return dict(sorted(counts.items()))


def _reason_counts_unique_startwise(intervals: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter()
    for interval in intervals:
        reasons = interval.get("all_reasons_union", ["unknown"])
        primary = reasons[0] if reasons else "unknown"
        counts.update([str(primary)])
    return dict(sorted(counts.items()))


def _run_metrics(sim: dict[str, object], *, case_name: str, plant_family: str, monitor_mode: str) -> tuple[dict[str, object], dict[str, int], dict[str, int]]:
    intervals = _collect_blocked_intervals(sim)
    blocked_steps_total = int(sum(int(interval["duration_steps"]) for interval in intervals))
    block_durations = [int(interval["duration_steps"]) for interval in intervals]
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    predicted_gap = np.asarray(sim.get("monitor_predicted_gap", []), dtype=float)
    monitor_actions = [str(v) for v in sim.get("monitor_action", [])]
    selector_switch_eligible = np.asarray(sim.get("selector_switch_eligible", []), dtype=bool)
    blocked_mask = np.asarray(
        [
            bool(selector_switch_eligible[idx]) and action in BLOCK_ACTIONS
            for idx, action in enumerate(monitor_actions)
        ],
        dtype=bool,
    )
    fallback_active = np.asarray(sim.get("fallback_active", []), dtype=bool)
    stepwise = _reason_counts_stepwise(sim, blocked_mask)
    unique_startwise = _reason_counts_unique_startwise(intervals)
    metrics = {
        "case_name": str(case_name),
        "plant_family": str(plant_family),
        "selection_mode": "active_lexicographic",
        "monitor_mode": str(monitor_mode),
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
        "hold_action_count": int(sum(1 for action in monitor_actions if action in {"hold_current", "extend_lockout"})),
        "fallback_activation_count": int(np.count_nonzero(fallback_active)),
        "mean_monitor_predicted_gap": float(np.mean(predicted_gap)) if predicted_gap.size else 0.0,
        "max_monitor_predicted_gap": float(np.max(predicted_gap)) if predicted_gap.size else 0.0,
    }
    return metrics, stepwise, unique_startwise


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_name",
        "plant_family",
        "selection_mode",
        "monitor_mode",
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
        "hold_action_count",
        "fallback_activation_count",
        "mean_monitor_predicted_gap",
        "max_monitor_predicted_gap",
        "reason_counts_stepwise",
        "reason_counts_unique_startwise",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grouped_plot(rows: list[dict[str, object]], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [f'{row["case_name"]}\n{row["plant_family"]}\n{row["monitor_mode"]}' for row in rows]
    values = [float(row[metric]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_edge_boundary_monitor_compare(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/edge_boundary_monitor_compare",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("edge_boundary_monitor_compare") if out_dir == "results/edge_boundary_monitor_compare" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = _edge_cases(cfg)
    sim_seed = int(cfg.seed) + 19721
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    rows: list[dict[str, object]] = []
    summary_cases: dict[str, object] = {}

    for case in cases:
        case_name = str(case["case_name"])
        x0 = np.asarray(case["x0"], dtype=float)
        lateral_offsets = case["tier1_lateral_offsets"]
        summary_cases[case_name] = {
            "x0": [float(v) for v in x0],
            "tier1_lateral_offsets": [float(v) for v in lateral_offsets],
            "families": {},
        }
        for plant_family in ["tier0", "tier1"]:
            family_runs: dict[str, object] = {}
            for monitor_mode in MONITOR_MODES:
                cfg_mode = _clone_cfg(
                    cfg,
                    plant_family=plant_family,
                    monitor_mode=monitor_mode,
                    lateral_offsets=lateral_offsets if plant_family == "tier1" else None,
                )
                sim = simulate_closed_loop(
                    cfg_mode,
                    x0=x0,
                    horizon=float(cfg_mode.system.horizon),
                    options=options,
                )
                metrics, stepwise, unique_startwise = _run_metrics(
                    sim,
                    case_name=case_name,
                    plant_family=plant_family,
                    monitor_mode=monitor_mode,
                )
                csv_row = dict(metrics)
                csv_row["reason_counts_stepwise"] = str(stepwise)
                csv_row["reason_counts_unique_startwise"] = str(unique_startwise)
                rows.append(csv_row)
                family_runs[monitor_mode] = {
                    "metrics": metrics,
                    "reason_counts_stepwise": stepwise,
                    "reason_counts_unique_startwise": unique_startwise,
                }

            disabled = family_runs["disabled"]["metrics"]
            shadow = family_runs["shadow"]["metrics"]
            active = family_runs["active"]["metrics"]
            summary_cases[case_name]["families"][plant_family] = {
                "disabled": family_runs["disabled"],
                "shadow": family_runs["shadow"],
                "active": family_runs["active"],
                "delta_active_minus_disabled": {
                    "switch_count": int(active["switch_count"]) - int(disabled["switch_count"]),
                    "transition_start_count": int(active["transition_start_count"]) - int(disabled["transition_start_count"]),
                    "blend_active_steps": int(active["blend_active_steps"]) - int(disabled["blend_active_steps"]),
                    "tracking_error_mean": float(active["tracking_error_mean"]) - float(disabled["tracking_error_mean"]),
                    "tracking_error_max": float(active["tracking_error_max"]) - float(disabled["tracking_error_max"]),
                    "max_raw_jump": float(active["max_raw_jump"]) - float(disabled["max_raw_jump"]),
                    "max_applied_jump": float(active["max_applied_jump"]) - float(disabled["max_applied_jump"]),
                    "blocked_steps_total": int(active["blocked_steps_total"]) - int(disabled["blocked_steps_total"]),
                    "unique_blocked_transition_starts": int(active["unique_blocked_transition_starts"])
                    - int(disabled["unique_blocked_transition_starts"]),
                    "mean_block_duration_steps": float(active["mean_block_duration_steps"]) - float(disabled["mean_block_duration_steps"]),
                    "hold_action_count": int(active["hold_action_count"]) - int(disabled["hold_action_count"]),
                    "fallback_activation_count": int(active["fallback_activation_count"]) - int(disabled["fallback_activation_count"]),
                },
                "delta_shadow_minus_disabled": {
                    "switch_count": int(shadow["switch_count"]) - int(disabled["switch_count"]),
                    "transition_start_count": int(shadow["transition_start_count"]) - int(disabled["transition_start_count"]),
                    "blend_active_steps": int(shadow["blend_active_steps"]) - int(disabled["blend_active_steps"]),
                    "tracking_error_mean": float(shadow["tracking_error_mean"]) - float(disabled["tracking_error_mean"]),
                    "tracking_error_max": float(shadow["tracking_error_max"]) - float(disabled["tracking_error_max"]),
                    "max_raw_jump": float(shadow["max_raw_jump"]) - float(disabled["max_raw_jump"]),
                    "max_applied_jump": float(shadow["max_applied_jump"]) - float(disabled["max_applied_jump"]),
                    "blocked_steps_total": int(shadow["blocked_steps_total"]) - int(disabled["blocked_steps_total"]),
                    "unique_blocked_transition_starts": int(shadow["unique_blocked_transition_starts"])
                    - int(disabled["unique_blocked_transition_starts"]),
                    "mean_block_duration_steps": float(shadow["mean_block_duration_steps"]) - float(disabled["mean_block_duration_steps"]),
                    "hold_action_count": int(shadow["hold_action_count"]) - int(disabled["hold_action_count"]),
                    "fallback_activation_count": int(shadow["fallback_activation_count"]) - int(disabled["fallback_activation_count"]),
                },
            }

    summary = {
        "cases": summary_cases,
        "selection_mode": "active_lexicographic",
        "monitor_modes": MONITOR_MODES,
        "evaluator_mode": "upstream_truncated",
        "shared_sim_seed": int(sim_seed),
    }
    dump_json(out / "edge_boundary_monitor_compare.json", summary)
    _write_csv(out / "edge_boundary_monitor_compare.csv", rows)

    _grouped_plot(
        rows,
        "switch_count",
        "Edge-boundary monitor comparison: switch count",
        "switch count",
        out / "switch_compare_by_case_monitor.png",
    )
    _grouped_plot(
        rows,
        "max_applied_jump",
        "Edge-boundary monitor comparison: max applied jump",
        "max applied jump",
        out / "jump_compare_by_case_monitor.png",
    )
    _grouped_plot(
        rows,
        "tracking_error_mean",
        "Edge-boundary monitor comparison: tracking error mean",
        "tracking error mean",
        out / "tracking_compare_by_case_monitor.png",
    )
    _grouped_plot(
        rows,
        "blocked_steps_total",
        "Edge-boundary monitor comparison: blocked steps",
        "blocked steps",
        out / "blocking_compare_by_case_monitor.png",
    )

    return out / "edge_boundary_monitor_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor comparison on edge-boundary scenarios.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/edge_boundary_monitor_compare")
    args = parser.parse_args()
    out = run_edge_boundary_monitor_compare(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

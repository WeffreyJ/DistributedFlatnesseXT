"""Compare current and redesigned monitor logic on Tier-1 edge-boundary cases."""

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

import src.control.closed_loop as closed_loop_module
import src.hybrid.runtime_monitor as runtime_monitor_module
from src.config import load_config
from src.control.closed_loop import SimOptions
from src.experiments.runtime_monitor_audit import _collect_blocked_intervals
from src.verify.utils import dump_json, make_results_dir


MONITOR_VARIANTS = ["disabled", "current_active", "redesigned_active"]
BLOCK_ACTIONS = {"hold_current", "extend_lockout", "fallback_fixed_order"}
RELATIVE_GAP_MULTIPLIER = 1.35
RECENT_GAP_WINDOW = 12


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


def _clone_cfg(cfg, *, monitor_variant: str, lateral_offsets: list[float]):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = "tier1"
    cfg_mode.selection.mode = "active_lexicographic"
    cfg_mode.evaluation.mode = "upstream_truncated"
    cfg_mode.system.tier1.geometry.lateral_offsets = [float(v) for v in lateral_offsets]
    cfg_mode.monitor.enabled = monitor_variant != "disabled"
    cfg_mode.monitor.mode = "active" if monitor_variant != "disabled" else "shadow"
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


def _run_metrics(sim: dict[str, object], *, case_name: str, monitor_variant: str) -> dict[str, object]:
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
    stepwise = _reason_counts_stepwise(sim, blocked_mask)
    unique_startwise = _reason_counts_unique_startwise(intervals)
    fallback_active = np.asarray(sim.get("fallback_active", []), dtype=bool)
    return {
        "case_name": str(case_name),
        "monitor_variant": str(monitor_variant),
        "selection_mode": "active_lexicographic",
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
        "reason_counts_stepwise": stepwise,
        "reason_counts_unique_startwise": unique_startwise,
    }


def _build_redesigned_monitor():
    recent_gaps: list[float] = []

    def redesigned_runtime_monitor_step(*, x, current_pi, candidate_pi, params, selector_meta, history, now_t, state=None):
        del x
        state = dict(state or {})
        base = runtime_monitor_module.evaluate_candidate_risk(
            x=None,
            current_pi=current_pi,
            candidate_pi=candidate_pi,
            params=params,
            selector_meta=selector_meta,
            history=history,
            now_t=now_t,
        )
        proposed_switch = bool(base.get("proposed_switch", False))
        predicted_gap = float(base.get("predicted_gap", 0.0))
        monitor_cfg = getattr(params, "monitor", None)
        gap_warn = float(getattr(monitor_cfg, "predicted_gap_warn", 0.25)) if monitor_cfg is not None else 0.25

        if proposed_switch:
            recent_gaps.append(predicted_gap)
            if len(recent_gaps) > RECENT_GAP_WINDOW:
                del recent_gaps[:-RECENT_GAP_WINDOW]

        local_baseline = max(gap_warn, float(np.median(recent_gaps)) if recent_gaps else gap_warn, 1.0e-9)
        relative_gap = predicted_gap / local_baseline

        reasons = [str(r) for r in base.get("risk_reasons", []) if str(r) != "large_predicted_gap"]
        level = str(base.get("risk_level", "low"))
        has_other_reasons = len(reasons) > 0
        if proposed_switch and predicted_gap > gap_warn and relative_gap > RELATIVE_GAP_MULTIPLIER:
            reasons.append("large_relative_predicted_gap")
            if level != "high":
                level = "moderate"
        elif not has_other_reasons:
            level = "low"

        base["risk_reasons"] = reasons
        base["risk_level"] = level if reasons else "low"

        high_risk_streak = int(state.get("high_risk_streak", 0))
        if str(base["risk_level"]) == "high" and proposed_switch:
            high_risk_streak += 1
        else:
            high_risk_streak = 0
        action = runtime_monitor_module.decide_monitor_action(base, params, high_risk_streak=high_risk_streak)
        fallback_cfg = getattr(params, "fallback", None)
        fallback_pi = (
            [int(v) for v in getattr(fallback_cfg, "fixed_order", [])]
            if fallback_cfg is not None and hasattr(fallback_cfg, "fixed_order")
            else []
        )
        out = dict(base)
        out["monitor_action"] = str(action)
        out["fallback_pi"] = fallback_pi
        out["high_risk_streak"] = high_risk_streak
        out["fallback_active"] = bool(action == "fallback_fixed_order")
        return out

    return redesigned_runtime_monitor_step


def _simulate_variant(cfg_mode, x0: np.ndarray, options: SimOptions, monitor_variant: str):
    original = closed_loop_module.runtime_monitor_step
    try:
        if monitor_variant == "redesigned_active":
            closed_loop_module.runtime_monitor_step = _build_redesigned_monitor()
        else:
            closed_loop_module.runtime_monitor_step = original
        return closed_loop_module.simulate_closed_loop(
            cfg_mode,
            x0=x0,
            horizon=float(cfg_mode.system.horizon),
            options=options,
        )
    finally:
        closed_loop_module.runtime_monitor_step = original


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_name",
        "monitor_variant",
        "selection_mode",
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


def _plot_metric(rows: list[dict[str, object]], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [f'{row["case_name"]}\n{row["monitor_variant"]}' for row in rows]
    values = [float(row[metric]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_edge_boundary_monitor_redesign_compare(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/edge_boundary_monitor_redesign_compare",
) -> Path:
    cfg = load_config(config_path)
    out = (
        make_results_dir("edge_boundary_monitor_redesign_compare")
        if out_dir == "results/edge_boundary_monitor_redesign_compare"
        else Path(out_dir)
    )
    out.mkdir(parents=True, exist_ok=True)

    cases = _edge_cases(cfg)
    sim_seed = int(cfg.seed) + 19821
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
            "variants": {},
        }
        for monitor_variant in MONITOR_VARIANTS:
            cfg_mode = _clone_cfg(cfg, monitor_variant=monitor_variant, lateral_offsets=lateral_offsets)
            sim = _simulate_variant(cfg_mode, x0, options, monitor_variant)
            metrics = _run_metrics(sim, case_name=case_name, monitor_variant=monitor_variant)
            csv_row = dict(metrics)
            csv_row["reason_counts_stepwise"] = str(metrics["reason_counts_stepwise"])
            csv_row["reason_counts_unique_startwise"] = str(metrics["reason_counts_unique_startwise"])
            rows.append(csv_row)
            summary_cases[case_name]["variants"][monitor_variant] = metrics

        disabled = summary_cases[case_name]["variants"]["disabled"]
        current = summary_cases[case_name]["variants"]["current_active"]
        redesigned = summary_cases[case_name]["variants"]["redesigned_active"]
        summary_cases[case_name]["delta_current_minus_disabled"] = {
            "switch_count": int(current["switch_count"]) - int(disabled["switch_count"]),
            "transition_start_count": int(current["transition_start_count"]) - int(disabled["transition_start_count"]),
            "blend_active_steps": int(current["blend_active_steps"]) - int(disabled["blend_active_steps"]),
            "tracking_error_mean": float(current["tracking_error_mean"]) - float(disabled["tracking_error_mean"]),
            "max_raw_jump": float(current["max_raw_jump"]) - float(disabled["max_raw_jump"]),
            "max_applied_jump": float(current["max_applied_jump"]) - float(disabled["max_applied_jump"]),
            "blocked_steps_total": int(current["blocked_steps_total"]) - int(disabled["blocked_steps_total"]),
        }
        summary_cases[case_name]["delta_redesigned_minus_disabled"] = {
            "switch_count": int(redesigned["switch_count"]) - int(disabled["switch_count"]),
            "transition_start_count": int(redesigned["transition_start_count"]) - int(disabled["transition_start_count"]),
            "blend_active_steps": int(redesigned["blend_active_steps"]) - int(disabled["blend_active_steps"]),
            "tracking_error_mean": float(redesigned["tracking_error_mean"]) - float(disabled["tracking_error_mean"]),
            "max_raw_jump": float(redesigned["max_raw_jump"]) - float(disabled["max_raw_jump"]),
            "max_applied_jump": float(redesigned["max_applied_jump"]) - float(disabled["max_applied_jump"]),
            "blocked_steps_total": int(redesigned["blocked_steps_total"]) - int(disabled["blocked_steps_total"]),
        }
        summary_cases[case_name]["delta_redesigned_minus_current"] = {
            "switch_count": int(redesigned["switch_count"]) - int(current["switch_count"]),
            "transition_start_count": int(redesigned["transition_start_count"]) - int(current["transition_start_count"]),
            "blend_active_steps": int(redesigned["blend_active_steps"]) - int(current["blend_active_steps"]),
            "tracking_error_mean": float(redesigned["tracking_error_mean"]) - float(current["tracking_error_mean"]),
            "max_raw_jump": float(redesigned["max_raw_jump"]) - float(current["max_raw_jump"]),
            "max_applied_jump": float(redesigned["max_applied_jump"]) - float(current["max_applied_jump"]),
            "blocked_steps_total": int(redesigned["blocked_steps_total"]) - int(current["blocked_steps_total"]),
        }

    summary = {
        "cases": summary_cases,
        "selection_mode": "active_lexicographic",
        "plant_family": "tier1",
        "monitor_variants": MONITOR_VARIANTS,
        "relative_gap_multiplier": float(RELATIVE_GAP_MULTIPLIER),
        "recent_gap_window": int(RECENT_GAP_WINDOW),
        "shared_sim_seed": int(sim_seed),
    }
    dump_json(out / "edge_boundary_monitor_redesign_compare.json", summary)
    _write_csv(out / "edge_boundary_monitor_redesign_compare.csv", rows)

    _plot_metric(
        rows,
        "max_applied_jump",
        "Edge-boundary monitor redesign: max applied jump",
        "max applied jump",
        out / "jump_compare_redesign.png",
    )
    _plot_metric(
        rows,
        "tracking_error_mean",
        "Edge-boundary monitor redesign: tracking error mean",
        "tracking error mean",
        out / "tracking_compare_redesign.png",
    )
    _plot_metric(
        rows,
        "blocked_steps_total",
        "Edge-boundary monitor redesign: blocked steps",
        "blocked steps",
        out / "blocking_compare_redesign.png",
    )

    return out / "edge_boundary_monitor_redesign_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor redesign comparison on Tier-1 edge-boundary scenarios.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/edge_boundary_monitor_redesign_compare")
    args = parser.parse_args()
    out = run_edge_boundary_monitor_redesign_compare(
        config_path=args.config,
        out_dir=args.out,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

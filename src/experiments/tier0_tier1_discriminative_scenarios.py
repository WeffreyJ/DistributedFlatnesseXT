"""Designed discriminative Tier-0 vs Tier-1 scenarios under frozen controller logic."""

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
from src.experiments.runtime_monitor_audit import _collect_blocked_intervals
from src.model.coupling import active_edges
from src.verify.utils import dump_json, make_results_dir


def _clone_cfg_for_case(cfg, *, plant_family: str, lateral_offsets: list[float] | None = None):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = str(plant_family)
    cfg_mode.evaluation.mode = "upstream_truncated"
    cfg_mode.selection.mode = "shadow_lexicographic"
    cfg_mode.monitor.enabled = False
    cfg_mode.monitor.mode = "shadow"
    if lateral_offsets is not None:
        cfg_mode.system.tier1.geometry.lateral_offsets = [float(v) for v in lateral_offsets]
    return cfg_mode


def _case_definitions(cfg) -> list[dict[str, object]]:
    gamma_edge = float(cfg.system.gamma_edge)
    return [
        {
            "case_name": "tie_small_sep_sym",
            "family": "tie_proximal",
            "x0": [0.045, 0.0, -0.24, 0.0, 0.0, 0.0],
        },
        {
            "case_name": "tie_small_sep_asym",
            "family": "tie_proximal",
            "x0": [0.05, 0.005, -0.22, -0.03, 0.025, 0.0],
        },
        {
            "case_name": "tie_crossing_near_boundary",
            "family": "tie_proximal",
            "x0": [0.06, 0.0, -0.26, -0.12, 0.12, 0.0],
        },
        {
            "case_name": "edge_radius_just_inside",
            "family": "edge_boundary",
            "x0": [gamma_edge + 0.015, 0.0, -0.20, 0.0, 0.0, 0.0],
            "tier1_lateral_offsets": [0.0, 2.45, -2.45],
        },
        {
            "case_name": "edge_radius_just_outside",
            "family": "edge_boundary",
            "x0": [gamma_edge + 0.015, 0.0, -0.20, 0.0, 0.0, 0.0],
            "tier1_lateral_offsets": [0.0, 2.55, -2.55],
        },
        {
            "case_name": "edge_longitudinal_just_above_gamma",
            "family": "edge_boundary",
            "x0": [gamma_edge + 0.005, 0.0, -0.20, 0.0, 0.0, 0.0],
        },
        {
            "case_name": "edge_longitudinal_just_below_gamma",
            "family": "edge_boundary",
            "x0": [gamma_edge - 0.005, 0.0, -0.20, 0.0, 0.0, 0.0],
        },
    ]


def _metrics_for_run(sim: dict[str, object], cfg, *, case_name: str, plant_family: str) -> dict[str, object]:
    intervals = _collect_blocked_intervals(sim)
    blocked_steps_total = int(sum(int(interval["duration_steps"]) for interval in intervals))
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    predicted_gap = np.asarray(sim.get("monitor_predicted_gap", []), dtype=float)
    return {
        "case_name": str(case_name),
        "plant_family": str(plant_family),
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in sim.get("switch_reason", []) if str(reason) == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
        "blocked_steps_total": blocked_steps_total,
        "mean_monitor_predicted_gap": float(np.mean(predicted_gap)) if predicted_gap.size else 0.0,
        "max_monitor_predicted_gap": float(np.max(predicted_gap)) if predicted_gap.size else 0.0,
    }


def _edge_pattern_signature(sim: dict[str, object], cfg) -> list[str]:
    x_hist = np.asarray(sim.get("x", []), dtype=float)
    signatures: list[str] = []
    for x in x_hist:
        edges = sorted((int(j), int(i)) for (j, i) in active_edges(np.asarray(x, dtype=float), cfg.system))
        signatures.append(str(edges))
    return signatures


def _case_delta(case_name: str, tier0: dict[str, object], tier1: dict[str, object], *, switch_pattern_differs: bool, candidate_pattern_differs: bool, active_edge_pattern_differs: bool) -> dict[str, object]:
    return {
        "case_name": str(case_name),
        "delta_switch_count": int(tier1["switch_count"]) - int(tier0["switch_count"]),
        "delta_transition_start_count": int(tier1["transition_start_count"]) - int(tier0["transition_start_count"]),
        "delta_blend_active_steps": int(tier1["blend_active_steps"]) - int(tier0["blend_active_steps"]),
        "delta_tracking_error_mean": float(tier1["tracking_error_mean"]) - float(tier0["tracking_error_mean"]),
        "delta_max_raw_jump": float(tier1["max_raw_jump"]) - float(tier0["max_raw_jump"]),
        "delta_max_applied_jump": float(tier1["max_applied_jump"]) - float(tier0["max_applied_jump"]),
        "delta_blocked_steps_total": int(tier1["blocked_steps_total"]) - int(tier0["blocked_steps_total"]),
        "switch_pattern_differs": bool(switch_pattern_differs),
        "candidate_pattern_differs": bool(candidate_pattern_differs),
        "active_edge_pattern_differs": bool(active_edge_pattern_differs),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_name",
        "delta_switch_count",
        "delta_transition_start_count",
        "delta_blend_active_steps",
        "delta_tracking_error_mean",
        "delta_max_raw_jump",
        "delta_max_applied_jump",
        "delta_blocked_steps_total",
        "switch_pattern_differs",
        "candidate_pattern_differs",
        "active_edge_pattern_differs",
        "tier0_switch_count",
        "tier1_switch_count",
        "tier0_transition_start_count",
        "tier1_transition_start_count",
        "tier0_blend_active_steps",
        "tier1_blend_active_steps",
        "tier0_tracking_error_mean",
        "tier1_tracking_error_mean",
        "tier0_tracking_error_max",
        "tier1_tracking_error_max",
        "tier0_max_raw_jump",
        "tier1_max_raw_jump",
        "tier0_max_applied_jump",
        "tier1_max_applied_jump",
        "tier0_blocked_steps_total",
        "tier1_blocked_steps_total",
        "tier0_mean_monitor_predicted_gap",
        "tier1_mean_monitor_predicted_gap",
        "tier0_max_monitor_predicted_gap",
        "tier1_max_monitor_predicted_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bar_plot(rows: list[dict[str, object]], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [str(row["case_name"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_tier0_tier1_discriminative_scenarios(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier0_tier1_discriminative_scenarios",
) -> Path:
    cfg = load_config(config_path)
    out = (
        make_results_dir("tier0_tier1_discriminative_scenarios")
        if out_dir == "results/tier0_tier1_discriminative_scenarios"
        else Path(out_dir)
    )
    out.mkdir(parents=True, exist_ok=True)

    sim_seed = int(cfg.seed) + 19521
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    cases = _case_definitions(cfg)
    csv_rows: list[dict[str, object]] = []
    summary_cases: dict[str, object] = {}

    for case in cases:
        case_name = str(case["case_name"])
        x0 = np.asarray(case["x0"], dtype=float)
        lateral_offsets = case.get("tier1_lateral_offsets")

        runs: dict[str, dict[str, object]] = {}
        raw_sims: dict[str, dict[str, object]] = {}
        raw_cfgs: dict[str, object] = {}
        for plant_family in ["tier0", "tier1"]:
            cfg_mode = _clone_cfg_for_case(
                cfg,
                plant_family=plant_family,
                lateral_offsets=lateral_offsets if plant_family == "tier1" else None,
            )
            sim = simulate_closed_loop(
                cfg_mode,
                x0=x0,
                horizon=float(cfg_mode.system.horizon),
                options=options,
            )
            runs[plant_family] = _metrics_for_run(sim, cfg_mode, case_name=case_name, plant_family=plant_family)
            raw_sims[plant_family] = sim
            raw_cfgs[plant_family] = cfg_mode

        switch_pattern_differs = list(raw_sims["tier0"].get("switch_steps", [])) != list(raw_sims["tier1"].get("switch_steps", []))
        candidate_pattern_differs = list(raw_sims["tier0"].get("pi_candidate", [])) != list(raw_sims["tier1"].get("pi_candidate", []))
        active_edge_pattern_differs = _edge_pattern_signature(raw_sims["tier0"], raw_cfgs["tier0"]) != _edge_pattern_signature(raw_sims["tier1"], raw_cfgs["tier1"])
        delta = _case_delta(
            case_name,
            runs["tier0"],
            runs["tier1"],
            switch_pattern_differs=switch_pattern_differs,
            candidate_pattern_differs=candidate_pattern_differs,
            active_edge_pattern_differs=active_edge_pattern_differs,
        )

        csv_rows.append(
            {
                "case_name": case_name,
                **delta,
                "tier0_switch_count": int(runs["tier0"]["switch_count"]),
                "tier1_switch_count": int(runs["tier1"]["switch_count"]),
                "tier0_transition_start_count": int(runs["tier0"]["transition_start_count"]),
                "tier1_transition_start_count": int(runs["tier1"]["transition_start_count"]),
                "tier0_blend_active_steps": int(runs["tier0"]["blend_active_steps"]),
                "tier1_blend_active_steps": int(runs["tier1"]["blend_active_steps"]),
                "tier0_tracking_error_mean": float(runs["tier0"]["tracking_error_mean"]),
                "tier1_tracking_error_mean": float(runs["tier1"]["tracking_error_mean"]),
                "tier0_tracking_error_max": float(runs["tier0"]["tracking_error_max"]),
                "tier1_tracking_error_max": float(runs["tier1"]["tracking_error_max"]),
                "tier0_max_raw_jump": float(runs["tier0"]["max_raw_jump"]),
                "tier1_max_raw_jump": float(runs["tier1"]["max_raw_jump"]),
                "tier0_max_applied_jump": float(runs["tier0"]["max_applied_jump"]),
                "tier1_max_applied_jump": float(runs["tier1"]["max_applied_jump"]),
                "tier0_blocked_steps_total": int(runs["tier0"]["blocked_steps_total"]),
                "tier1_blocked_steps_total": int(runs["tier1"]["blocked_steps_total"]),
                "tier0_mean_monitor_predicted_gap": float(runs["tier0"]["mean_monitor_predicted_gap"]),
                "tier1_mean_monitor_predicted_gap": float(runs["tier1"]["mean_monitor_predicted_gap"]),
                "tier0_max_monitor_predicted_gap": float(runs["tier0"]["max_monitor_predicted_gap"]),
                "tier1_max_monitor_predicted_gap": float(runs["tier1"]["max_monitor_predicted_gap"]),
            }
        )
        summary_cases[case_name] = {
            "family": str(case["family"]),
            "x0": [float(v) for v in x0],
            "tier1_lateral_offsets": [float(v) for v in lateral_offsets] if lateral_offsets is not None else None,
            "tier0": runs["tier0"],
            "tier1": runs["tier1"],
            "delta_tier1_minus_tier0": delta,
        }

    summary = {
        "selection_mode": "shadow_lexicographic",
        "monitor_mode": "disabled",
        "evaluator_mode": "upstream_truncated",
        "shared_sim_seed": int(sim_seed),
        "cases": summary_cases,
    }
    dump_json(out / "tier0_tier1_discriminative_scenarios.json", summary)
    _write_csv(out / "tier0_tier1_discriminative_scenarios.csv", csv_rows)

    _bar_plot(
        csv_rows,
        "delta_switch_count",
        "Tier-1 minus Tier-0 switch-count delta by case",
        "delta switch count",
        out / "switch_delta_by_case.png",
    )
    _bar_plot(
        csv_rows,
        "delta_max_applied_jump",
        "Tier-1 minus Tier-0 applied-jump delta by case",
        "delta max applied jump",
        out / "jump_delta_by_case.png",
    )
    _bar_plot(
        csv_rows,
        "delta_tracking_error_mean",
        "Tier-1 minus Tier-0 tracking-error-mean delta by case",
        "delta tracking error mean",
        out / "tracking_delta_by_case.png",
    )

    return out / "tier0_tier1_discriminative_scenarios.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Designed discriminative Tier-0 vs Tier-1 scenarios.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier0_tier1_discriminative_scenarios")
    args = parser.parse_args()
    out = run_tier0_tier1_discriminative_scenarios(
        config_path=args.config,
        out_dir=args.out,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

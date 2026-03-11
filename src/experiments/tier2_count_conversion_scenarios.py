"""Designed Tier-2 scenarios that try to convert timing shifts into burden deltas."""

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
from src.experiments.tier2_mismatch_sensitive_scenarios import _clone_cfg_for_variant
from src.experiments.tier2_residual_compare import _mismatch_metrics
from src.verify.utils import dump_json, make_results_dir


def _case_definitions(cfg) -> list[dict[str, object]]:
    gamma_edge = float(cfg.system.gamma_edge)
    return [
        {
            "case_name": "dwell_skew_bias_slow_crossing",
            "family": "prolonged_near_switch_dwell",
            "x0": [0.094, 0.0912, -0.205, -0.004, 0.004, 0.0],
            "tier1_lateral_offsets": [0.0, 1.85, -0.55],
        },
        {
            "case_name": "dwell_skew_opposed_slow_crossing",
            "family": "prolonged_near_switch_dwell",
            "x0": [0.094, 0.0910, -0.204, -0.005, 0.005, 0.0],
            "tier1_lateral_offsets": [0.0, 2.10, -1.90],
        },
        {
            "case_name": "repeated_grazing_bias",
            "family": "repeated_grazing",
            "x0": [0.0935, 0.0910, -0.205, -0.020, 0.018, 0.002],
            "tier1_lateral_offsets": [0.0, 1.95, -0.75],
        },
        {
            "case_name": "repeated_grazing_opposed",
            "family": "repeated_grazing",
            "x0": [0.0935, 0.0910, -0.205, -0.018, 0.020, -0.002],
            "tier1_lateral_offsets": [0.0, 2.15, -1.95],
        },
        {
            "case_name": "mixed_boundary_aligned_dwell",
            "family": "mixed_boundary_dwell",
            "x0": [gamma_edge + 0.012, 0.0, -0.210, -0.006, 0.006, 0.0],
            "tier1_lateral_offsets": [0.0, 2.45, -1.10],
        },
        {
            "case_name": "mixed_boundary_aligned_lockout_edge",
            "family": "mixed_boundary_dwell",
            "x0": [gamma_edge + 0.011, 0.0, -0.210, -0.010, 0.010, 0.0],
            "tier1_lateral_offsets": [0.0, 2.45, -1.10],
        },
        {
            "case_name": "slow_marginal_crossing_opposed_leaders",
            "family": "velocity_biased_marginal_crossing",
            "x0": [gamma_edge + 0.010, 0.0, -0.190, -0.004, 0.004, 0.0],
            "tier1_lateral_offsets": [0.0, 1.95, -1.95],
        },
        {
            "case_name": "slow_marginal_crossing_centerline",
            "family": "velocity_biased_marginal_crossing",
            "x0": [gamma_edge + 0.010, 0.002, -0.188, -0.003, 0.003, 0.0],
            "tier1_lateral_offsets": [0.0, 0.35, -0.40],
        },
    ]


def _metrics_for_run(sim: dict[str, object], cfg, *, case_name: str, case_family: str, variant: str) -> dict[str, object]:
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    out = {
        "case_name": str(case_name),
        "case_family": str(case_family),
        "variant": str(variant),
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


def _case_delta(
    case_name: str,
    tier1: dict[str, object],
    tier2: dict[str, object],
    *,
    switch_pattern_differs: bool,
    candidate_pattern_differs: bool,
    effective_candidate_pattern_differs: bool,
) -> dict[str, object]:
    return {
        "case_name": str(case_name),
        "delta_switch_count": int(tier2["switch_count"]) - int(tier1["switch_count"]),
        "delta_transition_start_count": int(tier2["transition_start_count"]) - int(tier1["transition_start_count"]),
        "delta_blend_active_steps": int(tier2["blend_active_steps"]) - int(tier1["blend_active_steps"]),
        "delta_tracking_error_mean": float(tier2["tracking_error_mean"]) - float(tier1["tracking_error_mean"]),
        "delta_tracking_error_max": float(tier2["tracking_error_max"]) - float(tier1["tracking_error_max"]),
        "delta_max_raw_jump": float(tier2["max_raw_jump"]) - float(tier1["max_raw_jump"]),
        "delta_max_applied_jump": float(tier2["max_applied_jump"]) - float(tier1["max_applied_jump"]),
        "delta_mean_nominal_vs_plant_gap_over_rollout": float(tier2["mean_nominal_vs_plant_gap_over_rollout"]) - float(tier1["mean_nominal_vs_plant_gap_over_rollout"]),
        "delta_max_nominal_vs_plant_gap_over_rollout": float(tier2["max_nominal_vs_plant_gap_over_rollout"]) - float(tier1["max_nominal_vs_plant_gap_over_rollout"]),
        "delta_mismatch_at_switch_steps_mean": float(tier2["mismatch_at_switch_steps_mean"]) - float(tier1["mismatch_at_switch_steps_mean"]),
        "switch_pattern_differs": bool(switch_pattern_differs),
        "candidate_pattern_differs": bool(candidate_pattern_differs),
        "effective_candidate_pattern_differs": bool(effective_candidate_pattern_differs),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_name",
        "family",
        "delta_switch_count",
        "delta_transition_start_count",
        "delta_blend_active_steps",
        "delta_tracking_error_mean",
        "delta_tracking_error_max",
        "delta_max_raw_jump",
        "delta_max_applied_jump",
        "delta_mean_nominal_vs_plant_gap_over_rollout",
        "delta_max_nominal_vs_plant_gap_over_rollout",
        "delta_mismatch_at_switch_steps_mean",
        "switch_pattern_differs",
        "candidate_pattern_differs",
        "effective_candidate_pattern_differs",
        "tier1_switch_count",
        "tier2_switch_count",
        "tier1_transition_start_count",
        "tier2_transition_start_count",
        "tier1_blend_active_steps",
        "tier2_blend_active_steps",
        "tier1_tracking_error_mean",
        "tier2_tracking_error_mean",
        "tier1_tracking_error_max",
        "tier2_tracking_error_max",
        "tier1_max_raw_jump",
        "tier2_max_raw_jump",
        "tier1_max_applied_jump",
        "tier2_max_applied_jump",
        "tier1_mean_nominal_vs_plant_gap_over_rollout",
        "tier2_mean_nominal_vs_plant_gap_over_rollout",
        "tier1_max_nominal_vs_plant_gap_over_rollout",
        "tier2_max_nominal_vs_plant_gap_over_rollout",
        "tier1_mismatch_at_switch_steps_mean",
        "tier2_mismatch_at_switch_steps_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bar_plot(rows: list[dict[str, object]], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [str(row["case_name"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_tier2_count_conversion_scenarios(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier2_count_conversion_scenarios",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("tier2_count_conversion_scenarios") if out_dir == "results/tier2_count_conversion_scenarios" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sim_seed = int(cfg.seed) + 21121
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    csv_rows: list[dict[str, object]] = []
    summary_cases: dict[str, object] = {}
    cases_with_count_deltas: list[str] = []
    for case in _case_definitions(cfg):
        case_name = str(case["case_name"])
        family = str(case["family"])
        x0 = np.asarray(case["x0"], dtype=float)
        lateral_offsets = [float(v) for v in case["tier1_lateral_offsets"]]

        runs: dict[str, dict[str, object]] = {}
        raw_sims: dict[str, dict[str, object]] = {}
        for variant in ["tier1_nominal", "tier2_residual_enabled"]:
            cfg_mode = _clone_cfg_for_variant(cfg, variant=variant, lateral_offsets=lateral_offsets)
            sim = simulate_closed_loop(
                cfg_mode,
                x0=x0,
                horizon=float(cfg_mode.system.horizon),
                options=options,
            )
            runs[variant] = _metrics_for_run(sim, cfg_mode, case_name=case_name, case_family=family, variant=variant)
            raw_sims[variant] = sim

        switch_pattern_differs = list(raw_sims["tier1_nominal"].get("switch_steps", [])) != list(raw_sims["tier2_residual_enabled"].get("switch_steps", []))
        candidate_pattern_differs = list(raw_sims["tier1_nominal"].get("pi_candidate", [])) != list(raw_sims["tier2_residual_enabled"].get("pi_candidate", []))
        effective_candidate_pattern_differs = list(raw_sims["tier1_nominal"].get("effective_pi_candidate", [])) != list(raw_sims["tier2_residual_enabled"].get("effective_pi_candidate", []))

        delta = _case_delta(
            case_name,
            runs["tier1_nominal"],
            runs["tier2_residual_enabled"],
            switch_pattern_differs=switch_pattern_differs,
            candidate_pattern_differs=candidate_pattern_differs,
            effective_candidate_pattern_differs=effective_candidate_pattern_differs,
        )
        if any(int(delta[key]) != 0 for key in ["delta_switch_count", "delta_transition_start_count", "delta_blend_active_steps"]):
            cases_with_count_deltas.append(case_name)

        csv_rows.append(
            {
                "case_name": case_name,
                "family": family,
                **delta,
                "tier1_switch_count": int(runs["tier1_nominal"]["switch_count"]),
                "tier2_switch_count": int(runs["tier2_residual_enabled"]["switch_count"]),
                "tier1_transition_start_count": int(runs["tier1_nominal"]["transition_start_count"]),
                "tier2_transition_start_count": int(runs["tier2_residual_enabled"]["transition_start_count"]),
                "tier1_blend_active_steps": int(runs["tier1_nominal"]["blend_active_steps"]),
                "tier2_blend_active_steps": int(runs["tier2_residual_enabled"]["blend_active_steps"]),
                "tier1_tracking_error_mean": float(runs["tier1_nominal"]["tracking_error_mean"]),
                "tier2_tracking_error_mean": float(runs["tier2_residual_enabled"]["tracking_error_mean"]),
                "tier1_tracking_error_max": float(runs["tier1_nominal"]["tracking_error_max"]),
                "tier2_tracking_error_max": float(runs["tier2_residual_enabled"]["tracking_error_max"]),
                "tier1_max_raw_jump": float(runs["tier1_nominal"]["max_raw_jump"]),
                "tier2_max_raw_jump": float(runs["tier2_residual_enabled"]["max_raw_jump"]),
                "tier1_max_applied_jump": float(runs["tier1_nominal"]["max_applied_jump"]),
                "tier2_max_applied_jump": float(runs["tier2_residual_enabled"]["max_applied_jump"]),
                "tier1_mean_nominal_vs_plant_gap_over_rollout": float(runs["tier1_nominal"]["mean_nominal_vs_plant_gap_over_rollout"]),
                "tier2_mean_nominal_vs_plant_gap_over_rollout": float(runs["tier2_residual_enabled"]["mean_nominal_vs_plant_gap_over_rollout"]),
                "tier1_max_nominal_vs_plant_gap_over_rollout": float(runs["tier1_nominal"]["max_nominal_vs_plant_gap_over_rollout"]),
                "tier2_max_nominal_vs_plant_gap_over_rollout": float(runs["tier2_residual_enabled"]["max_nominal_vs_plant_gap_over_rollout"]),
                "tier1_mismatch_at_switch_steps_mean": float(runs["tier1_nominal"]["mismatch_at_switch_steps_mean"]),
                "tier2_mismatch_at_switch_steps_mean": float(runs["tier2_residual_enabled"]["mismatch_at_switch_steps_mean"]),
            }
        )
        summary_cases[case_name] = {
            "family": family,
            "x0": [float(v) for v in x0],
            "tier1_lateral_offsets": lateral_offsets,
            "tier1_nominal": runs["tier1_nominal"],
            "tier2_residual_enabled": runs["tier2_residual_enabled"],
            "delta_tier2_enabled_minus_tier1": delta,
        }

    summary = {
        "selection_mode": str(getattr(getattr(cfg, "selection", object()), "mode", "unknown")),
        "monitor_mode": str(getattr(getattr(cfg, "monitor", object()), "mode", "disabled"))
        if bool(getattr(getattr(cfg, "monitor", object()), "enabled", False))
        else "disabled",
        "evaluator_mode": "upstream_truncated",
        "shared_sim_seed": int(sim_seed),
        "cases_with_count_deltas": cases_with_count_deltas,
        "cases": summary_cases,
    }
    dump_json(out / "tier2_count_conversion_scenarios.json", summary)
    _write_csv(out / "tier2_count_conversion_scenarios.csv", csv_rows)

    _bar_plot(
        csv_rows,
        "delta_switch_count",
        "Tier-2 residual minus Tier-1 switch-count delta by case",
        "delta switch count",
        out / "switch_delta_by_case.png",
    )
    _bar_plot(
        csv_rows,
        "delta_blend_active_steps",
        "Tier-2 residual minus Tier-1 blend-burden delta by case",
        "delta blend steps",
        out / "blend_delta_by_case.png",
    )
    _bar_plot(
        csv_rows,
        "delta_max_applied_jump",
        "Tier-2 residual minus Tier-1 applied-jump delta by case",
        "delta max applied jump",
        out / "jump_delta_by_case.png",
    )
    _bar_plot(
        csv_rows,
        "delta_mean_nominal_vs_plant_gap_over_rollout",
        "Tier-2 residual minus Tier-1 mean mismatch delta by case",
        "delta mean mismatch",
        out / "mismatch_delta_by_case.png",
    )

    return out / "tier2_count_conversion_scenarios.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Designed Tier-2 count-conversion scenarios.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier2_count_conversion_scenarios")
    args = parser.parse_args()
    out = run_tier2_count_conversion_scenarios(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

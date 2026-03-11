"""Tier-2 pattern-change mechanism audit on designed mismatch-sensitive cases."""

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
from src.experiments.tier2_mismatch_sensitive_scenarios import _case_definitions, _clone_cfg_for_variant
from src.flatness.evaluation_operator import compute_evaluator
from src.model.coupling import plant_delta_accel
from src.verify.utils import dump_json, make_results_dir


SWITCH_WINDOW = 2


def _as_pi_list(value) -> list[int]:
    return [int(v) for v in value]


def _target_cases(cfg) -> list[dict[str, object]]:
    wanted = {
        "transverse_skew_opposed_leaders",
        "near_switch_skew_bias",
        "near_switch_skew_opposed",
        "mixed_boundary_residual_aligned",
    }
    return [case for case in _case_definitions(cfg) if str(case["case_name"]) in wanted]


def _mismatch_series(sim: dict[str, object], cfg) -> np.ndarray:
    x_hist = np.asarray(sim.get("x", []), dtype=float)
    pi_hist = sim.get("pi", [])
    eval_mode = str(getattr(getattr(cfg, "evaluation", object()), "mode", "upstream_truncated"))
    mismatch = np.zeros(max(len(x_hist) - 1, 0), dtype=float)
    for k in range(mismatch.size):
        x_now = np.asarray(x_hist[k], dtype=float)
        pi_now = _as_pi_list(pi_hist[k])
        nominal = np.asarray(compute_evaluator(x_now, pi_now, cfg, mode=eval_mode), dtype=float)
        plant = np.asarray(plant_delta_accel(x_now, cfg.system), dtype=float)
        mismatch[k] = float(np.linalg.norm(plant - nominal))
    return mismatch


def _step_rows_for_case(case: dict[str, object], cfg, options: SimOptions) -> tuple[list[dict[str, object]], dict[str, object]]:
    case_name = str(case["case_name"])
    family = str(case["family"])
    x0 = np.asarray(case["x0"], dtype=float)
    lateral_offsets = [float(v) for v in case["tier1_lateral_offsets"]]

    cfg_t1 = _clone_cfg_for_variant(cfg, variant="tier1_nominal", lateral_offsets=lateral_offsets)
    cfg_t2 = _clone_cfg_for_variant(cfg, variant="tier2_residual_enabled", lateral_offsets=lateral_offsets)

    sim_t1 = simulate_closed_loop(cfg_t1, x0=x0, horizon=float(cfg_t1.system.horizon), options=options)
    sim_t2 = simulate_closed_loop(cfg_t2, x0=x0, horizon=float(cfg_t2.system.horizon), options=options)

    mismatch_t2 = _mismatch_series(sim_t2, cfg_t2)
    mismatch_mean_t2 = float(np.mean(mismatch_t2)) if mismatch_t2.size else 0.0

    switch_steps_t1 = {int(v) for v in sim_t1.get("switch_steps", [])}
    switch_steps_t2 = {int(v) for v in sim_t2.get("switch_steps", [])}
    switch_diff_steps = switch_steps_t1.symmetric_difference(switch_steps_t2)
    near_switch_steps: set[int] = set()
    for step in switch_steps_t1.union(switch_steps_t2):
        for offset in range(-SWITCH_WINDOW, SWITCH_WINDOW + 1):
            near_switch_steps.add(int(step + offset))

    rows: list[dict[str, object]] = []
    steps = min(
        len(sim_t1.get("pi", [])),
        len(sim_t2.get("pi", [])),
        len(sim_t1.get("pi_candidate", [])),
        len(sim_t2.get("pi_candidate", [])),
        len(sim_t1.get("effective_pi_candidate", [])),
        len(sim_t2.get("effective_pi_candidate", [])),
        len(sim_t1.get("switch_reason", [])),
        len(sim_t2.get("switch_reason", [])),
        len(sim_t1.get("selector_switch_eligible", [])),
        len(sim_t2.get("selector_switch_eligible", [])),
        len(sim_t1.get("tie_gap_min", [])),
        len(sim_t2.get("tie_gap_min", [])),
        len(sim_t1.get("J", [])),
        len(sim_t2.get("J", [])),
        len(sim_t1.get("blend_active", [])),
        len(sim_t2.get("blend_active", [])),
        len(mismatch_t2),
    )

    for k in range(steps):
        tier1_pi = _as_pi_list(sim_t1["pi"][k])
        tier2_pi = _as_pi_list(sim_t2["pi"][k])
        tier1_candidate = _as_pi_list(sim_t1["pi_candidate"][k])
        tier2_candidate = _as_pi_list(sim_t2["pi_candidate"][k])
        tier1_effective = _as_pi_list(sim_t1["effective_pi_candidate"][k])
        tier2_effective = _as_pi_list(sim_t2["effective_pi_candidate"][k])

        candidate_differs = tier1_candidate != tier2_candidate
        effective_candidate_differs = tier1_effective != tier2_effective
        current_pi_differs = tier1_pi != tier2_pi
        switch_eligibility_differs = bool(sim_t1["selector_switch_eligible"][k]) != bool(sim_t2["selector_switch_eligible"][k])
        switch_timing_differs = int(k) in switch_diff_steps
        pattern_difference = (
            candidate_differs
            or effective_candidate_differs
            or current_pi_differs
            or switch_eligibility_differs
            or switch_timing_differs
        )
        if not pattern_difference:
            continue

        tie_margin = float(min(float(sim_t1["tie_gap_min"][k]), float(sim_t2["tie_gap_min"][k])))
        mismatch_norm = float(mismatch_t2[k])
        rows.append(
            {
                "case_name": case_name,
                "family": family,
                "step": int(k),
                "time": float(sim_t2["t_control"][k]),
                "tier1_current_pi": str(tier1_pi),
                "tier2_current_pi": str(tier2_pi),
                "tier1_pi_candidate": str(tier1_candidate),
                "tier2_pi_candidate": str(tier2_candidate),
                "tier1_effective_pi_candidate": str(tier1_effective),
                "tier2_effective_pi_candidate": str(tier2_effective),
                "candidate_differs": bool(candidate_differs),
                "effective_candidate_differs": bool(effective_candidate_differs),
                "current_pi_differs": bool(current_pi_differs),
                "switch_eligibility_differs": bool(switch_eligibility_differs),
                "switch_timing_differs": bool(switch_timing_differs),
                "selector_switch_eligible_tier1": bool(sim_t1["selector_switch_eligible"][k]),
                "selector_switch_eligible_tier2": bool(sim_t2["selector_switch_eligible"][k]),
                "tier1_switch_reason": str(sim_t1["switch_reason"][k]),
                "tier2_switch_reason": str(sim_t2["switch_reason"][k]),
                "tier1_switch_event": bool(sim_t1["switch_event"][k]),
                "tier2_switch_event": bool(sim_t2["switch_event"][k]),
                "tie_margin": tie_margin,
                "mismatch_norm": mismatch_norm,
                "mismatch_relative_to_case_mean": (mismatch_norm / mismatch_mean_t2) if mismatch_mean_t2 > 0.0 else 0.0,
                "tier1_local_jump": float(sim_t1["J"][k]),
                "tier2_local_jump": float(sim_t2["J"][k]),
                "transition_active": bool(sim_t1["blend_active"][k]) or bool(sim_t2["blend_active"][k]),
                "near_switch_window": bool(int(k) in near_switch_steps),
            }
        )

    total = len(rows)
    candidate_count = sum(1 for row in rows if bool(row["candidate_differs"]))
    effective_count = sum(1 for row in rows if bool(row["effective_candidate_differs"]))
    switch_timing_count = sum(1 for row in rows if bool(row["switch_timing_differs"]))
    near_switch_count = sum(1 for row in rows if bool(row["near_switch_window"]))

    summary = {
        "family": family,
        "x0": [float(v) for v in x0],
        "tier1_lateral_offsets": lateral_offsets,
        "total_pattern_difference_steps": int(total),
        "mean_tie_margin_at_pattern_difference_steps": float(np.mean([row["tie_margin"] for row in rows])) if rows else 0.0,
        "mean_mismatch_at_pattern_difference_steps": float(np.mean([row["mismatch_norm"] for row in rows])) if rows else 0.0,
        "fraction_pattern_differences_near_switch_windows": float(near_switch_count / total) if total else 0.0,
        "fraction_pattern_differences_with_candidate_change": float(candidate_count / total) if total else 0.0,
        "fraction_pattern_differences_with_effective_candidate_change": float(effective_count / total) if total else 0.0,
        "fraction_pattern_differences_with_actual_switch_timing_difference": float(switch_timing_count / total) if total else 0.0,
    }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_name",
        "family",
        "step",
        "time",
        "tier1_current_pi",
        "tier2_current_pi",
        "tier1_pi_candidate",
        "tier2_pi_candidate",
        "tier1_effective_pi_candidate",
        "tier2_effective_pi_candidate",
        "candidate_differs",
        "effective_candidate_differs",
        "current_pi_differs",
        "switch_eligibility_differs",
        "switch_timing_differs",
        "selector_switch_eligible_tier1",
        "selector_switch_eligible_tier2",
        "tier1_switch_reason",
        "tier2_switch_reason",
        "tier1_switch_event",
        "tier2_switch_event",
        "tie_margin",
        "mismatch_norm",
        "mismatch_relative_to_case_mean",
        "tier1_local_jump",
        "tier2_local_jump",
        "transition_active",
        "near_switch_window",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bar_plot(cases: list[str], values: list[float], title: str, ylabel: str, out_path: Path) -> None:
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_tier2_pattern_change_audit(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier2_pattern_change_audit",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("tier2_pattern_change_audit") if out_dir == "results/tier2_pattern_change_audit" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sim_seed = int(cfg.seed) + 21021
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    all_rows: list[dict[str, object]] = []
    case_summaries: dict[str, object] = {}
    for case in _target_cases(cfg):
        rows, summary = _step_rows_for_case(case, cfg, options)
        all_rows.extend(rows)
        case_summaries[str(case["case_name"])] = summary

    total_rows = len(all_rows)
    summary = {
        "shared_sim_seed": int(sim_seed),
        "cases": case_summaries,
        "total_pattern_difference_steps_by_case": {
            case_name: int(case_summary["total_pattern_difference_steps"])
            for case_name, case_summary in case_summaries.items()
        },
        "mean_tie_margin_at_pattern_difference_steps": {
            case_name: float(case_summary["mean_tie_margin_at_pattern_difference_steps"])
            for case_name, case_summary in case_summaries.items()
        },
        "mean_mismatch_at_pattern_difference_steps": {
            case_name: float(case_summary["mean_mismatch_at_pattern_difference_steps"])
            for case_name, case_summary in case_summaries.items()
        },
        "fraction_pattern_differences_near_switch_windows": {
            case_name: float(case_summary["fraction_pattern_differences_near_switch_windows"])
            for case_name, case_summary in case_summaries.items()
        },
        "fraction_pattern_differences_with_candidate_change": {
            case_name: float(case_summary["fraction_pattern_differences_with_candidate_change"])
            for case_name, case_summary in case_summaries.items()
        },
        "fraction_pattern_differences_with_effective_candidate_change": {
            case_name: float(case_summary["fraction_pattern_differences_with_effective_candidate_change"])
            for case_name, case_summary in case_summaries.items()
        },
        "fraction_pattern_differences_with_actual_switch_timing_difference": {
            case_name: float(case_summary["fraction_pattern_differences_with_actual_switch_timing_difference"])
            for case_name, case_summary in case_summaries.items()
        },
        "global": {
            "total_pattern_difference_steps": int(total_rows),
            "fraction_pattern_differences_near_switch_windows": float(
                np.mean([bool(row["near_switch_window"]) for row in all_rows])
            )
            if all_rows
            else 0.0,
            "fraction_pattern_differences_with_candidate_change": float(
                np.mean([bool(row["candidate_differs"]) for row in all_rows])
            )
            if all_rows
            else 0.0,
            "fraction_pattern_differences_with_effective_candidate_change": float(
                np.mean([bool(row["effective_candidate_differs"]) for row in all_rows])
            )
            if all_rows
            else 0.0,
            "fraction_pattern_differences_with_actual_switch_timing_difference": float(
                np.mean([bool(row["switch_timing_differs"]) for row in all_rows])
            )
            if all_rows
            else 0.0,
        },
    }

    dump_json(out / "tier2_pattern_change_audit.json", summary)
    _write_csv(out / "tier2_pattern_change_audit.csv", all_rows)

    case_names = list(case_summaries.keys())
    _bar_plot(
        case_names,
        [float(case_summaries[name]["total_pattern_difference_steps"]) for name in case_names],
        "Tier-2 pattern-difference steps by case",
        "count",
        out / "pattern_difference_steps_by_case.png",
    )
    _bar_plot(
        case_names,
        [float(case_summaries[name]["mean_mismatch_at_pattern_difference_steps"]) for name in case_names],
        "Mean mismatch at pattern-difference steps",
        "mismatch norm",
        out / "mismatch_at_pattern_differences.png",
    )
    _bar_plot(
        case_names,
        [float(case_summaries[name]["mean_tie_margin_at_pattern_difference_steps"]) for name in case_names],
        "Mean tie margin at pattern-difference steps",
        "tie margin",
        out / "tie_margin_at_pattern_differences.png",
    )

    return out / "tier2_pattern_change_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-2 pattern-change mechanism audit.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier2_pattern_change_audit")
    args = parser.parse_args()
    out = run_tier2_pattern_change_audit(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

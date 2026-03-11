"""Mechanism audit for Tier-2 support-transition-bias pattern changes."""

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
from src.experiments.tier2_fourth_residual_compare import _clone_cfg_variant
from src.experiments.tier2_fourth_residual_scenarios import _case_definitions
from src.flatness.evaluation_operator import compute_evaluator
from src.model.coupling import plant_delta_accel
from src.model.residual_tier2 import support_transition_pair_terms_tier2
from src.verify.utils import dump_json, make_results_dir


SWITCH_WINDOW = 2
SIGNAL_TOL = 1.0e-9


def _as_pi_list(value) -> list[int]:
    return [int(v) for v in value]


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


def _dominant_transition_terms(x: np.ndarray, cfg) -> dict[str, object]:
    params = cfg.system
    residual = params.tier2.residual
    shell_tol = float(getattr(residual, "support_transition_shell_tol", 0.10))

    best: dict[str, object] | None = None
    n = int(params.N)
    for leader in range(n):
        for follower in range(n):
            if leader == follower:
                continue
            terms = support_transition_pair_terms_tier2(np.asarray(x, dtype=float), leader, follower, params)
            score = abs(float(terms["pair_residual"]))
            if best is None or score > float(best["score"]):
                best = {
                    "leader": int(leader),
                    "follower": int(follower),
                    "score": float(score),
                    **terms,
                }

    if best is None:
        return {
            "nearest_boundary_type": "none",
            "signed_boundary_distance": float("inf"),
            "crossing_direction_signal": 0.0,
            "crossing_direction_magnitude": 0.0,
            "transition_alignment": False,
            "entry_like_alignment": False,
            "exit_like_alignment": False,
            "shell_localized": False,
            "dominant_pair": "[]",
        }

    abs_d_ell = abs(float(best["d_ell"]))
    abs_d_rho = abs(float(best["d_rho"]))
    nearest_boundary_type = "longitudinal" if abs_d_ell <= abs_d_rho else "radial"
    if nearest_boundary_type == "longitudinal":
        signed_boundary_distance = float(best["d_ell"])
        crossing_direction_signal = float(best["d_ell_dot"])
        primary_term = abs(float(best["term_ell"]))
        secondary_term = abs(float(best["term_rho"]))
    else:
        signed_boundary_distance = float(best["d_rho"])
        crossing_direction_signal = float(best["d_rho_dot"])
        primary_term = abs(float(best["term_rho"]))
        secondary_term = abs(float(best["term_ell"]))

    shell_localized = bool(float(best["shell_strength"]) >= shell_tol)
    crossing_direction_magnitude = abs(crossing_direction_signal)
    transition_alignment = bool(
        shell_localized
        and primary_term >= secondary_term
        and crossing_direction_magnitude > SIGNAL_TOL
        and abs(float(best["pair_residual"])) > 0.0
    )

    return {
        "nearest_boundary_type": nearest_boundary_type,
        "signed_boundary_distance": signed_boundary_distance,
        "crossing_direction_signal": crossing_direction_signal,
        "crossing_direction_magnitude": float(crossing_direction_magnitude),
        "transition_alignment": transition_alignment,
        "entry_like_alignment": bool(transition_alignment and crossing_direction_signal > 0.0),
        "exit_like_alignment": bool(transition_alignment and crossing_direction_signal < 0.0),
        "shell_localized": shell_localized,
        "dominant_pair": str([int(best["leader"]), int(best["follower"])]),
    }


def _step_rows_for_case(case: dict[str, object], cfg, options: SimOptions) -> tuple[list[dict[str, object]], dict[str, object]]:
    case_name = str(case["case_name"])
    family = str(case["family"])
    x0 = np.asarray(case["x0"], dtype=float)
    lateral_offsets = [float(v) for v in case["tier1_lateral_offsets"]]

    cfg_t1 = _clone_cfg_variant(cfg, "tier1_nominal")
    cfg_t1.evaluation.mode = "upstream_truncated"
    cfg_t1.system.tier1.geometry.lateral_offsets = lateral_offsets

    cfg_t2 = _clone_cfg_variant(cfg, "tier2_support_transition_bias_enabled")
    cfg_t2.evaluation.mode = "upstream_truncated"
    cfg_t2.system.tier1.geometry.lateral_offsets = lateral_offsets

    sim_t1 = simulate_closed_loop(cfg_t1, x0=x0, horizon=float(cfg_t1.system.horizon), options=options)
    sim_t2 = simulate_closed_loop(cfg_t2, x0=x0, horizon=float(cfg_t2.system.horizon), options=options)

    mismatch_t2 = _mismatch_series(sim_t2, cfg_t2)
    mismatch_mean_t2 = float(np.mean(mismatch_t2)) if mismatch_t2.size else 0.0

    x_hist_t2 = np.asarray(sim_t2.get("x", []), dtype=float)
    dominant_terms = [
        _dominant_transition_terms(x_hist_t2[k], cfg_t2) for k in range(max(len(x_hist_t2) - 1, 0))
    ]

    switch_steps_t1 = {int(v) for v in sim_t1.get("switch_steps", [])}
    switch_steps_t2 = {int(v) for v in sim_t2.get("switch_steps", [])}
    switch_diff_steps = switch_steps_t1.symmetric_difference(switch_steps_t2)
    near_switch_steps: set[int] = set()
    for step in switch_steps_t1.union(switch_steps_t2):
        for offset in range(-SWITCH_WINDOW, SWITCH_WINDOW + 1):
            near_switch_steps.add(int(step + offset))

    steps = min(
        len(sim_t1.get("pi", [])),
        len(sim_t2.get("pi", [])),
        len(sim_t1.get("pi_candidate", [])),
        len(sim_t2.get("pi_candidate", [])),
        len(sim_t1.get("effective_pi_candidate", [])),
        len(sim_t2.get("effective_pi_candidate", [])),
        len(sim_t1.get("selector_switch_eligible", [])),
        len(sim_t2.get("selector_switch_eligible", [])),
        len(sim_t1.get("tie_gap_min", [])),
        len(sim_t2.get("tie_gap_min", [])),
        len(sim_t1.get("J", [])),
        len(sim_t2.get("J", [])),
        len(sim_t1.get("blend_active", [])),
        len(sim_t2.get("blend_active", [])),
        len(sim_t1.get("switch_event", [])),
        len(sim_t2.get("switch_event", [])),
        len(sim_t1.get("t_control", [])),
        len(mismatch_t2),
        len(dominant_terms),
    )

    rows: list[dict[str, object]] = []
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

        terms = dominant_terms[k]
        tie_margin = float(min(float(sim_t1["tie_gap_min"][k]), float(sim_t2["tie_gap_min"][k])))
        mismatch_norm = float(mismatch_t2[k])
        rows.append(
            {
                "case_name": case_name,
                "family": family,
                "step": int(k),
                "time": float(sim_t1["t_control"][k]),
                "tier1_current_pi": str(tier1_pi),
                "tier2_current_pi": str(tier2_pi),
                "tier1_pi_candidate": str(tier1_candidate),
                "tier2_pi_candidate": str(tier2_candidate),
                "tier1_effective_pi_candidate": str(tier1_effective),
                "tier2_effective_pi_candidate": str(tier2_effective),
                "candidate_history_difference": bool(candidate_differs),
                "effective_candidate_history_difference": bool(effective_candidate_differs),
                "current_pi_differs": bool(current_pi_differs),
                "switch_eligibility_differs": bool(switch_eligibility_differs),
                "actual_switch_timing_difference": bool(switch_timing_differs),
                "selector_switch_eligible_tier1": bool(sim_t1["selector_switch_eligible"][k]),
                "selector_switch_eligible_tier2": bool(sim_t2["selector_switch_eligible"][k]),
                "tier1_switch_event": bool(sim_t1["switch_event"][k]),
                "tier2_switch_event": bool(sim_t2["switch_event"][k]),
                "tie_margin": tie_margin,
                "mismatch_norm": mismatch_norm,
                "mismatch_relative_to_case_mean": (mismatch_norm / mismatch_mean_t2) if mismatch_mean_t2 > 0.0 else 0.0,
                "nearest_boundary_type": str(terms["nearest_boundary_type"]),
                "signed_boundary_distance": float(terms["signed_boundary_distance"]),
                "crossing_direction_signal": float(terms["crossing_direction_signal"]),
                "crossing_direction_magnitude": float(terms["crossing_direction_magnitude"]),
                "transition_alignment": bool(terms["transition_alignment"]),
                "entry_like_alignment": bool(terms["entry_like_alignment"]),
                "exit_like_alignment": bool(terms["exit_like_alignment"]),
                "shell_localized": bool(terms["shell_localized"]),
                "dominant_pair": str(terms["dominant_pair"]),
                "tier1_local_jump": float(sim_t1["J"][k]),
                "tier2_local_jump": float(sim_t2["J"][k]),
                "transition_active": bool(sim_t1["blend_active"][k]) or bool(sim_t2["blend_active"][k]),
                "near_switch_window": bool(int(k) in near_switch_steps),
            }
        )

    total = len(rows)
    summary = {
        "family": family,
        "x0": [float(v) for v in x0],
        "tier1_lateral_offsets": lateral_offsets,
        "total_pattern_difference_steps": int(total),
        "mean_tie_margin_at_pattern_difference_steps": float(np.mean([row["tie_margin"] for row in rows])) if rows else 0.0,
        "mean_mismatch_at_pattern_difference_steps": float(np.mean([row["mismatch_norm"] for row in rows])) if rows else 0.0,
        "fraction_pattern_differences_near_switch_windows": float(np.mean([bool(row["near_switch_window"]) for row in rows])) if rows else 0.0,
        "fraction_pattern_differences_with_candidate_change": float(np.mean([bool(row["candidate_history_difference"]) for row in rows])) if rows else 0.0,
        "fraction_pattern_differences_with_effective_candidate_change": float(np.mean([bool(row["effective_candidate_history_difference"]) for row in rows])) if rows else 0.0,
        "fraction_pattern_differences_with_actual_switch_timing_difference": float(np.mean([bool(row["actual_switch_timing_difference"]) for row in rows])) if rows else 0.0,
        "fraction_transition_aligned": float(np.mean([bool(row["transition_alignment"]) for row in rows])) if rows else 0.0,
        "counts_by_nearest_boundary_type": {
            "longitudinal": int(sum(1 for row in rows if str(row["nearest_boundary_type"]) == "longitudinal")),
            "radial": int(sum(1 for row in rows if str(row["nearest_boundary_type"]) == "radial")),
            "none": int(sum(1 for row in rows if str(row["nearest_boundary_type"]) == "none")),
        },
        "counts_by_entry_like_vs_exit_like_alignment": {
            "entry_like": int(sum(1 for row in rows if bool(row["entry_like_alignment"]))),
            "exit_like": int(sum(1 for row in rows if bool(row["exit_like_alignment"]))),
        },
        "mean_abs_signed_boundary_distance_at_pattern_difference_steps": float(
            np.mean([abs(float(row["signed_boundary_distance"])) for row in rows])
        )
        if rows
        else 0.0,
        "mean_crossing_direction_magnitude_at_pattern_difference_steps": float(
            np.mean([float(row["crossing_direction_magnitude"]) for row in rows])
        )
        if rows
        else 0.0,
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
        "candidate_history_difference",
        "effective_candidate_history_difference",
        "current_pi_differs",
        "switch_eligibility_differs",
        "actual_switch_timing_difference",
        "selector_switch_eligible_tier1",
        "selector_switch_eligible_tier2",
        "tier1_switch_event",
        "tier2_switch_event",
        "tie_margin",
        "mismatch_norm",
        "mismatch_relative_to_case_mean",
        "nearest_boundary_type",
        "signed_boundary_distance",
        "crossing_direction_signal",
        "crossing_direction_magnitude",
        "transition_alignment",
        "entry_like_alignment",
        "exit_like_alignment",
        "shell_localized",
        "dominant_pair",
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
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=22, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_tier2_fourth_residual_pattern_audit(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier2_fourth_residual_pattern_audit",
) -> Path:
    cfg = load_config(config_path)
    out = (
        make_results_dir("tier2_fourth_residual_pattern_audit")
        if out_dir == "results/tier2_fourth_residual_pattern_audit"
        else Path(out_dir)
    )
    out.mkdir(parents=True, exist_ok=True)

    sim_seed = int(cfg.seed) + 21158
    options = SimOptions(blending_on=True, noise_delta=0.0, seed=sim_seed, disable_switching=False)

    rows: list[dict[str, object]] = []
    case_summaries: dict[str, object] = {}
    for case in _case_definitions(cfg):
        case_rows, case_summary = _step_rows_for_case(case, cfg, options)
        rows.extend(case_rows)
        case_summaries[str(case["case_name"])] = case_summary

    _write_csv(out / "tier2_fourth_residual_pattern_audit.csv", rows)

    case_names = list(case_summaries.keys())
    _bar_plot(
        case_names,
        [int(case_summaries[name]["total_pattern_difference_steps"]) for name in case_names],
        "Tier-2 fourth residual audit: pattern difference steps",
        "count",
        out / "pattern_difference_steps_by_case.png",
    )
    _bar_plot(
        case_names,
        [float(case_summaries[name]["mean_mismatch_at_pattern_difference_steps"]) for name in case_names],
        "Tier-2 fourth residual audit: mismatch at divergence steps",
        "mismatch norm",
        out / "mismatch_at_pattern_differences.png",
    )
    _bar_plot(
        case_names,
        [float(case_summaries[name]["mean_tie_margin_at_pattern_difference_steps"]) for name in case_names],
        "Tier-2 fourth residual audit: tie margin at divergence steps",
        "tie margin",
        out / "tie_margin_at_pattern_differences.png",
    )
    _bar_plot(
        case_names,
        [float(case_summaries[name]["fraction_transition_aligned"]) for name in case_names],
        "Tier-2 fourth residual audit: transition alignment fraction",
        "fraction",
        out / "transition_alignment_fraction_by_case.png",
    )

    total = len(rows)
    global_summary = {
        "shared_sim_seed": int(sim_seed),
        "cases": case_summaries,
        "total_pattern_difference_steps": int(total),
        "fraction_near_switch_windows": float(np.mean([bool(row["near_switch_window"]) for row in rows])) if rows else 0.0,
        "fraction_with_actual_switch_timing_differences": float(
            np.mean([bool(row["actual_switch_timing_difference"]) for row in rows])
        )
        if rows
        else 0.0,
        "fraction_with_candidate_history_differences": float(
            np.mean([bool(row["candidate_history_difference"]) for row in rows])
        )
        if rows
        else 0.0,
        "fraction_with_effective_candidate_history_differences": float(
            np.mean([bool(row["effective_candidate_history_difference"]) for row in rows])
        )
        if rows
        else 0.0,
        "fraction_transition_aligned": float(np.mean([bool(row["transition_alignment"]) for row in rows])) if rows else 0.0,
        "mean_tie_margin_at_pattern_difference_steps": float(np.mean([float(row["tie_margin"]) for row in rows])) if rows else 0.0,
        "mean_mismatch_norm_at_pattern_difference_steps": float(np.mean([float(row["mismatch_norm"]) for row in rows])) if rows else 0.0,
        "mean_crossing_direction_magnitude_at_pattern_difference_steps": float(
            np.mean([float(row["crossing_direction_magnitude"]) for row in rows])
        )
        if rows
        else 0.0,
        "counts_by_nearest_boundary_type": {
            "longitudinal": int(sum(1 for row in rows if str(row["nearest_boundary_type"]) == "longitudinal")),
            "radial": int(sum(1 for row in rows if str(row["nearest_boundary_type"]) == "radial")),
            "none": int(sum(1 for row in rows if str(row["nearest_boundary_type"]) == "none")),
        },
        "counts_by_entry_like_vs_exit_like_alignment": {
            "entry_like": int(sum(1 for row in rows if bool(row["entry_like_alignment"]))),
            "exit_like": int(sum(1 for row in rows if bool(row["exit_like_alignment"]))),
        },
    }
    dump_json(out / "tier2_fourth_residual_pattern_audit.json", global_summary)
    return out / "tier2_fourth_residual_pattern_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit support-transition-bias pattern changes on designed cases.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier2_fourth_residual_pattern_audit")
    args = parser.parse_args()
    out = run_tier2_fourth_residual_pattern_audit(args.config, args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Decision-level selector mechanism audit on validated Tier-1 scenario families."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.hybrid.order_selection import admissible_order
from src.verify.utils import dump_json, make_results_dir


GAP_TOL = 1.0e-12
COND_TOL = 1.0e-12


def _case_definitions(cfg) -> list[dict[str, object]]:
    gamma_edge = float(cfg.system.gamma_edge)
    return [
        {
            "case_name": "mixed_boundary_inside_split",
            "family": "mixed_boundary",
            "x0": [gamma_edge + 0.012, 0.0, -0.21, 0.0, 0.0, 0.0],
            "tier1_lateral_offsets": [0.0, 2.45, -2.60],
        },
        {
            "case_name": "mixed_boundary_outside_split",
            "family": "mixed_boundary",
            "x0": [gamma_edge + 0.012, 0.0, -0.21, 0.0, 0.0, 0.0],
            "tier1_lateral_offsets": [0.0, 2.55, -2.35],
        },
        {
            "case_name": "asymmetric_geometry_skewed",
            "family": "asymmetric_geometry",
            "x0": [0.11, -0.015, -0.24, -0.04, 0.03, 0.01],
            "tier1_lateral_offsets": [0.0, 1.10, -2.40],
        },
        {
            "case_name": "asymmetric_crossing_mixed",
            "family": "asymmetric_geometry",
            "x0": [0.09, -0.005, -0.22, -0.09, 0.06, 0.0],
            "tier1_lateral_offsets": [0.0, 2.20, -1.40],
        },
    ]


def _clone_cfg(cfg, *, selection_mode: str, lateral_offsets: list[float]):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = "tier1"
    cfg_mode.selection.mode = str(selection_mode)
    cfg_mode.monitor.enabled = False
    cfg_mode.monitor.mode = "shadow"
    cfg_mode.evaluation.mode = "upstream_truncated"
    cfg_mode.system.tier1.geometry.lateral_offsets = [float(v) for v in lateral_offsets]
    return cfg_mode


def _as_pi(pi_obj) -> list[int]:
    return [int(v) for v in pi_obj]


def _find_candidate_index(candidates: list[list[int]], pi: list[int]) -> int | None:
    target = tuple(_as_pi(pi))
    for idx, candidate in enumerate(candidates):
        if tuple(_as_pi(candidate)) == target:
            return int(idx)
    return None


def _candidate_pool_details(
    *,
    candidates: list[list[int]],
    admissible_mask: list[bool],
    predicted_gap_scores: list[float],
    conditioning_scores: list[float],
    current_pi: list[int],
    live_candidate: list[int],
    active_selected: list[int],
    x_now: np.ndarray,
    cfg,
) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        admissible, extra = admissible_order(candidate, x_now, cfg)
        details.append(
            {
                "candidate_pi": _as_pi(candidate),
                "admissible": bool(admissible_mask[idx]) if idx < len(admissible_mask) else bool(admissible),
                "dag_ok": bool(extra.get("dag_ok", False)),
                "topo_ok": bool(extra.get("topo_ok", False)),
                "predicted_gap": float(predicted_gap_scores[idx]) if idx < len(predicted_gap_scores) else float("nan"),
                "conditioning_proxy": float(conditioning_scores[idx]) if idx < len(conditioning_scores) else float("nan"),
                "is_current": bool(_as_pi(candidate) == _as_pi(current_pi)),
                "is_live_candidate": bool(_as_pi(candidate) == _as_pi(live_candidate)),
                "is_active_selected": bool(_as_pi(candidate) == _as_pi(active_selected)),
            }
        )
    return details


def _classify_selector_difference(
    *,
    shadow_reason: str,
    current_pi: list[int],
    live_candidate: list[int],
    active_selected: list[int],
    live_admissible: bool,
    active_admissible: bool,
    live_gap: float,
    active_gap: float,
    live_cond: float,
    active_cond: float,
) -> str:
    if shadow_reason == "no_admissible_candidates_hold_current":
        return "no_admissible_hold_current"
    if (not live_admissible) and active_admissible:
        return "inadmissible_live_candidate_avoided"
    if (
        active_selected == current_pi
        and live_admissible
        and active_admissible
        and abs(live_gap - active_gap) <= GAP_TOL
    ):
        return "hold_current_no_switch_tie"
    if active_admissible and live_admissible and active_gap + GAP_TOL < live_gap:
        return "min_predicted_gap"
    if (
        active_admissible
        and live_admissible
        and abs(live_gap - active_gap) <= GAP_TOL
        and active_cond + COND_TOL < live_cond
    ):
        return "conditioning_tie_break"
    return "other"


def _save_reason_counts_by_case(case_reason_counts: dict[str, Counter[str]], out_path: Path) -> None:
    case_names = list(case_reason_counts.keys())
    categories = sorted({reason for counts in case_reason_counts.values() for reason in counts})
    if not case_names or not categories:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No selector-difference steps", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    x = np.arange(len(case_names))
    width = 0.8 / max(1, len(categories))
    fig, ax = plt.subplots(figsize=(12, 4))
    for idx, category in enumerate(categories):
        values = [int(case_reason_counts[case].get(category, 0)) for case in case_names]
        ax.bar(x + idx * width, values, width=width, label=category)
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(case_names, rotation=20, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Selector difference reasons by case")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_difference_steps_by_case(case_counts: dict[str, int], out_path: Path) -> None:
    labels = list(case_counts.keys())
    values = [int(case_counts[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("selector-difference steps")
    ax.set_title("Selector difference steps by case")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if values.size:
        ax.hist(values, bins=min(20, max(8, values.size)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No selector-difference steps", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_case_metrics(sim: dict[str, object]) -> dict[str, object]:
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    return {
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in sim.get("switch_reason", []) if str(reason) == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
    }


def run_tier1_selector_mechanism_audit(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier1_selector_mechanism_audit",
) -> Path:
    cfg = load_config(config_path)
    out = (
        make_results_dir("tier1_selector_mechanism_audit")
        if out_dir == "results/tier1_selector_mechanism_audit"
        else Path(out_dir)
    )
    out.mkdir(parents=True, exist_ok=True)

    cases = _case_definitions(cfg)
    sim_seed = int(cfg.seed) + 20521
    options = SimOptions(blending_on=True, noise_delta=0.0, seed=sim_seed, disable_switching=False)

    rows: list[dict[str, object]] = []
    case_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    case_difference_counts: dict[str, int] = {}
    case_summaries: dict[str, object] = {}

    for case in cases:
        case_name = str(case["case_name"])
        case_family = str(case["family"])
        x0 = np.asarray(case["x0"], dtype=float)
        lateral_offsets = [float(v) for v in case["tier1_lateral_offsets"]]

        cfg_legacy = _clone_cfg(cfg, selection_mode="legacy", lateral_offsets=lateral_offsets)
        cfg_active = _clone_cfg(cfg, selection_mode="active_lexicographic", lateral_offsets=lateral_offsets)

        sim_legacy = simulate_closed_loop(cfg_legacy, x0=x0, horizon=float(cfg_legacy.system.horizon), options=options)
        sim_active = simulate_closed_loop(cfg_active, x0=x0, horizon=float(cfg_active.system.horizon), options=options)

        legacy_metrics = _run_case_metrics(sim_legacy)
        active_metrics = _run_case_metrics(sim_active)

        t_control = np.asarray(sim_active.get("t_control", []), dtype=float)
        tie_i = np.asarray(sim_active.get("tie_i", []), dtype=int)
        tie_j = np.asarray(sim_active.get("tie_j", []), dtype=int)
        tie_margin = np.asarray(sim_active.get("tie_gap_min", []), dtype=float)
        current_pi_hist = sim_active.get("pi", [])
        live_candidate_hist = sim_active.get("pi_candidate", [])
        active_selected_hist = sim_active.get("selector_candidate_pi", [])
        shadow_differs = np.asarray(sim_active.get("shadow_differs_from_live", []), dtype=bool)
        selector_switch_eligible = np.asarray(sim_active.get("selector_switch_eligible", []), dtype=bool)
        shadow_candidates_hist = sim_active.get("shadow_candidates", [])
        shadow_admissible_hist = sim_active.get("shadow_admissible_mask", [])
        shadow_conditioning_hist = sim_active.get("shadow_conditioning_scores", [])
        shadow_gap_hist = sim_active.get("shadow_predicted_gap_scores", [])
        shadow_reason_hist = sim_active.get("shadow_reason", [])
        x_hist = np.asarray(sim_active.get("x", []), dtype=float)
        edges_hist = sim_active.get("edges", [])

        case_diff_steps = 0
        for k in range(len(live_candidate_hist)):
            if not bool(shadow_differs[k]):
                continue

            current_pi = _as_pi(current_pi_hist[k])
            live_candidate = _as_pi(live_candidate_hist[k])
            active_selected = _as_pi(active_selected_hist[k])
            candidates = [_as_pi(pi) for pi in shadow_candidates_hist[k]]
            admissible_mask = [bool(v) for v in shadow_admissible_hist[k]]
            conditioning_scores = [float(v) for v in shadow_conditioning_hist[k]]
            predicted_gap_scores = [float(v) for v in shadow_gap_hist[k]]
            live_idx = _find_candidate_index(candidates, live_candidate)
            active_idx = _find_candidate_index(candidates, active_selected)
            live_admissible = bool(admissible_mask[live_idx]) if live_idx is not None and live_idx < len(admissible_mask) else False
            active_admissible = bool(admissible_mask[active_idx]) if active_idx is not None and active_idx < len(admissible_mask) else False
            live_gap = float(predicted_gap_scores[live_idx]) if live_idx is not None and live_idx < len(predicted_gap_scores) else float("nan")
            active_gap = float(predicted_gap_scores[active_idx]) if active_idx is not None and active_idx < len(predicted_gap_scores) else float("nan")
            live_cond = float(conditioning_scores[live_idx]) if live_idx is not None and live_idx < len(conditioning_scores) else float("nan")
            active_cond = float(conditioning_scores[active_idx]) if active_idx is not None and active_idx < len(conditioning_scores) else float("nan")
            reason = _classify_selector_difference(
                shadow_reason=str(shadow_reason_hist[k]),
                current_pi=current_pi,
                live_candidate=live_candidate,
                active_selected=active_selected,
                live_admissible=live_admissible,
                active_admissible=active_admissible,
                live_gap=live_gap,
                active_gap=active_gap,
                live_cond=live_cond,
                active_cond=active_cond,
            )
            candidate_pool = _candidate_pool_details(
                candidates=candidates,
                admissible_mask=admissible_mask,
                predicted_gap_scores=predicted_gap_scores,
                conditioning_scores=conditioning_scores,
                current_pi=current_pi,
                live_candidate=live_candidate,
                active_selected=active_selected,
                x_now=np.asarray(x_hist[k], dtype=float),
                cfg=cfg_active,
            )
            row = {
                "case_name": case_name,
                "case_family": case_family,
                "step": int(k),
                "time_sec": float(t_control[k]) if k < len(t_control) else float(k),
                "current_pi": current_pi,
                "live_candidate_pi": live_candidate,
                "active_selected_pi": active_selected,
                "active_differs_from_live": True,
                "tie_pair": [int(tie_i[k]), int(tie_j[k])],
                "tie_margin": float(tie_margin[k]),
                "selector_switch_eligible": bool(selector_switch_eligible[k]),
                "active_edge_pattern_signature": str(sorted((int(a), int(b)) for (a, b) in edges_hist[k])),
                "shadow_reason": str(shadow_reason_hist[k]),
                "reason_category": reason,
                "live_candidate_admissible": bool(live_admissible),
                "active_selected_admissible": bool(active_admissible),
                "live_candidate_predicted_gap": float(live_gap),
                "active_selected_predicted_gap": float(active_gap),
                "predicted_gap_advantage": float(live_gap - active_gap),
                "live_candidate_conditioning_proxy": float(live_cond),
                "active_selected_conditioning_proxy": float(active_cond),
                "candidate_pool": candidate_pool,
            }
            rows.append(row)
            case_reason_counts[case_name].update([reason])
            case_diff_steps += 1

        case_difference_counts[case_name] = int(case_diff_steps)
        case_summaries[case_name] = {
            "family": case_family,
            "x0": [float(v) for v in x0],
            "tier1_lateral_offsets": lateral_offsets,
            "legacy_metrics": legacy_metrics,
            "active_metrics": active_metrics,
            "delta_active_minus_legacy": {
                key: float(active_metrics[key]) - float(legacy_metrics[key]) for key in legacy_metrics
            },
            "selector_difference_steps": int(case_diff_steps),
            "reason_counts": dict(sorted(case_reason_counts[case_name].items())),
        }

    csv_path = out / "tier1_selector_mechanism_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "case_family",
                "step",
                "time_sec",
                "current_pi",
                "live_candidate_pi",
                "active_selected_pi",
                "active_differs_from_live",
                "tie_pair",
                "tie_margin",
                "selector_switch_eligible",
                "active_edge_pattern_signature",
                "shadow_reason",
                "reason_category",
                "live_candidate_admissible",
                "active_selected_admissible",
                "live_candidate_predicted_gap",
                "active_selected_predicted_gap",
                "predicted_gap_advantage",
                "live_candidate_conditioning_proxy",
                "active_selected_conditioning_proxy",
                "candidate_pool",
            ],
        )
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["current_pi"] = json.dumps(row["current_pi"])
            csv_row["live_candidate_pi"] = json.dumps(row["live_candidate_pi"])
            csv_row["active_selected_pi"] = json.dumps(row["active_selected_pi"])
            csv_row["tie_pair"] = json.dumps(row["tie_pair"])
            csv_row["candidate_pool"] = json.dumps(row["candidate_pool"])
            writer.writerow(csv_row)

    global_reason_counts = Counter()
    for counter in case_reason_counts.values():
        global_reason_counts.update(counter)
    total_diffs = len(rows)
    no_switch_like_count = int(global_reason_counts.get("hold_current_no_switch_tie", 0) + global_reason_counts.get("no_admissible_hold_current", 0))
    predicted_gap_count = int(global_reason_counts.get("min_predicted_gap", 0))
    admissibility_rescue_count = int(global_reason_counts.get("inadmissible_live_candidate_avoided", 0))
    conditioning_count = int(global_reason_counts.get("conditioning_tie_break", 0))
    gap_advantages = np.asarray([float(row["predicted_gap_advantage"]) for row in rows], dtype=float) if rows else np.array([], dtype=float)
    tie_margins = np.asarray([float(row["tie_margin"]) for row in rows], dtype=float) if rows else np.array([], dtype=float)

    summary = {
        "total_selector_difference_steps": int(total_diffs),
        "selector_difference_steps_by_case": {case: int(count) for case, count in sorted(case_difference_counts.items())},
        "reason_counts_global": dict(sorted(global_reason_counts.items())),
        "reason_counts_by_case": {case: dict(sorted(counts.items())) for case, counts in sorted(case_reason_counts.items())},
        "fraction_differences_due_to_no_switch_like_behavior": float(no_switch_like_count / total_diffs) if total_diffs else 0.0,
        "fraction_differences_due_to_predicted_gap_advantage": float(predicted_gap_count / total_diffs) if total_diffs else 0.0,
        "fraction_differences_due_to_admissibility_rescue": float(admissibility_rescue_count / total_diffs) if total_diffs else 0.0,
        "fraction_differences_due_to_conditioning": float(conditioning_count / total_diffs) if total_diffs else 0.0,
        "mean_predicted_gap_advantage": float(np.mean(gap_advantages)) if gap_advantages.size else 0.0,
        "median_predicted_gap_advantage": float(np.median(gap_advantages)) if gap_advantages.size else 0.0,
        "mean_tie_margin_at_differences": float(np.mean(tie_margins)) if tie_margins.size else 0.0,
        "case_summaries": case_summaries,
        "selection_modes_compared": ["legacy", "active_lexicographic"],
        "monitor_mode": "disabled",
        "evaluator_mode": "upstream_truncated",
        "shared_sim_seed": int(sim_seed),
    }
    dump_json(out / "tier1_selector_mechanism_audit.json", summary)

    _save_reason_counts_by_case(case_reason_counts, out / "selector_reason_counts_by_case.png")
    _save_difference_steps_by_case(case_difference_counts, out / "selector_difference_steps_by_case.png")
    _save_hist(
        gap_advantages,
        "Predicted-gap advantage at selector differences",
        "live_gap - active_gap",
        out / "predicted_gap_advantage_hist.png",
    )
    _save_hist(
        tie_margins,
        "Tie margin at selector differences",
        "tie margin",
        out / "tie_margin_at_selector_differences.png",
    )

    return out / "tier1_selector_mechanism_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-1 selector mechanism audit.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier1_selector_mechanism_audit")
    args = parser.parse_args()
    out = run_tier1_selector_mechanism_audit(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

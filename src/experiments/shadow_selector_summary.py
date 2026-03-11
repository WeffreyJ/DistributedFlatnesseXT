"""Summarize and classify shadow-selector disagreements from one rollout."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir


GAP_TOL = 1.0e-12


def _base_x0(cfg) -> np.ndarray:
    n = int(cfg.system.N)
    return np.concatenate(
        [
            np.array(cfg.reference.base, dtype=float),
            np.zeros(n, dtype=float),
        ],
        axis=0,
    )


def _as_pi_list(pi_obj) -> list[int]:
    return [int(v) for v in pi_obj]


def _find_candidate_index(candidates: list[list[int]], pi: list[int]) -> int | None:
    target = tuple(_as_pi_list(pi))
    for idx, candidate in enumerate(candidates):
        if tuple(_as_pi_list(candidate)) == target:
            return int(idx)
    return None


def _save_gap_advantage_hist(values: np.ndarray, out_path: Path) -> None:
    if values.size == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No shadow/live disagreements", ha="center", va="center")
        ax.set_title("Predicted-gap advantage histogram")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(30, max(8, values.size)))
    ax.set_xlabel("predicted_gap_advantage = live_gap - shadow_gap")
    ax.set_ylabel("count")
    ax.set_title("Shadow selector predicted-gap advantage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _classify_difference(
    *,
    reason: str,
    current_pi: list[int],
    live_candidate: list[int],
    shadow_selected: list[int],
    live_candidate_admissible: bool,
    shadow_selected_admissible: bool,
    live_candidate_gap: float,
    shadow_selected_gap: float,
) -> str:
    if reason == "no_admissible_candidates_hold_current":
        return "no_admissible_hold_current"
    if (not live_candidate_admissible) and shadow_selected_admissible:
        return "live_inadmissible_shadow_admissible"
    if (
        shadow_selected == current_pi
        and live_candidate_admissible
        and shadow_selected_admissible
        and abs(live_candidate_gap - shadow_selected_gap) <= GAP_TOL
    ):
        return "no_switch_tie"
    if (
        live_candidate_admissible
        and shadow_selected_admissible
        and shadow_selected_gap + GAP_TOL < live_candidate_gap
    ):
        return "shadow_lower_gap"
    return "other"


def run_shadow_selector_summary(config_path: str = "configs/system.yaml") -> Path:
    cfg = load_config(config_path)
    out_dir = make_results_dir("shadow_selector_summary")
    sim = simulate_closed_loop(
        cfg,
        x0=_base_x0(cfg),
        horizon=float(cfg.system.horizon),
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(cfg.seed) + 9600),
    )

    selected = sim.get("shadow_pi_selected", [])
    live = sim.get("pi_candidate", [])
    differs = np.asarray(sim.get("shadow_differs_from_live", []), dtype=bool)
    reasons = sim.get("shadow_reason", [])
    candidates_hist = sim.get("shadow_candidates", [])
    admissible_hist = sim.get("shadow_admissible_mask", [])
    predicted_gap_hist = sim.get("shadow_predicted_gap_scores", [])
    t_control = np.asarray(sim.get("t_control", []), dtype=float)

    rows: list[dict[str, object]] = []
    for k in range(len(selected)):
        if not bool(differs[k]):
            continue

        candidates = [_as_pi_list(pi) for pi in candidates_hist[k]]
        shadow_selected = _as_pi_list(selected[k])
        live_candidate = _as_pi_list(live[k])
        current_pi = candidates[0] if candidates else []
        admissible_mask = [bool(v) for v in admissible_hist[k]]
        predicted_gap_scores = [float(v) for v in predicted_gap_hist[k]]

        live_candidate_index = _find_candidate_index(candidates, live_candidate)
        shadow_selected_index = _find_candidate_index(candidates, shadow_selected)
        live_candidate_admissible = (
            bool(admissible_mask[live_candidate_index]) if live_candidate_index is not None and live_candidate_index < len(admissible_mask) else False
        )
        shadow_selected_admissible = (
            bool(admissible_mask[shadow_selected_index])
            if shadow_selected_index is not None and shadow_selected_index < len(admissible_mask)
            else False
        )
        live_candidate_gap = (
            float(predicted_gap_scores[live_candidate_index])
            if live_candidate_index is not None and live_candidate_index < len(predicted_gap_scores)
            else float("nan")
        )
        shadow_selected_gap = (
            float(predicted_gap_scores[shadow_selected_index])
            if shadow_selected_index is not None and shadow_selected_index < len(predicted_gap_scores)
            else float("nan")
        )
        gap_advantage = float(live_candidate_gap - shadow_selected_gap)
        category = _classify_difference(
            reason=str(reasons[k]),
            current_pi=current_pi,
            live_candidate=live_candidate,
            shadow_selected=shadow_selected,
            live_candidate_admissible=live_candidate_admissible,
            shadow_selected_admissible=shadow_selected_admissible,
            live_candidate_gap=live_candidate_gap,
            shadow_selected_gap=shadow_selected_gap,
        )

        rows.append(
            {
                "step": int(k),
                "time_sec": float(t_control[k]) if k < len(t_control) else float(k),
                "current_pi": str(current_pi),
                "live_candidate": str(live_candidate),
                "shadow_selected": str(shadow_selected),
                "live_candidate_index": int(live_candidate_index) if live_candidate_index is not None else None,
                "shadow_selected_index": int(shadow_selected_index) if shadow_selected_index is not None else None,
                "live_candidate_admissible": bool(live_candidate_admissible),
                "shadow_selected_admissible": bool(shadow_selected_admissible),
                "live_candidate_predicted_gap": float(live_candidate_gap),
                "shadow_selected_predicted_gap": float(shadow_selected_gap),
                "predicted_gap_advantage": gap_advantage,
                "difference_category": category,
                "reason": str(reasons[k]),
                "candidates": str(candidates),
                "admissible_mask": str(admissible_mask),
                "predicted_gap_scores": str(predicted_gap_scores),
            }
        )

    csv_path = out_dir / "shadow_selector_diff_steps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "time_sec",
                "current_pi",
                "live_candidate",
                "shadow_selected",
                "live_candidate_index",
                "shadow_selected_index",
                "live_candidate_admissible",
                "shadow_selected_admissible",
                "live_candidate_predicted_gap",
                "shadow_selected_predicted_gap",
                "predicted_gap_advantage",
                "difference_category",
                "reason",
                "candidates",
                "admissible_mask",
                "predicted_gap_scores",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    category_counts = Counter(str(row["difference_category"]) for row in rows)
    total_diffs = max(len(rows), 1)
    gap_advantages = np.asarray([float(row["predicted_gap_advantage"]) for row in rows], dtype=float) if rows else np.array([], dtype=float)
    positive_adv = gap_advantages > GAP_TOL
    zero_adv = np.abs(gap_advantages) <= GAP_TOL

    representative_by_category: dict[str, dict[str, object]] = {}
    for category in sorted(category_counts):
        match = next((row for row in rows if str(row["difference_category"]) == category), None)
        if match is not None:
            representative_by_category[category] = {
                "step": int(match["step"]),
                "current_pi": str(match["current_pi"]),
                "live_candidate": str(match["live_candidate"]),
                "shadow_selected": str(match["shadow_selected"]),
                "predicted_gap_advantage": float(match["predicted_gap_advantage"]),
                "reason": str(match["reason"]),
            }

    payload = {
        "shadow_selection_enabled": bool(sim.get("shadow_selection_enabled", False)),
        "conditioning_proxy_note": "Current toy surrogate uses a constant conditioning proxy; lexicographic differences therefore come from admissibility, no-switch preference, and predicted-gap comparisons.",
        "num_steps": int(len(selected)),
        "num_shadow_differences": int(np.count_nonzero(differs)) if differs.size else 0,
        "shadow_difference_fraction": float(np.mean(differs)) if differs.size else 0.0,
        "first_difference_step": int(rows[0]["step"]) if rows else None,
        "reasons_seen": sorted({str(v) for v in reasons}),
        "difference_category_counts": dict(sorted(category_counts.items())),
        "difference_category_fractions": {
            key: float(value / total_diffs) for key, value in sorted(category_counts.items())
        },
        "mean_predicted_gap_advantage_on_differences": float(np.mean(gap_advantages)) if gap_advantages.size else 0.0,
        "median_predicted_gap_advantage_on_differences": float(np.median(gap_advantages)) if gap_advantages.size else 0.0,
        "fraction_differences_with_positive_gap_advantage": float(np.mean(positive_adv)) if gap_advantages.size else 0.0,
        "fraction_differences_with_zero_gap_advantage": float(np.mean(zero_adv)) if gap_advantages.size else 0.0,
        "representative_rows_by_category": representative_by_category,
    }
    summary_path = out_dir / "shadow_selector_summary.json"
    dump_json(summary_path, payload)

    _save_gap_advantage_hist(gap_advantages, out_dir / "predicted_gap_advantage_hist.png")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow selector rollout summary")
    parser.add_argument("--config", default="configs/system.yaml")
    args = parser.parse_args()
    path = run_shadow_selector_summary(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

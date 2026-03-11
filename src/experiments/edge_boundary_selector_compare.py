"""Selector-consequence study on edge-boundary scenarios with monitor disabled."""

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
from src.verify.utils import dump_json, make_results_dir


SELECTOR_MODES = ["legacy", "shadow_lexicographic", "active_lexicographic"]


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


def _clone_cfg(cfg, *, plant_family: str, selection_mode: str, lateral_offsets: list[float] | None = None):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = str(plant_family)
    cfg_mode.selection.mode = str(selection_mode)
    cfg_mode.monitor.enabled = False
    cfg_mode.monitor.mode = "shadow"
    cfg_mode.evaluation.mode = "upstream_truncated"
    if lateral_offsets is not None:
        cfg_mode.system.tier1.geometry.lateral_offsets = [float(v) for v in lateral_offsets]
    return cfg_mode


def _run_metrics(sim: dict[str, object], *, case_name: str, plant_family: str, selection_mode: str) -> dict[str, object]:
    switch_reason = [str(v) for v in sim.get("switch_reason", [])]
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    return {
        "case_name": str(case_name),
        "plant_family": str(plant_family),
        "selection_mode": str(selection_mode),
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in switch_reason if reason == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
        "effective_candidate_diff_steps": int(
            np.count_nonzero(np.asarray(sim.get("effective_candidate_differs_from_live", []), dtype=bool))
        ),
        "selector_candidate_differs_from_current_count": int(
            np.count_nonzero(np.asarray(sim.get("selector_candidate_differs_from_current", []), dtype=bool))
        ),
        "shadow_differs_from_live_count": int(
            np.count_nonzero(np.asarray(sim.get("shadow_differs_from_live", []), dtype=bool))
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_name",
        "plant_family",
        "selection_mode",
        "switch_count",
        "transition_start_count",
        "blend_active_steps",
        "tracking_error_mean",
        "tracking_error_max",
        "max_raw_jump",
        "max_applied_jump",
        "effective_candidate_diff_steps",
        "selector_candidate_differs_from_current_count",
        "shadow_differs_from_live_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grouped_plot(rows: list[dict[str, object]], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [f'{row["case_name"]}\n{row["plant_family"]}\n{row["selection_mode"]}' for row in rows]
    x = np.arange(len(labels))
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_edge_boundary_selector_compare(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/edge_boundary_selector_compare",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("edge_boundary_selector_compare") if out_dir == "results/edge_boundary_selector_compare" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = _edge_cases(cfg)
    sim_seed = int(cfg.seed) + 19621
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
            for selection_mode in SELECTOR_MODES:
                cfg_mode = _clone_cfg(
                    cfg,
                    plant_family=plant_family,
                    selection_mode=selection_mode,
                    lateral_offsets=lateral_offsets if plant_family == "tier1" else None,
                )
                sim = simulate_closed_loop(
                    cfg_mode,
                    x0=x0,
                    horizon=float(cfg_mode.system.horizon),
                    options=options,
                )
                metrics = _run_metrics(
                    sim,
                    case_name=case_name,
                    plant_family=plant_family,
                    selection_mode=selection_mode,
                )
                rows.append(metrics)
                family_runs[selection_mode] = metrics

            legacy = family_runs["legacy"]
            shadow = family_runs["shadow_lexicographic"]
            active = family_runs["active_lexicographic"]
            summary_cases[case_name]["families"][plant_family] = {
                "legacy": legacy,
                "shadow_lexicographic": shadow,
                "active_lexicographic": active,
                "delta_active_minus_legacy": {
                    "switch_count": int(active["switch_count"]) - int(legacy["switch_count"]),
                    "transition_start_count": int(active["transition_start_count"]) - int(legacy["transition_start_count"]),
                    "blend_active_steps": int(active["blend_active_steps"]) - int(legacy["blend_active_steps"]),
                    "tracking_error_mean": float(active["tracking_error_mean"]) - float(legacy["tracking_error_mean"]),
                    "tracking_error_max": float(active["tracking_error_max"]) - float(legacy["tracking_error_max"]),
                    "max_raw_jump": float(active["max_raw_jump"]) - float(legacy["max_raw_jump"]),
                    "max_applied_jump": float(active["max_applied_jump"]) - float(legacy["max_applied_jump"]),
                    "effective_candidate_diff_steps": int(active["effective_candidate_diff_steps"]) - int(legacy["effective_candidate_diff_steps"]),
                    "selector_candidate_differs_from_current_count": int(active["selector_candidate_differs_from_current_count"])
                    - int(legacy["selector_candidate_differs_from_current_count"]),
                    "shadow_differs_from_live_count": int(active["shadow_differs_from_live_count"]) - int(legacy["shadow_differs_from_live_count"]),
                },
                "delta_shadow_minus_legacy": {
                    "switch_count": int(shadow["switch_count"]) - int(legacy["switch_count"]),
                    "transition_start_count": int(shadow["transition_start_count"]) - int(legacy["transition_start_count"]),
                    "blend_active_steps": int(shadow["blend_active_steps"]) - int(legacy["blend_active_steps"]),
                    "tracking_error_mean": float(shadow["tracking_error_mean"]) - float(legacy["tracking_error_mean"]),
                    "tracking_error_max": float(shadow["tracking_error_max"]) - float(legacy["tracking_error_max"]),
                    "max_raw_jump": float(shadow["max_raw_jump"]) - float(legacy["max_raw_jump"]),
                    "max_applied_jump": float(shadow["max_applied_jump"]) - float(legacy["max_applied_jump"]),
                    "effective_candidate_diff_steps": int(shadow["effective_candidate_diff_steps"]) - int(legacy["effective_candidate_diff_steps"]),
                    "selector_candidate_differs_from_current_count": int(shadow["selector_candidate_differs_from_current_count"])
                    - int(legacy["selector_candidate_differs_from_current_count"]),
                    "shadow_differs_from_live_count": int(shadow["shadow_differs_from_live_count"]) - int(legacy["shadow_differs_from_live_count"]),
                },
            }

    summary = {
        "cases": summary_cases,
        "selection_modes": SELECTOR_MODES,
        "monitor_mode": "disabled",
        "evaluator_mode": "upstream_truncated",
        "shared_sim_seed": int(sim_seed),
    }
    dump_json(out / "edge_boundary_selector_compare.json", summary)
    _write_csv(out / "edge_boundary_selector_compare.csv", rows)

    _grouped_plot(
        rows,
        "max_applied_jump",
        "Edge-boundary selector comparison: max applied jump",
        "max applied jump",
        out / "jump_compare_by_case_mode.png",
    )
    _grouped_plot(
        rows,
        "tracking_error_mean",
        "Edge-boundary selector comparison: tracking error mean",
        "tracking error mean",
        out / "tracking_compare_by_case_mode.png",
    )
    _grouped_plot(
        rows,
        "effective_candidate_diff_steps",
        "Edge-boundary selector effect: effective candidate diff steps",
        "count",
        out / "selector_effect_compare.png",
    )

    return out / "edge_boundary_selector_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Selector comparison on edge-boundary scenarios.")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/edge_boundary_selector_compare")
    args = parser.parse_args()
    out = run_edge_boundary_selector_compare(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

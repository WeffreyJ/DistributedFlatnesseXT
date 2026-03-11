"""Tier-2 artifact regression runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.verify.utils import dump_json, make_results_dir


FAMILY_ARTIFACTS = {
    "transverse_skew": {
        "gap": "results/tier2_nominal_vs_plant_gap/tier2_nominal_vs_plant_gap_summary.json",
        "generic_compare": "results/tier2_residual_compare/tier2_residual_compare.json",
        "seed_sweep": "results/tier2_residual_seed_sweep/tier2_residual_seed_sweep.json",
        "designed": "results/tier2_mismatch_sensitive_scenarios/tier2_mismatch_sensitive_scenarios.json",
        "audit": "results/tier2_pattern_change_audit/tier2_pattern_change_audit.json",
    },
    "longitudinal_bias": {
        "gap": "results/tier2_second_residual_gap/tier2_second_residual_gap_summary.json",
        "generic_compare": "results/tier2_second_residual_compare/tier2_second_residual_compare.json",
        "seed_sweep": "results/tier2_second_residual_seed_sweep/tier2_second_residual_seed_sweep.json",
        "designed": "results/tier2_second_residual_scenarios/tier2_second_residual_scenarios.json",
        "audit": "results/tier2_second_residual_pattern_audit/tier2_second_residual_pattern_audit.json",
    },
    "edge_band_bias": {
        "gap": "results/tier2_third_residual_gap/tier2_third_residual_gap_summary.json",
        "generic_compare": "results/tier2_third_residual_compare/tier2_third_residual_compare.json",
        "seed_sweep": "results/tier2_third_residual_seed_sweep/tier2_third_residual_seed_sweep.json",
        "designed": "results/tier2_third_residual_scenarios/tier2_third_residual_scenarios.json",
        "audit": "results/tier2_third_residual_pattern_audit/tier2_third_residual_pattern_audit.json",
    },
    "support_transition_bias": {
        "gap": "results/tier2_fourth_residual_gap/tier2_fourth_residual_gap_summary.json",
        "generic_compare": "results/tier2_fourth_residual_compare/tier2_fourth_residual_compare.json",
        "seed_sweep": "results/tier2_fourth_residual_seed_sweep/tier2_fourth_residual_seed_sweep.json",
        "designed": "results/tier2_fourth_residual_scenarios/tier2_fourth_residual_scenarios.json",
        "audit": "results/tier2_fourth_residual_pattern_audit/tier2_fourth_residual_pattern_audit.json",
    },
}

PACKAGE_ARTIFACTS = {
    "comparison_json": "results/tier2_cross_family_compare/tier2_cross_family_compare.json",
    "table_rows_json": "results/tier2_cross_family_compare/tier2_cross_family_table_rows.json",
    "generic_plot_data": "results/tier2_cross_family_compare/tier2_cross_family_generic_plot_data.json",
    "designed_plot_data": "results/tier2_cross_family_compare/tier2_cross_family_designed_plot_data.json",
    "audit_plot_data": "results/tier2_cross_family_compare/tier2_cross_family_audit_plot_data.json",
}


def _exists(path: str) -> bool:
    return Path(path).exists()


def _load_json(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _reduction_anchor_status() -> tuple[str, list[str]]:
    path = "results/tier2_reduction_witness/tier2_reduction_witness.json"
    obj = _load_json(path)
    if obj is None:
        return "reduction_missing", [path]
    pointwise_ok = bool(obj.get("pointwise", {}).get("pointwise_nominal_match", False))
    rollout_ok = bool(obj.get("rollout", {}).get("rollout_match", False))
    if pointwise_ok and rollout_ok:
        return "exact_reduction_preserved", []
    return "reduction_inconsistent", []


def run_tier2_regression_runner(
    out_dir: str | Path = "results/tier2_regression_runner",
) -> Path:
    out = make_results_dir("tier2_regression_runner") if out_dir == "results/tier2_regression_runner" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reduction_anchor_status, missing_artifacts = _reduction_anchor_status()

    families: dict[str, Any] = {}
    for family_name, paths in FAMILY_ARTIFACTS.items():
        family_missing = []
        gap_exists = _exists(paths["gap"])
        generic_compare_exists = _exists(paths["generic_compare"])
        seed_sweep_exists = _exists(paths["seed_sweep"])
        designed_exists = _exists(paths["designed"])
        audit_exists = _exists(paths["audit"])
        if not gap_exists:
            family_missing.append(paths["gap"])
        if not generic_compare_exists:
            family_missing.append(paths["generic_compare"])
        if not seed_sweep_exists:
            family_missing.append(paths["seed_sweep"])
        if not designed_exists:
            family_missing.append(paths["designed"])
        if not audit_exists:
            family_missing.append(paths["audit"])
        missing_artifacts.extend(family_missing)
        families[family_name] = {
            "gap_exists": gap_exists,
            "generic_compare_exists": generic_compare_exists,
            "seed_sweep_exists": seed_sweep_exists,
            "designed_exists": designed_exists,
            "audit_exists": audit_exists,
            "status": "ok" if not family_missing else "missing_artifacts",
        }

    cross_family_package_exists = {name: _exists(path) for name, path in PACKAGE_ARTIFACTS.items()}
    for name, exists in cross_family_package_exists.items():
        if not exists:
            missing_artifacts.append(PACKAGE_ARTIFACTS[name])

    overall_status = "pass"
    if reduction_anchor_status != "exact_reduction_preserved" or missing_artifacts:
        overall_status = "fail"

    summary = {
        "phase_identity": "tier2_regression_runner",
        "reduction_anchor_status": reduction_anchor_status,
        "frozen_interface_policy": "unchanged_required",
        "families": families,
        "cross_family_package_exists": cross_family_package_exists,
        "project_position": "tier2_completed_four_family_ladder",
        "missing_artifacts": sorted(set(missing_artifacts)),
        "overall_status": overall_status,
    }
    dump_json(out / "tier2_regression_runner_summary.json", summary)
    return out / "tier2_regression_runner_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the canonical Tier-2 artifact ladder.")
    parser.add_argument("--out", default="results/tier2_regression_runner")
    args = parser.parse_args()
    out = run_tier2_regression_runner(args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

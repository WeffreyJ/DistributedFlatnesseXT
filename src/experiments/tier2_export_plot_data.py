"""Export figure-ready plot data from the Tier-2 cross-family comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.verify.utils import dump_json


def run_tier2_export_plot_data(
    compare_path: str | Path = "results/tier2_cross_family_compare/tier2_cross_family_compare.json",
    out_dir: str | Path = "results/tier2_cross_family_compare",
) -> list[Path]:
    compare_obj = json.loads(Path(compare_path).read_text(encoding="utf-8"))
    families = compare_obj.get("families", {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generic = {
        "phase_identity": "tier2_cross_family_generic_plot_data",
        "families": [
            {
                "family_name": name,
                "generic_mismatch_delta": family["generic"]["delta_mean_nominal_vs_plant_gap_over_rollout"],
                "max_raw_jump_delta": family["generic"]["delta_max_raw_jump"],
                "max_applied_jump_delta": family["generic"]["delta_max_applied_jump"],
                "tracking_error_mean_delta": family["generic"]["delta_tracking_error_mean"],
                "switch_count_delta": family["generic"]["delta_switch_count"],
                "blend_active_steps_delta": family["generic"]["delta_blend_active_steps"],
            }
            for name, family in families.items()
        ],
    }

    designed = {
        "phase_identity": "tier2_cross_family_designed_plot_data",
        "families": [
            {
                "family_name": name,
                "num_cases": family["designed"]["num_cases"],
                "switch_pattern_fraction": (
                    family["designed"]["num_switch_pattern_differs"] / max(family["designed"]["num_cases"], 1)
                ),
                "candidate_pattern_fraction": (
                    family["designed"]["num_candidate_pattern_differs"] / max(family["designed"]["num_cases"], 1)
                ),
                "effective_candidate_pattern_fraction": (
                    family["designed"]["num_effective_candidate_pattern_differs"] / max(family["designed"]["num_cases"], 1)
                ),
                "strongest_cases": family["designed"]["strongest_cases"],
                "burden_change_case_count": family["designed"]["num_cases_with_burden_change"],
            }
            for name, family in families.items()
        ],
    }

    audit = {
        "phase_identity": "tier2_cross_family_audit_plot_data",
        "families": [
            {
                "family_name": name,
                "fraction_near_switch_windows": family["audit"]["fraction_near_switch_windows"],
                "fraction_with_actual_switch_timing_differences": family["audit"]["fraction_with_actual_switch_timing_differences"],
                "fraction_with_candidate_history_differences": family["audit"]["fraction_with_candidate_history_differences"],
                "fraction_with_effective_candidate_history_differences": family["audit"]["fraction_with_effective_candidate_history_differences"],
                "mean_tie_margin_at_pattern_difference_steps": family["audit"]["mean_tie_margin_at_pattern_difference_steps"],
                "mean_mismatch_norm_at_pattern_difference_steps": family["audit"]["mean_mismatch_norm_at_pattern_difference_steps"],
                "fraction_transition_aligned": family["audit"].get("fraction_transition_aligned"),
                "counts_by_nearest_boundary_type": family["audit"].get("counts_by_nearest_boundary_type"),
            }
            for name, family in families.items()
        ],
    }

    paths = [
        out / "tier2_cross_family_generic_plot_data.json",
        out / "tier2_cross_family_designed_plot_data.json",
        out / "tier2_cross_family_audit_plot_data.json",
    ]
    dump_json(paths[0], generic)
    dump_json(paths[1], designed)
    dump_json(paths[2], audit)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export plot-ready Tier-2 cross-family data.")
    parser.add_argument("--compare", default="results/tier2_cross_family_compare/tier2_cross_family_compare.json")
    parser.add_argument("--out", default="results/tier2_cross_family_compare")
    args = parser.parse_args()
    paths = run_tier2_export_plot_data(args.compare, args.out)
    print(f"Wrote {paths[-1]}")


if __name__ == "__main__":
    main()

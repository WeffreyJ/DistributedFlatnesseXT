"""Cross-family Tier-2 comparison and synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.verify.utils import dump_json, make_results_dir


FAMILY_SPECS = {
    "transverse_skew": {
        "family_name": "transverse_skew",
        "mechanism_label": "lateral asymmetry mismatch",
        "support_label": "nominal active-edge support",
        "status_label": "first live mismatch family",
        "gap_path": "results/tier2_nominal_vs_plant_gap/tier2_nominal_vs_plant_gap_summary.json",
        "compare_path": "results/tier2_residual_compare/tier2_residual_compare.json",
        "compare_delta_key": "delta_tier2_enabled_minus_tier1",
        "seed_path": "results/tier2_residual_seed_sweep/tier2_residual_seed_sweep.json",
        "scenarios_path": "results/tier2_mismatch_sensitive_scenarios/tier2_mismatch_sensitive_scenarios.json",
        "audit_path": "results/tier2_pattern_change_audit/tier2_pattern_change_audit.json",
    },
    "longitudinal_bias": {
        "family_name": "longitudinal_bias",
        "mechanism_label": "persistent longitudinal-window bias",
        "support_label": "nominal active-edge support",
        "status_label": "strongest generic mismatch/severity family",
        "gap_path": "results/tier2_second_residual_gap/tier2_second_residual_gap_summary.json",
        "compare_path": "results/tier2_second_residual_compare/tier2_second_residual_compare.json",
        "compare_delta_key": "delta_tier2_longitudinal_bias_minus_tier1",
        "seed_path": "results/tier2_second_residual_seed_sweep/tier2_second_residual_seed_sweep.json",
        "scenarios_path": "results/tier2_second_residual_scenarios/tier2_second_residual_scenarios.json",
        "audit_path": "results/tier2_second_residual_pattern_audit/tier2_second_residual_pattern_audit.json",
    },
    "edge_band_bias": {
        "family_name": "edge_band_bias",
        "mechanism_label": "static edge-band boundary localization",
        "support_label": "support-boundary shell on active edges",
        "status_label": "support-boundary mechanism-matched family",
        "gap_path": "results/tier2_third_residual_gap/tier2_third_residual_gap_summary.json",
        "compare_path": "results/tier2_third_residual_compare/tier2_third_residual_compare.json",
        "compare_delta_key": "delta_tier2_edge_band_bias_minus_tier1",
        "seed_path": "results/tier2_third_residual_seed_sweep/tier2_third_residual_seed_sweep.json",
        "scenarios_path": "results/tier2_third_residual_scenarios/tier2_third_residual_scenarios.json",
        "audit_path": "results/tier2_third_residual_pattern_audit/tier2_third_residual_pattern_audit.json",
    },
    "support_transition_bias": {
        "family_name": "support_transition_bias",
        "mechanism_label": "support-transition entry/exit bias",
        "support_label": "longitudinal transition shell with radial relevance gating",
        "status_label": "transition-aligned count-neutral family",
        "gap_path": "results/tier2_fourth_residual_gap/tier2_fourth_residual_gap_summary.json",
        "compare_path": "results/tier2_fourth_residual_compare/tier2_fourth_residual_compare.json",
        "compare_delta_key": "delta_tier2_support_transition_bias_minus_tier1",
        "seed_path": "results/tier2_fourth_residual_seed_sweep/tier2_fourth_residual_seed_sweep.json",
        "scenarios_path": "results/tier2_fourth_residual_scenarios/tier2_fourth_residual_scenarios.json",
        "audit_path": "results/tier2_fourth_residual_pattern_audit/tier2_fourth_residual_pattern_audit.json",
    },
}


def _load_json(path: str, missing: list[str]) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        missing.append(path)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _weighted_case_mean(cases: dict[str, Any], value_key: str, count_key: str = "total_pattern_difference_steps") -> float:
    total_w = 0.0
    total_v = 0.0
    for case in cases.values():
        w = float(case.get(count_key, 0.0))
        total_w += w
        total_v += w * _float(case.get(value_key, 0.0))
    return total_v / total_w if total_w > 0.0 else 0.0


def _extract_gap(obj: dict[str, Any] | None) -> dict[str, Any]:
    if obj is None:
        return {
            "mean_nominal_vs_plant_gap": 0.0,
            "max_nominal_vs_plant_gap": 0.0,
            "mean_residual_order_gap": 0.0,
            "max_residual_order_gap": 0.0,
            "gap_liveness_label": "missing",
        }
    mean_gap = _float(obj.get("mean_nominal_vs_plant_gap", 0.0))
    max_gap = _float(obj.get("max_nominal_vs_plant_gap", 0.0))
    mean_order = _float(obj.get("mean_residual_order_gap", 0.0))
    max_order = _float(obj.get("max_residual_order_gap", 0.0))
    live = max(mean_gap, max_gap, mean_order, max_order)
    return {
        "mean_nominal_vs_plant_gap": mean_gap,
        "max_nominal_vs_plant_gap": max_gap,
        "mean_residual_order_gap": mean_order,
        "max_residual_order_gap": max_order,
        "gap_liveness_label": "live" if live > 1.0e-6 else "tolerance_level",
    }


def _extract_generic(obj: dict[str, Any] | None, delta_key: str) -> dict[str, Any]:
    delta = {} if obj is None else obj.get(delta_key, {})
    out = {
        "delta_switch_count": _float(delta.get("switch_count", 0.0)),
        "delta_transition_start_count": _float(delta.get("transition_start_count", 0.0)),
        "delta_blend_active_steps": _float(delta.get("blend_active_steps", 0.0)),
        "delta_blocked_steps_total": _float(delta.get("blocked_steps_total", 0.0)),
        "delta_tracking_error_mean": _float(delta.get("tracking_error_mean", 0.0)),
        "delta_max_raw_jump": _float(delta.get("max_raw_jump", 0.0)),
        "delta_max_applied_jump": _float(delta.get("max_applied_jump", 0.0)),
        "delta_mean_nominal_vs_plant_gap_over_rollout": _float(delta.get("mean_nominal_vs_plant_gap_over_rollout", 0.0)),
    }
    out["generic_burden_changing"] = any(
        abs(out[key]) > 0.0
        for key in [
            "delta_switch_count",
            "delta_transition_start_count",
            "delta_blend_active_steps",
            "delta_blocked_steps_total",
        ]
    )
    if out["generic_burden_changing"]:
        label = "generic_burden_relevant"
    elif max(
        abs(out["delta_mean_nominal_vs_plant_gap_over_rollout"]),
        abs(out["delta_max_raw_jump"]),
        abs(out["delta_max_applied_jump"]),
        abs(out["delta_tracking_error_mean"]),
    ) < 1.0e-4:
        label = "near_inert_generic"
    elif max(
        abs(out["delta_max_raw_jump"]),
        abs(out["delta_max_applied_jump"]),
        abs(out["delta_tracking_error_mean"]),
    ) > 1.0e-3:
        label = "generic_severity_perturbation"
    else:
        label = "generic_mismatch_perturbation"
    out["generic_interpretation_label"] = label
    return out


def _extract_seed(obj: dict[str, Any] | None) -> dict[str, Any]:
    ds = {} if obj is None else obj.get("delta_summary", {})

    def _frac(metric: str, which: str) -> float:
        return _float(ds.get(metric, {}).get(which, 0.0))

    out = {
        "switch_delta_zero_fraction": _frac("delta_switch_count", "zero_fraction"),
        "transition_delta_zero_fraction": _frac("delta_transition_start_count", "zero_fraction"),
        "blend_delta_zero_fraction": _frac("delta_blend_active_steps", "zero_fraction"),
        "positive_mismatch_fraction": _frac("delta_mean_nominal_vs_plant_gap", "positive_fraction"),
        "positive_raw_jump_fraction": _frac("delta_max_raw_jump", "positive_fraction"),
        "positive_applied_jump_fraction": _frac("delta_max_applied_jump", "positive_fraction"),
    }
    if out["switch_delta_zero_fraction"] == 1.0 and out["transition_delta_zero_fraction"] == 1.0 and out["blend_delta_zero_fraction"] == 1.0:
        out["seed_sweep_interpretation_label"] = (
            "near_inert_seed_sweep" if _frac("delta_mean_nominal_vs_plant_gap", "mean") < 1.0e-4 else "count_neutral_consistent_mismatch"
        )
    else:
        out["seed_sweep_interpretation_label"] = "burden_variable_seed_sweep"
    return out


def _extract_designed(obj: dict[str, Any] | None) -> dict[str, Any]:
    cases = {} if obj is None else obj.get("cases", {})
    num_cases = len(cases)
    num_switch = 0
    num_candidate = 0
    num_effective = 0
    num_burden = 0
    ranked: list[tuple[float, str]] = []
    for case_name, case in cases.items():
        delta_key = next((k for k in case.keys() if str(k).startswith("delta_tier2")), None)
        delta = {} if delta_key is None else case.get(delta_key, {})
        switch_diff = bool(delta.get("switch_pattern_differs", False))
        cand_diff = bool(delta.get("candidate_pattern_differs", False))
        eff_diff = bool(delta.get("effective_candidate_pattern_differs", False))
        num_switch += int(switch_diff)
        num_candidate += int(cand_diff)
        num_effective += int(eff_diff)
        burden_mag = sum(abs(_float(delta.get(key, 0.0))) for key in ["delta_switch_count", "delta_transition_start_count", "delta_blend_active_steps"])
        if burden_mag > 0.0:
            num_burden += 1
        score = burden_mag + (2.0 if switch_diff else 0.0) + (1.0 if cand_diff else 0.0) + abs(_float(delta.get("delta_mean_nominal_vs_plant_gap_over_rollout", 0.0)))
        ranked.append((score, str(case_name)))
    if num_burden > 0:
        label = "burden_relevant"
    elif num_switch > 0 or num_candidate > 0 or num_effective > 0:
        label = "pattern_relevant"
    else:
        label = "none"
    return {
        "num_cases": int(num_cases),
        "num_switch_pattern_differs": int(num_switch),
        "num_candidate_pattern_differs": int(num_candidate),
        "num_effective_candidate_pattern_differs": int(num_effective),
        "num_cases_with_burden_change": int(num_burden),
        "strongest_cases": [name for _, name in sorted(ranked, reverse=True)[:3]],
        "designed_relevance_label": label,
    }


def _extract_audit(obj: dict[str, Any] | None, family_name: str) -> dict[str, Any]:
    if obj is None:
        return {
            "total_pattern_difference_steps": 0,
            "fraction_near_switch_windows": 0.0,
            "fraction_with_actual_switch_timing_differences": 0.0,
            "fraction_with_candidate_history_differences": 0.0,
            "fraction_with_effective_candidate_history_differences": 0.0,
            "mean_tie_margin_at_pattern_difference_steps": 0.0,
            "mean_mismatch_norm_at_pattern_difference_steps": 0.0,
            "dominant_mechanism_label": "missing",
        }
    cases = obj.get("cases", {})
    if family_name == "transverse_skew":
        global_block = obj.get("global", {})
        out = {
            "total_pattern_difference_steps": int(global_block.get("total_pattern_difference_steps", 0)),
            "fraction_near_switch_windows": _float(global_block.get("fraction_pattern_differences_near_switch_windows", 0.0)),
            "fraction_with_actual_switch_timing_differences": _float(global_block.get("fraction_pattern_differences_with_actual_switch_timing_difference", 0.0)),
            "fraction_with_candidate_history_differences": _weighted_case_mean(cases, "fraction_pattern_differences_with_candidate_change"),
            "fraction_with_effective_candidate_history_differences": _weighted_case_mean(cases, "fraction_pattern_differences_with_effective_candidate_change"),
            "mean_tie_margin_at_pattern_difference_steps": _weighted_case_mean(cases, "mean_tie_margin_at_pattern_difference_steps"),
            "mean_mismatch_norm_at_pattern_difference_steps": _weighted_case_mean(cases, "mean_mismatch_at_pattern_difference_steps"),
        }
    elif family_name in {"longitudinal_bias", "edge_band_bias"}:
        prefix = "longitudinal_band" if family_name == "longitudinal_bias" else "edge_band"
        out = {
            "total_pattern_difference_steps": int(obj.get("total_pattern_difference_steps", 0)),
            "fraction_near_switch_windows": _float(obj.get("fraction_pattern_differences_near_switch_windows_global", 0.0)),
            "fraction_with_actual_switch_timing_differences": _float(obj.get("fraction_pattern_differences_with_actual_switch_timing_difference_global", 0.0)),
            "fraction_with_candidate_history_differences": _float(obj.get("fraction_pattern_differences_with_candidate_change_global", 0.0)),
            "fraction_with_effective_candidate_history_differences": _float(obj.get("fraction_pattern_differences_with_effective_candidate_change_global", 0.0)),
            "mean_tie_margin_at_pattern_difference_steps": _float(obj.get("mean_tie_margin_at_pattern_difference_steps_global", 0.0)),
            "mean_mismatch_norm_at_pattern_difference_steps": _float(obj.get("mean_mismatch_at_pattern_difference_steps_global", 0.0)),
            f"mean_{prefix}_distance_at_pattern_difference_steps": _float(obj.get(f"mean_{prefix}_distance_at_pattern_difference_steps_global", 0.0)),
        }
    else:
        out = {
            "total_pattern_difference_steps": int(obj.get("total_pattern_difference_steps", 0)),
            "fraction_near_switch_windows": _float(obj.get("fraction_near_switch_windows", 0.0)),
            "fraction_with_actual_switch_timing_differences": _float(obj.get("fraction_with_actual_switch_timing_differences", 0.0)),
            "fraction_with_candidate_history_differences": _float(obj.get("fraction_with_candidate_history_differences", 0.0)),
            "fraction_with_effective_candidate_history_differences": _float(obj.get("fraction_with_effective_candidate_history_differences", 0.0)),
            "mean_tie_margin_at_pattern_difference_steps": _float(obj.get("mean_tie_margin_at_pattern_difference_steps", 0.0)),
            "mean_mismatch_norm_at_pattern_difference_steps": _float(obj.get("mean_mismatch_norm_at_pattern_difference_steps", 0.0)),
            "fraction_transition_aligned": _float(obj.get("fraction_transition_aligned", 0.0)),
            "counts_by_nearest_boundary_type": obj.get("counts_by_nearest_boundary_type", {}),
            "counts_by_entry_like_vs_exit_like_alignment": obj.get("counts_by_entry_like_vs_exit_like_alignment", {}),
        }
    if family_name == "support_transition_bias" and _float(out.get("fraction_transition_aligned", 0.0)) >= 0.65:
        out["dominant_mechanism_label"] = "transition_aligned_support_crossing"
    elif out["fraction_with_actual_switch_timing_differences"] > max(out["fraction_with_candidate_history_differences"], out["fraction_with_effective_candidate_history_differences"]):
        out["dominant_mechanism_label"] = "switch_timing_dominant"
    else:
        out["dominant_mechanism_label"] = "candidate_mixed"
    return out


def _final_interpretation(family_name: str, generic: dict[str, Any], designed: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    burden_status = "observed" if generic["generic_burden_changing"] or designed["num_cases_with_burden_change"] > 0 else "none_observed"
    mechanism_distinct = "no"
    if family_name == "support_transition_bias":
        frac_t = _float(audit.get("fraction_transition_aligned", 0.0))
        frac_c = _float(audit.get("fraction_with_candidate_history_differences", 0.0))
        frac_e = _float(audit.get("fraction_with_effective_candidate_history_differences", 0.0))
        if frac_t >= 0.65 and frac_t >= frac_c + 0.20 and frac_t >= frac_e + 0.20:
            mechanism_distinct = "yes"
    elif family_name == "edge_band_bias":
        frac_s = _float(audit.get("fraction_with_actual_switch_timing_differences", 0.0))
        frac_c = _float(audit.get("fraction_with_candidate_history_differences", 0.0))
        frac_e = _float(audit.get("fraction_with_effective_candidate_history_differences", 0.0))
        if frac_s >= 0.65 and frac_s >= frac_c + 0.20 and frac_s >= frac_e + 0.20:
            mechanism_distinct = "yes"
    if burden_status == "observed" or mechanism_distinct == "yes":
        keep_status = "keep"
    elif designed["designed_relevance_label"] != "none" or generic["generic_interpretation_label"] != "near_inert_generic":
        keep_status = "keep"
    else:
        keep_status = "kill"
    sentences = {
        "transverse_skew": "First live Tier-2 mismatch family; designed cases show timing/pattern sensitivity without burden conversion.",
        "longitudinal_bias": "Strongest generic mismatch/severity family; designed cases broaden pattern relevance but still do not change burden.",
        "edge_band_bias": "Mechanism-matched support-boundary family; designed cases are pattern-relevant while generic rollouts stay near-inert.",
        "support_transition_bias": "Count-neutral family with a new transition-alignment signature concentrated on support entry/exit under the frozen controller.",
    }
    return {
        "burden_conversion_status": burden_status,
        "mechanism_distinct_status": mechanism_distinct,
        "keep_status": keep_status,
        "one_sentence_interpretation": sentences[family_name],
    }


def _ranking(families: dict[str, Any], key_fn) -> list[str]:
    return [name for name, _ in sorted(families.items(), key=lambda item: key_fn(item[1]), reverse=True)]


def run_tier2_cross_family_compare(out_dir: str | Path = "results/tier2_cross_family_compare") -> Path:
    out = make_results_dir("tier2_cross_family_compare") if out_dir == "results/tier2_cross_family_compare" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing_inputs: list[str] = []
    reduction = _load_json("results/tier2_reduction_witness/tier2_reduction_witness.json", missing_inputs)
    pointwise_ok = bool((reduction or {}).get("pointwise", {}).get("pointwise_nominal_match", False))
    rollout_ok = bool((reduction or {}).get("rollout", {}).get("rollout_match", False))
    reduction_anchor_status = "exact_reduction_preserved" if (pointwise_ok and rollout_ok) else ("missing" if reduction is None else "not_exact")

    families: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    for family_name, spec in FAMILY_SPECS.items():
        gap = _extract_gap(_load_json(spec["gap_path"], missing_inputs))
        generic = _extract_generic(_load_json(spec["compare_path"], missing_inputs), spec["compare_delta_key"])
        seed = _extract_seed(_load_json(spec["seed_path"], missing_inputs))
        designed = _extract_designed(_load_json(spec["scenarios_path"], missing_inputs))
        audit = _extract_audit(_load_json(spec["audit_path"], missing_inputs), family_name)
        final = _final_interpretation(family_name, generic, designed, audit)
        families[family_name] = {
            "identity": {
                "family_name": spec["family_name"],
                "mechanism_label": spec["mechanism_label"],
                "support_label": spec["support_label"],
                "status_label": spec["status_label"],
            },
            "gap": gap,
            "generic": generic,
            "seed_sweep": seed,
            "designed": designed,
            "audit": audit,
            "final_interpretation": final,
        }
        table_rows.append(
            {
                "family_name": family_name,
                "mechanism_label": spec["mechanism_label"],
                "status_label": spec["status_label"],
                "generic_mismatch_delta": generic["delta_mean_nominal_vs_plant_gap_over_rollout"],
                "generic_raw_jump_delta": generic["delta_max_raw_jump"],
                "designed_label": designed["designed_relevance_label"],
                "audit_mechanism": audit["dominant_mechanism_label"],
                "keep_status": final["keep_status"],
            }
        )

    rankings = {
        "generic_mismatch_strength_ranking": _ranking(families, lambda fam: abs(_float(fam["generic"]["delta_mean_nominal_vs_plant_gap_over_rollout"]))),
        "generic_severity_strength_ranking": _ranking(
            families,
            lambda fam: abs(_float(fam["generic"]["delta_max_raw_jump"])) + abs(_float(fam["generic"]["delta_max_applied_jump"])) + abs(_float(fam["generic"]["delta_tracking_error_mean"])),
        ),
        "designed_case_pattern_relevance_ranking": _ranking(
            families,
            lambda fam: (
                _float(fam["designed"]["num_switch_pattern_differs"])
                + _float(fam["designed"]["num_candidate_pattern_differs"])
                + _float(fam["designed"]["num_effective_candidate_pattern_differs"])
            )
            / max(_float(fam["designed"]["num_cases"]), 1.0),
        ),
        "mechanism_specificity_ranking": _ranking(
            families,
            lambda fam: _float(fam["audit"].get("fraction_transition_aligned", 0.0))
            if "fraction_transition_aligned" in fam["audit"]
            else _float(fam["audit"].get("fraction_with_actual_switch_timing_differences", 0.0))
            - max(
                _float(fam["audit"].get("fraction_with_candidate_history_differences", 0.0)),
                _float(fam["audit"].get("fraction_with_effective_candidate_history_differences", 0.0)),
            ),
        ),
        "burden_conversion_families": [name for name, fam in families.items() if fam["generic"]["generic_burden_changing"] or fam["designed"]["num_cases_with_burden_change"] > 0],
        "count_neutral_but_mechanism_distinct_families": [
            name
            for name, fam in families.items()
            if fam["final_interpretation"]["burden_conversion_status"] == "none_observed"
            and fam["final_interpretation"]["mechanism_distinct_status"] == "yes"
        ],
    }

    summary = {
        "phase_identity": "tier2_cross_family_compare",
        "reduction_anchor_status": reduction_anchor_status,
        "missing_inputs": missing_inputs,
        "families": families,
        "cross_family_rankings": rankings,
        "project_safe_conclusion": "Across four Tier-2 residual families, structured plant-side unmatchedness reproducibly produces nominal-vs-plant mismatch and mechanism-dependent timing/pattern perturbation before any aggregate burden conversion appears under the frozen controller.",
        "nonclaims": [
            "No aggregate burden-conversion claim is supported for Tier-2 under the frozen controller.",
            "Tier-2 does not yet justify controller redesign.",
            "Tier-2 does not constitute external physical validation.",
            "Four residual families do not exhaust all possible unmatchedness mechanisms.",
        ],
    }
    dump_json(out / "tier2_cross_family_compare.json", summary)
    dump_json(out / "tier2_cross_family_table_rows.json", table_rows)
    return out / "tier2_cross_family_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the Tier-2 cross-family comparison package.")
    parser.add_argument("--out", default="results/tier2_cross_family_compare")
    args = parser.parse_args()
    out = run_tier2_cross_family_compare(args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

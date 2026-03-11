"""Build the repo-local manuscript figure package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments.tier1_selector_mechanism_audit import run_tier1_selector_mechanism_audit


FIG_ROOT = Path("figures")
MANIFEST_PATH = FIG_ROOT / "manifest.json"


def _json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def _csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ensure_tree() -> None:
    for subdir in [
        FIG_ROOT / "matched_geometry",
        FIG_ROOT / "unmatchedness",
        FIG_ROOT / "protocol",
        FIG_ROOT / "appendix",
        FIG_ROOT / "_staging",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
    for keep in [FIG_ROOT / "protocol/.gitkeep", FIG_ROOT / "appendix/.gitkeep", FIG_ROOT / "_staging/.gitkeep"]:
        keep.touch(exist_ok=True)


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _manifest_entry(relative_path: str, *, status: str, purpose: str, source_artifacts: list[str], section: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "status": status,
        "purpose": purpose,
        "source_artifacts": source_artifacts,
        "source_script": "src/experiments/build_manuscript_figures.py",
        "manuscript_section_tag": section,
    }


def _build_operator_gap_identity(out_path: Path) -> dict[str, Any]:
    csv_path = "results/gate3_operator_mismatch_tier1/operator_gap_table_tier1.csv"
    rows = _csv_rows(csv_path)
    e_gap = np.asarray([float(row["E_gap"]) for row in rows], dtype=float)
    u_gap = np.asarray([float(row["u_gap"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    if e_gap.size and (np.max(np.abs(e_gap)) > 0.0 or np.max(np.abs(u_gap)) > 0.0):
        lim = float(max(np.max(np.abs(e_gap)), np.max(np.abs(u_gap))))
        ax.plot([0.0, lim], [0.0, lim], "k--", linewidth=1.0)
        ax.plot(e_gap, u_gap, ".", alpha=0.8)
        ax.set_xlim(0.0, lim * 1.05)
        ax.set_ylim(0.0, lim * 1.05)
    else:
        ax.plot([0.0], [0.0], "o")
        ax.annotate("all sampled rows collapse at the origin", xy=(0.0, 0.0), xytext=(0.05, 0.8), textcoords="axes fraction")
        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel(r"$||E_\pi-E_{\pi'}||$")
    ax.set_ylabel(r"$||u_\pi-u_{\pi'}||$")
    ax.set_title("Matched Reference Identity")
    _save(fig, out_path)
    return _manifest_entry(
        "figures/matched_geometry/operator_gap_identity.png",
        status="built",
        purpose="Exact same-snapshot identity for the matched reference evaluator.",
        source_artifacts=[csv_path],
        section="matched_geometry",
    )


def _build_approximate_evaluator_gap_hist(out_path: Path) -> dict[str, Any]:
    csv_path = "results/gate2_tier1_order_sensitivity/tier1_order_sensitivity_table.csv"
    rows = _csv_rows(csv_path)
    mode_to_vals: dict[str, list[float]] = {}
    for row in rows:
        mode = str(row["approx_mode"])
        mode_to_vals.setdefault(mode, []).append(float(row["E_gap_approx"]))

    modes = [mode for mode in ["upstream_truncated", "local_window"] if mode in mode_to_vals]
    if not modes:
        raise RuntimeError("No approximate evaluator modes found in Tier-1 order-sensitivity table")

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 20
    for mode in modes:
        ax.hist(mode_to_vals[mode], bins=bins, alpha=0.5, label=mode.replace("_", " "))
    ax.set_xlabel("approximate evaluator gap magnitude")
    ax.set_ylabel("count")
    ax.set_title("Approximate Evaluator Gaps")
    ax.legend()
    _save(fig, out_path)
    return _manifest_entry(
        "figures/matched_geometry/approximate_evaluator_gap_hist.png",
        status="built",
        purpose="Compare approximate evaluator gap signatures against the matched reference.",
        source_artifacts=[csv_path],
        section="matched_geometry",
    )


def _build_generic_compare(out_path: Path) -> dict[str, Any]:
    compare_path = "results/tier0_tier1_compare/tier0_tier1_compare.json"
    sweep_path = "results/tier0_tier1_seed_sweep/tier0_tier1_seed_sweep.json"
    compare_obj = _json(compare_path)
    sweep_obj = _json(sweep_path)
    delta = compare_obj["delta_tier1_minus_tier0"]
    metrics = [
        ("max_raw_jump", "raw jump"),
        ("max_applied_jump", "applied jump"),
        ("tracking_error_mean", "tracking mean"),
        ("switch_count", "switch count"),
        ("transition_start_count", "transition starts"),
        ("blend_active_steps", "blend-active steps"),
    ]
    vals = [float(delta[key]) for key, _ in metrics]
    mins = [float(sweep_obj["delta_summary"][f"delta_{key}"]["min"]) for key, _ in metrics]
    maxs = [float(sweep_obj["delta_summary"][f"delta_{key}"]["max"]) for key, _ in metrics]
    lower = np.asarray([v - mn for v, mn in zip(vals, mins)], dtype=float)
    upper = np.asarray([mx - v for v, mx in zip(vals, maxs)], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    ax.bar(x, vals, yerr=np.vstack([lower, upper]), capsize=4)
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=18, ha="right")
    ax.set_ylabel("Tier-1 minus Tier-0")
    ax.set_title("Generic Comparison")
    _save(fig, out_path)
    return _manifest_entry(
        "figures/matched_geometry/generic_compare.png",
        status="built",
        purpose="Generic matched-geometry comparison showing severity changes before burden changes.",
        source_artifacts=[compare_path, sweep_path],
        section="matched_geometry",
    )


def _build_designed_case_compare(out_path: Path) -> dict[str, Any]:
    json_path = "results/tier1_extended_discriminative_scenarios/tier1_extended_discriminative_scenarios.json"
    obj = _json(json_path)
    ranked = []
    for case_name, case in obj["cases"].items():
        delta = case["delta_tier1_minus_tier0"]
        score = abs(float(delta["delta_switch_count"])) + abs(float(delta["delta_blend_active_steps"])) + abs(float(delta["delta_max_applied_jump"]))
        ranked.append((score, case_name, delta))
    ranked = sorted(ranked, reverse=True)[:4]
    names = [name for _, name, _ in ranked]
    switch_vals = [float(delta["delta_switch_count"]) for _, _, delta in ranked]
    blend_vals = [float(delta["delta_blend_active_steps"]) for _, _, delta in ranked]
    jump_vals = [float(delta["delta_max_applied_jump"]) for _, _, delta in ranked]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, switch_vals, width=width, label="switch count")
    ax.bar(x, blend_vals, width=width, label="blend-active steps")
    ax.bar(x + width, jump_vals, width=width, label="max applied jump")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_title("Designed Cases")
    ax.legend(fontsize=8)
    _save(fig, out_path)
    return _manifest_entry(
        "figures/matched_geometry/designed_case_compare.png",
        status="built",
        purpose="Representative matched-geometry designed cases showing controller-relevant deltas.",
        source_artifacts=[json_path],
        section="matched_geometry",
    )


def _build_selector_compare(out_path: Path) -> dict[str, Any]:
    edge_path = "results/edge_boundary_selector_compare/edge_boundary_selector_compare.json"
    extended_path = "results/tier1_extended_selector_compare/tier1_extended_selector_compare.json"
    edge_obj = _json(edge_path)
    ext_obj = _json(extended_path)

    selected_cases = [
        ("edge_radius_just_inside", edge_obj["cases"]["edge_radius_just_inside"]["families"]["tier1"]),
        ("edge_radius_just_outside", edge_obj["cases"]["edge_radius_just_outside"]["families"]["tier1"]),
        ("mixed_boundary_inside_split", ext_obj["cases"]["mixed_boundary_inside_split"]["families"]["tier1"]),
        ("mixed_boundary_outside_split", ext_obj["cases"]["mixed_boundary_outside_split"]["families"]["tier1"]),
    ]
    modes = ["legacy", "shadow_lexicographic", "active_lexicographic"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    x = np.arange(len(selected_cases))
    width = 0.25
    for idx, mode in enumerate(modes):
        switch_vals = [float(case[mode]["switch_count"]) for _, case in selected_cases]
        blend_vals = [float(case[mode]["blend_active_steps"]) for _, case in selected_cases]
        axes[0].bar(x + (idx - 1) * width, switch_vals, width=width, label=mode)
        axes[1].bar(x + (idx - 1) * width, blend_vals, width=width, label=mode)
    labels = [name for name, _ in selected_cases]
    for ax, title in zip(axes, ["switch count", "blend-active steps"]):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Selector Comparison")
    _save(fig, out_path)
    return _manifest_entry(
        "figures/matched_geometry/selector_compare.png",
        status="built",
        purpose="Representative Tier-1 selector comparison across legacy, shadow, and active modes.",
        source_artifacts=[edge_path, extended_path],
        section="matched_geometry",
    )


def _ensure_selector_audit() -> str:
    json_path = "results/tier1_selector_mechanism_audit/tier1_selector_mechanism_audit.json"
    if not Path(json_path).exists():
        run_tier1_selector_mechanism_audit()
    return json_path


def _build_selector_mechanism_audit(out_path: Path) -> dict[str, Any]:
    json_path = _ensure_selector_audit()
    obj = _json(json_path)
    counts = obj.get("reason_counts_global", {})
    labels = [
        "hold_current_no_switch_tie",
        "predicted_gap_advantage",
        "admissibility_rescue",
        "conditioning",
    ]
    values = [
        int(counts.get("hold_current_no_switch_tie", 0)),
        int(counts.get("min_predicted_gap", 0)) + int(counts.get("predicted_gap_advantage", 0)),
        int(counts.get("inadmissible_live_candidate_avoided", 0)) + int(counts.get("admissibility_rescue", 0)),
        int(counts.get("conditioning_tie_break", 0)) + int(counts.get("conditioning", 0)),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Selector Mechanism Audit")
    for bar, value in zip(bars, values):
        ax.annotate(str(value), xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center")
    _save(fig, out_path)
    return _manifest_entry(
        "figures/matched_geometry/selector_mechanism_audit.png",
        status="built",
        purpose="Selector mechanism reason counts with explicit zero bars for non-dominant categories.",
        source_artifacts=[json_path],
        section="matched_geometry",
    )


def _build_generic_compare_all_families(out_path: Path) -> dict[str, Any]:
    json_path = "results/tier2_cross_family_compare/tier2_cross_family_generic_plot_data.json"
    obj = _json(json_path)
    families = obj["families"]
    labels = [entry["family_name"] for entry in families]
    metrics = [
        ("generic_mismatch_delta", "mismatch"),
        ("max_raw_jump_delta", "raw jump"),
        ("max_applied_jump_delta", "applied jump"),
        ("tracking_error_mean_delta", "tracking mean"),
        ("switch_count_delta", "switch count"),
        ("blend_active_steps_delta", "blend steps"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(labels))
    width = 0.12
    for idx, (key, label) in enumerate(metrics):
        vals = [float(entry[key]) for entry in families]
        ax.bar(x + (idx - (len(metrics) - 1) / 2.0) * width, vals, width=width, label=label)
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title("Generic Unmatchedness Comparison")
    ax.legend(fontsize=8, ncol=3)
    _save(fig, out_path)
    return _manifest_entry(
        "figures/unmatchedness/generic_compare_all_families.png",
        status="built",
        purpose="Cross-family generic unmatchedness comparison across families A/B/C/D.",
        source_artifacts=[json_path],
        section="unmatchedness",
    )


def _build_designed_compare_all_families(out_path: Path) -> dict[str, Any]:
    json_path = "results/tier2_cross_family_compare/tier2_cross_family_designed_plot_data.json"
    obj = _json(json_path)
    families = obj["families"]
    labels = [entry["family_name"] for entry in families]
    metrics = [
        ("switch_pattern_fraction", "switch pattern"),
        ("candidate_pattern_fraction", "candidate"),
        ("effective_candidate_pattern_fraction", "effective candidate"),
        ("burden_change_case_count", "burden-change cases"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    width = 0.18
    for idx, (key, label) in enumerate(metrics):
        vals = [float(entry[key]) for entry in families]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title("Designed Unmatchedness Comparison")
    ax.legend(fontsize=8)
    _save(fig, out_path)
    return _manifest_entry(
        "figures/unmatchedness/designed_compare_all_families.png",
        status="built",
        purpose="Cross-family designed-case pattern relevance comparison.",
        source_artifacts=[json_path],
        section="unmatchedness",
    )


def _build_pattern_audit_c_vs_d(out_path: Path) -> dict[str, Any]:
    json_path = "results/tier2_cross_family_compare/tier2_cross_family_audit_plot_data.json"
    obj = _json(json_path)
    by_name = {entry["family_name"]: entry for entry in obj["families"]}
    c = by_name["edge_band_bias"]
    d = by_name["support_transition_bias"]
    metrics = [
        ("fraction_near_switch_windows", "near switch"),
        ("fraction_with_actual_switch_timing_differences", "switch timing"),
        ("fraction_with_candidate_history_differences", "candidate"),
        ("fraction_with_effective_candidate_history_differences", "effective"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(metrics))
    width = 0.35
    axes[0].bar(x - width / 2.0, [float(c[key]) for key, _ in metrics], width=width, label="edge_band_bias")
    axes[0].bar(x + width / 2.0, [float(d[key]) for key, _ in metrics], width=width, label="support_transition_bias")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label for _, label in metrics], rotation=18, ha="right")
    axes[0].set_title("C vs D common audit metrics")
    axes[0].legend(fontsize=8)

    axes[1].bar([0], [float(d["fraction_transition_aligned"])], width=0.5)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["transition aligned"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Family D transition alignment")
    _save(fig, out_path)
    return _manifest_entry(
        "figures/unmatchedness/pattern_audit_C_vs_D.png",
        status="built",
        purpose="Mechanism comparison between edge-band bias and support-transition bias.",
        source_artifacts=[json_path],
        section="cross_family",
    )


def _build_familyd_transition_detail(out_path: Path) -> dict[str, Any]:
    json_path = "results/tier2_fourth_residual_pattern_audit/tier2_fourth_residual_pattern_audit.json"
    obj = _json(json_path)
    boundary = obj["counts_by_nearest_boundary_type"]
    align = obj["counts_by_entry_like_vs_exit_like_alignment"]
    transition_fraction = float(obj["fraction_transition_aligned"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(list(boundary.keys()), [int(v) for v in boundary.values()])
    axes[0].set_title("nearest boundary type")

    align_labels = ["entry_like", "exit_like", "transition_aligned_fraction"]
    align_values = [int(align.get("entry_like", 0)), int(align.get("exit_like", 0)), transition_fraction]
    axes[1].bar(align_labels, align_values)
    axes[1].set_title("transition detail")
    axes[1].tick_params(axis="x", rotation=18)
    _save(fig, out_path)
    return _manifest_entry(
        "figures/unmatchedness/familyD_transition_detail.png",
        status="built",
        purpose="Detailed support-transition mechanism view for family D.",
        source_artifacts=[json_path],
        section="unmatchedness",
    )


def run_build_manuscript_figures() -> Path:
    _ensure_tree()
    manifest: list[dict[str, Any]] = []
    built: list[str] = []
    reserved: list[str] = []
    missing: list[dict[str, Any]] = []

    builders = [
        ("figures/matched_geometry/operator_gap_identity.png", _build_operator_gap_identity),
        ("figures/matched_geometry/approximate_evaluator_gap_hist.png", _build_approximate_evaluator_gap_hist),
        ("figures/matched_geometry/generic_compare.png", _build_generic_compare),
        ("figures/matched_geometry/designed_case_compare.png", _build_designed_case_compare),
        ("figures/matched_geometry/selector_compare.png", _build_selector_compare),
        ("figures/matched_geometry/selector_mechanism_audit.png", _build_selector_mechanism_audit),
        ("figures/unmatchedness/generic_compare_all_families.png", _build_generic_compare_all_families),
        ("figures/unmatchedness/designed_compare_all_families.png", _build_designed_compare_all_families),
        ("figures/unmatchedness/pattern_audit_C_vs_D.png", _build_pattern_audit_c_vs_d),
        ("figures/unmatchedness/familyD_transition_detail.png", _build_familyd_transition_detail),
    ]

    for rel_path, builder in builders:
        try:
            entry = builder(Path(rel_path))
            manifest.append(entry)
            built.append(rel_path)
        except FileNotFoundError as exc:
            manifest.append(
                _manifest_entry(
                    rel_path,
                    status="reserved",
                    purpose="Reserved because one or more source artifacts are missing.",
                    source_artifacts=[str(exc)],
                    section="matched_geometry" if "matched_geometry" in rel_path else "unmatchedness",
                )
            )
            reserved.append(rel_path)
            missing.append({"figure": rel_path, "missing_source": str(exc)})
        except Exception as exc:
            manifest.append(
                _manifest_entry(
                    rel_path,
                    status="reserved",
                    purpose=f"Reserved because build failed: {exc}",
                    source_artifacts=[],
                    section="matched_geometry" if "matched_geometry" in rel_path else "unmatchedness",
                )
            )
            reserved.append(rel_path)
            missing.append({"figure": rel_path, "build_error": str(exc)})

    MANIFEST_PATH.write_text(json.dumps({"figures": manifest}, indent=2, sort_keys=True), encoding="utf-8")

    print("Built figures:")
    for path in built:
        print(f"  - {path}")
    print("Reserved figures:")
    for path in reserved:
        print(f"  - {path}")
    if missing:
        print("Missing source artifacts:")
        for item in missing:
            print(f"  - {item}")
    return MANIFEST_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the manuscript figure package.")
    parser.parse_args()
    out = run_build_manuscript_figures()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

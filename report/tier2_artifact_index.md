# Tier-2 Artifact Index

## Purpose

This is the canonical human-readable map of the Tier-2 evidence ladder.
It is meant to keep the completed Tier-2 branch discoverable and stable.

## Reading order

1. `results/tier2_reduction_witness/tier2_reduction_witness.json`
2. Family gap witnesses
3. Family generic compares
4. Family seed sweeps
5. Family designed-scenario summaries
6. Family pattern audits
7. `results/tier2_cross_family_compare/tier2_cross_family_compare.json`
8. `report/tier2_controller_research_conclusion.md`
9. `report/tier2_cross_family_interpretation_note.md`

## Reduction anchor

### `results/tier2_reduction_witness/tier2_reduction_witness.json`
- Category: `reduction_anchor`
- Load-bearing: `yes`
- Purpose: prove residual-disabled Tier-2 reduces exactly to Tier-1 nominal.
- Conclusion: Tier-2 inherits the matched nominal branch exactly before any residual is activated.

## Family A: `transverse_skew`

### `results/tier2_nominal_vs_plant_gap/tier2_nominal_vs_plant_gap_summary.json`
- Category: `family_gap_witness`
- Load-bearing: `yes`
- Purpose: establish first live Tier-2 mismatch object.
- Conclusion: plant-side skew produces measurable nominal-vs-plant mismatch.

### `results/tier2_residual_compare/tier2_residual_compare.json`
- Category: `generic_compare`
- Load-bearing: `yes`
- Purpose: generic frozen-rollout comparison.
- Conclusion: mismatch/severity changes appear before burden change.

### `results/tier2_residual_seed_sweep/tier2_residual_seed_sweep.json`
- Category: `seed_robustness`
- Load-bearing: `yes`
- Purpose: paired-seed robustness check.
- Conclusion: generic burden remains unchanged across seeds.

### `results/tier2_mismatch_sensitive_scenarios/tier2_mismatch_sensitive_scenarios.json`
- Category: `designed_case_relevance`
- Load-bearing: `yes`
- Purpose: reveal designed-case timing/pattern sensitivity.
- Conclusion: family A is pattern-relevant on mismatch-sensitive cases.

### `results/tier2_pattern_change_audit/tier2_pattern_change_audit.json`
- Category: `mechanism_audit`
- Load-bearing: `yes`
- Purpose: explain how family A changes hybrid behavior.
- Conclusion: the dominant mechanism is switch-timing perturbation near marginal switch windows.

## Family B: `longitudinal_bias`

### `results/tier2_second_residual_gap/tier2_second_residual_gap_summary.json`
- Category: `family_gap_witness`
- Load-bearing: `yes`
- Purpose: show the stronger longitudinal-window residual is live and distinct.
- Conclusion: family B is a stronger generic mismatch candidate than family A.

### `results/tier2_second_residual_compare/tier2_second_residual_compare.json`
- Category: `generic_compare`
- Load-bearing: `yes`
- Purpose: generic frozen-rollout comparison.
- Conclusion: family B is the strongest generic mismatch/severity family without burden conversion.

### `results/tier2_second_residual_seed_sweep/tier2_second_residual_seed_sweep.json`
- Category: `seed_robustness`
- Load-bearing: `yes`
- Purpose: paired-seed robustness check.
- Conclusion: the stronger generic perturbation remains count-neutral across seeds.

### `results/tier2_second_residual_scenarios/tier2_second_residual_scenarios.json`
- Category: `designed_case_relevance`
- Load-bearing: `yes`
- Purpose: mechanism-matched designed scenarios for family B.
- Conclusion: family B broadens designed-case pattern relevance.

### `results/tier2_second_residual_pattern_audit/tier2_second_residual_pattern_audit.json`
- Category: `mechanism_audit`
- Load-bearing: `yes`
- Purpose: audit longitudinal-band mechanism.
- Conclusion: family B remains timing/pattern sensitive near low-margin switching regions.

## Family C: `edge_band_bias`

### `results/tier2_third_residual_gap/tier2_third_residual_gap_summary.json`
- Category: `family_gap_witness`
- Load-bearing: `yes`
- Purpose: show static support-boundary shell localization is live.
- Conclusion: family C is shell-localized and order-sensitive through mismatch.

### `results/tier2_third_residual_compare/tier2_third_residual_compare.json`
- Category: `generic_compare`
- Load-bearing: `yes`
- Purpose: generic frozen-rollout comparison.
- Conclusion: family C is near-inert generically at current default parameters.

### `results/tier2_third_residual_seed_sweep/tier2_third_residual_seed_sweep.json`
- Category: `seed_robustness`
- Load-bearing: `yes`
- Purpose: paired-seed robustness check.
- Conclusion: generic near-inertness is stable across seeds.

### `results/tier2_third_residual_scenarios/tier2_third_residual_scenarios.json`
- Category: `designed_case_relevance`
- Load-bearing: `yes`
- Purpose: edge-band-sensitive designed cases.
- Conclusion: family C is clearly pattern-relevant on mechanism-matched cases.

### `results/tier2_third_residual_pattern_audit/tier2_third_residual_pattern_audit.json`
- Category: `mechanism_audit`
- Load-bearing: `yes`
- Purpose: audit edge-band-localized divergence.
- Conclusion: family C is count-neutral but mechanism-distinct through support-boundary timing effects.

## Family D: `support_transition_bias`

### `results/tier2_fourth_residual_gap/tier2_fourth_residual_gap_summary.json`
- Category: `family_gap_witness`
- Load-bearing: `yes`
- Purpose: show support-transition bias is live and shell-localized.
- Conclusion: family D is a nontrivial transition-sensitive residual.

### `results/tier2_fourth_residual_compare/tier2_fourth_residual_compare.json`
- Category: `generic_compare`
- Load-bearing: `yes`
- Purpose: generic frozen-rollout comparison.
- Conclusion: family D remains count-neutral generically.

### `results/tier2_fourth_residual_seed_sweep/tier2_fourth_residual_seed_sweep.json`
- Category: `seed_robustness`
- Load-bearing: `yes`
- Purpose: paired-seed robustness check.
- Conclusion: generic count-neutrality persists across seeds while mismatch remains nonzero.

### `results/tier2_fourth_residual_scenarios/tier2_fourth_residual_scenarios.json`
- Category: `designed_case_relevance`
- Load-bearing: `yes`
- Purpose: support-transition designed cases.
- Conclusion: family D is designed-case pattern-relevant without burden conversion.

### `results/tier2_fourth_residual_pattern_audit/tier2_fourth_residual_pattern_audit.json`
- Category: `mechanism_audit`
- Load-bearing: `yes`
- Purpose: audit transition-alignment mechanism.
- Conclusion: family D is count-neutral but mechanism-distinct through support entry/exit alignment.

## Cross-family synthesis

### `results/tier2_cross_family_compare/tier2_cross_family_compare.json`
- Category: `cross_family_synthesis`
- Load-bearing: `yes`
- Purpose: canonical machine-readable Tier-2 comparison across families A/B/C/D.
- Conclusion: Tier-2 is a completed four-family ladder with no burden conversion under the frozen controller.

### `results/tier2_cross_family_compare/tier2_cross_family_table_rows.json`
- Category: `cross_family_synthesis`
- Load-bearing: `no`
- Purpose: simplified table-ready rows.
- Conclusion: exposes one-row-per-family synthesis data.

### `results/tier2_cross_family_compare/tier2_cross_family_generic_plot_data.json`
- Category: `cross_family_synthesis`
- Load-bearing: `no`
- Purpose: figure-ready generic comparison data.
- Conclusion: supports generic mismatch/severity ranking across families.

### `results/tier2_cross_family_compare/tier2_cross_family_designed_plot_data.json`
- Category: `cross_family_synthesis`
- Load-bearing: `no`
- Purpose: figure-ready designed-case relevance data.
- Conclusion: supports cross-family designed-case pattern relevance ranking.

### `results/tier2_cross_family_compare/tier2_cross_family_audit_plot_data.json`
- Category: `cross_family_synthesis`
- Load-bearing: `no`
- Purpose: figure-ready audit-mechanism data.
- Conclusion: supports mechanism-specificity comparisons across families.

## Project-safe conclusion artifacts

### `report/tier2_controller_research_conclusion.md`
- Category: `project_safe_conclusion`
- Load-bearing: `yes`
- Purpose: short statement of what Tier-2 supports and does not support.
- Conclusion: negative burden-conversion results are structured findings, not failure.

### `report/tier2_cross_family_interpretation_note.md`
- Category: `project_safe_conclusion`
- Load-bearing: `yes`
- Purpose: project-facing interpretation of the four-family ladder.
- Conclusion: family D closes the residual-family search by default.

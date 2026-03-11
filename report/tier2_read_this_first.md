# Tier-2 Read This First

If you continue this project, read in this order before touching runtime code:

1. Distributed Flatness Full Handoff Report
2. [tier2_controller_research_conclusion.md](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/report/tier2_controller_research_conclusion.md)
3. [tier2_cross_family_interpretation_note.md](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/report/tier2_cross_family_interpretation_note.md)
4. [tier2_cross_family_compare.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/tier2_cross_family_compare/tier2_cross_family_compare.json)
5. [tier2_fourth_residual_pattern_audit.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/tier2_fourth_residual_pattern_audit/tier2_fourth_residual_pattern_audit.json)
6. Only then read source files if implementation work is actually needed

## Current position

- Tier-2 is complete as a four-family ladder.
- Do not default to a fifth residual family.
- Do not default to controller redesign.
- Higher-fidelity / external model transfer is later work, after the current story is stabilized.

## Frozen interfaces

Unless a future phase explicitly targets them, keep these semantically frozen:

- selector logic
- runtime monitor logic
- candidate richness
- blend law
- controller-side evaluator semantics
- high-level closed-loop structure

## Default next moves

Near-term:
- manuscript consolidation
- artifact maintenance
- scope discipline

Later:
- higher-fidelity / external plant-model transfer, if the project decides to reopen the science branch

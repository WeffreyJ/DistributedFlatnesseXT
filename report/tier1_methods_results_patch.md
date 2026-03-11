# Tier-1 Methods And Results Patch

## Purpose

This note organizes the existing Tier-1 evidence into a manuscript-facing methods/results structure.
It is a patch/outline artifact, not final prose.

The goal is to make the Tier-1 section read as a disciplined extension of the Tier-0 formal package rather than as an unstructured pile of experiments.

## Core narrative

The Tier-1 story should be presented in six steps.

1. Define the Tier-1 matched kernel and evaluator family.
2. Verify that the matched `full` Tier-1 reference is live and exact.
3. Show that approximate Tier-1 evaluator modes are nontrivially order-sensitive relative to `full`.
4. Show that generic rollouts expose mainly jump-severity effects.
5. Show that designed edge-boundary cases reveal the first geometry-induced controller consequences.
6. Rank selector and monitor consequences on that edge-boundary family.

That is now the cleanest Tier-1 results arc.

## Recommended methods structure

### Methods subsection: Tier-1 matched geometry-aware extension

State:

- the Tier-1 plant retains the matched second-order form
- the controller still reconstructs through the evaluator object
- `full` is the matched reference evaluator
- `upstream_truncated` and `local_window` are approximate order-sensitive modes

This subsection should point back to [tier1_matched_reconstruction_spec.md](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/report/tier1_matched_reconstruction_spec.md) for internal design provenance.

### Methods subsection: Tier-1 diagnostic protocol

State the three diagnostic layers explicitly:

1. matched verification
2. order-sensitivity relative to `full`
3. frozen-controller comparison under shared rollout conditions

Then state the designed-scenario methodology:

- tie-proximal deterministic cases
- edge-boundary deterministic cases

### Methods subsection: controller consequence protocol

State that controller comparisons were performed only after the Tier-1 approximation layer had shown nontrivial order sensitivity.

Separate:

- selector consequence study on edge-boundary cases
- monitor consequence study on the same cases with selector fixed to `active_lexicographic`

This ordering matters and should be made explicit.

## Recommended results structure

### Result 1: matched Tier-1 reference is exact but hybrid-trivial

Use the current C.2a artifact:

- [operator_gap_summary_tier1.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/gate3_operator_mismatch_tier1/operator_gap_summary_tier1.json)

Main statement:

- same-snapshot identity holds exactly at dataset scale
- `full` is permutation-invariant on the collected matched dataset
- therefore matched verification alone does not create hybrid discrimination

That motivates the next diagnostic.

### Result 2: approximate Tier-1 evaluators are nontrivially order-sensitive

Use the C.2b artifact:

- [tier1_order_sensitivity_summary.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/gate2_tier1_order_sensitivity/tier1_order_sensitivity_summary.json)

Main statements:

- `full` remains the zero-gap anchor
- `upstream_truncated` is sparse but larger-amplitude
- `local_window` is broader but smaller-amplitude

This is the first nontrivial Tier-1 hybrid result.

### Result 3: generic Tier-0 vs Tier-1 rollouts show mostly severity effects

Use the generic frozen-controller artifacts:

- [tier0_tier1_compare.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/tier0_tier1_compare/tier0_tier1_compare.json)
- [tier0_tier1_seed_sweep.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/tier0_tier1_seed_sweep/tier0_tier1_seed_sweep.json)

Main statements:

- Tier-1 robustly changes jump severity
- Tier-1 slightly changes mean tracking error
- Tier-1 does not change switch/blend/block counts on the generic rollout family

Interpretation:

- the generic family is not the right probe for discrete Tier-1 controller consequences

### Result 4: designed edge-boundary cases reveal the first geometry-induced Tier-1 consequence

Use:

- [tier0_tier1_discriminative_scenarios.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/tier0_tier1_discriminative_scenarios/tier0_tier1_discriminative_scenarios.json)

Main statements:

- tie-proximal cases still do not unlock discrete Tier-1 differences
- edge-radius boundary cases do:
  - `active_edge_pattern_differs = true`
- these cases materially change jump severity and slightly change mean tracking

Interpretation:

- the edge-boundary family is the first correct Tier-1 controller-relevant scenario family

### Result 5: selector activation is the primary useful refinement on the edge family

Use:

- [edge_boundary_selector_compare.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/edge_boundary_selector_compare/edge_boundary_selector_compare.json)

Main statements:

- `shadow_lexicographic` is behaviorally identical to legacy on the tested edge cases
- `active_lexicographic` becomes substantially more meaningful for Tier-1 than it was on the generic rollout family
- the decisive case is `edge_radius_just_outside`, Tier-1:
  - legacy: `switch_count = 9`, `blend_active_steps = 18`
  - active selector: `switch_count = 0`, `blend_active_steps = 0`

Interpretation:

- selector activation is the main useful controller-layer refinement on the current Tier-1 edge family

### Result 6: monitor activation remains secondary on the edge family

Use:

- [edge_boundary_monitor_compare.json](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/results/edge_boundary_monitor_compare/edge_boundary_monitor_compare.json)

Main statements:

- `edge_radius_just_outside`, Tier-1:
  - monitor adds nothing beyond the active selector
- `edge_radius_just_inside`, Tier-1:
  - monitor becomes more interpretable than on the generic baseline family
  - but it does not reduce switch count further
  - it trades modest jump reduction for tracking cost
  - blocked intervals are still dominated by `large_predicted_gap`

Interpretation:

- monitor activation is currently secondary and not yet a recommended default on the Tier-1 edge family

## Recommended final interpretation subsection

The final Tier-1 interpretation subsection should state the following clearly.

### Recommended current claim

Tier-1 is a credible matched geometry-aware extension whose controller consequences are scenario-dependent.

### Supported controller ranking

For the current Tier-1 family:

- `active_lexicographic` is the strongest controller-layer refinement on the edge-boundary family
- monitor activation is more interpretable there than on generic rollouts but remains secondary

### Important limitation

These conclusions are currently scenario-conditioned, not universal.
Generic rollout families still show mostly severity differences rather than discrete event-count changes.

## Suggested manuscript section order

Insert the Tier-1 material into the paper in this order.

1. Tier-1 matched kernel construction
2. Matched Tier-1 verification
3. Tier-1 order sensitivity relative to `full`
4. Generic Tier-0 vs Tier-1 frozen-controller comparison
5. Designed edge-boundary scenarios
6. Selector comparison on edge family
7. Monitor comparison on edge family
8. Controller conclusion: selector primary, monitor secondary

## Suggested figure/table mapping

### Table A: Tier-1 diagnostic ladder

Rows:

- matched verification
- order sensitivity
- generic frozen comparison
- designed edge-boundary comparison
- selector comparison
- monitor comparison

Columns:

- key question
- primary artifact
- main outcome

### Figure B: generic vs designed Tier-1 consequences

Contrast:

- generic rollout jump-amplitude effect
- edge-boundary active-edge-pattern effect

### Figure C: selector vs monitor on edge family

Show:

- `edge_radius_just_inside`
- `edge_radius_just_outside`

Metrics:

- switch count
- blend burden
- max applied jump
- tracking error mean

## Recommended current default stance section

Add a short explicit stance paragraph near the end of the results/discussion section:

- Tier-0 baseline remains the main regression anchor.
- Tier-1 is now a validated extension with scenario-dependent controller consequences.
- On the current evidence, `active_lexicographic` is the preferred controller-layer refinement for the Tier-1 edge-boundary family.
- Active monitor is not yet recommended as a default.

## What should remain outside the current paper claim

Do not overclaim the following yet:

- that Tier-1 generically changes switching counts across broad rollout families
- that monitor activation is broadly beneficial on Tier-1
- that the current monitor threshold logic is final
- that the edge-boundary family exhausts the Tier-1 scenario space

Those are reasonable next-step research directions, not current conclusions.

## Short summary

If a one-paragraph summary is needed for an abstracted results bridge, use this:

> The Tier-1 matched geometry-aware extension is exact under the full evaluator and nontrivially order-sensitive under approximate evaluators. Generic frozen-controller comparisons show that Tier-1 robustly changes transition severity but not discrete event counts. Designed edge-boundary scenarios reveal the first geometry-induced controller consequences: active lexicographic selection becomes substantially more effective for Tier-1 than on generic rollouts, whereas runtime monitoring becomes more interpretable but remains secondary and is not yet a recommended default.

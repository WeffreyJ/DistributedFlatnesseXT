# Tier-1 Controller Conclusion

## Status

This note is a synthesis artifact for the current Tier-1 path in `hybrid_flatness_ext`.
It does not change code behavior.
It states the strongest controller conclusion supported by the current evidence.

## Current evidence chain

The current Tier-1 result package now has five layers.

1. The matched Tier-1 kernel is implemented and live.
2. The matched `full` Tier-1 evaluator is exact and permutation-invariant on the collected matched dataset.
3. Tier-1 approximate evaluator modes are nontrivially order-sensitive relative to matched `full`.
4. Generic Tier-0 vs Tier-1 rollouts show a robust Tier-1 jump-severity effect but no change in switch/blend/block counts.
5. Designed edge-boundary cases reveal the first Tier-1 scenario family where geometry changes the active-edge pattern and controller consequences become meaningful.

That progression is enough to rank the controller-layer objects for the current Tier-1 family.

## Recommended current default stance

The current recommended stance is:

- Keep the Tier-0 baseline intact.
- Treat Tier-1 as a credible, scenario-dependent extension rather than a blanket replacement claim.
- On the current evidence, `active_lexicographic` is the primary useful controller-layer refinement for the Tier-1 edge-boundary family.
- Keep runtime monitor activation optional and secondary.
- Do not recommend active monitor as a default for the current Tier-1 family.

This is an empirical controller recommendation, not a theorem.

## What is supported on generic rollouts

On the generic rollout family, Tier-1 changes severity, not discrete event structure.

Supported statements:

- Tier-1 robustly increases `max_raw_jump`.
- Tier-1 robustly increases `max_applied_jump`.
- Tier-1 slightly increases mean tracking error.
- Tier-1 does not change `switch_count`, `transition_start_count`, `blend_active_steps`, or blocking burden on the tested generic rollout family.

Interpretation:

- The generic rollout family is adequate for exposing continuous transition severity differences.
- It is not adequate for judging discrete controller-layer consequences.

## What is supported on designed edge-boundary scenarios

The first Tier-1 controller-relevant family is the edge-radius boundary family:

- `edge_radius_just_inside`
- `edge_radius_just_outside`

These are the first cases where Tier-1 changes the underlying active-edge pattern relative to Tier-0.

Supported statements:

- `active_edge_pattern_differs = true` on the edge-radius boundary cases.
- The same cases do not automatically change candidate or switch patterns under the frozen baseline controller.
- They do materially change jump severity and slightly change mean tracking.

This makes the edge-boundary family the correct scenario family for controller-consequence testing.

## Selector conclusion

### Main conclusion

Selector activation is the primary useful controller-layer refinement on the current Tier-1 edge-boundary family.

### Why

`shadow_lexicographic` remained identical to legacy on the tested edge-boundary cases.
So the relevant comparison is legacy versus `active_lexicographic`.

The decisive result is:

- `edge_radius_just_outside`, Tier-1
  - legacy: `switch_count = 9`, `blend_active_steps = 18`
  - active selector: `switch_count = 0`, `blend_active_steps = 0`

Tier-0 does not show the same selector consequence on that case.

This is the clearest current controller-layer win in the Tier-1 branch.

### Practical interpretation

For the current Tier-1 family:

- selector activation can be materially useful on scenario families where the geometry changes the active-edge structure,
- especially on edge-boundary cases,
- and that usefulness appears before runtime monitor activation becomes compelling.

## Monitor conclusion

### Main conclusion

Monitor activation is more interpretable on the Tier-1 edge-boundary family than it was on the generic baseline family, but it is still secondary and not yet compelling as a default controller refinement.

### Why

There are two distinct cases.

#### `edge_radius_just_outside`, Tier-1

- active selector already suppresses all switches,
- monitor modes `disabled`, `shadow`, and `active` are identical,
- therefore the monitor adds no value beyond the active selector.

This is a selector-dominant case.

#### `edge_radius_just_inside`, Tier-1

- active monitor does not reduce switch count beyond the active selector baseline,
- it produces some blocked intervals,
- it reduces jump severity modestly,
- but it increases mean tracking error noticeably,
- and its blocked intervals are still dominated by `large_predicted_gap`.

This is more interpretable than the old generic-rollout monitor story, but it is not a strong monitor win.

### Practical interpretation

For the current Tier-1 edge-boundary family:

- active monitor is not yet recommended as a default,
- monitor behavior is still largely a predicted-gap veto mechanism,
- and the benefit-cost tradeoff is weaker than the selector tradeoff.

## What remains empirical

The following are empirical conclusions, not proven controller guarantees:

- that `active_lexicographic` is the best current controller-layer refinement on the Tier-1 edge family,
- that monitor activation remains secondary,
- that edge-boundary scenarios are the correct regime for Tier-1 controller evaluation,
- that the current generic rollout family is not discriminative enough for Tier-1 discrete effects.

These statements are evidence-backed but scenario-conditioned.

## Current recommended interpretation for the roadmap

The roadmap should currently treat the controller layers as follows.

### Tier-0 baseline

- preserve unchanged
- remain the regression and theorem sandbox

### Tier-1 generic rollout family

- use primarily to report jump-severity consequences
- do not overclaim discrete controller consequences there

### Tier-1 edge-boundary family

- use as the primary controller-consequence family
- treat `active_lexicographic` as the main useful refinement
- keep monitor activation optional and secondary

## What should remain frozen right now

Before any further controller redesign, keep the following frozen:

- Tier-1 kernel parameters
- evaluator definitions
- candidate richness
- blend law
- plant-family dispatch
- current edge-boundary scenario definitions

The next technical change, if desired, should be justified against this conclusion rather than mixed into it.

## Short recommendation

If a single paper-facing sentence is needed now, use this:

> On the current Tier-1 family, controller consequences are scenario-dependent: generic rollouts show mainly jump-severity differences, whereas edge-boundary scenarios reveal that active lexicographic selection is the primary useful controller-layer refinement; runtime monitoring becomes more interpretable there but remains secondary and is not yet recommended as a default.

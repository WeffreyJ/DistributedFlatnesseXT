# Tier-2 Controller and Research Conclusion

## Status

This note summarizes the current Tier-2 results in `hybrid_flatness_ext`.
It is a synthesis artifact only.
It does not modify code behavior.

Tier-2 is currently framed as a disciplined nominal-plus-residual extension:
\[
\ddot s_i = u_i + \Delta_i^{\pi,\mathrm{nom}}(x) + r_i^\pi(x),
\qquad
u_i^\pi = v_i - \Delta_i^{\pi,\mathrm{nom}}(x).
\]

The controller continues to reconstruct against the nominal evaluator, while the plant applies a plant-side residual term.
This creates a controlled partial-unmatchedness setting.

## Current evidence chain

The Tier-2 evidence now has a complete internal ladder.

### 1. Reduction witness

Residual-disabled Tier-2 reduces exactly to Tier-1 nominal behavior.

Supported facts:
- pointwise nominal interaction matches Tier-1 nominal exactly,
- plant-side residual is exactly zero when disabled,
- controller-side evaluator behavior is unchanged,
- rollout-level switching, blending, jump, and tracking series match exactly.

Interpretation:
- Tier-2 was introduced without breaking the Tier-1 nominal baseline.
- This keeps the Tier-2 branch scientifically interpretable.

### 2. First residual family: `transverse_skew`

The first structured residual is plant-side only and geometry-aware.
It shares Tier-1 active-edge support and biases the plant interaction with a lateral-asymmetry-dependent skew.

Supported facts:
- the residual is live, structured, and nonzero,
- nominal-vs-plant mismatch is measurable at both pointwise and rollout levels,
- mismatch is order-sensitive through the object
  \[
  \Delta_{\mathrm{plant}}(x) - E_{\pi}^{\mathrm{nom}}(x),
  \]
- on generic rollouts and paired seed sweeps, the residual robustly increases mismatch burden,
  slightly worsens tracking, and slightly increases jump severity,
- it does not change `switch_count`, `transition_start_count`, or `blend_active_steps`.

Interpretation:
- `transverse_skew` is a valid Tier-2 residual, but under the frozen controller it behaves first as a mismatch/severity perturbation, not as a burden-changing mechanism.

### 3. Designed mismatch-sensitive scenarios for `transverse_skew`

Designed cases reveal that the first residual is not only a generic severity perturbation.

Supported facts:
- several designed cases produce:
  - switch-pattern differences,
  - candidate-history differences,
  - effective-candidate-history differences,
- these effects occur without changing aggregate switching counts or blend burden.

Interpretation:
- the first residual can perturb hybrid timing and pattern structure in a scenario-dependent way,
- but those perturbations do not convert into burden changes under the current frozen controller.

### 4. Pattern audit for `transverse_skew`

The pattern audit sharpens the mechanism.

Supported facts:
- pattern differences are concentrated near marginal switch windows,
- the dominant effect is switch-timing perturbation,
- candidate-history perturbations are present but secondary,
- pattern-difference steps occur at low tie margins and elevated mismatch values.

Interpretation:
- the first Tier-2 residual primarily shifts marginal switching timing rather than globally scrambling controller logic.

### 5. Second residual family: `longitudinal_bias`

The second residual is also plant-side only and uses Tier-1 active-edge support.
It applies a localized longitudinal-band bias inside the active interaction window.

Supported facts:
- the residual is live, structured, nonzero, and distinct from `transverse_skew`,
- generic rollout and seed-sweep diagnostics show that it is stronger than `transverse_skew` in:
  - mismatch increase,
  - jump increase,
  - mean tracking degradation,
- it still does not change `switch_count`, `transition_start_count`, or `blend_active_steps`.

Interpretation:
- `longitudinal_bias` is a stronger partial-unmatchedness residual, but still not a burden-changing mechanism on generic frozen rollouts.

### 6. Designed scenarios and pattern audit for `longitudinal_bias`

Designed longitudinal-band scenarios reveal stronger hybrid consequences.

Supported facts:
- all designed cases produced:
  - switch-pattern differences,
  - candidate-pattern differences,
  - effective-candidate-pattern differences,
- nevertheless, aggregate counts remain unchanged,
- the pattern audit shows:
  - pattern changes are still concentrated near marginal switching regions,
  - the dominant mechanism is again switch-timing perturbation,
  - candidate/effective-candidate changes are secondary,
  - pattern-difference steps are associated with low tie margins, elevated mismatch, and proximity to the longitudinal bias band.

Interpretation:
- the second residual provides the clearest Tier-2 mechanism result so far:
  structured plant-side mismatch alters hybrid timing and pattern behavior in a geometry- and scenario-dependent way, but still does not convert to switching burden changes under the frozen controller.

## Main Tier-2 conclusion

The strongest current Tier-2 conclusion is:

- Tier-2 is now a credible nominal-plus-residual extension of the hybrid flatness framework.
- Both residual families are interpretable and scientifically useful.
- Their first robust effect is not aggregate burden change.
- Their first robust effect is:
  - mismatch increase,
  - severity increase,
  - scenario-dependent hybrid timing/pattern perturbation near marginal switch regions.

This is the correct current Tier-2 claim.

## What Tier-2 currently supports

The current evidence supports the following statements.

### Supported statements

- Residual-disabled Tier-2 reduces exactly to Tier-1 nominal.
- Plant-side residual mismatch can be introduced without changing controller-side reconstruction semantics.
- Both residual families produce measurable nominal-vs-plant mismatch.
- Both residual families robustly increase mismatch burden and slightly worsen tracking/jump severity.
- Designed scenarios show that Tier-2 residuals can perturb hybrid timing and pattern structure.
- These perturbations are concentrated near low-margin switching regions.
- The stronger `longitudinal_bias` residual yields broader and more consistent pattern changes than `transverse_skew`.

### Not yet supported

The current evidence does not support claims that:
- Tier-2 residuals change aggregate switching burden under the frozen controller,
- Tier-2 residuals justify controller redesign already,
- Tier-2 residuals improve realism in a validated external-model sense,
- or Tier-2 residuals produce stronger controller-layer consequences than the Tier-1 selector result.

## Controller interpretation

Tier-2 is not currently a new controller-win story.
It is primarily a controlled plant-mismatch story.

Recommended interpretation:

- Tier-1 remains the main controller-consequence branch.
- Tier-2 currently extends the paper by showing that the hybrid mechanism remains interpretable under disciplined partial unmatchedness.
- The frozen controller is sensitive in timing and pattern terms, but robust in aggregate burden terms.

This is a valuable result because it distinguishes:
- burden sensitivity,
from
- timing/pattern sensitivity.

## Recommended paper-facing claim

If a single paper-facing claim is needed now, use:

> Under disciplined plant-side residual mismatch, Tier-2 preserves the nominal controller reconstruction while inducing structured nominal-vs-plant mismatch. Across two residual families, the dominant closed-loop effect is scenario-dependent perturbation of hybrid timing and candidate-pattern structure near low-margin switch regions, whereas aggregate switch and blend burdens remain unchanged under the frozen controller.

## Recommended roadmap interpretation

### Tier-1 role
- main controller-consequence branch,
- selector result remains primary.

### Tier-2 role
- partial-unmatchedness branch,
- demonstrates interpretable hybrid sensitivity under plant mismatch,
- expands the scientific scope of the paper beyond matched nominal behavior.

## What should remain frozen right now

Before any further Tier-2 redesign, keep frozen:
- selector logic,
- runtime monitor logic,
- candidate richness,
- blend law,
- evaluator semantics,
- Tier-2 nominal definition,
- the two current residual families,
- current designed scenario suites.

## Best next research move after this synthesis

The next move should not be controller tuning.

The best next move, if further Tier-2 work is desired, is one of:
1. external-model-grounded residual design,
2. a third residual family only if the paper truly needs burden conversion,
3. or stopping Tier-2 here and using it as a strong manuscript section on structured partial unmatchedness.

## Short conclusion

Tier-2 now supports a disciplined and manuscript-worthy story:
plant-side residual mismatch does not yet change switching burden under the frozen controller, but it does induce structured, scenario-dependent timing and pattern perturbations concentrated near marginal switching regions, with the longitudinal-bias residual providing the strongest current evidence.

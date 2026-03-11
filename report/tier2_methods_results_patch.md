# Tier-2 Methods and Results Patch

## Purpose

This note organizes the current Tier-2 evidence into a manuscript-facing methods/results structure.

Tier-2 should not be presented as a vague “more realistic extension.”
It should be presented as a disciplined nominal-plus-residual extension that preserves controller-side nominal reconstruction while introducing controlled plant-side mismatch.

That framing is the key to making the Tier-2 section coherent.

## Core Tier-2 narrative

The Tier-2 results should be presented in the following arc.

1. Define Tier-2 as a nominal-plus-residual extension.
2. Verify exact reduction to Tier-1 nominal when residuals are disabled.
3. Introduce the first residual family and show that it produces measurable mismatch.
4. Show that the first residual affects severity robustly but does not change burden on generic rollouts.
5. Use designed scenarios to show that the first residual perturbs hybrid timing and pattern structure.
6. Introduce the second residual family as a stronger burden-change candidate.
7. Show that the second residual increases mismatch and pattern perturbation further, but still does not change burden.
8. Conclude that Tier-2 currently demonstrates interpretable partial unmatchedness and hybrid timing sensitivity rather than burden conversion.

This is the cleanest Tier-2 section logic.

## Recommended methods structure

### Methods subsection: Tier-2 nominal-plus-residual formulation

State the Tier-2 plant explicitly:

\[
\ddot s_i = u_i + \Delta_i^{\pi,\mathrm{nom}}(x) + r_i^\pi(x).
\]

State the controller-side nominal reconstruction:

\[
u_i^\pi = v_i - \Delta_i^{\pi,\mathrm{nom}}(x).
\]

Then explain the conceptual split:

- the controller reconstructs against the nominal evaluator,
- the plant applies the nominal interaction plus a structured residual,
- therefore Tier-2 introduces disciplined partial unmatchedness without changing the nominal controller architecture.

This subsection should also state that the main Tier-2 object is the mismatch:

\[
M_\pi(x) := \Delta_{\mathrm{plant}}(x) - E_\pi^{\mathrm{nom}}(x).
\]

### Methods subsection: Tier-2 reduction witness

State that the first Tier-2 verification step is exact reduction with residual disabled.

Purpose:
- prove that Tier-2 inherits the Tier-1 nominal branch exactly before any residual is activated.

This subsection should mention both:
- pointwise reduction checks,
- rollout-level exact equivalence checks.

### Methods subsection: residual family 1 — transverse skew

Describe:
- plant-side only,
- Tier-1 active-edge support,
- lateral-asymmetry-dependent skew term,
- small structured perturbation.

Make clear that this was designed as the first interpretable unmatched residual, not as a burden-maximizing perturbation.

### Methods subsection: residual family 2 — longitudinal bias

Describe:
- plant-side only,
- Tier-1 active-edge support,
- localized longitudinal-band bias,
- stronger persistence inside sensitive longitudinal regions.

Make clear that this second family was introduced after the first residual failed to convert timing perturbations into burden changes.

### Methods subsection: Tier-2 diagnostic protocol

Organize the Tier-2 diagnostics in layers.

#### Layer 1: pointwise and matched reduction diagnostics
- reduction witness,
- nominal-vs-plant gap.

#### Layer 2: generic closed-loop comparison
- frozen-controller shared rollout comparison,
- paired seed sweep.

#### Layer 3: designed scenario comparison
- mismatch-sensitive scenarios,
- count-conversion scenarios for the first residual,
- longitudinal-band scenarios for the second residual.

#### Layer 4: pattern-change mechanism audit
- pattern-difference steps,
- switch-window proximity,
- tie-margin statistics,
- mismatch at pattern-difference steps,
- residual-mechanism-aligned geometric diagnostics.

This diagnostic layering should be explicit in the paper.

## Recommended results structure

### Result 1: Tier-2 reduces exactly to Tier-1 nominal when residuals are disabled

Use:
- `tier2_reduction_witness.json`

Main statements:
- controller-side nominal evaluator remains unchanged,
- plant-side residual is zero,
- Tier-2 rollout matches Tier-1 nominal exactly.

Interpretation:
- Tier-2 was introduced without corrupting the nominal branch.

### Result 2: the first Tier-2 residual yields measurable nominal-vs-plant mismatch

Use:
- `tier2_nominal_vs_plant_gap_summary.json`

Main statements:
- the `transverse_skew` residual is live and structured,
- mismatch is nonzero and order-sensitive through the mismatch object,
- plant-order gap remains zero at the snapshot level while the nominal mismatch remains order-sensitive.

Interpretation:
- the relevant Tier-2 object is mismatch against the nominal evaluator, not a plant-order-dependent interaction law.

### Result 3: on generic rollouts, the first residual changes severity but not burden

Use:
- `tier2_residual_compare.json`
- `tier2_residual_seed_sweep.json`

Main statements:
- mismatch burden increases robustly,
- tracking error and jump severity increase slightly but consistently,
- switch count, transition count, and blend burden remain unchanged.

Interpretation:
- the first residual behaves as a robust severity/mismatch perturbation on generic rollouts.

### Result 4: designed scenarios reveal timing/pattern effects for the first residual

Use:
- `tier2_mismatch_sensitive_scenarios.json`
- `tier2_pattern_change_audit.json`

Main statements:
- designed cases produce switch-pattern and candidate-history differences,
- effects concentrate near marginal switch windows,
- timing perturbation is the dominant mechanism,
- candidate-history changes are secondary.

Interpretation:
- the first residual can perturb hybrid structure without changing burden.

### Result 5: count-conversion attempts fail for the first residual

Use:
- `tier2_count_conversion_scenarios.json`

Main statements:
- even amplified dwell/grazing scenarios do not change aggregate burden,
- pattern differences persist without count conversion.

Interpretation:
- the first residual is exhausted as a burden-changing mechanism.

### Result 6: the second residual is stronger in mismatch and timing-pattern consequences

Use:
- `tier2_second_residual_gap_summary.json`
- `tier2_second_residual_compare.json`
- `tier2_second_residual_seed_sweep.json`

Main statements:
- `longitudinal_bias` is stronger than `transverse_skew` in mismatch magnitude,
  jump increase, and tracking degradation,
- generic rollouts and seeds still show no burden changes.

Interpretation:
- stronger mismatch alone is not sufficient to produce burden conversion under the frozen controller.

### Result 7: designed longitudinal-bias scenarios yield broader pattern changes but still no burden change

Use:
- `tier2_second_residual_scenarios.json`
- `tier2_second_residual_pattern_audit.json`

Main statements:
- all designed longitudinal-bias cases produce switch/candidate/effective-candidate pattern differences,
- burden counts remain unchanged,
- the dominant mechanism is marginal switch-timing perturbation,
- candidate-history perturbation is secondary,
- pattern-difference steps occur at low tie margins, elevated mismatch, and proximity to the residual’s longitudinal bias band.

Interpretation:
- the second residual gives the strongest current Tier-2 mechanism result.

## Recommended final Tier-2 interpretation subsection

The final interpretation subsection should say:

### Main claim

Tier-2 demonstrates that the hybrid framework remains interpretable under disciplined plant-side residual mismatch.

### Strongest supported reading

- residual mismatch robustly changes severity metrics,
- designed scenarios reveal structured hybrid timing and pattern sensitivity,
- these perturbations are concentrated near marginal switch regions,
- even stronger residuals do not produce aggregate burden changes under the frozen controller.

### What this means scientifically

Tier-2 should be read as a partial-unmatchedness robustness/sensitivity section, not as a controller-improvement section.

## Recommended discussion stance

Use a short explicit stance paragraph:

- Tier-1 remains the primary controller-consequence branch.
- Tier-2 extends the framework to controlled plant-side partial unmatchedness.
- The dominant Tier-2 effect is structured timing/pattern perturbation under mismatch.
- Aggregate burden robustness remains intact under the frozen controller.

## Suggested figure and table mapping

### Table A: Tier-2 diagnostic ladder
Rows:
- reduction witness,
- first residual gap diagnostic,
- first residual generic comparison,
- first residual designed scenario audit,
- count-conversion attempt,
- second residual gap diagnostic,
- second residual generic comparison,
- second residual designed scenario audit.

Columns:
- question,
- primary artifact,
- key finding.

### Figure B: Tier-2 generic vs designed consequences
Contrast:
- generic mismatch/severity increase,
- designed timing/pattern perturbation.

### Figure C: first vs second residual family
Show:
- mean mismatch increase,
- max applied jump increase,
- count/burden delta remains zero.

### Figure D: pattern-change mechanism
Show:
- mismatch at pattern-difference steps,
- tie margin at pattern-difference steps,
- mechanism-aligned geometry statistic
  (switch-window proximity for the first residual, longitudinal-band distance for the second residual).

## What should remain outside the current claim

Do not claim yet that:
- Tier-2 residuals change switching burden,
- Tier-2 justifies controller redesign,
- Tier-2 is validated against an external high-fidelity plant,
- or the second residual is the final plant-mismatch model.

Those are next-step research directions, not current results.

## Short summary paragraph

If a compact bridge paragraph is needed, use:

> Tier-2 extends the nominal matched controller with disciplined plant-side residual mismatch while preserving nominal controller reconstruction. Residual-disabled Tier-2 reduces exactly to Tier-1 nominal behavior. Across two structured residual families, generic rollouts show robust increases in mismatch burden and severity metrics without changing aggregate switching burden. Designed scenarios reveal the first nontrivial Tier-2 hybrid consequences: residual mismatch perturbs switch timing and candidate-pattern histories in low-margin regions, with the stronger longitudinal-bias residual producing the clearest mechanism-aligned pattern changes. These results position Tier-2 as an interpretable partial-unmatchedness extension rather than a burden-changing controller regime.

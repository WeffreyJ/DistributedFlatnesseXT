# Tier-2 Second Residual Spec

## Purpose

The first Tier-2 residual family, `transverse_skew`, is now exhausted as a burden-changing mechanism.

What it established:
- it is live, structured, and order-sensitive through the nominal-vs-plant mismatch object
- it robustly increases mismatch burden and slightly worsens tracking/jump severity
- it can perturb switch timing and pattern histories on designed scenarios
- it does not change `switch_count`, `transition_start_count`, or `blend_active_steps` under the frozen controller

That makes the next question specific: what plant-side residual structure is more likely to perturb switching burden rather than only timing?

## New Mechanism

The second residual mode is `longitudinal_bias`.

Target mechanism:
- keep the Tier-2 plant-side residual on the Tier-1 active-edge support
- bias the plant interaction inside a longitudinal band of the active coupling window
- make the mismatch strongest where marginal switching decisions are likely to persist for several steps

This is structurally different from `transverse_skew`:
- `transverse_skew` depends on lateral asymmetry
- `longitudinal_bias` depends on where the edge sits along the active longitudinal window

The intent is to create a more persistent plant-side bias near marginal switching regions, which is a more plausible route to burden changes than pure transverse skew.

## Mathematical Form

Tier-2 remains:

\[
\ddot s_i = u_i + \Delta_i^{\pi,\mathrm{nom}}(x) + r_i^\pi(x),
\qquad
u_i^\pi = v_i - \Delta_i^{\pi,\mathrm{nom}}(x).
\]

For `longitudinal_bias`, the plant-side residual is:

\[
r_i^\pi(x) = -\alpha_{\mathrm{res}}
\sum_{(j,i)\in \mathcal E_{\mathrm{T1}}(x)}
g_{\mathrm{long}}(d_{ji})
\exp(-d_{ji}/L_{\mathrm{long}}),
\]

where:
- the support is the Tier-1 active-edge set
- \(d_{ji}\) is the effective longitudinal separation inside the active window
- \(g_{\mathrm{long}}\) is a localized longitudinal-band weighting centered at a configurable bias center

Implementation choice:
- use a Gaussian longitudinal window centered at `longitudinal_bias_center`
- scale it by `longitudinal_scale`
- multiply by a configurable `longitudinal_bias_gain`

## Why This Is The Right Next Residual

This residual is a plausible next burden-changing candidate because it can:
- persist over several steps while an edge stays in a sensitive longitudinal band
- perturb the plant interaction without changing controller-side evaluator semantics
- act directly in the same geometry dimension that already influences ordering and switching logic

It is still:
- deterministic
- structured
- easy to disable
- comparable to the first residual through the same Tier-2 diagnostics

## First Diagnostic Ladder

1. Add `longitudinal_bias` as a second residual mode in `residual_tier2.py`.
2. Add config support in `system.yaml`.
3. Run a nominal-vs-plant gap diagnostic for the new mode.
4. Verify that the residual is:
   - live
   - structured
   - nonzero
   - measurably order-sensitive through the mismatch object
5. Only then run frozen-controller rollout comparisons.

## Frozen Interfaces

The following stay frozen initially:
- selector logic
- monitor logic
- candidate richness
- blend law
- controller-side evaluator semantics
- closed-loop structure

## Non-Claims

Before the first diagnostic ladder is complete, do not claim that `longitudinal_bias`:
- changes burden
- improves realism materially
- or creates a better controller regime than `transverse_skew`

At this phase it is only a new structured residual candidate.

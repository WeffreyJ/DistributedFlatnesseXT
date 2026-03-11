# Tier-2 Extension Specification

## Status

This document defines the Tier-2 research branch for `hybrid_flatness_ext`.
It is a design/specification artifact only.
It does not change plant code, controller code, selector logic, monitor logic, or verification scripts.

Tier-1 is now mature enough to support manuscript assembly:

- the matched geometry-aware kernel is live,
- the matched `full` path is verified,
- approximate modes are nontrivially order-sensitive relative to `full`,
- generic rollouts primarily expose severity effects,
- designed Tier-1 scenario families expose controller-relevant regimes,
- `active_lexicographic` is the primary useful controller-layer refinement,
- monitor behavior remains secondary on the current validated Tier-1 families.

Tier-2 should therefore be treated as a new research branch, not as an ad hoc extension of Tier-1.

## Purpose

### What Tier-1 accomplished

Tier-1 answered the matched-extension question:

1. A geometry-aware interaction kernel can be added while preserving the matched reconstruction architecture.
2. The evaluator layer can remain the central abstraction boundary.
3. The same hybrid controller stack can be compared meaningfully across Tier-0 and Tier-1.
4. Selector consequences are scenario-conditioned, and on validated Tier-1 families the dominant useful mechanism is conservative no-switch retention.

### What Tier-1 still leaves unresolved

Tier-1 still uses an exactly matched controller/plant pair.
That means the controller’s evaluator is also the plant’s effective interaction model.
This is analytically clean, but it does not test whether the verification-first hybrid story survives when:

- the plant contains structured interaction effects that are not fully compensated by the controller,
- the evaluator is only a nominal model of the plant,
- and controller-side exact identities must be interpreted alongside plant-side model mismatch.

### Tier-2 scientific target

Tier-2 should answer this question:

> Can the verification-first hybrid-control story remain interpretable and useful when the controller uses a nominal matched interaction model, but the plant contains an additional structured residual interaction term that is not compensated in reconstruction?

This is the cleanest next step because it changes one conceptually meaningful layer:

- Tier-1: matched geometry-aware nominal interaction
- Tier-2: nominal matched interaction plus structured residual / unmatched interaction

That keeps attribution clear.

## Tier-2 Mathematical Target

### Recommended Tier-2 identity

Tier-2 should introduce a structured residual interaction layer while preserving the basic second-order channel form:

\[
\ddot s_i
=
u_i
+ \Delta_i^{\pi,\mathrm{nom}}(x;\theta_{\mathrm{nom}})
+ r_i^{\pi}(x;\theta_{\mathrm{res}}).
\]

Interpretation:

- `\Delta_i^{\pi,\mathrm{nom}}` is the Tier-1-style nominal interaction model used by the controller,
- `r_i^{\pi}` is a structured residual term representing interaction physics not captured by the nominal evaluator,
- `s_i` remains the scalar along-track state,
- `a_i(x_i)=0` and `b_i(x_i)=1` remain fixed in the first Tier-2 step.

This is the preferred Tier-2 target because it introduces partial unmatchedness without simultaneously changing:

- the single-agent channel structure,
- the inversion algebra,
- the selector interface,
- or the high-level simulation loop.

### Nominal controller reconstruction

The controller should continue to use the nominal matched reconstruction

\[
u_i^\pi = v_i - \Delta_i^{\pi,\mathrm{nom}}(x),
\]

or in evaluator form

\[
u^\pi(x) = v(x) - E_\pi^{\mathrm{nom}}(x),
\qquad
E_\pi^{\mathrm{nom}}(x) := \Delta^{\pi,\mathrm{nom}}(x).
\]

The plant, however, evolves under

\[
\ddot s_i
=
v_i - \Delta_i^{\pi,\mathrm{nom}}(x)
+ \Delta_i^{\pi,\mathrm{nom}}(x)
+ r_i^{\pi}(x)
=
v_i + r_i^{\pi}(x).
\]

That makes the residual term explicit in the closed-loop error budget.

### Recommended residual family

The first Tier-2 residual should be structured and bounded, not arbitrary.
A suitable first family is:

\[
r_i^{\pi}(x;\theta_{\mathrm{res}})
=
\rho_{\mathrm{res}}
\sum_{j \in \mathrm{Up}_\pi(i)}
\widetilde w_{ji}(x;\theta_{\mathrm{res}}),
\]

where `\rho_res` is a scalar amplitude knob and `\widetilde w_{ji}` differs from the nominal Tier-1 kernel by one conceptually meaningful effect, for example:

- stronger anisotropy in lateral vs vertical geometry,
- a finite-support shoulder near the interaction boundary,
- orientation-sensitive gain,
- or a mild skew/nonreciprocity term.

The important constraint is that Tier-2 should not introduce many unrelated effects at once.
It should add one structured residual family with a clear interpretation.

## Evaluator Interpretation

Tier-2 requires an explicit distinction between:

1. the controller nominal evaluator,
2. the plant reference interaction,
3. and approximate control-side evaluator modes.

### Controller nominal evaluator

The controller-facing evaluator remains

\[
E_\pi^{\mathrm{nom}}(x).
\]

This is the interaction model used in reconstruction and in all current controller-side objects:

- raw mismatch,
- predicted-gap proxy,
- selector candidate comparison,
- monitor predicted-gap field,
- and operator-mismatch diagnostics.

### Plant reference interaction

The plant-side reference interaction is

\[
E_\pi^{\mathrm{plant}}(x)
=
E_\pi^{\mathrm{nom}}(x) + r^\pi(x).
\]

This object should exist for diagnostics, not for the initial controller path.
It answers the question:

> what interaction is actually acting on the plant, beyond what the controller compensates?

### Approximate evaluator modes in Tier-2

The existing evaluator modes should remain, but their meaning must be stated carefully:

- `full`: full nominal interaction `E_\pi^{\mathrm{nom,full}}`
- `upstream_truncated`: nominal order-truncated interaction
- `local_window`: nominal locally truncated interaction

Those modes remain controller-side approximations of the nominal model.
They do not automatically include the plant residual.

That separation is deliberate.
It keeps controller identities and plant mismatch diagnostics conceptually distinct.

## Exact / Approximate / Residual Case Split

Tier-2 should be defined through three subcases.

### Case T2.0: residual disabled witness

\[
r^\pi(x) \equiv 0.
\]

This must reduce exactly to the Tier-1 nominal matched model.
It is the first witness stage.

### Case T2.1: exact nominal matched controller with diagnostic plant reference

Controller and plant both use `E_\pi^{\mathrm{nom}}`, while `E_\pi^{\mathrm{plant}}` is computed only as an offline diagnostic object.
This is useful for validating that the Tier-2 nominal branch is wired correctly before activating the residual in the plant dynamics.

### Case T2.2: structured residual / partially unmatched Tier-2

Plant uses

\[
E_\pi^{\mathrm{plant}} = E_\pi^{\mathrm{nom}} + r^\pi,
\]

while controller reconstruction still uses only `E_\pi^{\mathrm{nom}}`.

This is the first true Tier-2 scientific case and should be the main implementation target after the witness stages.

### Which case should be implemented first

Implementation order must be:

1. `T2.0` residual-disabled reduction to Tier-1
2. `T2.1` nominal/reference diagnostic wiring
3. `T2.2` residual-enabled plant dynamics

This prevents Tier-2 from skipping the path-integrity checks that made Tier-1 tractable.

## What Remains Exact And What Becomes Approximate

### Exact controller-side objects that survive unchanged

As long as the controller still reconstructs with `E_\pi^{\mathrm{nom}}`, the following remain exact on the controller side:

1. same-snapshot nominal raw mismatch identity
   \[
   \|u^\pi - u^{\pi'}\|
   =
   \|E_\pi^{\mathrm{nom}} - E_{\pi'}^{\mathrm{nom}}\|
   \]
2. the current operator-gap and predicted-gap definitions
3. the applied-jump decomposition under blending
4. the selector’s same-snapshot predicted-gap comparisons

### Objects that become plant-side mismatch diagnostics

Once the residual is enabled in the plant, the following are no longer exact plant-side statements:

- controller raw mismatch is no longer the full story of plant transition severity,
- controller-side evaluator-gap bounds no longer fully describe closed-loop acceleration mismatch,
- tracking degradation now depends explicitly on the residual term.

The plant-side residual should therefore be measured through explicit diagnostics such as:

\[
\|r^\pi(x)\|,
\qquad
\|E_\pi^{\mathrm{plant}} - E_\pi^{\mathrm{nom}}\|,
\qquad
\|u^\pi_{\mathrm{nom}} - u^\pi_{\mathrm{matched\ plant}}\|,
\]

where the last quantity is diagnostic-only if a plant-matched compensator is constructed offline.

## Frozen Interfaces

The following must remain frozen during the first Tier-2 implementation ladder:

- selector logic
- monitor logic
- candidate richness / candidate-set construction
- blend law
- high-level `simulate_closed_loop(...)` structure
- Tier-1 baseline artifacts and comparison scripts
- controller-facing evaluator API shape

This means Tier-2 is not allowed to simultaneously become:

- a new plant,
- a new selector,
- a new monitor,
- and a new blending strategy.

Tier-2 should first answer the residual-model question only.

## Gate 2 Interpretation In Tier-2

For the first Tier-2 step, `a_i=0` and `b_i=1` remain unchanged.
Therefore Gate 2 should still not be presented as a nontrivial decoupling-invertibility theorem.

Instead, Gate 2 should be reinterpreted in two layers:

1. **nominal inversion layer**
   - still trivial from the standpoint of `b_i=1`
2. **residual sensitivity layer**
   - quantify the size of the plant residual relative to the nominal evaluator
   - report bounded residual-to-output sensitivity on the operating envelope

This is a disciplined extension, not a full nonlinear decoupling proof.

## Verification-First Ladder

Tier-2 should be implemented and validated in this order.

### Stage 1: config and dispatch scaffolding

- add a `tier2` config block
- add a plant-family switch `tier2`
- preserve `tier0` and `tier1` behavior unchanged

### Stage 2: Tier-2 reduction witness

Residual disabled:

- verify `tier2(residual_off)` reproduces Tier-1 nominal results
- confirm no silent fallback to Tier-1 dispatch

### Stage 3: nominal/reference diagnostic witness

- compute `E^{nom}` and `E^{plant}` side by side
- verify `E^{plant} = E^{nom}` when residual amplitude is zero
- verify nonzero residual norm when amplitude is positive

### Stage 4: first Tier-2 theorem-support diagnostic

Before any controller comparison, add a diagnostic that measures:

- nominal evaluator gap
- plant-reference interaction gap
- residual norm
- nominal-vs-plant interaction discrepancy

This is the Tier-2 analogue of the early Tier-1 theorem-support step.

### Stage 5: frozen-controller Tier-1 vs Tier-2 comparison

Only after the witness and diagnostic stages:

- compare Tier-1 vs Tier-2 under frozen controller settings
- start with generic rollouts
- then move to designed Tier-1-relevant scenario families

### Stage 6: scenario search for Tier-2-specific consequences

If Tier-2 only amplifies severity again, search for new designed cases.
If Tier-2 changes discrete behavior, then and only then revisit controller consequences.

## File-By-File Implementation Order

The expected first implementation order is:

1. `configs/system.yaml`
   - add `system.plant_family: tier2`
   - add `system.tier2.nominal`
   - add `system.tier2.residual`
   - add a residual amplitude toggle and witness-friendly default values

2. `src/model/coupling_tier2_nominal.py`
   - implement the Tier-2 nominal interaction branch
   - initially this may wrap the Tier-1 nominal kernel or a lightly extended nominal kernel

3. `src/model/coupling_tier2_residual.py`
   - implement the structured residual term only
   - keep geometry and residual logic modular

4. `src/model/dynamics.py`
   - dispatch Tier-2 plant dynamics
   - plant acceleration becomes nominal plus residual

5. `src/model/coupling.py`
   - expose shared dispatch helpers for Tier-2 nominal and plant-reference interaction

6. `src/flatness/evaluation_operator.py`
   - preserve current controller-facing evaluator modes on the nominal Tier-2 branch
   - add diagnostic-only support for plant-reference interaction if needed, without destabilizing the control API

7. `src/model/graph.py`
   - keep the first Tier-2 graph definition tied to the nominal/controller branch unless a later experiment explicitly promotes plant-reference graphing

8. first Tier-2 verify scripts
   - reduction witness
   - nominal/reference discrepancy diagnostic

9. first Tier-1 vs Tier-2 frozen-controller comparison

## Non-Claims And Discipline

Before the first Tier-2 verification stages are complete, the project must not claim:

- that Tier-2 preserves exact matched reconstruction
- that Tier-2 operator-gap results fully describe plant behavior
- that selector or monitor conclusions automatically transfer unchanged from Tier-1
- that Gate 2 has become a nontrivial conditioning theorem
- that Tier-2 is a high-fidelity aerodynamic model

Tier-2 should initially be described as:

> a verification-first nominal-plus-residual extension that tests whether the hybrid-control conclusions remain interpretable under structured plant-side model mismatch.

That is enough.

## Recommended First Implementation Step After This Spec

The first coding step after this document should be:

1. add the Tier-2 config block and plant-family dispatch scaffold,
2. implement a residual-disabled witness path that reproduces Tier-1 nominal behavior exactly,
3. stop and verify that reduction witness before enabling any plant-side residual term.

That keeps Tier-2 attributable from the first line of code.

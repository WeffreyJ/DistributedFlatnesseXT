# Tier-1 Matched-Reconstruction Design Specification

## Status

This document is a design/specification artifact for the next plant transition in `hybrid_flatness_ext`.
It does not change controller behavior, plant code, selector logic, or monitor logic.

The current workspace has already established, for the scalar wake-surrogate model:

- an explicit evaluator layer,
- an exact same-snapshot raw-mismatch identity,
- a conservative component-aware evaluator-gap bound,
- an exact applied-jump decomposition under blending,
- selector and runtime-monitor diagnostics,
- and a negative calibration result showing that monitor tuning alone is not the next bottleneck.

The next bottleneck is the informativeness of the plant/evaluator family itself.
This document defines the intended Tier-1 plant family and the matched reconstruction that should accompany it.

## Scope

This specification answers four questions:

1. What Tier-1 plant family should replace the current scalar wake surrogate?
2. How does the matched reconstruction object change?
3. Which current identities remain exact, and which become approximate?
4. What concrete file-by-file implementation order should be used in the future coding phase?

It also states what must remain frozen during the Tier-1 transition so the project does not lose attribution.

## Current Baseline To Preserve

The current plant and controller fit the exact matched form

\[
\ddot y_i = u_i + \Delta_i^{\pi}(x),
\qquad
u_i^{\pi} = v_i - \Delta_i^{\pi}(x),
\]

with:

- `a_i(x_i) = 0`,
- `b_i(x_i) = 1`,
- `y_i = x_{1,i}`,
- `\Delta_i^{\pi}(x)` implemented by the current evaluator/coupling surrogate.

Because the controller and simulated plant use the same state-dependent coupling term, the following are exact for the current surrogate:

1. Same-snapshot evaluator difference is exactly measurable.
2. Same-snapshot raw mismatch satisfies
   \[
   \|u^\pi(x)-u^{\pi'}(x)\| = \|E_\pi(x)-E_{\pi'}(x)\|.
   \]
3. The applied-jump decomposition under blending is exact at the controller level.

The Tier-1 design should preserve that exactness whenever the controller and plant share the same Tier-1 coupling law.

## Tier-1 Plant Family

### Design objective

Tier-1 should add geometric richness without blowing up the rest of the stack.
The preferred first Tier-1 family keeps:

- second-order per-agent channel structure,
- scalar tracked output per agent,
- direct control effectiveness `b_i = 1`,
- and order-dependent coupling only through the evaluator/coupling operator.

This avoids changing the core recursion architecture during the first plant upgrade.

### Recommended state interpretation

Keep the controller-facing dynamic state as

\[
x_i = (s_i, \dot s_i),
\]

where `s_i` is a scalar along-track progress coordinate.

Add fixed or slowly varying geometry descriptors per agent as parameters, not dynamic states, in the first Tier-1 step:

- lateral offset `\ell_i`,
- vertical offset `h_i`,
- optional heading/alignment parameter `\psi_i` if needed later.

This yields a plant that is still second-order in `s_i` but whose pairwise coupling depends on richer relative geometry:

\[
\ddot s_i = u_i + \Delta_i^{\pi}(x; \theta_{\mathrm{geom}}).
\]

This is the least disruptive path because it leaves `src/control/flat_tracking.py`, the recursion order, and the closed-loop integrator shape intact.

### Coupling structure

Tier-1 coupling should be defined through pairwise directed terms

\[
w_{ji}^{\mathrm{T1}}(x;\theta)
=
k_{\mathrm{ind}}
\cdot
K_{\perp}(\ell_j-\ell_i,\; h_j-h_i;\theta_\perp)
\cdot
K_{\parallel}(s_j-s_i;\theta_\parallel)
\cdot
\mathbf{1}\!\left[\gamma_{\mathrm{edge}} < s_j-s_i < R_x\right].
\]

Recommended interpretation:

- `K_parallel` is a smooth longitudinal decay or transport kernel,
- `K_perp` penalizes large lateral/vertical separation and regularizes near-core singularity,
- the indicator keeps the existing finite interaction window,
- order dependence enters through which directed terms are included by the evaluator.

The per-agent coupling term is then

\[
\Delta_i^{\pi}(x;\theta) =
\sum_{j \in \mathrm{Up}_\pi(i)} w_{ji}^{\mathrm{T1}}(x;\theta)
\]

for the order-truncated case, with obvious `full` and `local_window` variants.

### Recommended initial kernel family

The first Tier-1 implementation should use a closed-form reduced kernel, not tables:

\[
K_{\parallel}(\Delta s)
=
\exp\!\left(-\frac{\max(\Delta s-\gamma_{\mathrm{edge}},0)}{L_s}\right),
\]

\[
K_{\perp}(\Delta \ell,\Delta h)
=
\frac{1}{\left(r_c^2 + \Delta \ell^2 + \alpha_h \Delta h^2\right)^{p/2}}.
\]

Design notes:

- `r_c > 0` is a finite core-radius regularizer,
- `\alpha_h` weights vertical separation,
- `p \in [1,2]` controls how quickly the induced effect decays cross-stream,
- this is still cheap enough for Monte Carlo gates.

This family is not meant to be a full aircraft wake model.
It is meant to be a more informative reduced interaction family than the current purely longitudinal exponential surrogate.

## Order Dependence In Tier-1

Order dependence should continue to enter only through the evaluation/inclusion operator, not through the basic single-agent channel:

\[
E_\pi^{\mathrm{T1}}(x) :=
\begin{bmatrix}
\sum_{j \in \mathrm{Up}_\pi(1)} w_{j1}^{\mathrm{T1}}(x) \\
\vdots \\
\sum_{j \in \mathrm{Up}_\pi(N)} w_{jN}^{\mathrm{T1}}(x)
\end{bmatrix}.
\]

The same evaluation modes should remain:

- `full`,
- `upstream_truncated`,
- `local_window`.

What changes is the pairwise kernel, not the evaluator API.

That is a key transition constraint:
`src/flatness/evaluation_operator.py` must remain the controller-facing abstraction boundary.

## Matched Reconstruction Object

### Target matched family

The Tier-1 matched family should be written as

\[
\ddot y_i = a_i(x_i) + b_i(x_i) u_i + \Delta_i^{\pi}(x),
\]

with preferred first-step choice

\[
a_i(x_i)=0,\qquad b_i(x_i)=1.
\]

Then the matched reconstruction is

\[
u_i^\pi = v_i - \Delta_i^{\pi}(x).
\]

In vector form:

\[
u^\pi(x) = v(x) - E_\pi^{\mathrm{T1}}(x).
\]

This is intentionally the same algebraic structure as the current surrogate.
That is the design target.

### Why this matters

If Tier-1 keeps `a_i=0` and `b_i=1`, then:

- the recursion layer still only needs the evaluator output vector,
- the same-snapshot raw-mismatch identity remains exact under the Tier-1 controller model,
- the current operator-gap and applied-jump package still applies with the new kernel substituted in.

If Tier-1 does not keep `b_i=1`, the project immediately becomes a Gate-2-first redesign problem.
That is not the preferred first Tier-1 step.

## Exact vs Approximate Claims In Tier-1

### Case 1: fully matched Tier-1 simulation

If the plant simulation and controller reconstruction use the same Tier-1 evaluator/coupling model, then the following remain exact:

1. Same-snapshot control difference identity
   \[
   \|u^\pi-u^{\pi'}\| = \|E_\pi^{\mathrm{T1}}-E_{\pi'}^{\mathrm{T1}}\|.
   \]
2. The controller-level applied-jump decomposition under blending.
3. Any evaluator-gap bound proved directly for the Tier-1 kernel family.

This is the preferred first implementation mode.

### Case 2: approximate controller model vs richer plant

If the controller uses a reduced evaluator `\widehat E_\pi` while the plant uses a richer or table-driven `E_\pi^{\mathrm{plant}}`, then:

- the raw mismatch identity remains exact for controller reconstructions relative to `\widehat E_\pi`,
- but the plant is no longer exactly matched to that controller model,
- and the residual
  \[
  r_i(x) = E_{\pi,i}^{\mathrm{plant}}(x) - \widehat E_{\pi,i}(x)
  \]
  becomes an explicit model mismatch term.

Then statements split into:

- exact controller-side identities,
- empirical plant-side mismatch diagnostics.

This split must be stated explicitly if/when a table-driven plant is introduced.

## Meaning Of Existing Results In Tier-1

### Operator-gap result

Under a fully matched Tier-1 model, the operator-gap result means:

- the difference in reconstructed controls across orderings is still exactly the difference in Tier-1 evaluator outputs,
- but the bound must now use the Tier-1 kernel magnitude bound rather than the scalar surrogate’s longitudinal-only bound.

The current component-aware structure still applies:

\[
\|E_\pi - E_{\pi'}\|_2
\le
\bar w_{\mathrm{T1}}
\left(\sum_i |D_i(\pi,\pi')|^2\right)^{1/2},
\]

provided a valid uniform pairwise bound `|w_{ji}^{\mathrm{T1}}| \le \bar w_{\mathrm{T1}}` is available on the operating envelope.

### Applied-jump result

Under a fully matched Tier-1 model, the exact applied-jump decomposition remains exact at the controller level:

\[
\|u_{\mathrm{app}}[k]-u_{\mathrm{app}}[k-1]\|
\le
|\beta_k-\beta_{k-1}|\,\|u_{\mathrm{new}}[k]-u_{\mathrm{old}}[k]\|
+
V_{\mathrm{intra}}^\star[k].
\]

What changes is the size distribution of the raw-gap term because the Tier-1 evaluator is richer.

If controller and plant are not matched, the decomposition is still exact for the controller-applied command sequence, but interpretation of closed-loop performance must include model mismatch.

## Gate 2 Reinterpretation

### Immediate Tier-1 case: still trivial invertibility, nontrivial modeling

If Tier-1 keeps `b_i=1`, then Gate 2 should not suddenly claim nontrivial decoupling rank theory.
Instead Gate 2 should be reinterpreted as:

1. matched-inversion well-posedness check,
2. evaluator regularity and boundedness check on the operating envelope,
3. geometry-kernel non-singularity/regularization check,
4. optional finite-difference sensitivity proxy for future richer plants.

This is more honest than overselling rank/conditioning in a system that still has unit input effectiveness.

### Future nontrivial Gate 2

Only after a later plant introduces nonconstant `b_i(x_i)` or richer decoupling structure should Gate 2 be elevated to:

- bounded invertibility certification,
- decoupling-condition conditioning bounds,
- mismatch amplification factors.

## New Verification Outputs Needed For Tier-1

The current gates should not be deleted.
Tier-1 should add or extend diagnostics with the following new outputs:

1. Pairwise geometry statistics:
   - distributions of `\Delta s`, `\Delta \ell`, `\Delta h`,
   - active interaction counts by geometry regime.

2. Tier-1 evaluator diagnostics:
   - `E_\pi^{\mathrm{T1}}` magnitude distribution,
   - per-follower contribution counts,
   - envelope bound constant `\bar w_{\mathrm{T1}}`.

3. Matched-vs-approximate model diagnostics:
   - if a reduced controller model differs from plant simulation, log residual `E^{\mathrm{plant}}-\widehat E`.

4. Gate 2 outputs:
   - regularization/core-radius checks,
   - bounded evaluator sensitivity estimates,
   - domain violations for geometry assumptions.

5. Gate R updates:
   - envelope escape by geometry regime,
   - escape under Tier-0 vs Tier-1.

## File-By-File Future Implementation Roadmap

This section is the future coding order.
It is intentionally sequential.

### Step 1: add Tier-1 config blocks

Primary file:

- `configs/system.yaml`

Add:

- `system.plant_family: wake_surrogate_tier0 | wake_geometry_tier1`
- `tier1.geometry.lateral_offsets`
- `tier1.geometry.vertical_offsets`
- `tier1.kernel.k_ind`
- `tier1.kernel.L_s`
- `tier1.kernel.r_core`
- `tier1.kernel.alpha_h`
- `tier1.kernel.p`
- `tier1.matching.mode: exact | reduced_controller_vs_richer_plant`

No controller logic should change in this step.

### Step 2: add new model modules

New files:

- `src/model/wake_geometry.py`
- `src/model/coupling_tier1.py`

Responsibilities:

- compute pairwise geometry descriptors,
- compute Tier-1 pairwise kernel values,
- expose active directed pairwise contributions in a reusable form.

Do not modify `src/control/closed_loop.py` yet beyond optional imports guarded by config.

### Step 3: extend plant dynamics

Primary file:

- `src/model/dynamics.py`

Change:

- dispatch `delta_accel` by `system.plant_family`,
- keep the state layout unchanged for the first Tier-1 step,
- preserve `f(x,u,params)` signature.

### Step 4: extend graph construction

Primary file:

- `src/model/graph.py`

Goal:

- keep `coupling_graph(...)` API stable,
- ensure graph edges are built from the Tier-1 active interaction rule under config dispatch.

### Step 5: extend evaluator layer

Primary file:

- `src/flatness/evaluation_operator.py`

Change:

- preserve public evaluator API,
- switch pairwise weight computation to a plant-family dispatch,
- for Tier-1 exact-matched mode, make `E_\pi^{\mathrm{T1}}` identical to the plant-side coupling aggregation.

This file remains the abstraction boundary for order-dependent coupling evaluation.

### Step 6: keep recursion algebra unchanged if matched family preserved

Primary file:

- `src/flatness/recursion.py`

Desired outcome:

- no structural redesign,
- `build_phi(..., evaluator_output=...)` continues to consume the Tier-1 evaluator vector,
- `phi[(i,3)] = v_i - E_i` remains valid.

If this step requires a different formula, the Tier-1 design has already violated the preferred matched family and the project should stop for a Gate-2 reconsideration.

### Step 7: add Tier-1 verification diagnostics

Primary files:

- `src/verify/gate3_operator_mismatch.py`
- `src/verify/gateB_applied_jump_bound.py`
- `src/verify/gate2_flat_output_rank.py`
- `src/verify/gateR_envelope_invariance.py`

Required changes:

- dispatch reported kernel constants by plant family,
- add Tier-1 geometry/model-mismatch fields,
- reinterpret Gate 2 as matched-inversion well-posedness for the first Tier-1 step.

### Step 8: add Tier-0 vs Tier-1 comparison experiment

New file:

- `src/experiments/tier1_transition_compare.py`

Purpose:

- compare Tier-0 and Tier-1 under the same selector/monitor settings,
- ensure the plant transition is attributable before any controller redesign.

## What Must Remain Frozen During The Tier-1 Transition

The following should stay frozen during the first Tier-1 coding phase:

1. Selector logic in `src/hybrid/order_selection.py`
2. Runtime monitor policy in `src/hybrid/runtime_monitor.py`
3. Candidate-generation richness
4. Blend law and applied-jump analysis
5. `simulate_closed_loop(...)` high-level calling pattern
6. Existing Tier-0 gates and their baseline artifacts

Reason:

If plant and controller logic change simultaneously, attribution is lost.
The Tier-1 phase must isolate plant/evaluator informativeness from controller-policy changes.

## Proven vs Empirical Status Table

### Proven/exact under exact matched Tier-1 implementation

- controller reconstruction formula `u^\pi = v - E_\pi^{\mathrm{T1}}`,
- same-snapshot raw mismatch identity,
- controller-level applied-jump decomposition,
- any operator-gap bound proved directly for the Tier-1 kernel on the envelope.

### Empirical / model-dependent

- whether the Tier-1 reduced kernel is aerodynamically informative enough,
- whether the chosen geometry descriptors are sufficient,
- whether envelope preservation improves or worsens,
- whether selector/monitor behavior becomes more meaningful under Tier-1.

### Approximate if controller and plant are not matched

- interpretation of operator-gap as true plant mismatch,
- interpretation of raw mismatch as actual closed-loop acceleration mismatch,
- any theorem using controller evaluator in place of plant coupling without a residual term.

## Recommended Future Implementation Order

This is the exact implementation order to follow later:

1. `configs/system.yaml` Tier-1 config fields
2. `src/model/wake_geometry.py`
3. `src/model/coupling_tier1.py`
4. `src/model/dynamics.py` dispatch
5. `src/model/graph.py` dispatch
6. `src/flatness/evaluation_operator.py` dispatch
7. `src/flatness/recursion.py` verification that no algebra change is needed
8. Tier-1 verification extensions
9. Tier-0 vs Tier-1 comparison experiment

If step 7 fails, stop and revisit the plant family before touching selector/monitor logic.

## Immediate Recommendation

The next coding phase should not be more monitor tuning.
It should be the first exact-matched Tier-1 implementation with:

- unchanged second-order channel structure,
- unchanged controller reconstruction algebra,
- richer geometry-aware pairwise evaluator/coupling,
- and explicit Tier-0 vs Tier-1 verification outputs.

That is the cleanest way to test whether the current bottleneck is the surrogate itself rather than local controller policy.

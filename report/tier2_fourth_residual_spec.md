# Tier-2 Fourth Residual Spec

## Purpose

The first three Tier-2 residual families now define the limit of the current partial-unmatchedness story:
- `transverse_skew` perturbs mismatch through lateral asymmetry
- `longitudinal_bias` strengthens mismatch through a persistent longitudinal window
- `edge_band_bias` localizes mismatch near static support-boundary regions

All three families remain plant-side only, structured, and scientifically useful. None of them changes switching burden under the frozen controller. The next mechanism question is therefore sharper:

> Is the missing unmatchedness mechanism transition-local asymmetry during support entry and exit, rather than another static spatial bias?

The fourth residual family answers that question.

## Residual Family

Implementation name:
- `support_transition_bias`

Scientific role:
- plant-side only
- current-state only in the first implementation
- no plant memory or hysteresis state machine
- nominal controller semantics unchanged
- support-shell localized rather than globally active
- transition-sensitive through signed boundary-distance rates

## Nominal Support Geometry

This family binds to the existing Tier-1 nominal support set already used in the code:
- longitudinal support: `gamma_edge < ds < wake_Rx`
- radial support: `transverse_radius <= tier1.edge.transverse_radius`

Let:
- `ell = ds`
- `ell_min = gamma_edge`
- `ell_max = wake_Rx`
- `rho = transverse_radius`
- `rho_max = tier1.edge.transverse_radius`

Then the signed distances to the nominal support boundaries are:

- longitudinal nearest-boundary distance:
  `d_ell = min(ell - ell_min, ell_max - ell)`
- radial boundary distance:
  `d_rho = rho_max - rho`

Sign convention:
- positive: inside nominal support
- zero: on nominal support boundary
- negative: outside nominal support

## Crossing-Direction Proxies

Let `ell_dot` be the relative longitudinal velocity already available from the current state:
- `ell_dot = xdot_leader - xdot_follower`

The nearest-boundary longitudinal distance rate is:
- `d_ell_dot = ell_dot` if the lower boundary is nearer
- `d_ell_dot = -ell_dot` if the upper boundary is nearer

So:
- `d_ell_dot > 0` means entry-like or deeper-into-support motion
- `d_ell_dot < 0` means exit-like motion toward or through the nearest longitudinal boundary

The current scaffold uses static lateral and vertical offsets, so true radial crossing velocities are unavailable in the first implementation. For that reason the radial crossing proxy is kept explicit and zero:
- `d_rho_dot = 0`

This is not hidden. It means the first implementation is expected to be longitudinal-transition dominated, while still retaining a radial shell gate and radial boundary typing in the audit.

## Shell Definitions

The family acts on a thin near-support shell rather than only on already-active edges.

Smooth shell gates:
- `g_ell = sigmoid(k_shell * (d_ell + m_ell))`
- `g_rho = sigmoid(k_shell * (d_rho + m_rho))`

Boundary-centered windows:
- `w_ell = exp(-(d_ell^2) / (2 sigma_ell^2))`
- `w_rho = exp(-(d_rho^2) / (2 sigma_rho^2))`

These keep the residual strongest near nominal support boundaries, including slightly outside the support set.

## Residual Definition

For ordered pair `(j, i)`, define the pairwise residual contribution:

`r_ji^STB = A_stb * eta_ji * (alpha_ell * g_rho * w_ell * phi_ell + alpha_rho * g_ell * w_rho * phi_rho)`

with:
- `phi_ell = tanh(k_v * d_ell_dot)`
- `phi_rho = tanh(k_v * d_rho_dot)`
- default `eta_ji = 1`

The first implementation also multiplies this by a bounded nominal-geometry activity scale derived from the existing Tier-1 kernel conventions. This keeps the residual tied to the nominal geometry without forcing it to vanish just outside support.

The subsystem residual is the sum of pairwise contributions over ordered pairs near the support shell.

## Parameters

Config surface:
- `support_transition_sigma_ell`
- `support_transition_sigma_rho`
- `support_transition_m_ell`
- `support_transition_m_rho`
- `support_transition_alpha_ell`
- `support_transition_alpha_rho`
- `support_transition_k_v`
- `support_transition_k_shell`
- `support_transition_rho_core`
- `support_transition_rho_floor`
- `support_transition_shell_tol`
- `support_transition_gain`

Interpretation:
- `sigma_*`: shell widths
- `m_*`: small outside-shell margins
- `alpha_*`: longitudinal vs radial weighting
- `k_v`: transition-modulator sharpness
- `k_shell`: shell-gate sharpness
- `rho_core`, `rho_floor`: numerical regularization
- `support_transition_gain`: mode-specific gain multiplier inside the global Tier-2 residual amplitude

## Expected Signatures

### Generic rollout
Likely weak or mixed. If the baseline trajectory does not spend time entering or exiting the support shell, the family may remain count-neutral on the generic rollout.

### Designed scenarios
The family should be strongest on:
- slow longitudinal entry
- slow longitudinal exit
- grazing or chatter near support thresholds
- mixed cases where one edge is near entry while another is near exit
- near-switch cases with opposed transition directions

### Pattern audit
The family is only worth keeping if the audit shows more than generic near-switch localization. The expected distinct signature is:
- divergence steps concentrated near support shells
- strong alignment with crossing-direction signal
- meaningful counts by nearest-boundary type
- transition-alignment stronger than candidate-history fractions

## Non-Claims

This phase does not claim:
- external-model realism
- controller improvement
- burden conversion in general
- radial-transition fidelity beyond the current static-offset scaffold

The first implementation is a disciplined mechanism probe, not a final plant model.

## Keep / Kill Criteria

### Hard requirements
- residual-disabled Tier-2 still reduces exactly to Tier-1 nominal
- frozen controller interfaces stay semantically unchanged
- family is live and nontrivial in the gap witness
- geometry/sign conventions bind to the existing nominal support set

### Strong success
Any of:
- burden conversion on designed scenarios
- burden conversion on seed sweep
- changed transition-start count under designed support-transition cases

### Count-neutral but worth keeping
Keep the family if:
- `fraction_transition_aligned >= 0.65`
- and transition-aligned fraction exceeds both candidate-history and effective-candidate-history fractions by at least `0.20`
- and the audit shows clear boundary-type / crossing-direction structure beyond generic near-switch localization

### Kill criteria
Kill the family if:
- the gap witness is trivial
- generic and designed cases are both inert
- the audit looks like `edge_band_bias` with relabeled shell distance
- meaningful controller-side semantic changes would be required

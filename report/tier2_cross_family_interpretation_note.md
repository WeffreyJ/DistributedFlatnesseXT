# Tier-2 Cross-Family Interpretation Note

## Purpose

This note consolidates the completed Tier-2 residual-family ladder into one stable interpretation.
It is a synthesis artifact.
It does not change code behavior.

## Family summary

### A. `transverse_skew`

This is the first live Tier-2 mismatch family.
Its main value is historical and structural: it established that plant-side residual mismatch can produce measurable nominal-vs-plant divergence and designed-case timing/pattern sensitivity without burden conversion.

### B. `longitudinal_bias`

This is the strongest generic mismatch/severity family in the current Tier-2 ladder.
It increases generic mismatch and jump/tracking severity more clearly than family A, and its designed cases broaden pattern relevance, but burden remains unchanged under the frozen controller.

### C. `edge_band_bias`

This is the clearest support-boundary-localized family.
Its main contribution is mechanism specificity on designed cases: the pattern audit shows boundary-local switch-timing relevance even though generic rollouts remain near-inert and burden stays unchanged.

### D. `support_transition_bias`

This is the final justified Tier-2 family in the current ladder.
It remains count-neutral, but it adds a new mechanism signature: transition-aligned divergence concentrated on support entry/exit rather than only static shell occupancy.

## Cross-family interpretation

Across families A/B/C/D, the stable Tier-2 reading is:

- structured plant-side unmatchedness reliably creates nominal-vs-plant mismatch,
- generic burden remains unchanged under the frozen controller,
- designed cases reveal timing/pattern sensitivity before burden conversion,
- and the family search now spans four distinct mechanism classes:
  - lateral asymmetry,
  - longitudinal window bias,
  - static support-boundary localization,
  - support-transition alignment.

## Why family D closes the search by default

Family D does not produce burden conversion, but it does produce a new transition-alignment signature that family C did not capture.
That is enough to justify keeping it in the ladder.
It is not enough to justify a fifth residual family by default.

The default next move is therefore not another residual search.
It is to treat Tier-2 as a completed four-family ladder unless a later phase introduces a sharply different external-model-grounded unmatchedness question.

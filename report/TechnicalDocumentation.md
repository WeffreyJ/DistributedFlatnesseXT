# Technical Documentation Source

This repository implementation is based on the technical draft:

- `report/DistributedFlatnessPaperTechnicalDraft.pdf`

Title referenced in the draft:

- **Technical Report: Hybrid State-Dependent Ordering for Distributed Flatness**

This file is the canonical documentation pointer so future changes can be traced back to the source draft.

## Wake Surrogate Mode

The repository includes an additional coupling mode:

- `system.coupling_mode: "wake_surrogate"`

Interpretation (1D along-track surrogate):

- `x1_i`: along-track position
- `x2_i`: along-track velocity
- `u_i`: along-track acceleration command

Active wake edges are leader-to-follower only:

- `(j -> i)` active when `gamma_edge < (x_j - x_i) < wake_Rx`

This enforces DAG-consistent monotonicity with ordering metric `s_i = x1_i`.

Wake disturbance surrogate:

- `Delta_i = -sum_j k_wake * exp(-(x_j - x_i)/wake_decay_L)` over active `(j -> i)` edges.

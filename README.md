# Hybrid Flatness Extension (Verification-First)

Verification-first reference implementation of the toy hybrid state-dependent ordering model from the technical draft.

## Quickstart

```bash
cd hybrid_flatness_ext
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run gates in order:

```bash
python -m src.verify.reachset --config configs/system.yaml
python -m src.verify.gate1_graph --config configs/system.yaml
python -m src.verify.gate2_flat_output_rank --config configs/system.yaml
python -m src.verify.gate3_constants --config configs/system.yaml
python -m src.verify.gate3p5_satisfiability --config configs/system.yaml
python -m src.verify.gate4_stability_inequality --config configs/experiments.yaml
```

Artifacts are written under `results/`.

## Math-to-Code Mapping

- Inverse maps `h_k^i` (implicit): [`src/flatness/recursion.py`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/src/flatness/recursion.py)
- Recursion maps `Phi_k^i`: [`src/flatness/recursion.py`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/src/flatness/recursion.py)
- Coupling terms `Delta_k^i`: [`src/model/coupling.py`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/src/model/coupling.py)
- Reconstruction `Psi = Phi_{r+1}`: [`src/flatness/recursion.py`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/src/flatness/recursion.py)
- Ordering metric `s_i(x)`, permutation `pi(x)`, hysteresis + lockout: [`src/hybrid/ordering.py`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/src/hybrid/ordering.py)
- Active coupling graph `G(x)`, edge set `E(x)`: [`src/model/graph.py`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/src/model/graph.py), [`src/model/coupling.py`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/src/model/coupling.py)

## Notes

- Plant dynamics include coupling in acceleration (`x2dot = u + Delta`), and inversion subtracts `Delta`.
- TODOs for discrete-time lockout, noise, and blending details are embedded in relevant modules.

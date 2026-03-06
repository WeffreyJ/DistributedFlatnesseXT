# Hybrid Flatness Extension (Verification-First)

Verification-first reference implementation of the toy hybrid state-dependent ordering model from the technical draft.

## Full Documentation

For a complete reviewer walkthrough (environment setup, gate pipeline, CLI/Python usage, plotting, artifact/hash checks):

- [`REPRODUCIBILITY_GUIDE.md`](/Users/jeffreywalker/Downloads/DistributedFlatnessExtension/hybrid_flatness_ext/REPRODUCIBILITY_GUIDE.md)

## Quickstart

```bash
cd hybrid_flatness_ext
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run gates in order:

```bash
python3 run_gate.py --gate R --samples 25 --method bbox
python3 run_gate.py --gate 1
python3 run_gate.py --gate 2
python3 run_gate.py --gate 3
python3 run_gate.py --gate 3.5
python3 run_experiments.py --config configs/experiments.yaml
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

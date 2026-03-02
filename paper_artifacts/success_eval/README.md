# Success Evaluation Bundle

## Claim Summary
- π-upstream wake surrogate yields large raw mismatch (`J_raw ≈ 0.657`) and without blending applied jumps are large (`J_no_blend ≈ 0.688`).
- With blending enabled, applied jumps reduce (`J_blend ≈ 0.031`) and jump-ratio at `tau_d=0, noise=0` drops from `1.047` to `0.087`.

## Contents
- `gate1/`: DAG/topological compatibility and switching well-posedness artifacts.
- `gate3/`: constants table and jump diagnostics.
- `gate4/`: Monte-Carlo trend CSV and plots (`J`, `jump_ratio`, switch metrics, spike metrics).
- `baseline_demo/`: fixed-order/no-switch baseline vs hybrid switching+blending comparison.
- `snapshot/`: exact configs, git state, and key source-file hashes used to generate this bundle.

## Reproduce
```bash
python -m tools.export_success_eval --out paper_artifacts/success_eval
```

Optional flags:
- `--no_run_gates` to reuse existing `results/` files.
- `--no_baseline_demo` to skip the baseline comparison run.

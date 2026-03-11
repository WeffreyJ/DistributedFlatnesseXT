"""Convenience dispatcher for verification gates."""

from __future__ import annotations

import argparse
from pathlib import Path


def _norm_gate(text: str) -> str:
    tok = text.strip().lower().replace("gate", "")
    if tok in {"r", "reachset"}:
        return "R"
    if tok in {"r_env", "rcheck", "renv"}:
        return "R_ENV"
    if tok in {"1", "g1"}:
        return "1"
    if tok in {"2", "g2"}:
        return "2"
    if tok in {"3", "g3"}:
        return "3"
    if tok in {"3_op", "3op", "g3op", "op", "operator_mismatch"}:
        return "3_OP"
    if tok in {"3.5", "3p5", "g3p5"}:
        return "3.5"
    if tok in {"4", "g4"}:
        return "4"
    if tok in {"s", "gates"}:
        return "S"
    if tok in {"b", "gateb"}:
        return "B"
    if tok in {"b_bound", "bbound", "gatebbound", "blend_bound"}:
        return "B_BOUND"
    if tok == "all":
        return "ALL"
    raise ValueError(f"Unsupported gate selector: {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single verification gate (or all gates).")
    parser.add_argument("--gate", required=True, help="R|R_env|1|2|3|3_op|3.5|4|S|B|B_bound|all")
    parser.add_argument("--config", default="configs/system.yaml", help="System config path")
    parser.add_argument(
        "--experiments_config",
        default="configs/experiments.yaml",
        help="Experiments config path for Gate 4",
    )
    parser.add_argument("--samples", type=int, default=None, help="Override Monte Carlo count for Gate R (reachset)")
    parser.add_argument("--method", default=None, help="Reachset method metadata (e.g., bbox)")
    parser.add_argument("--quantiles", type=float, nargs=2, default=None, metavar=("Q_LOW", "Q_HIGH"))
    parser.add_argument("--export_negative_vignette", action="store_true", help="Enable Gate 1 vignette export")
    parser.add_argument("--vignette_out_dir", default=None)
    parser.add_argument("--vignette_prefer", default=None, choices=["cycle", "topo", "either"])
    parser.add_argument("--vignette_graph_mode", default=None, choices=["sim", "physical_forward_only", "physical_all_edges"])
    parser.add_argument("--vignette_window", type=int, default=None)
    parser.add_argument("--vignette_max_episodes", type=int, default=None)
    args = parser.parse_args()

    gate = _norm_gate(args.gate)
    paths: list[Path] = []

    if gate == "R":
        from src.verify.reachset import run_reachset

        paths.append(
            run_reachset(
                cfg_path=args.config,
                samples=args.samples,
                method=args.method,
                quantiles=[float(args.quantiles[0]), float(args.quantiles[1])] if args.quantiles is not None else None,
            )
        )
    elif gate == "R_ENV":
        from src.verify.gateR_envelope_invariance import run_gateR

        paths.append(run_gateR(args.config))
    elif gate == "1":
        from src.verify.gate1_graph import run_gate1

        paths.append(
            run_gate1(
                cfg_path=args.config,
                export_negative_vignette=args.export_negative_vignette if args.export_negative_vignette else None,
                vignette_out_dir=args.vignette_out_dir,
                vignette_prefer=args.vignette_prefer,
                vignette_window=args.vignette_window,
                vignette_max_episodes=args.vignette_max_episodes,
                vignette_graph_mode=args.vignette_graph_mode,
            )
        )
    elif gate == "2":
        from src.verify.gate2_flat_output_rank import run_gate2

        paths.append(run_gate2(args.config))
    elif gate == "3":
        from src.verify.gate3_constants import run_gate3

        paths.append(run_gate3(args.config))
    elif gate == "3_OP":
        from src.verify.gate3_operator_mismatch import run_gate3_operator_mismatch

        paths.append(run_gate3_operator_mismatch(args.config))
    elif gate == "3.5":
        from src.verify.gate3p5_satisfiability import run_gate35

        paths.append(run_gate35(args.config))
    elif gate == "4":
        from src.verify.gate4_stability_inequality import run_gate4

        paths.append(run_gate4(args.experiments_config))
    elif gate == "S":
        from src.verify.gateS_sensitivity import run_gateS

        paths.append(run_gateS(args.config))
    elif gate == "B":
        from src.verify.gateB_blend_transient import run_gateB

        paths.append(run_gateB(args.config))
    elif gate == "B_BOUND":
        from src.verify.gateB_applied_jump_bound import run_gateB_applied_jump_bound

        paths.append(run_gateB_applied_jump_bound(args.config))
    elif gate == "ALL":
        from src.verify.gate1_graph import run_gate1
        from src.verify.gate2_flat_output_rank import run_gate2
        from src.verify.gate3_constants import run_gate3
        from src.verify.gate3p5_satisfiability import run_gate35
        from src.verify.gate4_stability_inequality import run_gate4
        from src.verify.gateR_envelope_invariance import run_gateR
        from src.verify.reachset import run_reachset

        paths.append(
            run_reachset(
                cfg_path=args.config,
                samples=args.samples,
                method=args.method,
                quantiles=[float(args.quantiles[0]), float(args.quantiles[1])] if args.quantiles is not None else None,
            )
        )
        paths.append(run_gateR(args.config))
        paths.append(
            run_gate1(
                cfg_path=args.config,
                export_negative_vignette=args.export_negative_vignette if args.export_negative_vignette else None,
                vignette_out_dir=args.vignette_out_dir,
                vignette_prefer=args.vignette_prefer,
                vignette_window=args.vignette_window,
                vignette_max_episodes=args.vignette_max_episodes,
                vignette_graph_mode=args.vignette_graph_mode,
            )
        )
        paths.append(run_gate2(args.config))
        paths.append(run_gate3(args.config))
        paths.append(run_gate35(args.config))
        paths.append(run_gate4(args.experiments_config))

    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()

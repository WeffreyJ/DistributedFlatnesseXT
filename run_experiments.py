"""Convenience entrypoint for experiment sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment sweeps and optional audits.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Experiments config (Gate 4)")
    parser.add_argument(
        "--run_gate4_selftest",
        action="store_true",
        help="Run Gate 4 self-test after sweep and fail on assertion failures",
    )
    parser.add_argument("--run_metric_ablation", action="store_true", help="Run metric ablation audit")
    parser.add_argument("--metric_config", default="configs/system.yaml", help="System config for metric ablation")
    parser.add_argument("--metric_out", default="results/metric_ablation", help="Metric ablation output directory")
    parser.add_argument(
        "--export_success_eval",
        action="store_true",
        help="Export paper_artifacts-style bundle after experiments",
    )
    parser.add_argument("--success_eval_out", default="paper_artifacts/success_eval")
    parser.add_argument("--no_baseline_demo", action="store_true")
    parser.add_argument("--no_run_gates", action="store_true")
    args = parser.parse_args()

    from src.verify.gate4_stability_inequality import run_gate4

    gate4_csv = run_gate4(args.config)
    print(f"Wrote {gate4_csv}")

    if args.run_gate4_selftest:
        from src.verify.gate4_selftest import run_selftest

        rc = run_selftest(args.config)
        if rc != 0:
            raise SystemExit(rc)

    if args.run_metric_ablation:
        from src.experiments.metric_ablation import run_metric_ablation

        metric_csv = run_metric_ablation(args.metric_config, out=args.metric_out)
        print(f"Wrote {metric_csv}")

    if args.export_success_eval:
        from tools.export_success_eval import export_success_eval

        out = export_success_eval(
            out=Path(args.success_eval_out),
            run_gates=not args.no_run_gates,
            run_baseline=not args.no_baseline_demo,
            run_metric_ablation=args.run_metric_ablation,
        )
        print(f"Wrote success-eval bundle to {out}")


if __name__ == "__main__":
    main()

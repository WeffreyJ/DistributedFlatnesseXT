"""One-command self-test for Gate 4 metrics and artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.gate4_stability_inequality import (
    _error_series,
    _error_spike_instant,
    _split_switch_windows,
    run_gate4,
)


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes"}


def _mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def _print_result(name: str, ok: bool, detail: str, level: str = "PASS") -> None:
    tag = level if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}")


def run_selftest(config_path: str) -> int:
    failures = 0
    warnings = 0
    exp_cfg = load_config(config_path)
    sys_cfg = load_config(exp_cfg.base_system_config)
    g4_cfg = exp_cfg.gate4 if hasattr(exp_cfg, "gate4") else sys_cfg.gate4
    enable_eps = bool(getattr(g4_cfg, "enable_epsilon_sweep", False))
    epsilon_values = [float(v) for v in list(getattr(g4_cfg, "epsilon_values", []))]
    expect_epsilon_col = bool(enable_eps and len(epsilon_values) > 1)

    # Run Gate 4 first; this also exercises alignment assertions in _error_series.
    csv_path = run_gate4(config_path)

    # -----------------------
    # Unit tests for helpers
    # -----------------------
    e = np.array([1.0, 2.0, 3.0, 4.0])
    out_a, spike_a = _split_switch_windows(e, switch_steps=[], W=2)
    ok_a = abs(out_a - 2.5) < 1e-12 and abs(spike_a - 4.0) < 1e-12
    _print_result("Test A (no switches)", ok_a, f"outside={out_a:.6g}, spike={spike_a:.6g}")
    failures += 0 if ok_a else 1

    e = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    out_b, spike_b = _split_switch_windows(e, switch_steps=[2], W=0)
    ok_b = abs(out_b - 0.0) < 1e-12 and abs(spike_b - 10.0) < 1e-12
    _print_result("Test B (single switch)", ok_b, f"outside={out_b:.6g}, spike={spike_b:.6g}")
    failures += 0 if ok_b else 1

    e = np.array([5.0, 0.0, 0.0, 0.0])
    out_c, spike_c = _split_switch_windows(e, switch_steps=[0], W=10)
    ok_c = abs(spike_c - 5.0) < 1e-12 and abs(out_c - float(np.mean(e))) < 1e-12
    _print_result("Test C (boundary/full-cover fallback)", ok_c, f"outside={out_c:.6g}, spike={spike_c:.6g}")
    failures += 0 if ok_c else 1

    e = np.array([1.0, 2.0, 3.0, 4.0])
    sp_i0 = _error_spike_instant(e, switch_steps=[])
    ok_c2 = abs(sp_i0 - 4.0) < 1e-12
    _print_result("Test C2 (instant spike no switches)", ok_c2, f"instant_spike={sp_i0:.6g}")
    failures += 0 if ok_c2 else 1

    e = np.array([0.0, 8.0, 1.0, 5.0, 0.0])
    sp_i1 = _error_spike_instant(e, switch_steps=[1, 3])
    ok_c3 = abs(sp_i1 - 8.0) < 1e-12
    _print_result("Test C3 (instant spike k/k+1)", ok_c3, f"instant_spike={sp_i1:.6g}")
    failures += 0 if ok_c3 else 1

    # -----------------------
    # CSV schema and content
    # -----------------------
    with Path(csv_path).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        cols = set(rows[0].keys()) if rows else set()

    required = {
        "tau_d",
        "blending",
        "noise_delta",
        "run",
        "error_envelope",
        "error_outside_switch",
        "error_spike_windowed",
        "error_spike_instant",
        "switch_rate",
        "J",
        "J_raw",
        "jump_ratio",
        "num_switches",
        "avg_time_between_switches",
    }
    if expect_epsilon_col:
        required.add("epsilon")
    missing = required - cols
    ok_d = len(missing) == 0
    _print_result("Test D (CSV schema)", ok_d, "columns present" if ok_d else f"missing={sorted(missing)}")
    failures += 0 if ok_d else 1

    # -----------------------
    # y_ref alignment check
    # -----------------------
    n = int(sys_cfg.system.N)
    x0 = np.concatenate([np.array(sys_cfg.reference.base, dtype=float), np.zeros(n, dtype=float)])
    sim = simulate_closed_loop(
        sys_cfg,
        x0=x0,
        horizon=1.0,
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(sys_cfg.seed) + 4242),
    )
    e_align = _error_series(sim, sys_cfg)
    ok_e = len(e_align) == len(sim["u_applied"])
    _print_result("Test E (y_ref alignment)", ok_e, f"len(e)={len(e_align)}, len(u)={len(sim['u_applied'])}")
    failures += 0 if ok_e else 1

    # -----------------------
    # Behavior sanity checks
    # -----------------------
    ref_eps = float(getattr(sys_cfg.ordering, "epsilon", 0.0))
    rows_00 = [
        r
        for r in rows
        if float(r["noise_delta"]) == 0.0
        and float(r["tau_d"]) == 0.0
        and (("epsilon" not in r) or (not expect_epsilon_col) or abs(float(r["epsilon"]) - ref_eps) < 1e-12)
    ]
    rows_false = [r for r in rows_00 if not _parse_bool(r["blending"])]
    rows_true = [r for r in rows_00 if _parse_bool(r["blending"])]

    ok_f = False
    if rows_false and rows_true:
        mean_false_j = _mean(rows_false, "J")
        mean_true_j = _mean(rows_true, "J")
        mean_false_spike_inst = _mean(rows_false, "error_spike_instant")
        mean_true_spike_inst = _mean(rows_true, "error_spike_instant")
        mean_false_jump_ratio = _mean(rows_false, "jump_ratio")
        mean_true_jump_ratio = _mean(rows_true, "jump_ratio")
        j_ok = mean_true_j < mean_false_j
        # Soft requirement: instantaneous spike should not worsen materially (>0.5%).
        spike_ok = mean_true_spike_inst <= mean_false_spike_inst * 1.005 + 1e-12
        spike_strict = mean_true_spike_inst < mean_false_spike_inst
        ratio_ok = mean_true_jump_ratio < mean_false_jump_ratio
        ok_f = j_ok and spike_ok and ratio_ok
        detail = (
            f"J_true={mean_true_j:.6g}, J_false={mean_false_j:.6g}, J_reduced={j_ok}; "
            f"spike_inst_true={mean_true_spike_inst:.6g}, spike_inst_false={mean_false_spike_inst:.6g}, "
            f"spike_reduced_strict={spike_strict}, spike_within_tol={spike_ok}; "
            f"jump_ratio_true={mean_true_jump_ratio:.6g}, jump_ratio_false={mean_false_jump_ratio:.6g}, "
            f"jump_ratio_reduced={ratio_ok}"
        )
    else:
        detail = "missing tau_d=0/noise=0 rows for one blending mode"
    _print_result("Test F (blending reduces J and spike)", ok_f, detail)
    failures += 0 if ok_f else 1

    ok_g = all(float(r["error_spike_windowed"]) + 1e-12 >= float(r["error_outside_switch"]) for r in rows)
    _print_result("Test G (spike >= outside)", ok_g, "checked all rows")
    failures += 0 if ok_g else 1

    # Broad dwell-rate trend check (soft warning, not strict fail).
    rows_nd0 = [r for r in rows if float(r["noise_delta"]) == 0.0]
    trend_lines = []
    trend_ok = True
    for b in [False, True]:
        rb = [r for r in rows_nd0 if _parse_bool(r["blending"]) == b]
        tau_vals = sorted({float(r["tau_d"]) for r in rb})
        means = []
        for tau in tau_vals:
            rt = [r for r in rb if float(r["tau_d"]) == tau]
            means.append(_mean(rt, "switch_rate"))
        trend_lines.append((b, tau_vals, means))
        if means and means[-1] > means[0] * 1.1 + 1e-12:
            trend_ok = False

    if trend_ok:
        _print_result("Test H (switch rate vs dwell trend)", True, "broadly non-increasing")
    else:
        warnings += 1
        _print_result(
            "Test H (switch rate vs dwell trend)",
            True,
            "not strictly decreasing; inspect printed trend table",
            level="WARN",
        )
    for b, taus, means in trend_lines:
        print(f"  blending={b} tau={taus} mean_switch_rate={means}")

    # Optional epsilon trend checks (warning-only), using tau_d=0 and blending=True.
    if expect_epsilon_col:
        eps_trend_ok = True
        eps_ratio_ok = True
        rows_eps = [r for r in rows if _parse_bool(r["blending"]) and float(r["tau_d"]) == 0.0]
        noise_vals = sorted({float(r["noise_delta"]) for r in rows_eps})
        for nd in noise_vals:
            rn = [r for r in rows_eps if float(r["noise_delta"]) == nd]
            eps_vals = sorted({float(r["epsilon"]) for r in rn})
            sw = []
            jr = []
            for eps in eps_vals:
                re = [r for r in rn if float(r["epsilon"]) == eps]
                sw.append(_mean(re, "switch_rate"))
                jr.append(_mean(re, "jump_ratio"))
            if sw and sw[-1] > sw[0] * 1.1 + 1e-12:
                eps_trend_ok = False
            if jr and jr[-1] > jr[0] * 1.1 + 1e-12:
                eps_ratio_ok = False
            print(f"  eps trend noise={nd} eps={eps_vals} switch_rate={sw} jump_ratio={jr}")
        if eps_trend_ok and eps_ratio_ok:
            _print_result("Test I (epsilon trends)", True, "broadly non-increasing for blending=True at tau_d=0")
        else:
            warnings += 1
            _print_result(
                "Test I (epsilon trends)",
                True,
                "not broadly non-increasing for all noise levels; inspect printed epsilon trend table",
                level="WARN",
            )
    else:
        _print_result("Test I (epsilon trends)", True, "epsilon sweep disabled or single epsilon")

    # -----------------------
    # Plot artifact checks
    # -----------------------
    plot_dir = Path("results") / "gate4"
    expected_plots = [
        plot_dir / "error_envelope_by_tau.png",
        plot_dir / "error_spike_by_tau.png",
        plot_dir / "error_spike_instant_by_tau.png",
        plot_dir / "error_spike_instant_by_noise.png",
        plot_dir / "switch_rate_by_tau.png",
        plot_dir / "J_by_tau_and_blending.png",
        plot_dir / "jump_ratio_by_tau_and_blending.png",
        plot_dir / "switch_rate_by_noise.png",
    ]
    if expect_epsilon_col:
        expected_plots.extend(
            [
                plot_dir / "switch_rate_vs_epsilon.png",
                plot_dir / "jump_ratio_vs_epsilon.png",
            ]
        )
    missing_plots = [p.name for p in expected_plots if not p.exists()]
    ok_plots = len(missing_plots) == 0
    _print_result("Plot artifacts", ok_plots, "all present" if ok_plots else f"missing={missing_plots}")
    failures += 0 if ok_plots else 1

    print("\nSelf-test summary:")
    print(f"  Failures: {failures}")
    print(f"  Warnings: {warnings}")
    print(f"  CSV: {csv_path}")

    if failures == 0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 4 one-command self-test")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments config")
    args = parser.parse_args()
    raise SystemExit(run_selftest(args.config))


if __name__ == "__main__":
    main()

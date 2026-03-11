"""Tier-2 reduction witness: residual-disabled Tier-2 must reduce to Tier-1 nominal."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.flatness.evaluation_operator import compute_evaluator
from src.model.coupling import delta_accel, plant_delta_accel, residual_delta_accel
from src.verify.utils import dump_json, make_results_dir


TOL = 1.0e-12


def _clone_cfg(cfg, plant_family: str):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.system.plant_family = str(plant_family)
    if plant_family == "tier2":
        cfg_mode.system.tier2.residual.enabled = False
        cfg_mode.system.tier2.residual.amplitude = 0.0
    return cfg_mode


def _base_x0(cfg) -> np.ndarray:
    n = int(cfg.system.N)
    return np.concatenate(
        [
            np.array(cfg.reference.base, dtype=float),
            np.zeros(n, dtype=float),
        ],
        axis=0,
    )


def _metrics(sim: dict[str, object]) -> dict[str, float | int]:
    e_norm = np.asarray(sim.get("e_norm", [0.0]), dtype=float)
    return {
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in sim.get("switch_reason", []) if str(reason) == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
    }


def _allclose(a, b, tol: float = TOL) -> bool:
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=0.0))


def run_tier2_reduction_witness(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/tier2_reduction_witness",
) -> Path:
    cfg = load_config(config_path)
    out = (
        make_results_dir("tier2_reduction_witness")
        if out_dir == "results/tier2_reduction_witness"
        else Path(out_dir)
    )
    out.mkdir(parents=True, exist_ok=True)

    cfg_tier1 = _clone_cfg(cfg, "tier1")
    cfg_tier2 = _clone_cfg(cfg, "tier2")

    x_witness = np.array([0.09, 0.0, -0.21, 0.02, -0.01, 0.03], dtype=float)
    pi_witness = [0, 1, 2]
    eval_mode = str(getattr(getattr(cfg, "evaluation", None), "mode", "upstream_truncated"))

    E_tier1 = np.asarray(compute_evaluator(x_witness, pi_witness, cfg_tier1, mode=eval_mode), dtype=float)
    E_tier2 = np.asarray(compute_evaluator(x_witness, pi_witness, cfg_tier2, mode=eval_mode), dtype=float)
    delta_tier1 = np.asarray(delta_accel(x_witness, cfg_tier1.system), dtype=float)
    delta_tier2_nom = np.asarray(delta_accel(x_witness, cfg_tier2.system), dtype=float)
    residual_tier2 = np.asarray(residual_delta_accel(x_witness, cfg_tier2.system), dtype=float)
    plant_delta_tier2 = np.asarray(plant_delta_accel(x_witness, cfg_tier2.system), dtype=float)

    pointwise = {
        "x_witness": [float(v) for v in x_witness],
        "pi_witness": [int(v) for v in pi_witness],
        "evaluation_mode": eval_mode,
        "pointwise_nominal_match": bool(_allclose(E_tier1, E_tier2) and _allclose(delta_tier1, delta_tier2_nom)),
        "pointwise_residual_zero": bool(_allclose(residual_tier2, np.zeros_like(residual_tier2))),
        "pointwise_plant_matches_nominal": bool(_allclose(plant_delta_tier2, delta_tier2_nom)),
        "tier1_evaluator": [float(v) for v in E_tier1],
        "tier2_evaluator": [float(v) for v in E_tier2],
        "tier1_nominal_delta": [float(v) for v in delta_tier1],
        "tier2_nominal_delta": [float(v) for v in delta_tier2_nom],
        "tier2_residual_delta": [float(v) for v in residual_tier2],
        "tier2_plant_delta": [float(v) for v in plant_delta_tier2],
    }

    x0 = _base_x0(cfg)
    sim_seed = int(cfg.seed) + 20621
    options = SimOptions(blending_on=True, noise_delta=0.0, seed=sim_seed, disable_switching=False)
    sim_tier1 = simulate_closed_loop(cfg_tier1, x0=x0, horizon=float(cfg_tier1.system.horizon), options=options)
    sim_tier2 = simulate_closed_loop(cfg_tier2, x0=x0, horizon=float(cfg_tier2.system.horizon), options=options)

    metrics_tier1 = _metrics(sim_tier1)
    metrics_tier2 = _metrics(sim_tier2)
    metric_checks = {key: bool(_allclose(metrics_tier1[key], metrics_tier2[key])) for key in metrics_tier1}
    rollout_checks = {
        "switch_steps_match": bool(list(sim_tier1.get("switch_steps", [])) == list(sim_tier2.get("switch_steps", []))),
        "transition_reasons_match": bool(list(sim_tier1.get("switch_reason", [])) == list(sim_tier2.get("switch_reason", []))),
        "pi_candidate_match": bool(list(sim_tier1.get("pi_candidate", [])) == list(sim_tier2.get("pi_candidate", []))),
        "effective_pi_candidate_match": bool(list(sim_tier1.get("effective_pi_candidate", [])) == list(sim_tier2.get("effective_pi_candidate", []))),
        "J_raw_series_match": bool(_allclose(sim_tier1.get("J_raw", []), sim_tier2.get("J_raw", []))),
        "J_series_match": bool(_allclose(sim_tier1.get("J", []), sim_tier2.get("J", []))),
        "e_norm_series_match": bool(_allclose(sim_tier1.get("e_norm", []), sim_tier2.get("e_norm", []))),
    }
    rollout_match = bool(all(metric_checks.values()) and all(rollout_checks.values()))

    payload = {
        "pointwise": pointwise,
        "rollout": {
            "shared_x0": [float(v) for v in x0],
            "shared_sim_seed": int(sim_seed),
            "tier1_metrics": metrics_tier1,
            "tier2_metrics": metrics_tier2,
            "metric_exact_match": metric_checks,
            "series_and_event_match": rollout_checks,
            "rollout_match": rollout_match,
        },
        "tier2_residual_disabled": True,
    }
    dump_json(out / "tier2_reduction_witness.json", payload)
    return out / "tier2_reduction_witness.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-2 reduction witness")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/tier2_reduction_witness")
    args = parser.parse_args()
    out = run_tier2_reduction_witness(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

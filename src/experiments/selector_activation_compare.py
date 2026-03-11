"""Compare legacy and active lexicographic selector modes on one rollout."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir


def _sample_x0(cfg, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(cfg.system.N)
    x1 = rng.uniform(float(cfg.x0.x1[0]), float(cfg.x0.x1[1]), size=n)
    x2 = rng.uniform(float(cfg.x0.x2[0]), float(cfg.x0.x2[1]), size=n)
    return np.concatenate([x1, x2], axis=0)


def _clone_with_selection_mode(cfg, mode: str):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.selection.mode = str(mode)
    return cfg_mode


def _sim_metrics(sim: dict[str, object]) -> dict[str, float | int | str]:
    switch_reason = [str(v) for v in sim.get("switch_reason", [])]
    return {
        "selection_mode": str(sim.get("selection_mode", ["unknown"])[0]) if len(sim.get("selection_mode", [])) else "unknown",
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in switch_reason if reason == "transition_start")),
        "immediate_switch_count": int(sum(1 for reason in switch_reason if reason == "immediate_switch")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "max_J_raw": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "mean_J_raw": float(np.mean(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
        "mean_applied_jump": float(np.mean(np.asarray(sim.get("J", [0.0]), dtype=float))),
        "max_E_gap_same_step": float(np.max(np.asarray(sim.get("E_gap_same_step", [0.0]), dtype=float))),
        "mean_E_gap_same_step": float(np.mean(np.asarray(sim.get("E_gap_same_step", [0.0]), dtype=float))),
        "effective_candidate_diff_steps": int(
            np.count_nonzero(np.asarray(sim.get("effective_candidate_differs_from_live", []), dtype=bool))
        ),
    }


def run_selector_activation_compare(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/selector_activation_compare",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("selector_activation_compare") if out_dir == "results/selector_activation_compare" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = _sample_x0(cfg, seed=int(cfg.seed) + 14001)
    sim_seed = int(cfg.seed) + 14011
    sim_options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    mode_payload: dict[str, dict[str, float | int | str]] = {}
    csv_rows: list[dict[str, float | int | str]] = []

    for mode in ("legacy", "active_lexicographic"):
        cfg_mode = _clone_with_selection_mode(cfg, mode)
        sim = simulate_closed_loop(
            cfg_mode,
            x0=x0,
            horizon=float(cfg_mode.system.horizon),
            options=sim_options,
        )
        metrics = _sim_metrics(sim)
        mode_payload[mode] = metrics
        csv_rows.append(metrics)

    legacy = mode_payload["legacy"]
    active = mode_payload["active_lexicographic"]
    comparison = {
        "shared_x0": [float(v) for v in x0],
        "shared_seed": int(sim_seed),
        "legacy": legacy,
        "active_lexicographic": active,
        "deltas_active_minus_legacy": {
            "switch_count": int(active["switch_count"]) - int(legacy["switch_count"]),
            "transition_start_count": int(active["transition_start_count"]) - int(legacy["transition_start_count"]),
            "immediate_switch_count": int(active["immediate_switch_count"]) - int(legacy["immediate_switch_count"]),
            "blend_active_steps": int(active["blend_active_steps"]) - int(legacy["blend_active_steps"]),
            "max_J_raw": float(active["max_J_raw"]) - float(legacy["max_J_raw"]),
            "mean_J_raw": float(active["mean_J_raw"]) - float(legacy["mean_J_raw"]),
            "max_applied_jump": float(active["max_applied_jump"]) - float(legacy["max_applied_jump"]),
            "mean_applied_jump": float(active["mean_applied_jump"]) - float(legacy["mean_applied_jump"]),
            "max_E_gap_same_step": float(active["max_E_gap_same_step"]) - float(legacy["max_E_gap_same_step"]),
            "mean_E_gap_same_step": float(active["mean_E_gap_same_step"]) - float(legacy["mean_E_gap_same_step"]),
            "effective_candidate_diff_steps": int(active["effective_candidate_diff_steps"])
            - int(legacy["effective_candidate_diff_steps"]),
        },
    }

    dump_json(out / "selector_activation_compare.json", comparison)

    with (out / "selector_activation_compare.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selection_mode",
                "switch_count",
                "transition_start_count",
                "immediate_switch_count",
                "blend_active_steps",
                "max_J_raw",
                "mean_J_raw",
                "max_applied_jump",
                "mean_applied_jump",
                "max_E_gap_same_step",
                "mean_E_gap_same_step",
                "effective_candidate_diff_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    return out / "selector_activation_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy and active selector modes")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/selector_activation_compare")
    args = parser.parse_args()
    out = run_selector_activation_compare(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Compare baseline, shadow-monitor, and active-monitor runs."""

from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir


RISK_LEVEL_NUM = {"low": 0.0, "moderate": 1.0, "high": 2.0}


def _sample_x0(cfg, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(cfg.system.N)
    x1 = rng.uniform(float(cfg.x0.x1[0]), float(cfg.x0.x1[1]), size=n)
    x2 = rng.uniform(float(cfg.x0.x2[0]), float(cfg.x0.x2[1]), size=n)
    return np.concatenate([x1, x2], axis=0)


def _clone_cfg(cfg, *, monitor_enabled: bool, monitor_mode: str):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.monitor.enabled = bool(monitor_enabled)
    cfg_mode.monitor.mode = str(monitor_mode)
    return cfg_mode


def _monitor_metrics(sim: dict[str, object], label: str) -> dict[str, float | int | str]:
    switch_reason = [str(v) for v in sim.get("switch_reason", [])]
    monitor_actions = [str(v) for v in sim.get("monitor_action", [])]
    selector_switch_eligible = np.asarray(sim.get("selector_switch_eligible", []), dtype=bool)
    rejected_mask = np.asarray(
        [
            (action in {"hold_current", "extend_lockout", "fallback_fixed_order"}) and bool(selector_switch_eligible[idx])
            for idx, action in enumerate(monitor_actions)
        ],
        dtype=bool,
    )
    tie_margin = np.asarray(sim.get("monitor_tie_margin", []), dtype=float)
    predicted_gap = np.asarray(sim.get("monitor_predicted_gap", []), dtype=float)
    switch_rate = np.asarray(sim.get("monitor_switch_rate_recent", []), dtype=float)
    e_norm = np.asarray(sim.get("e_norm", []), dtype=float)
    fallback_active = np.asarray(sim.get("fallback_active", []), dtype=bool)
    blocked_runs = 0
    blocked_lengths: list[int] = []
    run_len = 0
    for is_blocked in rejected_mask:
        if bool(is_blocked):
            run_len += 1
        elif run_len > 0:
            blocked_runs += 1
            blocked_lengths.append(run_len)
            run_len = 0
    if run_len > 0:
        blocked_runs += 1
        blocked_lengths.append(run_len)

    return {
        "label": label,
        "selection_mode": str(sim.get("selection_mode", ["unknown"])[0]) if len(sim.get("selection_mode", [])) else "unknown",
        "monitor_mode": str(sim.get("monitor_mode", ["disabled"])[0]) if len(sim.get("monitor_mode", [])) else "disabled",
        "switch_count": int(len(sim.get("switch_steps", []))),
        "transition_start_count": int(sum(1 for reason in switch_reason if reason == "transition_start")),
        "blend_active_steps": int(np.count_nonzero(np.asarray(sim.get("blend_active", []), dtype=bool))),
        "rejected_switch_count": int(np.count_nonzero(rejected_mask)),
        "blocked_steps_total": int(np.count_nonzero(rejected_mask)),
        "unique_blocked_transition_starts": int(blocked_runs),
        "mean_block_duration_steps": float(np.mean(blocked_lengths)) if blocked_lengths else 0.0,
        "hold_action_count": int(sum(1 for action in monitor_actions if action in {"hold_current", "extend_lockout"})),
        "fallback_activation_count": int(np.count_nonzero(fallback_active)),
        "mean_tie_margin_at_rejections": float(np.mean(tie_margin[rejected_mask])) if np.any(rejected_mask) else 0.0,
        "mean_predicted_gap_at_rejections": float(np.mean(predicted_gap[rejected_mask])) if np.any(rejected_mask) else 0.0,
        "mean_switch_rate_recent_at_rejections": float(np.mean(switch_rate[rejected_mask])) if np.any(rejected_mask) else 0.0,
        "tracking_error_mean": float(np.mean(e_norm)) if e_norm.size else 0.0,
        "tracking_error_max": float(np.max(e_norm)) if e_norm.size else 0.0,
        "max_raw_jump": float(np.max(np.asarray(sim.get("J_raw", [0.0]), dtype=float))),
        "max_applied_jump": float(np.max(np.asarray(sim.get("J", [0.0]), dtype=float))),
    }


def _plot_action_counts(payload: dict[str, dict[str, object]], out_path: Path) -> None:
    labels = list(payload.keys())
    actions = sorted({action for item in payload.values() for action in item["action_counts"]})
    x = np.arange(len(labels))
    width = 0.8 / max(len(actions), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, action in enumerate(actions):
        vals = [int(payload[label]["action_counts"].get(action, 0)) for label in labels]
        ax.bar(x + idx * width - 0.4 + width / 2.0, vals, width=width, label=action)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("count")
    ax.set_title("Monitor action counts")
    if actions:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_risk_timeline(raw_sims: dict[str, dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, sim in raw_sims.items():
        levels = [RISK_LEVEL_NUM.get(str(v), 0.0) for v in sim.get("monitor_risk_level", [])]
        t = np.asarray(sim.get("t_control", np.arange(len(levels))), dtype=float)
        ax.plot(t[: len(levels)], levels, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("risk level")
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["low", "moderate", "high"])
    ax.set_title("Runtime-monitor risk level timeline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_switches_vs_rejections(metrics_rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    labels = [str(row["label"]) for row in metrics_rows]
    x = np.arange(len(labels))
    width = 0.35
    switches = [int(row["switch_count"]) for row in metrics_rows]
    rejected = [int(row["rejected_switch_count"]) for row in metrics_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2.0, switches, width=width, label="switch_count")
    ax.bar(x + width / 2.0, rejected, width=width, label="rejected_switch_count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("count")
    ax.set_title("Switches vs monitor rejections")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_runtime_monitor_compare(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/runtime_monitor_compare",
) -> Path:
    cfg = load_config(config_path)
    out = make_results_dir("runtime_monitor_compare") if out_dir == "results/runtime_monitor_compare" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = _sample_x0(cfg, seed=int(cfg.seed) + 15001)
    sim_seed = int(cfg.seed) + 15021
    options = SimOptions(
        blending_on=True,
        noise_delta=0.0,
        seed=sim_seed,
        disable_switching=False,
    )

    configs = {
        "baseline_current_controller": _clone_cfg(cfg, monitor_enabled=False, monitor_mode="shadow"),
        "shadow_monitor": _clone_cfg(cfg, monitor_enabled=True, monitor_mode="shadow"),
        "active_monitor": _clone_cfg(cfg, monitor_enabled=True, monitor_mode="active"),
    }

    raw_sims: dict[str, dict[str, object]] = {}
    payload: dict[str, dict[str, object]] = {}
    metrics_rows: list[dict[str, float | int | str]] = []

    for label, cfg_mode in configs.items():
        sim = simulate_closed_loop(
            cfg_mode,
            x0=x0,
            horizon=float(cfg_mode.system.horizon),
            options=options,
        )
        raw_sims[label] = sim
        metrics = _monitor_metrics(sim, label)
        metrics_rows.append(metrics)
        payload[label] = {
            "metrics": metrics,
            "action_counts": dict(Counter(str(v) for v in sim.get("monitor_action", []))),
        }

    summary = {
        "shared_x0": [float(v) for v in x0],
        "shared_seed": int(sim_seed),
        "runs": {label: payload[label]["metrics"] for label in payload},
    }
    dump_json(out / "runtime_monitor_compare.json", summary)

    with (out / "runtime_monitor_compare.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "selection_mode",
                "monitor_mode",
                "switch_count",
                "transition_start_count",
                "blend_active_steps",
                "rejected_switch_count",
                "blocked_steps_total",
                "unique_blocked_transition_starts",
                "mean_block_duration_steps",
                "hold_action_count",
                "fallback_activation_count",
                "mean_tie_margin_at_rejections",
                "mean_predicted_gap_at_rejections",
                "mean_switch_rate_recent_at_rejections",
                "tracking_error_mean",
                "tracking_error_max",
                "max_raw_jump",
                "max_applied_jump",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    _plot_action_counts(payload, out / "monitor_action_counts.png")
    _plot_risk_timeline(raw_sims, out / "risk_level_timeline.png")
    _plot_switches_vs_rejections(metrics_rows, out / "switches_vs_rejections.png")

    return out / "runtime_monitor_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runtime-monitor configurations")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/runtime_monitor_compare")
    args = parser.parse_args()
    out = run_runtime_monitor_compare(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

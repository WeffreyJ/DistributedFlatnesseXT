"""Audit blocked runtime-monitor intervals for calibration work."""

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


BLOCK_ACTIONS = {"hold_current", "extend_lockout", "fallback_fixed_order"}


def _sample_x0(cfg, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(cfg.system.N)
    x1 = rng.uniform(float(cfg.x0.x1[0]), float(cfg.x0.x1[1]), size=n)
    x2 = rng.uniform(float(cfg.x0.x2[0]), float(cfg.x0.x2[1]), size=n)
    return np.concatenate([x1, x2], axis=0)


def _clone_active_monitor_cfg(cfg):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.monitor.enabled = True
    cfg_mode.monitor.mode = "active"
    return cfg_mode


def _blocked_mask(sim: dict[str, object]) -> np.ndarray:
    selector_switch_eligible = np.asarray(sim.get("selector_switch_eligible", []), dtype=bool)
    monitor_actions = [str(v) for v in sim.get("monitor_action", [])]
    return np.asarray(
        [
            bool(selector_switch_eligible[idx]) and action in BLOCK_ACTIONS
            for idx, action in enumerate(monitor_actions)
        ],
        dtype=bool,
    )


def _collect_blocked_intervals(sim: dict[str, object]) -> list[dict[str, object]]:
    blocked = _blocked_mask(sim)
    t_control = np.asarray(sim.get("t_control", np.arange(len(blocked))), dtype=float)
    tie_margin = np.asarray(sim.get("monitor_tie_margin", []), dtype=float)
    predicted_gap = np.asarray(sim.get("monitor_predicted_gap", []), dtype=float)
    switch_rate = np.asarray(sim.get("monitor_switch_rate_recent", []), dtype=float)
    edge_churn = np.asarray(sim.get("monitor_edge_churn_recent", []), dtype=float)
    e_norm = np.asarray(sim.get("e_norm", []), dtype=float)
    reason_lists = [[str(r) for r in reasons] for reasons in sim.get("monitor_risk_reasons", [])]

    intervals: list[dict[str, object]] = []
    start = None
    for idx, is_blocked in enumerate(blocked):
        if is_blocked and start is None:
            start = idx
        is_end = start is not None and (not is_blocked or idx == len(blocked) - 1)
        if not is_end:
            continue

        end = idx - 1 if not is_blocked else idx
        step_slice = slice(start, end + 1)
        interval_reasons = reason_lists[step_slice]
        all_reasons_flat = [reason for reasons in interval_reasons for reason in reasons]
        primary_reason = interval_reasons[0][0] if interval_reasons and interval_reasons[0] else "unknown"
        all_reasons_union = sorted(set(all_reasons_flat)) if all_reasons_flat else ["unknown"]
        intervals.append(
            {
                "start_step": int(start),
                "end_step": int(end),
                "duration_steps": int(end - start + 1),
                "start_time": float(t_control[start]) if start < len(t_control) else float(start),
                "end_time": float(t_control[end]) if end < len(t_control) else float(end),
                "primary_reason": str(primary_reason),
                "all_reasons_union": all_reasons_union,
                "mean_tie_margin": float(np.mean(tie_margin[step_slice])) if tie_margin.size else 0.0,
                "mean_predicted_gap": float(np.mean(predicted_gap[step_slice])) if predicted_gap.size else 0.0,
                "mean_switch_rate_recent": float(np.mean(switch_rate[step_slice])) if switch_rate.size else 0.0,
                "mean_edge_churn_recent": float(np.mean(edge_churn[step_slice])) if edge_churn.size else 0.0,
                "mean_tracking_error": float(np.mean(e_norm[step_slice])) if e_norm.size else 0.0,
                "max_tracking_error": float(np.max(e_norm[step_slice])) if e_norm.size else 0.0,
            }
        )
        start = None

    return intervals


def _reason_counts_stepwise(sim: dict[str, object], blocked: np.ndarray) -> dict[str, int]:
    counts: Counter[str] = Counter()
    reason_lists = [[str(r) for r in reasons] for reasons in sim.get("monitor_risk_reasons", [])]
    for idx, is_blocked in enumerate(blocked):
        if not is_blocked:
            continue
        if idx < len(reason_lists) and reason_lists[idx]:
            counts.update(reason_lists[idx])
        else:
            counts.update(["unknown"])
    return dict(sorted(counts.items()))


def _reason_counts_unique_startwise(intervals: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(interval["primary_reason"]) for interval in intervals)
    return dict(sorted(counts.items()))


def _plot_reason_counts(stepwise: dict[str, int], startwise: dict[str, int], out_path: Path) -> None:
    reasons = sorted(set(stepwise) | set(startwise))
    x = np.arange(len(reasons))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2.0, [int(stepwise.get(r, 0)) for r in reasons], width=width, label="stepwise")
    ax.bar(x + width / 2.0, [int(startwise.get(r, 0)) for r in reasons], width=width, label="unique starts")
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=20)
    ax.set_ylabel("count")
    ax.set_title("Blocked-interval reason counts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_duration_hist(intervals: list[dict[str, object]], out_path: Path) -> None:
    durations = np.asarray([int(interval["duration_steps"]) for interval in intervals], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    if durations.size:
        ax.hist(durations, bins=min(20, max(5, durations.size)))
    else:
        ax.text(0.5, 0.5, "No blocked intervals", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("duration [steps]")
    ax.set_ylabel("count")
    ax.set_title("Blocked interval duration histogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_tracking_error_during_blocks(intervals: list[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if intervals:
        start_times = np.asarray([float(interval["start_time"]) for interval in intervals], dtype=float)
        mean_err = np.asarray([float(interval["mean_tracking_error"]) for interval in intervals], dtype=float)
        max_err = np.asarray([float(interval["max_tracking_error"]) for interval in intervals], dtype=float)
        ax.plot(start_times, mean_err, marker="o", label="mean error during block")
        ax.plot(start_times, max_err, marker="x", label="max error during block")
        ax.set_xlabel("block start time [s]")
        ax.set_ylabel("tracking error")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No blocked intervals", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Tracking error during blocked intervals")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_runtime_monitor_audit(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/runtime_monitor_audit",
) -> Path:
    cfg = load_config(config_path)
    cfg_mode = _clone_active_monitor_cfg(cfg)
    out = make_results_dir("runtime_monitor_audit") if out_dir == "results/runtime_monitor_audit" else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = _sample_x0(cfg_mode, seed=int(cfg_mode.seed) + 15101)
    sim_seed = int(cfg_mode.seed) + 15121
    sim = simulate_closed_loop(
        cfg_mode,
        x0=x0,
        horizon=float(cfg_mode.system.horizon),
        options=SimOptions(
            blending_on=True,
            noise_delta=0.0,
            seed=sim_seed,
            disable_switching=False,
        ),
    )

    blocked = _blocked_mask(sim)
    intervals = _collect_blocked_intervals(sim)
    stepwise_counts = _reason_counts_stepwise(sim, blocked)
    startwise_counts = _reason_counts_unique_startwise(intervals)
    tracking_error = np.asarray(sim.get("e_norm", []), dtype=float)

    with (out / "blocked_intervals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "start_step",
                "end_step",
                "duration_steps",
                "start_time",
                "end_time",
                "primary_reason",
                "all_reasons_union",
                "mean_tie_margin",
                "mean_predicted_gap",
                "mean_switch_rate_recent",
                "mean_edge_churn_recent",
                "mean_tracking_error",
                "max_tracking_error",
            ],
        )
        writer.writeheader()
        for interval in intervals:
            row = dict(interval)
            row["all_reasons_union"] = "|".join(str(v) for v in interval["all_reasons_union"])
            writer.writerow(row)

    blocked_error = tracking_error[blocked] if tracking_error.size and blocked.size else np.array([], dtype=float)
    summary = {
        "shared_seed": int(sim_seed),
        "shared_x0": [float(v) for v in x0],
        "blocked_steps_total": int(np.count_nonzero(blocked)),
        "unique_blocked_transition_starts": int(len(intervals)),
        "blocked_step_to_unique_start_ratio": float(np.count_nonzero(blocked) / max(len(intervals), 1)),
        "mean_block_duration_steps": float(np.mean([int(interval["duration_steps"]) for interval in intervals])) if intervals else 0.0,
        "max_block_duration_steps": int(max([int(interval["duration_steps"]) for interval in intervals], default=0)),
        "reason_counts_stepwise": stepwise_counts,
        "reason_counts_unique_startwise": startwise_counts,
        "tracking_error_during_blocks_mean": float(np.mean(blocked_error)) if blocked_error.size else 0.0,
        "tracking_error_during_blocks_max": float(np.max(blocked_error)) if blocked_error.size else 0.0,
        "top_blocked_intervals": intervals[:10],
    }
    dump_json(out / "runtime_monitor_audit_summary.json", summary)

    _plot_reason_counts(stepwise_counts, startwise_counts, out / "blocked_reason_counts.png")
    _plot_duration_hist(intervals, out / "blocked_duration_hist.png")
    _plot_tracking_error_during_blocks(intervals, out / "tracking_error_during_blocks.png")

    return out / "runtime_monitor_audit_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit blocked runtime-monitor intervals")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/runtime_monitor_audit")
    args = parser.parse_args()
    out = run_runtime_monitor_audit(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

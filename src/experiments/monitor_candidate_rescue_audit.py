"""Audit whether blocked monitor starts are rescuable by enriched local candidates."""

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
from src.experiments.runtime_monitor_audit import _blocked_mask, _sample_x0
from src.hybrid.order_selection import admissible_order, predicted_raw_gap_proxy
from src.verify.utils import dump_json, make_results_dir


def _clone_active_monitor_cfg(cfg):
    cfg_mode = copy.deepcopy(cfg)
    cfg_mode.monitor.enabled = True
    cfg_mode.monitor.mode = "active"
    return cfg_mode


def _as_pi(seq) -> list[int]:
    return [int(v) for v in seq]


def _adjacent_swaps(pi: list[int]) -> list[list[int]]:
    swaps: list[list[int]] = []
    for idx in range(len(pi) - 1):
        cand = _as_pi(pi)
        cand[idx], cand[idx + 1] = cand[idx + 1], cand[idx]
        swaps.append(cand)
    return swaps


def _tie_swap(pi: list[int], tie_pair: tuple[int, int]) -> list[int] | None:
    a_id, b_id = int(tie_pair[0]), int(tie_pair[1])
    if a_id < 0 or b_id < 0:
        return None
    out = _as_pi(pi)
    try:
        a = out.index(a_id)
        b = out.index(b_id)
    except ValueError:
        return None
    if abs(a - b) != 1:
        return None
    out[a], out[b] = out[b], out[a]
    return out


def _zeta_from_row(zeta_r_row: np.ndarray, cfg) -> dict[str, np.ndarray]:
    layout = cfg.flat.zeta_layout
    return {
        "y": np.asarray(zeta_r_row[int(layout.y[0]) : int(layout.y[1])], dtype=float),
        "ydot": np.asarray(zeta_r_row[int(layout.ydot[0]) : int(layout.ydot[1])], dtype=float),
        "v": np.asarray(zeta_r_row[int(layout.v[0]) : int(layout.v[1])], dtype=float),
    }


def _blocked_interval_starts(sim: dict[str, object]) -> list[int]:
    blocked = _blocked_mask(sim)
    starts: list[int] = []
    prev = False
    for idx, val in enumerate(blocked):
        if bool(val) and not prev:
            starts.append(int(idx))
        prev = bool(val)
    return starts


def _build_candidate_pool(
    current_pi: list[int],
    blocked_selector_candidate: list[int],
    tie_pair: tuple[int, int],
) -> list[tuple[list[int], str]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[list[int], str]] = []

    def add(pi: list[int] | None, ctype: str) -> None:
        if pi is None:
            return
        key = tuple(_as_pi(pi))
        if key in seen:
            return
        seen.add(key)
        out.append((list(key), ctype))

    add(current_pi, "current_order")
    add(blocked_selector_candidate, "blocked_selector_candidate")
    add(_tie_swap(current_pi, tie_pair), "tie_pair_swap_current")
    add(_tie_swap(blocked_selector_candidate, tie_pair), "tie_pair_swap_blocked")
    for pi in _adjacent_swaps(current_pi):
        add(pi, "adjacent_swap_current")
    for pi in _adjacent_swaps(blocked_selector_candidate):
        add(pi, "adjacent_swap_blocked")
    return out


def _plot_rescue_exists(rows: list[dict[str, object]], out_path: Path) -> None:
    yes = sum(1 for row in rows if bool(row["rescue_exists"]))
    no = len(rows) - yes
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["no_rescue", "rescue_exists"], [no, yes])
    ax.set_ylabel("count")
    ax.set_title("Blocked starts with rescue candidates")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rescue_improvement(rows: list[dict[str, object]], out_path: Path) -> None:
    vals = np.asarray([float(row["rescue_gap_improvement"]) for row in rows if bool(row["rescue_exists"])], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    if vals.size:
        ax.hist(vals, bins=min(20, max(5, vals.size)))
        ax.set_xlabel("rescue gap improvement")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No rescuable starts", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Rescue gap improvement histogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_candidate_type_counts(rows: list[dict[str, object]], out_path: Path) -> None:
    counts = Counter(str(row["rescue_candidate_type"]) for row in rows if bool(row["rescue_exists"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    if counts:
        keys = list(sorted(counts))
        ax.bar(keys, [int(counts[k]) for k in keys])
        ax.set_xticklabels(keys, rotation=20)
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No rescuable starts", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Rescue candidate type counts")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_monitor_candidate_rescue_audit(
    config_path: str = "configs/system.yaml",
    out_dir: str | Path = "results/monitor_candidate_rescue_audit",
) -> Path:
    cfg = load_config(config_path)
    cfg_mode = _clone_active_monitor_cfg(cfg)
    out = make_results_dir("monitor_candidate_rescue_audit") if out_dir == "results/monitor_candidate_rescue_audit" else Path(out_dir)
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

    starts = _blocked_interval_starts(sim)
    x_hist = np.asarray(sim["x"], dtype=float)
    zeta_r = np.asarray(sim["zeta_r"], dtype=float)
    t_control = np.asarray(sim["t_control"], dtype=float)
    current_pis = [_as_pi(v) for v in sim["pi"]]
    selector_candidates = [_as_pi(v) for v in sim["selector_candidate_pi"]]
    tie_i = np.asarray(sim["tie_i"], dtype=int)
    tie_j = np.asarray(sim["tie_j"], dtype=int)

    rows: list[dict[str, object]] = []
    for step in starts:
        current_pi = current_pis[step]
        blocked_candidate = selector_candidates[step]
        tie_pair = (int(tie_i[step]), int(tie_j[step]))
        x_now = np.asarray(x_hist[step], dtype=float)
        zeta_now = _zeta_from_row(np.asarray(zeta_r[step], dtype=float), cfg_mode)

        pool = _build_candidate_pool(current_pi, blocked_candidate, tie_pair)
        admissible_count = 0
        blocked_gap = float(predicted_raw_gap_proxy(current_pi, blocked_candidate, x_now, zeta_now, cfg_mode))
        best_rescue_pi: list[int] | None = None
        best_rescue_gap = float("inf")
        best_rescue_type = "none"

        for candidate_pi, candidate_type in pool:
            admissible, _ = admissible_order(candidate_pi, x_now, cfg_mode)
            if not admissible:
                continue
            admissible_count += 1
            if candidate_pi == blocked_candidate or candidate_pi == current_pi:
                continue
            gap = float(predicted_raw_gap_proxy(current_pi, candidate_pi, x_now, zeta_now, cfg_mode))
            if gap + 1.0e-12 < blocked_gap and gap < best_rescue_gap:
                best_rescue_gap = gap
                best_rescue_pi = candidate_pi
                best_rescue_type = candidate_type

        rescue_exists = best_rescue_pi is not None
        rows.append(
            {
                "step": int(step),
                "time": float(t_control[step]) if step < len(t_control) else float(step),
                "current_pi": str(current_pi),
                "blocked_selector_candidate": str(blocked_candidate),
                "blocked_candidate_predicted_gap": blocked_gap,
                "candidate_pool_size": int(len(pool)),
                "num_admissible_candidates": int(admissible_count),
                "rescue_exists": bool(rescue_exists),
                "best_rescue_pi": str(best_rescue_pi) if rescue_exists else "",
                "best_rescue_predicted_gap": float(best_rescue_gap) if rescue_exists else "",
                "rescue_gap_improvement": float(blocked_gap - best_rescue_gap) if rescue_exists else 0.0,
                "rescue_candidate_type": str(best_rescue_type) if rescue_exists else "none",
            }
        )

    with (out / "blocked_start_rescue_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "time",
                "current_pi",
                "blocked_selector_candidate",
                "blocked_candidate_predicted_gap",
                "candidate_pool_size",
                "num_admissible_candidates",
                "rescue_exists",
                "best_rescue_pi",
                "best_rescue_predicted_gap",
                "rescue_gap_improvement",
                "rescue_candidate_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    rescue_improvements = np.asarray([float(row["rescue_gap_improvement"]) for row in rows if bool(row["rescue_exists"])], dtype=float)
    rescue_type_counts = dict(sorted(Counter(str(row["rescue_candidate_type"]) for row in rows if bool(row["rescue_exists"])).items()))
    summary = {
        "shared_x0": [float(v) for v in x0],
        "shared_seed": int(sim_seed),
        "num_blocked_starts": int(len(rows)),
        "num_rescuable_starts": int(sum(1 for row in rows if bool(row["rescue_exists"]))),
        "num_nonrescuable_starts": int(sum(1 for row in rows if not bool(row["rescue_exists"]))),
        "rescue_fraction": float(np.mean([bool(row["rescue_exists"]) for row in rows])) if rows else 0.0,
        "mean_rescue_gap_improvement": float(np.mean(rescue_improvements)) if rescue_improvements.size else 0.0,
        "median_rescue_gap_improvement": float(np.median(rescue_improvements)) if rescue_improvements.size else 0.0,
        "rescue_candidate_type_counts": rescue_type_counts,
        "top_rows": rows[:10],
    }
    dump_json(out / "monitor_candidate_rescue_summary.json", summary)

    _plot_rescue_exists(rows, out / "rescue_exists_bar.png")
    _plot_rescue_improvement(rows, out / "rescue_gap_improvement_hist.png")
    _plot_candidate_type_counts(rows, out / "rescue_candidate_type_counts.png")

    return out / "monitor_candidate_rescue_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rescuable blocked monitor starts from enriched local candidates")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--out", default="results/monitor_candidate_rescue_audit")
    args = parser.parse_args()
    out = run_monitor_candidate_rescue_audit(config_path=args.config, out_dir=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

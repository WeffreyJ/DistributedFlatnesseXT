"""Gate 1: Graph compatibility and switching well-posedness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir, seed_rng


def _sample_x0(cfg, rng: np.random.Generator) -> np.ndarray:
    n = cfg.system.N
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def run_gate1(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gate1")
    rng = seed_rng(int(cfg.seed) + 101)

    mc_runs = int(cfg.gate1.mc_runs)
    dag_vals: list[float] = []
    topo_vals: list[float] = []
    all_inter_switch: list[float] = []
    all_switch_rates: list[float] = []
    cycle_snapshots: list[dict] = []
    topo_failures: list[dict] = []

    for run in range(mc_runs):
        x0 = _sample_x0(cfg, rng)
        sim = simulate_closed_loop(
            cfg,
            x0=x0,
            options=SimOptions(blending_on=False, seed=int(cfg.seed) + 1000 + run),
        )

        dag = np.asarray(sim["dag"], dtype=float)
        topo = np.asarray(sim["topo"], dtype=float)
        dag_vals.append(float(np.mean(dag)))
        topo_vals.append(float(np.mean(topo)))

        # TODO: richer cycle-state debug snapshots for larger models.
        for k, is_dag in enumerate(sim["dag"]):
            if not bool(is_dag):
                cycle_snapshots.append(
                    {
                        "run": run,
                        "step": int(k),
                        "edges": [list(e) for e in sim["edges"][k]],
                    }
                )
        for fail in sim.get("topo_failures", []):
            item = dict(fail)
            item["run"] = int(run)
            topo_failures.append(item)

        switch_times = np.array(sim["switch_times"], dtype=float)
        if switch_times.size >= 2:
            dts = np.diff(switch_times)
            all_inter_switch.extend(dts.tolist())
        rate = float(switch_times.size / max(float(sim["horizon"]), 1e-8))
        all_switch_rates.append(rate)

    min_inter_switch = float(min(all_inter_switch)) if all_inter_switch else float("inf")
    payload = {
        "gate": "Gate 1",
        "mc_runs": mc_runs,
        "dag_rate_mean": float(np.mean(dag_vals)),
        "topo_pass_rate_mean": float(np.mean(topo_vals)),
        "min_inter_switch_time": min_inter_switch,
        "switches_per_second_mean": float(np.mean(all_switch_rates)),
        "num_cycle_snapshots": len(cycle_snapshots),
        "cycle_snapshots": cycle_snapshots[:25],
        "num_topo_failures": len(topo_failures),
    }

    dump_json(out_dir / "gate1_summary.json", payload)
    topo_path = out_dir / "topo_failures.jsonl"
    with topo_path.open("w", encoding="utf-8") as f:
        for row in topo_failures:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["DAG rate", "Topo pass"], [payload["dag_rate_mean"], payload["topo_pass_rate_mean"]])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Gate 1 Compatibility Rates")
    fig.tight_layout()
    fig.savefig(out_dir / "gate1_rates.png", dpi=150)
    plt.close(fig)

    if all_inter_switch:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(all_inter_switch, bins=min(20, len(all_inter_switch)))
        ax.set_title("Inter-switch Times")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "gate1_interswitch_hist.png", dpi=150)
        plt.close(fig)

    return out_dir / "gate1_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 1 verification")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gate1(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

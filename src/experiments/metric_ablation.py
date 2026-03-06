"""Metric ablation study across Gate outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.verify.gate1_graph import run_gate1
from src.verify.gate4_stability_inequality import run_gate4
from src.verify.gateB_blend_transient import run_gateB
from src.verify.gateS_sensitivity import run_gateS
from src.verify.utils import make_results_dir

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("metric_ablation requires PyYAML") from exc


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _gate4_stats(csv_path: Path) -> tuple[float, float, float]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return 0.0, 0.0, 0.0
    jump_ratio = np.asarray([float(r["jump_ratio"]) for r in rows], dtype=float)
    switch_rate = np.asarray([float(r["switch_rate"]) for r in rows], dtype=float)
    success_rate = float(np.mean(jump_ratio < 1.0))
    return float(np.median(jump_ratio)), float(np.mean(switch_rate)), success_rate


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, out_path: Path) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_status_aware_scatter(rows: list[dict[str, float | str]], out_path: Path) -> None:
    if not rows:
        return

    finite_jump = [
        float(r["Gate4_jump_ratio_median"])
        for r in rows
        if np.isfinite(float(r["Gate4_jump_ratio_median"]))
    ]
    finite_s = [
        float(r.get("GateS_effective_p90", r.get("GateS_S_p90_nonzero", r["GateS_S_p90"])))
        for r in rows
        if np.isfinite(float(r.get("GateS_effective_p90", r.get("GateS_S_p90_nonzero", r["GateS_S_p90"]))))
    ]
    if finite_jump:
        y_fail = max(max(finite_jump) * 1.15, 1.0)
    else:
        y_fail = 1.0
    if finite_s:
        x_min = min(finite_s)
        x_max = max(finite_s)
    else:
        x_min, x_max = 0.0, 1.0
    x_span = max(x_max - x_min, 1.0e-3)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(y_fail, color="crimson", linestyle="--", linewidth=1.0, alpha=0.6, label="Gate4 unavailable")

    for idx, row in enumerate(rows):
        name = str(row["metric_name"])
        status = str(row["status"])
        x_raw = float(row.get("GateS_effective_p90", row.get("GateS_S_p90_nonzero", row["GateS_S_p90"])))
        y_raw = float(row["Gate4_jump_ratio_median"])
        x = x_raw if np.isfinite(x_raw) else (x_min - 0.05 * x_span)
        # Deterministic small jitter makes overlapping labels visible with small metric sets.
        jitter = ((idx % 5) - 2) * 0.008 * x_span
        x_plot = x + jitter

        if np.isfinite(y_raw):
            marker = "o" if status == "ok" else "^"
            color = "tab:blue" if status == "ok" else "tab:orange"
            y_plot = y_raw
        else:
            marker = "x"
            color = "crimson"
            y_plot = y_fail

        ax.scatter([x_plot], [y_plot], marker=marker, color=color, s=52, alpha=0.9)
        ax.annotate(name, (x_plot, y_plot), fontsize=8, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("GateS effective p90 (swap or candidate)")
    ax.set_ylabel("Gate4 median jump_ratio")
    ax.set_title("Metric Ablation: Status-aware Sensitivity vs Jump Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_metric_ablation(system_config: str, out: str | None = None) -> Path:
    sys_path = Path(system_config)
    cfg = load_config(sys_path)

    out_dir = Path(out) if out is not None else make_results_dir("metric_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    base_system = _load_yaml(sys_path)
    exp_path = Path("configs/experiments.yaml")
    base_experiments = _load_yaml(exp_path)

    mab = getattr(cfg, "metric_ablation", None)
    metrics = list(getattr(mab, "metrics", ["s_metric", "wake_influence", "random_fixed", "constant"]))
    ablation_runs = int(getattr(mab, "mc_runs", 6))
    gate4_mc_runs = int(getattr(mab, "gate4_mc_runs", 3))
    max_samples_gateS = int(getattr(mab, "max_samples_gateS", 1000))

    rows: list[dict[str, float | str]] = []

    for metric in metrics:
        sys_dict = copy.deepcopy(base_system)
        sys_dict.setdefault("ordering", {})
        sys_dict["ordering"]["metric"] = metric
        sys_dict["ordering"].setdefault("metric_seed", int(base_system.get("seed", 0)) + 99)
        sys_dict["ordering"].setdefault("wake_metric_gain", 1.0)

        sys_dict.setdefault("gate1", {})
        sys_dict["gate1"]["mc_runs"] = ablation_runs
        sys_dict.setdefault("verify", {})
        sys_dict["verify"].setdefault("gateS", {})
        sys_dict["verify"]["gateS"]["mc_runs"] = ablation_runs
        sys_dict["verify"]["gateS"]["max_samples"] = max_samples_gateS
        if getattr(mab, "tie_gap_delta", None) is not None:
            sys_dict["verify"]["gateS"]["tie_gap_delta"] = float(getattr(mab, "tie_gap_delta"))
        sys_dict["verify"].setdefault("gateB", {})
        sys_dict["verify"]["gateB"]["mc_runs"] = ablation_runs

        sys_tmp = tmp_dir / f"system_{metric}.yaml"
        _write_yaml(sys_tmp, sys_dict)

        exp_dict = copy.deepcopy(base_experiments)
        exp_dict["base_system_config"] = str(sys_tmp)
        exp_dict.setdefault("gate4", {})
        exp_dict["gate4"]["mc_runs"] = gate4_mc_runs
        exp_tmp = tmp_dir / f"experiments_{metric}.yaml"
        _write_yaml(exp_tmp, exp_dict)

        errors: list[str] = []
        try:
            run_gate1(str(sys_tmp))
        except Exception as exc:  # pragma: no cover
            errors.append(f"gate1:{exc.__class__.__name__}:{exc}")
        try:
            run_gateS(str(sys_tmp))
        except Exception as exc:  # pragma: no cover
            errors.append(f"gateS:{exc.__class__.__name__}:{exc}")
        try:
            run_gateB(str(sys_tmp))
        except Exception as exc:  # pragma: no cover
            errors.append(f"gateB:{exc.__class__.__name__}:{exc}")
        try:
            run_gate4(str(exp_tmp))
            jump_ratio_med, switch_rate_mean, success_rate = _gate4_stats(Path("results/gate4/gate4_summary.csv"))
        except Exception as exc:  # pragma: no cover
            errors.append(f"gate4:{exc.__class__.__name__}:{exc}")
            jump_ratio_med, switch_rate_mean, success_rate = float("nan"), float("nan"), 0.0

        g1 = {} if any(e.startswith("gate1:") for e in errors) else _safe_read_json(Path("results/gate1/gate1_summary.json"))
        gs = {} if any(e.startswith("gateS:") for e in errors) else _safe_read_json(Path("results/gateS/gateS_sensitivity.json"))
        gb = {} if any(e.startswith("gateB:") for e in errors) else _safe_read_json(Path("results/gateB/gateB.json"))

        s_p90 = float(gs.get("S_p90", 0.0))
        s_p90_nonzero = float(gs.get("S_p90_nonzero", 0.0))
        s_candidate_p90_nonzero = float(gs.get("S_candidate_p90_nonzero", 0.0))
        rows.append(
            {
                "metric_name": metric,
                "status": "ok" if not errors else "error",
                "errors": ";".join(errors),
                "Gate1_DAG_pass_rate": float(g1.get("dag_rate_mean", 0.0)),
                "Gate1_topo_pass_rate": float(g1.get("topo_pass_rate_mean", 0.0)),
                "GateS_S_p90": s_p90,
                "GateS_S_p90_nonzero": s_p90_nonzero,
                "GateS_candidate_p90_nonzero": s_candidate_p90_nonzero,
                "GateS_effective_p90": max(s_p90_nonzero, s_candidate_p90_nonzero),
                "GateS_S_nonzero_fraction": float(gs.get("S_nonzero_fraction", 0.0)),
                "GateS_S_max": float(gs.get("S_max", 0.0)),
                "GateB_max_e_blend_p90": float(gb.get("max_e_p90", 0.0)),
                "Gate4_jump_ratio_median": jump_ratio_med,
                "Gate4_switch_rate_mean": switch_rate_mean,
                "Gate4_success_rate": success_rate,
            }
        )

    csv_path = out_dir / "metric_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric_name",
                "status",
                "errors",
                "Gate1_DAG_pass_rate",
                "Gate1_topo_pass_rate",
                "GateS_S_p90",
                "GateS_S_p90_nonzero",
                "GateS_candidate_p90_nonzero",
                "GateS_effective_p90",
                "GateS_S_nonzero_fraction",
                "GateS_S_max",
                "GateB_max_e_blend_p90",
                "Gate4_jump_ratio_median",
                "Gate4_switch_rate_mean",
                "Gate4_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": "1.0",
        "metrics": rows,
        "mc_runs": ablation_runs,
        "gate4_mc_runs": gate4_mc_runs,
    }
    json_path = out_dir / "metric_ablation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    valid_rows = [r for r in rows if np.isfinite(float(r["Gate4_jump_ratio_median"])) and np.isfinite(float(r["GateS_S_p90"]))]
    labels = [str(r["metric_name"]) for r in valid_rows]
    jump_vals = [float(r["Gate4_jump_ratio_median"]) for r in valid_rows]
    s_vals = [float(r["GateS_S_p90"]) for r in valid_rows]

    _plot_bar(
        labels,
        jump_vals,
        title="Metric Ablation: Median jump_ratio",
        ylabel="median jump_ratio",
        out_path=out_dir / "metric_compare_bar_jump_ratio.png",
    )
    _plot_bar(
        labels,
        s_vals,
        title="Metric Ablation: GateS S_p90",
        ylabel="S_p90",
        out_path=out_dir / "metric_compare_bar_Sp90.png",
    )
    s_nonzero_vals = [float(r["GateS_S_p90_nonzero"]) for r in valid_rows]
    _plot_bar(
        labels,
        s_nonzero_vals,
        title="Metric Ablation: GateS S_p90 (nonzero-only)",
        ylabel="S_p90_nonzero",
        out_path=out_dir / "metric_compare_bar_Sp90_nonzero.png",
    )
    s_effective_vals = [float(r["GateS_effective_p90"]) for r in valid_rows]
    _plot_bar(
        labels,
        s_effective_vals,
        title="Metric Ablation: GateS effective p90 (swap/candidate)",
        ylabel="effective_p90",
        out_path=out_dir / "metric_compare_bar_S_effective_p90.png",
    )

    if valid_rows:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(s_vals, jump_vals, "o")
        for i, name in enumerate(labels):
            ax.annotate(name, (s_vals[i], jump_vals[i]), fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("GateS S_p90")
        ax.set_ylabel("Gate4 median jump_ratio")
        ax.set_title("Metric Ablation: Sensitivity vs Jump Ratio")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "metric_compare_scatter_Sp90_vs_jump_ratio.png", dpi=150)
        plt.close(fig)
    _plot_status_aware_scatter(rows, out_dir / "metric_compare_scatter_status_aware.png")

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Metric ablation study")
    parser.add_argument("--config", required=True, help="Path to system config")
    parser.add_argument("--out", default="results/metric_ablation", help="Output directory")
    args = parser.parse_args()

    path = run_metric_ablation(args.config, out=args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

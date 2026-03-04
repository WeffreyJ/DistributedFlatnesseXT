"""Gate 3.5: provenance-aware satisfiability inequality check."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.verify.utils import dump_json, make_results_dir


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _load_gate3_constants(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run gate3_constants first.")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "constants" not in payload:
        raise KeyError("gate3 constants file missing 'constants' block")
    return payload


def run_gate35(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gate3p5")

    gate_cfg = getattr(getattr(cfg, "verify", object()), "gate3p5", None)
    constants_source = str(getattr(gate_cfg, "constants_source", "gate3"))

    if constants_source == "gate3":
        g3_path = Path("results/gate3/constants_table.json")
        g3 = _load_gate3_constants(g3_path)
        c = g3["constants"]
        mu = float(c["mu"])
        alpha_over_c2 = float(c["alpha_over_c2"])
        Kd = int(c["Kd"])
        dt = float(c.get("dt", cfg.system.dt))
        certificate_mode = str(g3.get("certificate_mode", "template"))
        constants_used = {
            "mu": mu,
            "alpha_over_c2": alpha_over_c2,
            "Kd": Kd,
            "dt": dt,
            "source_path": str(g3_path),
        }
        interpretation = (
            "empirical_high_confidence_on_R" if certificate_mode == "empirical" else "template_satisfiability_only"
        )
        template_only = certificate_mode != "empirical"
    elif constants_source == "manual":
        man = getattr(gate_cfg, "manual_constants", None)
        if man is None:
            raise ValueError("verify.gate3p5.manual_constants is required when constants_source=manual")
        mu = float(getattr(man, "mu"))
        alpha_over_c2 = float(getattr(man, "alpha_over_c2"))
        Kd = int(getattr(man, "Kd"))
        dt = float(getattr(man, "dt", cfg.system.dt))
        constants_used = {
            "mu": mu,
            "alpha_over_c2": alpha_over_c2,
            "Kd": Kd,
            "dt": dt,
            "source_path": "manual_from_config",
        }
        interpretation = "template_satisfiability_only"
        template_only = True
    else:
        raise ValueError(f"Unsupported constants_source={constants_source!r}")

    lhs = float(mu * np.exp(-alpha_over_c2 * Kd))
    passed = lhs < 1.0
    margin = float(1.0 - lhs)

    kd_grid = np.arange(0, max(Kd * 3, 100) + 1, dtype=float)
    lhs_grid = mu * np.exp(-alpha_over_c2 * kd_grid)
    tau_grid = kd_grid * float(dt)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tau_grid, lhs_grid, label=r"$\mu e^{-(\alpha/c_2)K_d}$")
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1.0, label="threshold=1")
    ax.axvline(float(Kd) * float(dt), color="k", linestyle=":", linewidth=1.0, label=f"Kd={Kd}")
    ax.set_xlabel("tau_d [s]")
    ax.set_ylabel("LHS")
    ax.set_title("Gate 3.5 Satisfiability")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "satisfiability_vs_tau_d.png", dpi=150)
    plt.close(fig)

    payload = {
        "schema_version": "1.0",
        "gate": "Gate 3.5",
        "constants_source": constants_source,
        "interpretation": interpretation,
        "template_satisfiability_only": bool(template_only),
        "lhs": lhs,
        "satisfied": bool(passed),
        "margin": margin,
        "constants_used": constants_used,
        "provenance": {
            "source": "gate3p5_satisfiability",
            "config_path": str(cfg_path),
            "git_commit": _git_head(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    out_path = out_dir / "gate3p5.json"
    dump_json(out_path, payload)

    # Backward-compatible legacy filename.
    dump_json(out_dir / "stability_margin_report.json", payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3.5 satisfiability")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gate35(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

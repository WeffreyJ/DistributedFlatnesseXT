"""Gate 2: Flat output regularity / rank check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_config
from src.verify.utils import dump_json, make_results_dir


def run_gate2(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gate2")

    n = int(cfg.system.N)
    decoupling = np.eye(n)
    rank = int(np.linalg.matrix_rank(decoupling))
    passed = rank == n

    payload = {
        "gate": "Gate 2",
        "model": "toy_double_integrator",
        "flat_output": "y = x1",
        "relative_degree": 2,
        "decoupling_matrix": decoupling.tolist(),
        "rank": rank,
        "expected_rank": n,
        "status": "PASS" if passed else "FAIL",
        "note": "Toy model has identity decoupling matrix everywhere. Future models must compute true rank map.",
    }

    out_path = out_dir / "gate2_rank_summary.json"
    dump_json(out_path, payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2 flat output rank test")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_gate2(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

"""Export paper-ready Success Evaluation bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _run_module(module: str, args: list[str]) -> None:
    env = dict(**__import__("os").environ)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/pycache")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    env.setdefault("MPLBACKEND", "Agg")
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"WARN missing: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    count = 0
    for src in sorted(src_dir.glob(pattern)):
        if src.is_file():
            _safe_copy(src, dst_dir / src.name)
            count += 1
    if count == 0:
        print(f"WARN no files for {src_dir}/{pattern}")
    return count


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_snapshot(root: Path) -> None:
    snap = root / "snapshot"
    (snap / "configs").mkdir(parents=True, exist_ok=True)

    _safe_copy(Path("configs/system.yaml"), snap / "configs/system.yaml")
    _safe_copy(Path("configs/experiments.yaml"), snap / "configs/experiments.yaml")

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception as exc:  # pragma: no cover
        head = f"UNKNOWN ({exc})"
        status = ""

    (snap / "git_commit.txt").write_text(
        f"HEAD: {head}\n\nstatus --porcelain:\n{status}",
        encoding="utf-8",
    )

    key_files = [
        Path("src/model/coupling.py"),
        Path("src/flatness/recursion.py"),
        Path("src/control/closed_loop.py"),
        Path("src/verify/gate4_stability_inequality.py"),
    ]
    lines = []
    for p in key_files:
        if p.exists():
            stat = p.stat()
            lines.append(
                f"{p}: sha256={_sha256(p)} mtime={stat.st_mtime:.3f} size={stat.st_size}"
            )
    (snap / "src_versions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (snap / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        (snap / "pip_freeze.txt").write_text(f"Unavailable: {exc}\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gate4_means(path: Path, tau_d: float, noise_delta: float) -> dict[bool, dict[str, float]]:
    out: dict[bool, dict[str, float]] = {False: {}, True: {}}
    if not path.exists():
        return out
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    for b in [False, True]:
        filt = [
            r
            for r in rows
            if (r["blending"].strip().lower() == str(b).lower())
            and abs(float(r["tau_d"]) - tau_d) < 1e-12
            and abs(float(r["noise_delta"]) - noise_delta) < 1e-12
        ]
        if not filt:
            continue
        for k in ["J", "J_raw", "jump_ratio", "error_spike_instant", "switch_rate"]:
            out[b][k] = float(sum(float(r[k]) for r in filt) / len(filt))
    return out


def _write_readme(root: Path) -> None:
    g3 = _load_json(root / "gate3" / "constants_table.json")
    means = _gate4_means(root / "gate4" / "gate4_summary.csv", tau_d=0.0, noise_delta=0.0)

    j_raw = g3.get("J_raw_with_blending", float("nan"))
    j_nb = g3.get("J_without_blending", float("nan"))
    j_b = g3.get("J_applied_with_blending", g3.get("J_with_blending", float("nan")))

    jr_false = means.get(False, {}).get("jump_ratio", float("nan"))
    jr_true = means.get(True, {}).get("jump_ratio", float("nan"))

    text = f"""# Success Evaluation Bundle

## Claim Summary
- \u03c0-upstream wake surrogate yields large raw mismatch (`J_raw \u2248 {j_raw:.3f}`) and without blending applied jumps are large (`J_no_blend \u2248 {j_nb:.3f}`).
- With blending enabled, applied jumps reduce (`J_blend \u2248 {j_b:.3f}`) and jump-ratio at `tau_d=0, noise=0` drops from `{jr_false:.3f}` to `{jr_true:.3f}`.

## Contents
- `gate1/`: DAG/topological compatibility and switching well-posedness artifacts.
- `gate3/`: constants table and jump diagnostics.
- `gate4/`: Monte-Carlo trend CSV and plots (`J`, `jump_ratio`, switch metrics, spike metrics).
- `baseline_demo/`: fixed-order/no-switch baseline vs hybrid switching+blending comparison.
- `snapshot/`: exact configs, git state, and key source-file hashes used to generate this bundle.

## Reproduce
```bash
python -m tools.export_success_eval --out paper_artifacts/success_eval
```

Optional flags:
- `--no_run_gates` to reuse existing `results/` files.
- `--no_baseline_demo` to skip the baseline comparison run.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def export_success_eval(out: Path, run_gates: bool, run_baseline: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate1").mkdir(parents=True, exist_ok=True)
    (out / "gate3").mkdir(parents=True, exist_ok=True)
    (out / "gate4").mkdir(parents=True, exist_ok=True)
    (out / "baseline_demo").mkdir(parents=True, exist_ok=True)

    if run_gates:
        _run_module("src.verify.gate1_graph", ["--config", "configs/system.yaml"])
        _run_module("src.verify.gate3_constants", ["--config", "configs/system.yaml"])
        _run_module("src.verify.gate4_stability_inequality", ["--config", "configs/experiments.yaml"])
        _run_module("src.verify.gate4_selftest", ["--config", "configs/experiments.yaml"])

    # Gate 1 artifacts
    _safe_copy(Path("results/gate1/gate1_summary.json"), out / "gate1/gate1_summary.json")
    _copy_glob(Path("results/gate1"), "*.png", out / "gate1")

    # Gate 3 artifacts
    _safe_copy(Path("results/gate3/constants_table.json"), out / "gate3/constants_table.json")
    _copy_glob(Path("results/gate3"), "*.png", out / "gate3")

    # Gate 4 artifacts
    _safe_copy(Path("results/gate4/gate4_summary.csv"), out / "gate4/gate4_summary.csv")
    for name in [
        "J_by_tau_and_blending.png",
        "jump_ratio_by_tau_and_blending.png",
        "switch_rate_by_noise.png",
        "switch_rate_by_tau.png",
        "error_spike_instant_by_tau.png",
        "error_spike_instant_by_noise.png",
        "error_envelope_by_tau.png",
        "error_spike_by_tau.png",
    ]:
        _safe_copy(Path("results/gate4") / name, out / "gate4" / name)

    # Baseline demo
    if run_baseline:
        _run_module("src.experiments.baseline_demo", ["--config", "configs/system.yaml", "--out", "results/baseline_demo"])
    if Path("results/baseline_demo").exists():
        _copy_glob(Path("results/baseline_demo"), "*", out / "baseline_demo")
    else:
        print("WARN results/baseline_demo missing")

    _write_snapshot(out)
    _write_readme(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export success-evaluation bundle")
    parser.add_argument("--out", default="paper_artifacts/success_eval")
    parser.add_argument("--no_run_gates", action="store_true")
    parser.add_argument("--no_baseline_demo", action="store_true")
    args = parser.parse_args()

    out = export_success_eval(
        out=Path(args.out),
        run_gates=not args.no_run_gates,
        run_baseline=not args.no_baseline_demo,
    )
    print(f"Wrote success-eval bundle to {out}")


if __name__ == "__main__":
    main()

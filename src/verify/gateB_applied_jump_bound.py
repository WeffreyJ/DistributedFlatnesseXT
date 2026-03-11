"""Gate B bound support: row-wise applied-jump inequality diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir


ABS_TOL = 1.0e-9
REL_TOL = 1.0e-7


def _base_x0(cfg) -> np.ndarray:
    n = int(cfg.system.N)
    return np.concatenate(
        [
            np.array(cfg.reference.base, dtype=float),
            np.zeros(n, dtype=float),
        ],
        axis=0,
    )


def _bound_tol(lhs: float, rhs: float) -> float:
    scale = max(abs(lhs), abs(rhs), 1.0)
    return ABS_TOL + REL_TOL * scale


def _save_empty_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_with_diagonal(
    xvals: np.ndarray,
    yvals: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if xvals.size == 0 or yvals.size == 0:
        _save_empty_plot(out_path, title, "No blend-active rows available")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xvals, yvals, ".", alpha=0.75)
    lo = float(min(np.min(xvals), np.min(yvals)))
    hi = float(max(np.max(xvals), np.max(yvals)))
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hist_plot(values: np.ndarray, *, xlabel: str, title: str, out_path: Path) -> None:
    if values.size == 0:
        _save_empty_plot(out_path, title, "No blend-active rows available")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(30, max(8, values.size)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _collect_rows(sim: dict) -> list[dict[str, object]]:
    """Collect one-step blend-active rows using the exact logged progress `w_blend`."""
    blend_active = np.asarray(sim["blend_active"], dtype=bool)
    blend_id = np.asarray(sim["blend_id"], dtype=int)
    blend_step = np.asarray(sim["blend_step"], dtype=int)
    w_blend = np.asarray(sim["w_blend"], dtype=float)
    u_old = np.asarray(sim["u_old"], dtype=float)
    u_new = np.asarray(sim["u_new"], dtype=float)
    u_applied = np.asarray(sim["u_applied"], dtype=float)
    t_control = np.asarray(sim["t_control"], dtype=float)
    J_raw = np.asarray(sim["J_raw"], dtype=float)
    E_gap = np.asarray(sim["E_gap_same_step"], dtype=float) if "E_gap_same_step" in sim else None
    tie_gap_min = np.asarray(sim["tie_gap_min"], dtype=float) if "tie_gap_min" in sim else None
    switch_reason = [str(v) for v in sim["switch_reason"]]

    rows: list[dict[str, object]] = []
    for k in range(1, len(blend_active)):
        if not bool(blend_active[k]):
            continue
        bid = int(blend_id[k])
        if bid < 0:
            continue

        same_blend_prev = bool(blend_active[k - 1]) and int(blend_id[k - 1]) == bid
        beta_k = float(w_blend[k])
        beta_prev = float(w_blend[k - 1]) if same_blend_prev else 0.0
        beta_increment = float(abs(beta_k - beta_prev))
        raw_gap_k = float(J_raw[k]) if 0 <= k < len(J_raw) else float(np.linalg.norm(u_new[k] - u_old[k]))
        applied_jump_k = float(np.linalg.norm(u_applied[k] - u_applied[k - 1]))
        u_old_variation = float(np.linalg.norm(u_old[k] - u_old[k - 1]))
        u_new_variation = float(np.linalg.norm(u_new[k] - u_new[k - 1]))
        raw_gap_term_k = float(beta_increment * raw_gap_k)
        V_intra_simplified_k = float((1.0 - beta_k) * u_old_variation + beta_k * u_new_variation)
        rhs_bound_simplified_k = float(raw_gap_term_k + V_intra_simplified_k)
        V_intra_exact_k = float((1.0 - beta_prev) * u_old_variation + beta_prev * u_new_variation)
        rhs_bound_exact_k = float(raw_gap_term_k + V_intra_exact_k)
        tol_simplified = _bound_tol(applied_jump_k, rhs_bound_simplified_k)
        tol_exact = _bound_tol(applied_jump_k, rhs_bound_exact_k)
        margin_simplified = float(rhs_bound_simplified_k - applied_jump_k)
        margin_exact = float(rhs_bound_exact_k - applied_jump_k)
        rows.append(
            {
                "step": int(k),
                "time_sec": float(t_control[k]),
                "blend_id": bid,
                "blend_step": int(blend_step[k]),
                "beta_k": beta_k,
                "beta_prev": beta_prev,
                "beta_increment": beta_increment,
                "raw_gap_k": raw_gap_k,
                "raw_gap_term_k": raw_gap_term_k,
                "applied_jump_k": applied_jump_k,
                "u_old_variation": u_old_variation,
                "u_new_variation": u_new_variation,
                "V_intra_simplified_k": V_intra_simplified_k,
                "rhs_bound_simplified_k": rhs_bound_simplified_k,
                "bound_margin_simplified": margin_simplified,
                "bound_satisfied_simplified": bool(applied_jump_k <= rhs_bound_simplified_k + tol_simplified),
                "bound_tolerance_simplified": float(tol_simplified),
                "V_intra_exact_k": V_intra_exact_k,
                "rhs_bound_exact_k": rhs_bound_exact_k,
                "bound_margin_exact": margin_exact,
                "bound_satisfied_exact": bool(applied_jump_k <= rhs_bound_exact_k + tol_exact),
                "bound_tolerance_exact": float(tol_exact),
                "E_gap_k": float(E_gap[k]) if E_gap is not None else 0.0,
                "tie_gap_min": float(tie_gap_min[k]) if tie_gap_min is not None else float("inf"),
                "switch_reason": switch_reason[k],
                "prev_same_blend": bool(same_blend_prev),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "time_sec",
        "blend_id",
        "blend_step",
        "beta_k",
        "beta_prev",
        "beta_increment",
        "raw_gap_k",
        "raw_gap_term_k",
        "applied_jump_k",
        "u_old_variation",
        "u_new_variation",
        "V_intra_simplified_k",
        "rhs_bound_simplified_k",
        "bound_margin_simplified",
        "bound_satisfied_simplified",
        "bound_tolerance_simplified",
        "V_intra_exact_k",
        "rhs_bound_exact_k",
        "bound_margin_exact",
        "bound_satisfied_exact",
        "bound_tolerance_exact",
        "E_gap_k",
        "tie_gap_min",
        "switch_reason",
        "prev_same_blend",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    applied = np.asarray([float(r["applied_jump_k"]) for r in rows], dtype=float)
    raw_gap = np.asarray([float(r["raw_gap_k"]) for r in rows], dtype=float)
    raw_gap_term = np.asarray([float(r["raw_gap_term_k"]) for r in rows], dtype=float)
    V_intra_simplified = np.asarray([float(r["V_intra_simplified_k"]) for r in rows], dtype=float)
    V_intra_exact = np.asarray([float(r["V_intra_exact_k"]) for r in rows], dtype=float)
    rhs_simplified = np.asarray([float(r["rhs_bound_simplified_k"]) for r in rows], dtype=float)
    rhs_exact = np.asarray([float(r["rhs_bound_exact_k"]) for r in rows], dtype=float)
    beta_inc = np.asarray([float(r["beta_increment"]) for r in rows], dtype=float)
    beta_k = np.asarray([float(r["beta_k"]) for r in rows], dtype=float)
    margin_simplified = np.asarray([float(r["bound_margin_simplified"]) for r in rows], dtype=float)
    margin_exact = np.asarray([float(r["bound_margin_exact"]) for r in rows], dtype=float)
    satisfied_simplified = np.asarray([bool(r["bound_satisfied_simplified"]) for r in rows], dtype=bool)
    satisfied_exact = np.asarray([bool(r["bound_satisfied_exact"]) for r in rows], dtype=bool)
    violation_simplified = np.maximum(applied - rhs_simplified, 0.0)
    violation_exact = np.maximum(applied - rhs_exact, 0.0)
    terminal_like_simplified_violation = (~satisfied_simplified) & (beta_k >= 1.0 - 1.0e-12) & (beta_inc >= 1.0 - 1.0e-12)
    terminal_like_exact_violation = (~satisfied_exact) & (beta_k >= 1.0 - 1.0e-12) & (beta_inc >= 1.0 - 1.0e-12)

    worst_rows_simplified = [
        {
            "step": int(r["step"]),
            "blend_id": int(r["blend_id"]),
            "blend_step": int(r["blend_step"]),
            "applied_jump_k": float(r["applied_jump_k"]),
            "rhs_bound_simplified_k": float(r["rhs_bound_simplified_k"]),
            "bound_margin_simplified": float(r["bound_margin_simplified"]),
            "raw_gap_term_k": float(r["raw_gap_term_k"]),
            "V_intra_simplified_k": float(r["V_intra_simplified_k"]),
        }
        for r in sorted(rows, key=lambda row: float(row["bound_margin_simplified"]))[:5]
    ]
    worst_rows_exact = [
        {
            "step": int(r["step"]),
            "blend_id": int(r["blend_id"]),
            "blend_step": int(r["blend_step"]),
            "applied_jump_k": float(r["applied_jump_k"]),
            "rhs_bound_exact_k": float(r["rhs_bound_exact_k"]),
            "bound_margin_exact": float(r["bound_margin_exact"]),
            "raw_gap_term_k": float(r["raw_gap_term_k"]),
            "V_intra_exact_k": float(r["V_intra_exact_k"]),
        }
        for r in sorted(rows, key=lambda row: float(row["bound_margin_exact"]))[:5]
    ]

    return {
        "gate": "Gate B applied jump bound",
        "num_rows": int(len(rows)),
        "exact_bound_satisfaction_fraction": float(np.mean(satisfied_exact)) if satisfied_exact.size else 1.0,
        "num_exact_bound_violations": int(np.count_nonzero(~satisfied_exact)) if satisfied_exact.size else 0,
        "num_terminal_like_exact_violations": int(np.count_nonzero(terminal_like_exact_violation)) if terminal_like_exact_violation.size else 0,
        "max_exact_violation": float(np.max(violation_exact)) if violation_exact.size else 0.0,
        "simplified_bound_satisfaction_fraction": float(np.mean(satisfied_simplified)) if satisfied_simplified.size else 1.0,
        "num_simplified_bound_violations": int(np.count_nonzero(~satisfied_simplified)) if satisfied_simplified.size else 0,
        "num_terminal_like_simplified_violations": int(np.count_nonzero(terminal_like_simplified_violation)) if terminal_like_simplified_violation.size else 0,
        "max_simplified_violation": float(np.max(violation_simplified)) if violation_simplified.size else 0.0,
        "max_applied_jump": float(np.max(applied)) if applied.size else 0.0,
        "max_rhs_bound_exact": float(np.max(rhs_exact)) if rhs_exact.size else 0.0,
        "max_rhs_bound_simplified": float(np.max(rhs_simplified)) if rhs_simplified.size else 0.0,
        "mean_raw_gap": float(np.mean(raw_gap)) if raw_gap.size else 0.0,
        "mean_raw_gap_term": float(np.mean(raw_gap_term)) if raw_gap_term.size else 0.0,
        "mean_V_intra_exact": float(np.mean(V_intra_exact)) if V_intra_exact.size else 0.0,
        "mean_V_intra_simplified": float(np.mean(V_intra_simplified)) if V_intra_simplified.size else 0.0,
        "mean_beta_increment": float(np.mean(beta_inc)) if beta_inc.size else 0.0,
        "min_bound_margin_exact": float(np.min(margin_exact)) if margin_exact.size else 0.0,
        "min_bound_margin_simplified": float(np.min(margin_simplified)) if margin_simplified.size else 0.0,
        "beta_source": "logged w_blend from simulate_closed_loop; this is the exact progress used by blend_progress.",
        "exact_bound_form": "applied_jump_k <= beta_increment * raw_gap_k + V_intra_exact_k",
        "simplified_bound_form": "applied_jump_k <= beta_increment * raw_gap_k + V_intra_simplified_k",
        "exact_bound_note": (
            "The exact bound resolved the prior simplified terminal-step violations."
            if violation_exact.size and np.count_nonzero(~satisfied_exact) == 0 and np.count_nonzero(~satisfied_simplified) > 0
            else "Exact bound still has violations; inspect worst_rows_exact."
            if np.count_nonzero(~satisfied_exact) > 0
            else "No exact-bound violations observed."
        ),
        "simplified_bound_note": (
            "All observed simplified-bound violations occur on terminal-like rows with beta_k=1 and beta_increment=1, so the gap is in the simplified bound form rather than progress reconstruction."
            if violation_simplified.size
            and np.count_nonzero(~satisfied_simplified) > 0
            and np.count_nonzero(terminal_like_simplified_violation) == np.count_nonzero(~satisfied_simplified)
            else "No clear single simplified-bound violation mode identified."
            if np.count_nonzero(~satisfied_simplified) > 0
            else "No simplified-bound violations observed."
        ),
        "worst_rows_exact": worst_rows_exact,
        "worst_rows_simplified": worst_rows_simplified,
    }


def run_gateB_applied_jump_bound(config_path: str = "configs/system.yaml") -> Path:
    cfg = load_config(config_path)
    out_dir = make_results_dir("gateB_applied_jump_bound")

    sim = simulate_closed_loop(
        cfg,
        x0=_base_x0(cfg),
        horizon=float(cfg.system.horizon),
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(cfg.seed) + 8600),
    )
    rows = _collect_rows(sim)

    table_path = out_dir / "applied_jump_bound_table.csv"
    summary_path = out_dir / "applied_jump_bound_summary.json"
    _write_csv(table_path, rows)
    dump_json(summary_path, _summary_payload(rows))

    applied = np.asarray([float(r["applied_jump_k"]) for r in rows], dtype=float)
    rhs_exact = np.asarray([float(r["rhs_bound_exact_k"]) for r in rows], dtype=float)
    rhs_simplified = np.asarray([float(r["rhs_bound_simplified_k"]) for r in rows], dtype=float)
    raw_gap_term = np.asarray([float(r["raw_gap_term_k"]) for r in rows], dtype=float)
    V_intra_exact = np.asarray([float(r["V_intra_exact_k"]) for r in rows], dtype=float)
    margin_exact = np.asarray([float(r["bound_margin_exact"]) for r in rows], dtype=float)
    margin_simplified = np.asarray([float(r["bound_margin_simplified"]) for r in rows], dtype=float)

    _scatter_with_diagonal(
        xvals=rhs_exact,
        yvals=applied,
        xlabel="rhs_bound_exact_k",
        ylabel="applied_jump_k",
        title="Applied jump vs exact blend-bound RHS",
        out_path=out_dir / "applied_jump_vs_rhs.png",
    )
    _scatter_with_diagonal(
        xvals=rhs_simplified,
        yvals=applied,
        xlabel="rhs_bound_simplified_k",
        ylabel="applied_jump_k",
        title="Applied jump vs simplified blend-bound RHS",
        out_path=out_dir / "applied_jump_vs_rhs_simplified.png",
    )
    _scatter_with_diagonal(
        xvals=raw_gap_term,
        yvals=applied,
        xlabel="beta_increment * raw_gap_k",
        ylabel="applied_jump_k",
        title="Applied jump vs raw-gap term",
        out_path=out_dir / "applied_jump_vs_raw_gap_term.png",
    )
    _hist_plot(
        V_intra_exact,
        xlabel="V_intra_exact_k",
        title="Exact intra-mode variation term histogram",
        out_path=out_dir / "V_intra_hist.png",
    )
    _hist_plot(
        margin_exact,
        xlabel="bound_margin_exact",
        title="Exact applied-jump bound margin histogram",
        out_path=out_dir / "margin_hist.png",
    )
    _hist_plot(
        margin_simplified,
        xlabel="bound_margin_simplified",
        title="Simplified applied-jump bound margin histogram",
        out_path=out_dir / "margin_hist_simplified.png",
    )

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B applied-jump bound diagnostics")
    parser.add_argument("--config", default="configs/system.yaml")
    args = parser.parse_args()
    path = run_gateB_applied_jump_bound(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

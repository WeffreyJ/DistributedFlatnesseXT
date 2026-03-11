"""Gate 3 operator-mismatch diagnostics for same-snapshot evaluator/control gaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.flatness.evaluation_operator import (
    compute_evaluator,
    get_evaluator_mode,
    get_local_window_agents,
    supports_explicit_evaluator,
)
from src.flatness.recursion import build_phi, psi
from src.model.coupling import active_edges
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


def _as_pi_list(pi_hist_entry) -> list[int]:
    return [int(v) for v in pi_hist_entry]


def _inverse_permutation(pi: list[int]) -> list[int]:
    inv = np.zeros(len(pi), dtype=int)
    for rank, agent in enumerate(pi):
        inv[int(agent)] = int(rank)
    return [int(v) for v in inv]


def _zeta_from_log(sim: dict, k: int, cfg) -> dict[str, np.ndarray]:
    layout = getattr(getattr(cfg, "flat", object()), "zeta_layout", None)
    zeta_r = np.asarray(sim["zeta_r"][k], dtype=float)
    if layout is None:
        n = int(cfg.system.N)
        return {
            "y": zeta_r[:n].copy(),
            "ydot": zeta_r[n : 2 * n].copy(),
            "v": zeta_r[2 * n : 3 * n].copy(),
        }
    return {
        "y": zeta_r[int(layout.y[0]) : int(layout.y[1])].copy(),
        "ydot": zeta_r[int(layout.ydot[0]) : int(layout.ydot[1])].copy(),
        "v": zeta_r[int(layout.v[0]) : int(layout.v[1])].copy(),
    }


def reconstruct_same_snapshot_pair(
    x: np.ndarray,
    zeta: dict[str, np.ndarray],
    pi_old: list[int],
    pi_new: list[int],
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct old/new controls from one snapshot using the explicit evaluator path."""
    if not supports_explicit_evaluator(cfg):
        raise ValueError("Operator-mismatch gate requires an evaluator-supported coupling mode.")
    eval_mode = get_evaluator_mode(cfg)
    E_old = compute_evaluator(x=x, pi=pi_old, cfg=cfg, mode=eval_mode)
    E_new = compute_evaluator(x=x, pi=pi_new, cfg=cfg, mode=eval_mode)
    phi_old = build_phi(x=x, zeta=zeta, pi=pi_old, sys=cfg, params=cfg.system, evaluator_output=E_old)
    phi_new = build_phi(x=x, zeta=zeta, pi=pi_new, sys=cfg, params=cfg.system, evaluator_output=E_new)
    u_old = psi(phi_old, params=cfg.system)
    u_new = psi(phi_new, params=cfg.system)
    return u_old, u_new, E_old, E_new


def _included_pair_terms(x: np.ndarray, pi: list[int], cfg, eval_mode: str) -> set[tuple[int, int]]:
    """Return active pairwise terms included by the evaluator for one ordering.

    The simple count bound later uses the symmetric-difference size between these
    included directed terms under two orderings at the same snapshot.
    """
    params = cfg.system
    full_terms = {(int(j), int(i)) for (j, i) in active_edges(x, params)}
    if eval_mode == "full":
        return full_terms

    rank = _inverse_permutation(pi)
    window = get_local_window_agents(cfg) if eval_mode == "local_window" else None
    included: set[tuple[int, int]] = set()
    for leader, follower in full_terms:
        if rank[leader] >= rank[follower]:
            continue
        if window is not None and (rank[follower] - rank[leader]) > int(window):
            continue
        included.add((int(leader), int(follower)))
    return included


def _followerwise_diff_count_vector(
    old_terms: set[tuple[int, int]],
    new_terms: set[tuple[int, int]],
    n_agents: int,
) -> np.ndarray:
    """Return the follower-indexed count vector for differing included terms."""
    counts = np.zeros(int(n_agents), dtype=int)
    for _, follower in old_terms.symmetric_difference(new_terms):
        counts[int(follower)] += 1
    return counts


def _consistency_ok(u_gap: float, E_gap: float) -> bool:
    scale = max(abs(u_gap), abs(E_gap), 1.0)
    return abs(u_gap - E_gap) <= ABS_TOL + REL_TOL * scale


def _infer_event_rows(sim: dict, cfg) -> list[dict[str, object]]:
    steps = int(len(sim["k"]))
    t_control = np.asarray(sim["t_control"], dtype=float)
    x_hist = np.asarray(sim["x"], dtype=float)
    blend_active = np.asarray(sim["blend_active"], dtype=bool)
    switch_reason = [str(v) for v in sim["switch_reason"]]
    pi_hist = [_as_pi_list(v) for v in sim["pi"]]
    pi_candidate_hist = [_as_pi_list(v) for v in sim["pi_candidate"]]
    eval_mode = str(get_evaluator_mode(cfg))

    logged_u_old = np.asarray(sim["u_old"], dtype=float) if "u_old" in sim else None
    logged_u_new = np.asarray(sim["u_new"], dtype=float) if "u_new" in sim else None
    logged_E_old = np.asarray(sim["E_old_order"], dtype=float) if "E_old_order" in sim else None
    logged_E_new = np.asarray(sim["E_new_order"], dtype=float) if "E_new_order" in sim else None

    kernel_single_term_bound = float(
        getattr(cfg.system, "k_wake", 0.0) * np.exp(-float(getattr(cfg.system, "gamma_edge", 0.0)) / max(float(getattr(cfg.system, "wake_decay_L", 1.0)), 1.0e-12))
    )

    rows: list[dict[str, object]] = []
    mode_before = pi_hist[0].copy()
    transition_pair: tuple[list[int], list[int]] | None = None

    for k in range(steps):
        candidate = pi_candidate_hist[k].copy()
        if switch_reason[k] == "transition_start":
            transition_pair = (mode_before.copy(), candidate.copy())

        if transition_pair is not None and blend_active[k]:
            pi_old = transition_pair[0].copy()
            pi_new = transition_pair[1].copy()
        else:
            pi_old = mode_before.copy()
            pi_new = candidate.copy()

        event_type: str | None = None
        if switch_reason[k] == "transition_start":
            event_type = "transition_start"
        elif switch_reason[k] == "immediate_switch":
            event_type = "immediate_switch"
        elif transition_pair is not None and blend_active[k]:
            event_type = "transition_active"
        elif candidate != mode_before:
            event_type = "candidate_change"

        if event_type is not None:
            x_snapshot = np.asarray(x_hist[k], dtype=float)
            if (
                logged_u_old is not None
                and logged_u_new is not None
                and logged_E_old is not None
                and logged_E_new is not None
            ):
                u_old = np.asarray(logged_u_old[k], dtype=float)
                u_new = np.asarray(logged_u_new[k], dtype=float)
                E_old = np.asarray(logged_E_old[k], dtype=float)
                E_new = np.asarray(logged_E_new[k], dtype=float)
            else:
                zeta = _zeta_from_log(sim, k, cfg)
                u_old, u_new, E_old, E_new = reconstruct_same_snapshot_pair(
                    x=x_snapshot,
                    zeta=zeta,
                    pi_old=pi_old,
                    pi_new=pi_new,
                    cfg=cfg,
                )

            full_terms = {(int(j), int(i)) for (j, i) in active_edges(x_snapshot, cfg.system)}
            old_terms = _included_pair_terms(x_snapshot, pi_old, cfg, eval_mode)
            new_terms = _included_pair_terms(x_snapshot, pi_new, cfg, eval_mode)
            follower_diff_count_vector = _followerwise_diff_count_vector(
                old_terms=old_terms,
                new_terms=new_terms,
                n_agents=int(cfg.system.N),
            )
            m_bound_count_total = int(np.sum(follower_diff_count_vector))
            E_gap = float(np.linalg.norm(E_new - E_old))
            u_gap = float(np.linalg.norm(u_new - u_old))
            abs_error = float(abs(u_gap - E_gap))
            gap_rel_error = float(abs_error / max(E_gap, 1.0e-12))

            tie_gap_min = float(sim["tie_gap_min"][k]) if "tie_gap_min" in sim else np.inf
            tie_pair = (
                int(sim["tie_i"][k]) if "tie_i" in sim else -1,
                int(sim["tie_j"][k]) if "tie_j" in sim else -1,
            )

            rows.append(
                {
                    "step": int(k),
                    "time_sec": float(t_control[k]),
                    "event_type": event_type,
                    "pi_old": json.dumps([int(v) for v in pi_old]),
                    "pi_new": json.dumps([int(v) for v in pi_new]),
                    "E_mode": eval_mode,
                    "E_gap": E_gap,
                    "u_gap": u_gap,
                    "gap_abs_error": abs_error,
                    "gap_rel_error": gap_rel_error,
                    "tie_gap_min": tie_gap_min,
                    "tie_pair": json.dumps([int(tie_pair[0]), int(tie_pair[1])]),
                    "blend_active": bool(blend_active[k]),
                    "switching_context": switch_reason[k],
                    "active_edge_count_full": int(len(full_terms)),
                    "active_edge_count_old": int(len(old_terms)),
                    "active_edge_count_new": int(len(new_terms)),
                    "kernel_single_term_bound": kernel_single_term_bound if m_bound_count_total > 0 else 0.0,
                    "m_bound_count_total": m_bound_count_total,
                    "follower_diff_count_vector": json.dumps([int(v) for v in follower_diff_count_vector]),
                    "E_gap_bound_simple": float(np.sqrt(m_bound_count_total) * kernel_single_term_bound)
                    if m_bound_count_total > 0
                    else 0.0,
                    "E_gap_bound_refined": float(
                        kernel_single_term_bound * np.sqrt(np.sum(follower_diff_count_vector.astype(float) ** 2.0))
                    )
                    if m_bound_count_total > 0
                    else 0.0,
                }
            )

        if transition_pair is not None and (k == steps - 1 or not bool(blend_active[k + 1])):
            transition_pair = None
        mode_before = pi_hist[k].copy()

    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "time_sec",
        "event_type",
        "pi_old",
        "pi_new",
        "E_mode",
        "E_gap",
        "u_gap",
        "gap_abs_error",
        "gap_rel_error",
        "tie_gap_min",
        "tie_pair",
        "blend_active",
        "switching_context",
        "active_edge_count_full",
        "active_edge_count_old",
        "active_edge_count_new",
        "kernel_single_term_bound",
        "m_bound_count_total",
        "follower_diff_count_vector",
        "E_gap_bound_simple",
        "E_gap_bound_refined",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        _save_empty_plot(out_path, title, "No events available")
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


def _tie_margin_plot(rows: list[dict[str, object]], out_path: Path) -> None:
    tie_gap = np.asarray([float(r["tie_gap_min"]) for r in rows], dtype=float)
    u_gap = np.asarray([float(r["u_gap"]) for r in rows], dtype=float)
    mask = np.isfinite(tie_gap)
    if not np.any(mask):
        _save_empty_plot(out_path, "u_gap vs tie margin", "No finite tie margins logged")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tie_gap[mask], u_gap[mask], ".", alpha=0.75)
    ax.set_xlabel("tie_gap_min")
    ax.set_ylabel("u_gap")
    ax.set_title("Same-snapshot raw gap vs tie margin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hist_plot(values: np.ndarray, *, xlabel: str, title: str, out_path: Path) -> None:
    if values.size == 0:
        _save_empty_plot(out_path, title, "No events available")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(30, max(8, values.size)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_payload(rows: list[dict[str, object]], cfg, eval_mode: str) -> dict[str, object]:
    E_gap = np.asarray([float(r["E_gap"]) for r in rows], dtype=float)
    u_gap = np.asarray([float(r["u_gap"]) for r in rows], dtype=float)
    abs_error = np.asarray([float(r["gap_abs_error"]) for r in rows], dtype=float)
    simple_bounds = np.asarray([float(r["E_gap_bound_simple"]) for r in rows], dtype=float)
    refined_bounds = np.asarray([float(r["E_gap_bound_refined"]) for r in rows], dtype=float)
    consistent = np.asarray([_consistency_ok(float(r["u_gap"]), float(r["E_gap"])) for r in rows], dtype=bool)
    simple_positive_bound = simple_bounds > 0.0
    refined_positive_bound = refined_bounds > 0.0
    simple_bound_ratio = (
        E_gap[simple_positive_bound] / simple_bounds[simple_positive_bound] if np.any(simple_positive_bound) else np.array([], dtype=float)
    )
    refined_bound_ratio = (
        E_gap[refined_positive_bound] / refined_bounds[refined_positive_bound] if np.any(refined_positive_bound) else np.array([], dtype=float)
    )
    simple_bound_ok = simple_bound_ratio <= (1.0 + 1.0e-9) if simple_bound_ratio.size else np.array([], dtype=bool)
    refined_bound_ok = refined_bound_ratio <= (1.0 + 1.0e-9) if refined_bound_ratio.size else np.array([], dtype=bool)

    inconsistent_rows = [
        {
            "step": int(r["step"]),
            "event_type": str(r["event_type"]),
            "gap_abs_error": float(r["gap_abs_error"]),
            "E_gap": float(r["E_gap"]),
            "u_gap": float(r["u_gap"]),
        }
        for r in sorted(rows, key=lambda row: float(row["gap_abs_error"]), reverse=True)[:5]
        if not _consistency_ok(float(r["u_gap"]), float(r["E_gap"]))
    ]
    bound_violation_examples = [
        {
            "step": int(r["step"]),
            "event_type": str(r["event_type"]),
            "E_gap": float(r["E_gap"]),
            "E_gap_bound_simple": float(r["E_gap_bound_simple"]),
            "m_bound_count_total": int(r["m_bound_count_total"]),
            "pi_old": str(r["pi_old"]),
            "pi_new": str(r["pi_new"]),
        }
        for r in sorted(
            [row for row in rows if float(row["E_gap_bound_simple"]) > 0.0],
            key=lambda row: float(row["E_gap"]) / float(row["E_gap_bound_simple"]),
            reverse=True,
        )[:5]
        if float(r["E_gap_bound_simple"]) > 0.0 and float(r["E_gap"]) > float(r["E_gap_bound_simple"]) + 1.0e-9
    ]
    refined_bound_violation_examples = [
        {
            "step": int(r["step"]),
            "event_type": str(r["event_type"]),
            "E_gap": float(r["E_gap"]),
            "E_gap_bound_refined": float(r["E_gap_bound_refined"]),
            "follower_diff_count_vector": str(r["follower_diff_count_vector"]),
            "pi_old": str(r["pi_old"]),
            "pi_new": str(r["pi_new"]),
        }
        for r in sorted(
            [row for row in rows if float(row["E_gap_bound_refined"]) > 0.0],
            key=lambda row: float(row["E_gap"]) / float(row["E_gap_bound_refined"]),
            reverse=True,
        )[:5]
        if float(r["E_gap_bound_refined"]) > 0.0 and float(r["E_gap"]) > float(r["E_gap_bound_refined"]) + 1.0e-9
    ]

    return {
        "gate": "Gate 3 operator mismatch",
        "num_events": int(len(rows)),
        "evaluator_mode": str(eval_mode),
        "consistency_abs_tol": float(ABS_TOL),
        "consistency_rel_tol": float(REL_TOL),
        "E_gap_mean": float(np.mean(E_gap)) if E_gap.size else 0.0,
        "E_gap_median": float(np.median(E_gap)) if E_gap.size else 0.0,
        "E_gap_max": float(np.max(E_gap)) if E_gap.size else 0.0,
        "u_gap_mean": float(np.mean(u_gap)) if u_gap.size else 0.0,
        "u_gap_median": float(np.median(u_gap)) if u_gap.size else 0.0,
        "u_gap_max": float(np.max(u_gap)) if u_gap.size else 0.0,
        "gap_abs_error_max": float(np.max(abs_error)) if abs_error.size else 0.0,
        "gap_abs_error_mean": float(np.mean(abs_error)) if abs_error.size else 0.0,
        "gap_identity_fraction": float(np.mean(consistent)) if consistent.size else 1.0,
        "num_identity_failures": int(np.count_nonzero(~consistent)),
        "max_E_gap_to_simple_bound_ratio": float(np.max(simple_bound_ratio)) if simple_bound_ratio.size else 0.0,
        "max_E_gap_to_refined_bound_ratio": float(np.max(refined_bound_ratio)) if refined_bound_ratio.size else 0.0,
        "rows_with_positive_simple_bound": int(np.count_nonzero(simple_positive_bound)),
        "rows_with_positive_refined_bound": int(np.count_nonzero(refined_positive_bound)),
        "simple_bound_satisfaction_fraction": float(np.mean(simple_bound_ok)) if simple_bound_ok.size else 1.0,
        "num_simple_bound_violations": int(np.count_nonzero(~simple_bound_ok)) if simple_bound_ok.size else 0,
        "refined_bound_satisfaction_fraction": float(np.mean(refined_bound_ok)) if refined_bound_ok.size else 1.0,
        "num_refined_bound_violations": int(np.count_nonzero(~refined_bound_ok)) if refined_bound_ok.size else 0,
        "caveats": [
            "Rows use same-snapshot u_old/u_new and E_old/E_new from simulate_closed_loop logs when available.",
            "follower_diff_count_vector is indexed by follower agent and counts differing included directed terms terminating at that follower.",
            "The refined bound uses the current exponential surrogate kernel k_wake * exp(-gamma_edge / L) with follower-component-aware aggregation.",
            "The simple sqrt(m) bound is retained only as a looser comparison baseline.",
            "The simple sqrt(m) bound can underbound rows where multiple differing pairwise terms stack on the same follower component.",
        ],
        "unsupported_mode_caveat": None if supports_explicit_evaluator(cfg) else "Explicit evaluator unsupported for coupling_mode='upstream_u'.",
        "inconsistent_examples": inconsistent_rows,
        "simple_bound_violation_examples": bound_violation_examples,
        "refined_bound_violation_examples": refined_bound_violation_examples,
    }


def run_gate3_operator_mismatch(config_path: str = "configs/system.yaml") -> Path:
    cfg = load_config(config_path)
    if not supports_explicit_evaluator(cfg):
        raise ValueError("Gate 3 operator mismatch requires an evaluator-supported coupling mode.")

    out_dir = make_results_dir("gate3_operator_mismatch")
    sim = simulate_closed_loop(
        cfg,
        x0=_base_x0(cfg),
        horizon=float(getattr(cfg.gate3, "run_horizon", cfg.system.horizon)),
        options=SimOptions(blending_on=True, noise_delta=0.0, seed=int(cfg.seed) + 9100),
    )

    eval_mode = str(get_evaluator_mode(cfg))
    rows = _infer_event_rows(sim, cfg)

    table_path = out_dir / "operator_gap_table.csv"
    summary_path = out_dir / "operator_gap_summary.json"
    _write_csv(table_path, rows)
    dump_json(summary_path, _summary_payload(rows, cfg, eval_mode))

    E_gap = np.asarray([float(r["E_gap"]) for r in rows], dtype=float)
    u_gap = np.asarray([float(r["u_gap"]) for r in rows], dtype=float)
    simple_bounds = np.asarray([float(r["E_gap_bound_simple"]) for r in rows], dtype=float)
    refined_bounds = np.asarray([float(r["E_gap_bound_refined"]) for r in rows], dtype=float)

    _scatter_with_diagonal(
        xvals=E_gap,
        yvals=u_gap,
        xlabel="E_gap",
        ylabel="u_gap",
        title="Same-snapshot raw gap vs evaluator gap",
        out_path=out_dir / "u_gap_vs_E_gap.png",
    )
    _tie_margin_plot(rows, out_dir / "u_gap_vs_tie_margin.png")
    _hist_plot(E_gap, xlabel="E_gap", title="Evaluator gap histogram", out_path=out_dir / "E_gap_hist.png")
    _scatter_with_diagonal(
        xvals=simple_bounds,
        yvals=E_gap,
        xlabel="E_gap_bound_simple",
        ylabel="E_gap",
        title="Measured evaluator gap vs simple bound",
        out_path=out_dir / "bound_vs_measured.png",
    )
    _scatter_with_diagonal(
        xvals=refined_bounds,
        yvals=E_gap,
        xlabel="E_gap_bound_refined",
        ylabel="E_gap",
        title="Measured evaluator gap vs refined bound",
        out_path=out_dir / "refined_bound_vs_measured.png",
    )

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3 operator-mismatch diagnostics")
    parser.add_argument("--config", default="configs/system.yaml")
    args = parser.parse_args()
    path = run_gate3_operator_mismatch(args.config)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

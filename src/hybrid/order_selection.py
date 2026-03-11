"""Shadow-mode candidate generation and lexicographic order selection."""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.flatness.evaluation_operator import compute_evaluator, supports_explicit_evaluator
from src.flatness.recursion import build_phi, psi
from src.hybrid.ordering import compute_ordering_metric
from src.model.graph import coupling_graph


def _as_pi(pi: list[int]) -> list[int]:
    return [int(v) for v in pi]


def _swap_pair_in_pi(pi: list[int], tie_pair: tuple[int, int] | None) -> list[int] | None:
    if tie_pair is None:
        return None
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


def candidate_orders(
    current_pi: list[int],
    pi_candidate: list[int],
    x: np.ndarray,
    cfg,
    *,
    tie_pair: tuple[int, int] | None = None,
) -> list[list[int]]:
    """Return a small auditable candidate set for shadow evaluation."""
    del x
    sel_cfg = getattr(cfg, "selection", None)
    max_candidates = int(getattr(sel_cfg, "max_candidates", 3)) if sel_cfg is not None else 3
    use_tie_swap = bool(getattr(sel_cfg, "use_tie_pair_swap_candidate", True)) if sel_cfg is not None else True

    seen: set[tuple[int, ...]] = set()
    candidates: list[list[int]] = []

    def add(pi: list[int] | None) -> None:
        if pi is None:
            return
        key = tuple(_as_pi(pi))
        if key in seen:
            return
        seen.add(key)
        candidates.append(list(key))

    add(current_pi)
    add(pi_candidate)
    if use_tie_swap:
        add(_swap_pair_in_pi(current_pi, tie_pair))
        if len(candidates) < max_candidates:
            add(_swap_pair_in_pi(pi_candidate, tie_pair))

    return candidates[:max_candidates]


def admissible_order(pi: list[int], x: np.ndarray, cfg) -> tuple[bool, dict[str, object]]:
    """Check DAG/topological consistency for one candidate ordering."""
    sys = cfg.system
    ord_cfg = cfg.ordering
    s_eval = compute_ordering_metric(x, sys, ord_cfg)
    g = coupling_graph(x, sys, mode={"pi": pi, "epsilon": float(ord_cfg.epsilon)}, s=s_eval)
    dag_ok = bool(nx.is_directed_acyclic_graph(g))
    pos = {int(agent): idx for idx, agent in enumerate(pi)}
    violating = [(int(u), int(v)) for (u, v) in g.edges() if pos[int(u)] >= pos[int(v)]]
    topo_ok = len(violating) == 0
    reconstruction_ok = True
    details = {
        "dag_ok": dag_ok,
        "topo_ok": topo_ok,
        "reconstruction_ok": reconstruction_ok,
        "num_edges": int(g.number_of_edges()),
        "violating_edges": [[int(a), int(b)] for (a, b) in violating[:10]],
    }
    return bool(dag_ok and topo_ok and reconstruction_ok), details


def conditioning_proxy(pi: list[int], x: np.ndarray, zeta: dict[str, np.ndarray], cfg) -> float:
    """Placeholder hook for future plants; constant for the current toy surrogate."""
    del pi, x, zeta, cfg
    return 0.0


def predicted_raw_gap_proxy(
    current_pi: list[int],
    candidate_pi: list[int],
    x: np.ndarray,
    zeta: dict[str, np.ndarray],
    cfg,
) -> float:
    """Compute same-snapshot raw mismatch between two candidate orders."""
    sys = cfg.system
    if _as_pi(current_pi) == _as_pi(candidate_pi):
        return 0.0
    if supports_explicit_evaluator(cfg):
        E_curr = compute_evaluator(x=x, pi=current_pi, cfg=cfg)
        E_cand = compute_evaluator(x=x, pi=candidate_pi, cfg=cfg)
        phi_curr = build_phi(x=x, zeta=zeta, pi=current_pi, sys=cfg, params=sys, evaluator_output=E_curr)
        phi_cand = build_phi(x=x, zeta=zeta, pi=candidate_pi, sys=cfg, params=sys, evaluator_output=E_cand)
    else:
        phi_curr = build_phi(x=x, zeta=zeta, pi=current_pi, sys=cfg, params=sys)
        phi_cand = build_phi(x=x, zeta=zeta, pi=candidate_pi, sys=cfg, params=sys)
    u_curr = psi(phi_curr, params=sys)
    u_cand = psi(phi_cand, params=sys)
    return float(np.linalg.norm(u_cand - u_curr))


def lexicographic_shadow_select(
    current_pi: list[int],
    pi_candidate: list[int],
    x: np.ndarray,
    zeta: dict[str, np.ndarray],
    cfg,
    *,
    tie_pair: tuple[int, int] | None = None,
    tie_margin: float | None = None,
) -> dict[str, object]:
    """Compute a shadow order choice without mutating controller state."""
    candidates = candidate_orders(current_pi, pi_candidate, x, cfg, tie_pair=tie_pair)
    evaluated: list[dict[str, object]] = []
    for pi in candidates:
        admissible, details = admissible_order(pi, x, cfg)
        cond = conditioning_proxy(pi, x, zeta, cfg)
        gap = predicted_raw_gap_proxy(current_pi, pi, x, zeta, cfg)
        evaluated.append(
            {
                "pi": _as_pi(pi),
                "admissible": bool(admissible),
                "conditioning": float(cond),
                "predicted_gap": float(gap),
                "details": details,
            }
        )

    admissible_eval = [row for row in evaluated if bool(row["admissible"])]
    reason = "no_admissible_candidates_hold_current"
    selected = _as_pi(current_pi)
    if admissible_eval:
        best_cond = min(float(row["conditioning"]) for row in admissible_eval)
        cond_filtered = [row for row in admissible_eval if abs(float(row["conditioning"]) - best_cond) <= 1.0e-12]
        current_match = next((row for row in cond_filtered if _as_pi(row["pi"]) == _as_pi(current_pi)), None)
        if current_match is not None:
            selected = _as_pi(current_pi)
            reason = "current_retained_after_admissibility_and_conditioning"
        else:
            best_gap_row = min(cond_filtered, key=lambda row: float(row["predicted_gap"]))
            selected = _as_pi(best_gap_row["pi"])
            reason = "selected_min_predicted_gap_after_admissibility_and_conditioning"

    selected_row = next((row for row in evaluated if _as_pi(row["pi"]) == _as_pi(selected)), None)
    selected_details = dict(selected_row["details"]) if selected_row is not None else {}
    selected_admissible = bool(selected_row["admissible"]) if selected_row is not None else False
    selected_conditioning = float(selected_row["conditioning"]) if selected_row is not None else float("inf")
    selected_gap = float(selected_row["predicted_gap"]) if selected_row is not None else float("inf")

    return {
        "selected_pi": selected,
        "current_pi": _as_pi(current_pi),
        "candidate_pis": [_as_pi(row["pi"]) for row in evaluated],
        "candidate_set": [_as_pi(row["pi"]) for row in evaluated],
        "admissible_mask": [bool(row["admissible"]) for row in evaluated],
        "conditioning_scores": [float(row["conditioning"]) for row in evaluated],
        "predicted_gap_scores": [float(row["predicted_gap"]) for row in evaluated],
        "details": [row["details"] for row in evaluated],
        "reason": reason,
        "selection_reason": reason,
        "differs_from_live": bool(_as_pi(selected) != _as_pi(pi_candidate)),
        "tie_margin": float(tie_margin) if tie_margin is not None else float("inf"),
        "predicted_gap": selected_gap,
        "conditioning_proxy": selected_conditioning,
        "admissible": selected_admissible,
        "dag_ok": bool(selected_details.get("dag_ok", False)),
        "topo_ok": bool(selected_details.get("topo_ok", False)),
    }

"""Closed-loop simulator for hybrid ordering flatness control."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.control.flat_tracking import virtual_input
from src.experiments.scenario_crossing import sinusoidal_crossing_reference
from src.flatness.recursion import build_phi, psi
from src.hybrid.blending import blend, blend_progress
from src.hybrid.ordering import (
    OrderingState,
    compute_ordering_metric,
    compute_pi,
    rho_global,
    tie_gap_info,
)
from src.model.dynamics import f, split_state
from src.model.graph import coupling_graph


@dataclass
class SimOptions:
    """Simulation runtime options."""

    blending_on: bool = False
    noise_delta: float = 0.0
    seed: int = 0
    lockout_sec_override: float | None = None
    force_fixed_pi: bool = False
    fixed_pi: list[int] | None = None
    disable_switching: bool = False


def _inverse_permutation(pi: list[int]) -> list[int]:
    inv = np.zeros(len(pi), dtype=int)
    for rank, agent in enumerate(pi):
        inv[int(agent)] = int(rank)
    return [int(v) for v in inv]


def simulate_closed_loop(
    cfg,
    x0: np.ndarray | None,
    horizon: float | None = None,
    options: SimOptions | None = None,
) -> dict[str, np.ndarray | list | str | float]:
    """Simulate closed loop and return rich logs for verification gates."""
    options = options or SimOptions()

    sys = cfg.system
    ord_cfg = cfg.ordering
    ctrl = cfg.controller
    ref_cfg = cfg.reference

    dt = float(sys.dt)
    T = float(horizon if horizon is not None else sys.horizon)
    steps = int(np.floor(T / dt))

    rng = np.random.default_rng(options.seed)
    if x0 is None:
        # Deterministic fallback for quick diagnostics.
        x = np.concatenate(
            [
                np.array(ref_cfg.base, dtype=float),
                np.zeros(sys.N, dtype=float),
            ],
            axis=0,
        )
    else:
        x = x0.astype(float).copy()

    t_hist = np.linspace(0.0, steps * dt, steps + 1)
    x_hist = np.zeros((steps + 1, x.size), dtype=float)
    u_applied_hist = np.zeros((steps, sys.N), dtype=float)
    u_old_hist = np.zeros((steps, sys.N), dtype=float)
    u_new_hist = np.zeros((steps, sys.N), dtype=float)
    y_ref_hist = np.zeros((steps, sys.N), dtype=float)
    ydot_ref_hist = np.zeros((steps, sys.N), dtype=float)
    yddot_ref_hist = np.zeros((steps, sys.N), dtype=float)
    pi_hist: list[list[int]] = []
    pi_inv_hist: list[list[int]] = []
    pi_candidate_hist: list[list[int]] = []
    rho_hist = np.zeros(steps, dtype=float)
    dag_hist = np.zeros(steps, dtype=bool)
    topo_hist = np.zeros(steps, dtype=bool)
    edges_hist: list[list[tuple[int, int]]] = []
    num_edges_hist = np.zeros(steps, dtype=int)
    switch_times: list[float] = []
    switch_steps: list[int] = []
    topo_failures: list[dict] = []

    # v2 logging fields.
    s_hist = np.zeros((steps, sys.N), dtype=float)
    s_sorted_hist = np.zeros((steps, sys.N), dtype=float)
    tie_gap_min_hist = np.full(steps, np.inf, dtype=float)
    tie_i_hist = -np.ones(steps, dtype=int)
    tie_j_hist = -np.ones(steps, dtype=int)

    switch_event_hist = np.zeros(steps, dtype=bool)
    switch_reason_hist: list[str] = []
    lockout_remaining_hist = np.zeros(steps, dtype=int)

    w_blend_hist = np.zeros(steps, dtype=float)
    blend_active_hist = np.zeros(steps, dtype=bool)
    blend_id_hist = -np.ones(steps, dtype=int)
    blend_step_hist = -np.ones(steps, dtype=int)

    J_raw_hist = np.zeros(steps, dtype=float)
    J_hist = np.zeros(steps, dtype=float)
    jump_ratio_hist = np.zeros(steps, dtype=float)
    du_old_hist = np.zeros(steps, dtype=float)
    du_new_hist = np.zeros(steps, dtype=float)

    y_hist = np.zeros((steps, sys.N), dtype=float)
    ydot_hist = np.zeros((steps, sys.N), dtype=float)
    zeta_r_hist = np.zeros((steps, 3 * sys.N), dtype=float)
    e_hist = np.zeros((steps, 2 * sys.N), dtype=float)
    e_norm_hist = np.zeros(steps, dtype=float)

    info_ok_new_upstream_hist = np.ones(steps, dtype=bool)
    ordering_consensus_ok_hist = np.ones(steps, dtype=bool)
    consensus_rounds_used_hist = np.zeros(steps, dtype=int)

    phi_snapshots: list[dict[str, object]] = []
    logging_cfg = getattr(cfg, "logging", None)
    store_phi_all = bool(getattr(logging_cfg, "store_phi", False)) if logging_cfg is not None else False
    store_phi_debug = bool(getattr(logging_cfg, "store_phi_debug", False)) if logging_cfg is not None else False
    tie_gap_delta = float(getattr(ord_cfg, "tie_gap_delta", 0.05))

    x_hist[0] = x

    mode_state = OrderingState()
    eps = float(ord_cfg.epsilon)
    eta = float(ord_cfg.eta)
    lockout_samples = int(ord_cfg.lockout_samples)
    transition_blend_enable = bool(
        getattr(ord_cfg, "transition_blend_enable", getattr(ord_cfg, "blend_enabled", True))
    )
    transition_blend_sec = float(
        getattr(ord_cfg, "transition_blend_sec", getattr(ord_cfg, "blend_sec", 0.2))
    )
    transition_M = max(1, int(np.ceil(transition_blend_sec / max(dt, 1.0e-9))))
    eps_jump = float(getattr(getattr(cfg, "blending", object()), "eps_jump", 1.0e-6))

    blend_id_counter = 0
    active_blend_id = -1

    for k in range(steps):
        t = k * dt

        x1, x2 = split_state(x, sys.N)
        s_true = compute_ordering_metric(x, sys, ord_cfg)
        s_hat = s_true + rng.uniform(-options.noise_delta, options.noise_delta, size=sys.N)

        pi_candidate = compute_pi(s_hat)
        if options.force_fixed_pi:
            if options.fixed_pi is not None:
                pi_fixed = [int(v) for v in options.fixed_pi]
            elif mode_state.current_pi is None:
                pi_fixed = pi_candidate.copy()
            else:
                pi_fixed = mode_state.current_pi.copy()
            pi_candidate = pi_fixed.copy()

        if mode_state.current_pi is None:
            mode_state.current_pi = pi_candidate.copy()
            mode_state.last_switch_t = t
            mode_state.last_switch_k = k
        pi_mode = mode_state.current_pi.copy()

        # Evaluate topological consistency against a single consistent snapshot.
        s_eval = s_hat.copy()
        pi_eval = pi_candidate
        g = coupling_graph(
            x,
            sys,
            mode={"pi": pi_eval, "epsilon": eps},
            s=s_eval,
        )
        dag = nx.is_directed_acyclic_graph(g)
        rank = {node: idx for idx, node in enumerate(pi_eval)}
        violations = []
        for (j, i) in g.edges():
            if rank[j] >= rank[i]:
                violations.append(
                    {
                        "j": int(j),
                        "i": int(i),
                        "sj": float(s_eval[j]),
                        "si": float(s_eval[i]),
                        "ds": float(s_eval[j] - s_eval[i]),
                        "rank_j": int(rank[j]),
                        "rank_i": int(rank[i]),
                    }
                )
        topo_ok = len(violations) == 0
        if not topo_ok:
            topo_failures.append(
                {
                    "k": int(k),
                    "t": float(t),
                    "s": [float(v) for v in s_eval],
                    "pi_eval": [int(v) for v in pi_eval],
                    "pi_mode": [int(v) for v in pi_mode],
                    "edges": [[int(a), int(b)] for (a, b) in g.edges()],
                    "violations": violations[:10],
                }
            )

        y_ref, ydot_ref, yddot_ref = sinusoidal_crossing_reference(t, ref_cfg)
        y_ref_hist[k] = y_ref
        ydot_ref_hist[k] = ydot_ref
        yddot_ref_hist[k] = yddot_ref

        v = virtual_input(
            y=x1,
            ydot=x2,
            y_ref=y_ref,
            ydot_ref=ydot_ref,
            yddot_ref=yddot_ref,
            kp=float(ctrl.kp),
            kd=float(ctrl.kd),
        )
        zeta = {"y": x1, "ydot": x2, "v": v}

        y_hist[k] = x1
        ydot_hist[k] = x2
        zeta_r_hist[k] = np.concatenate([x1, x2, v], axis=0)
        e_vec = np.concatenate([x1 - y_ref, x2 - ydot_ref], axis=0)
        e_hist[k] = e_vec
        e_norm_hist[k] = float(np.linalg.norm(e_vec))

        s_hist[k] = s_hat
        pi_sorted = compute_pi(s_hat)
        s_sorted_hist[k] = s_hat[pi_sorted]
        tie_gap_min, tie_pair = tie_gap_info(s_hat, pi_sorted)
        tie_gap_min_hist[k] = tie_gap_min
        tie_i_hist[k] = int(tie_pair[0])
        tie_j_hist[k] = int(tie_pair[1])

        rho = rho_global(s_hat)
        rho_hist[k] = rho

        lockout_sec = (
            float(options.lockout_sec_override)
            if options.lockout_sec_override is not None
            else float(ord_cfg.lockout_sec)
        )
        lockout_time_active = (t - mode_state.last_switch_t) < lockout_sec
        lockout_sample_active = (k - mode_state.last_switch_k) < lockout_samples
        lockout_remaining = max(0, lockout_samples - max(0, k - mode_state.last_switch_k))
        lockout_remaining_hist[k] = int(lockout_remaining)

        can_switch = (
            (not mode_state.transition_active)
            and pi_candidate != pi_mode
            and (not lockout_time_active)
            and (not lockout_sample_active)
            and rho >= eps
        )
        if options.disable_switching:
            can_switch = False

        switched_immediate = False
        switch_event = False
        switch_reason = "none"
        if can_switch:
            if options.blending_on and transition_blend_enable and transition_M > 0:
                mode_state.transition_active = True
                mode_state.transition_target_pi = pi_candidate.copy()
                mode_state.transition_start_k = k
                mode_state.transition_M = transition_M
                active_blend_id = blend_id_counter
                blend_id_counter += 1
                # For blended runs, switch event is transition start.
                switch_times.append(t)
                switch_steps.append(k)
                switch_event = True
                switch_reason = "transition_start"
            else:
                mode_state.current_pi = pi_candidate.copy()
                mode_state.last_switch_t = t
                mode_state.last_switch_k = k
                switch_times.append(t)
                switch_steps.append(k)
                switched_immediate = True
                switch_event = True
                switch_reason = "immediate_switch"

        if mode_state.transition_active:
            old_pi = mode_state.current_pi.copy()
            new_pi = (
                mode_state.transition_target_pi.copy()
                if mode_state.transition_target_pi is not None
                else pi_candidate.copy()
            )
            phi_old = build_phi(x=x, zeta=zeta, pi=old_pi, sys=sys, params=sys)
            u_old = psi(phi=phi_old, params=sys)
            phi_new = build_phi(x=x, zeta=zeta, pi=new_pi, sys=sys, params=sys)
            u_new = psi(phi=phi_new, params=sys)

            elapsed = max(0, k - mode_state.transition_start_k)
            progress = float(np.clip(elapsed / float(max(mode_state.transition_M, 1)), 0.0, 1.0))
            u_applied = blend_progress(u_old=u_old, u_new=u_new, progress=progress)

            blend_active = True
            blend_step = int(elapsed)
            blend_id = int(active_blend_id)
            w_blend = progress

            if elapsed >= mode_state.transition_M:
                mode_state.transition_active = False
                mode_state.current_pi = new_pi.copy()
                mode_state.last_switch_t = t
                mode_state.last_switch_k = k
                mode_state.transition_target_pi = None
                mode_state.transition_start_k = -1
                mode_state.transition_M = 0
                active_blend_id = -1
        else:
            phi_old = build_phi(x=x, zeta=zeta, pi=pi_mode, sys=sys, params=sys)
            u_old = psi(phi=phi_old, params=sys)
            phi_new = build_phi(x=x, zeta=zeta, pi=pi_candidate, sys=sys, params=sys)
            u_new = psi(phi=phi_new, params=sys)

            if switched_immediate:
                u_applied = u_new
            elif pi_candidate == pi_mode:
                u_applied = u_old
            elif options.blending_on and not transition_blend_enable:
                # Optional rho-band fallback if transition blending is disabled.
                u_applied = blend(
                    u_old=u_old,
                    u_new=u_new,
                    rho=rho,
                    eps=eps,
                    eta=eta,
                )
            else:
                # Keep persistent-mode control until immediate switch or transition starts.
                u_applied = u_old

            blend_active = False
            blend_step = -1
            blend_id = -1
            w_blend = 0.0

        # Same-snapshot raw mismatch and applied jump metrics.
        J_raw = float(np.linalg.norm(u_new - u_old))
        if k == 0:
            J = 0.0
            du_old = 0.0
            du_new = 0.0
        else:
            J = float(np.linalg.norm(u_applied - u_applied_hist[k - 1]))
            du_old = float(np.linalg.norm(u_old - u_old_hist[k - 1]))
            du_new = float(np.linalg.norm(u_new - u_new_hist[k - 1]))

        J_raw_hist[k] = J_raw
        J_hist[k] = J
        du_old_hist[k] = du_old
        du_new_hist[k] = du_new
        if J_raw < 1.0e-12:
            jump_ratio_hist[k] = 0.0
        else:
            jump_ratio_hist[k] = float(J / (J_raw + eps_jump))

        if store_phi_all or tie_gap_min <= tie_gap_delta:
            snap: dict[str, object] = {
                "k": int(k),
                "t": float(t),
                "x": x.copy(),
                "zeta_r": zeta_r_hist[k].copy(),
                "pi": [int(v) for v in pi_mode],
                "pi_candidate": [int(v) for v in pi_candidate],
                "tie_gap_min": float(tie_gap_min),
                "tie_pair": (int(tie_pair[0]), int(tie_pair[1])),
            }
            if store_phi_debug:
                snap["phi_old_k3"] = np.array([float(phi_old[(i, 3)]) for i in range(sys.N)], dtype=float)
                snap["phi_new_k3"] = np.array([float(phi_new[(i, 3)]) for i in range(sys.N)], dtype=float)
                snap["u_old"] = u_old.copy()
                snap["u_new"] = u_new.copy()
            phi_snapshots.append(snap)

        # Euler integration with fixed step.
        xdot = f(x, u_applied, sys)
        x = x + dt * xdot

        x_hist[k + 1] = x
        u_applied_hist[k] = u_applied
        u_old_hist[k] = u_old
        u_new_hist[k] = u_new
        pi_hist.append(mode_state.current_pi.copy())
        pi_inv_hist.append(_inverse_permutation(mode_state.current_pi))
        pi_candidate_hist.append(pi_candidate.copy())

        dag_hist[k] = dag
        topo_hist[k] = topo_ok
        edges_hist.append(list(g.edges()))
        num_edges_hist[k] = int(g.number_of_edges())

        switch_event_hist[k] = switch_event
        switch_reason_hist.append(switch_reason)
        w_blend_hist[k] = w_blend
        blend_active_hist[k] = blend_active
        blend_id_hist[k] = int(blend_id)
        blend_step_hist[k] = int(blend_step)

    return {
        # Existing keys (v1-compatible)
        "t": t_hist,
        "x": x_hist,
        "u_applied": u_applied_hist,
        "u_old": u_old_hist,
        "u_new": u_new_hist,
        "y_ref": y_ref_hist,
        "ydot_ref": ydot_ref_hist,
        "yddot_ref": yddot_ref_hist,
        "pi": pi_hist,
        "pi_candidate": pi_candidate_hist,
        "rho": rho_hist,
        "dag": dag_hist,
        "topo": topo_hist,
        "edges": edges_hist,
        "num_edges": num_edges_hist,
        "switch_times": switch_times,
        "switch_steps": switch_steps,
        "topo_failures": topo_failures,
        "dt": dt,
        "horizon": T,

        # v2 log schema additions
        "log_schema_version": "2.0",
        # State-indexed timeline arrays have length steps+1; control-indexed arrays have length steps.
        "k": np.arange(steps, dtype=int),
        "k_state": np.arange(steps + 1, dtype=int),
        "t_state": t_hist,
        "t_control": t_hist[:-1].copy(),
        "dt_series": np.full(steps, dt, dtype=float),
        "s": s_hist,
        "s_sorted": s_sorted_hist,
        "tie_gap_min": tie_gap_min_hist,
        "tie_i": tie_i_hist,
        "tie_j": tie_j_hist,
        "pi_inv": pi_inv_hist,
        "switch_event": switch_event_hist,
        "switch_reason": switch_reason_hist,
        "lockout_remaining_steps": lockout_remaining_hist,
        "w_blend": w_blend_hist,
        "blend_active": blend_active_hist,
        "blend_id": blend_id_hist,
        "blend_step": blend_step_hist,
        "J_raw": J_raw_hist,
        "J": J_hist,
        "jump_ratio": jump_ratio_hist,
        "du_old": du_old_hist,
        "du_new": du_new_hist,
        "y": y_hist,
        "ydot": ydot_hist,
        "zeta_r": zeta_r_hist,
        "e": e_hist,
        "e_norm": e_norm_hist,
        "G_edges": edges_hist,
        "is_DAG": dag_hist,
        "topo_ok": topo_hist,
        "info_ok_new_upstream": info_ok_new_upstream_hist,
        "ordering_consensus_ok": ordering_consensus_ok_hist,
        "consensus_rounds_used": consensus_rounds_used_hist,
        "phi_snapshots": phi_snapshots,
    }

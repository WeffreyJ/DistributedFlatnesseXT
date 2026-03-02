"""Closed-loop simulator for hybrid ordering flatness control."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.control.flat_tracking import virtual_input
from src.experiments.scenario_crossing import sinusoidal_crossing_reference
from src.flatness.recursion import compute_phi, psi
from src.hybrid.blending import blend, blend_progress
from src.hybrid.ordering import OrderingState, compute_pi, rho_global
from src.model.coupling import s_metric
from src.model.dynamics import f, split_state
from src.model.graph import coupling_graph


@dataclass
class SimOptions:
    """Simulation runtime options."""

    blending_on: bool = False
    noise_delta: float = 0.0
    seed: int = 0
    lockout_sec_override: float | None = None


def simulate_closed_loop(
    cfg,
    x0: np.ndarray,
    horizon: float | None = None,
    options: SimOptions | None = None,
) -> dict[str, np.ndarray | list]:
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
    x = x0.astype(float).copy()

    t_hist = np.linspace(0.0, steps * dt, steps + 1)
    x_hist = np.zeros((steps + 1, x.size), dtype=float)
    u_applied_hist = np.zeros((steps, sys.N), dtype=float)
    u_old_hist = np.zeros((steps, sys.N), dtype=float)
    u_new_hist = np.zeros((steps, sys.N), dtype=float)
    pi_hist: list[list[int]] = []
    pi_candidate_hist: list[list[int]] = []
    rho_hist = np.zeros(steps, dtype=float)
    dag_hist = np.zeros(steps, dtype=bool)
    topo_hist = np.zeros(steps, dtype=bool)
    edges_hist: list[list[tuple[int, int]]] = []
    switch_times: list[float] = []
    switch_steps: list[int] = []
    topo_failures: list[dict] = []

    x_hist[0] = x

    mode_state = OrderingState()
    eps = float(ord_cfg.epsilon)
    eta = float(ord_cfg.eta)
    lockout_samples = int(ord_cfg.lockout_samples)
    blend_enabled_cfg = bool(getattr(ord_cfg, "blend_enabled", True))
    blend_sec = float(getattr(ord_cfg, "blend_sec", 0.2))
    blend_steps = max(1, int(np.ceil(blend_sec / max(dt, 1.0e-9))))

    for k in range(steps):
        t = k * dt

        x1, x2 = split_state(x, sys.N)
        s_true = s_metric(x, sys)
        s_hat = s_true + rng.uniform(-options.noise_delta, options.noise_delta, size=sys.N)

        pi_candidate = compute_pi(s_hat)
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
        rho = rho_global(s_hat)

        lockout_sec = (
            float(options.lockout_sec_override)
            if options.lockout_sec_override is not None
            else float(ord_cfg.lockout_sec)
        )
        lockout_time_active = (t - mode_state.last_switch_t) < lockout_sec
        lockout_sample_active = (k - mode_state.last_switch_k) < lockout_samples
        can_switch = (
            (not mode_state.blend_active)
            and pi_candidate != pi_mode
            and (not lockout_time_active)
            and (not lockout_sample_active)
            and rho >= eps
        )
        if can_switch:
            if options.blending_on and blend_enabled_cfg and blend_steps > 0:
                mode_state.blend_active = True
                mode_state.blend_steps_remaining = blend_steps
                mode_state.blend_total_steps = blend_steps
                mode_state.blend_old_pi = pi_mode.copy()
                mode_state.blend_new_pi = pi_candidate.copy()
            else:
                mode_state.current_pi = pi_candidate.copy()
                mode_state.last_switch_t = t
                mode_state.last_switch_k = k
                switch_times.append(t)
                switch_steps.append(k)

        if mode_state.blend_active:
            old_pi = mode_state.blend_old_pi if mode_state.blend_old_pi is not None else pi_mode
            new_pi = mode_state.blend_new_pi if mode_state.blend_new_pi is not None else pi_candidate
            phi_old = compute_phi(zeta=zeta, x=x, pi=old_pi, params=sys)
            u_old = psi(phi=phi_old, params=sys)
            phi_new = compute_phi(zeta=zeta, x=x, pi=new_pi, params=sys)
            u_new = psi(phi=phi_new, params=sys)
            progress = 1.0 - (
                float(mode_state.blend_steps_remaining) / float(max(mode_state.blend_total_steps, 1))
            )
            u_applied = blend_progress(u_old=u_old, u_new=u_new, progress=progress)
            mode_state.blend_steps_remaining -= 1
            if mode_state.blend_steps_remaining <= 0:
                mode_state.blend_active = False
                mode_state.current_pi = new_pi.copy()
                mode_state.last_switch_t = t
                mode_state.last_switch_k = k
                switch_times.append(t)
                switch_steps.append(k)
                mode_state.blend_old_pi = None
                mode_state.blend_new_pi = None
                mode_state.blend_total_steps = 0
                mode_state.blend_steps_remaining = 0
        else:
            phi_old = compute_phi(zeta=zeta, x=x, pi=pi_mode, params=sys)
            u_old = psi(phi=phi_old, params=sys)
            phi_new = compute_phi(zeta=zeta, x=x, pi=pi_candidate, params=sys)
            u_new = psi(phi=phi_new, params=sys)
            if pi_candidate == pi_mode:
                u_applied = u_old
            elif options.blending_on and not blend_enabled_cfg:
                # Optional rho-band blending fallback if transition blending is disabled.
                u_applied = blend(
                    u_old=u_old,
                    u_new=u_new,
                    rho=rho,
                    eps=eps,
                    eta=eta,
                )
            else:
                u_applied = u_new

        # Euler integration with fixed step.
        xdot = f(x, u_applied, sys)
        x = x + dt * xdot

        x_hist[k + 1] = x
        u_applied_hist[k] = u_applied
        u_old_hist[k] = u_old
        u_new_hist[k] = u_new
        pi_hist.append(mode_state.current_pi.copy())
        pi_candidate_hist.append(pi_candidate.copy())
        rho_hist[k] = rho
        dag_hist[k] = dag
        topo_hist[k] = topo_ok
        edges_hist.append(list(g.edges()))

    return {
        "t": t_hist,
        "x": x_hist,
        "u_applied": u_applied_hist,
        "u_old": u_old_hist,
        "u_new": u_new_hist,
        "pi": pi_hist,
        "pi_candidate": pi_candidate_hist,
        "rho": rho_hist,
        "dag": dag_hist,
        "topo": topo_hist,
        "edges": edges_hist,
        "switch_times": switch_times,
        "switch_steps": switch_steps,
        "topo_failures": topo_failures,
        "dt": dt,
        "horizon": T,
    }

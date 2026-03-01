"""Closed-loop simulator for hybrid ordering flatness control."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.control.flat_tracking import virtual_input
from src.experiments.scenario_crossing import sinusoidal_crossing_reference
from src.flatness.recursion import compute_phi, psi
from src.hybrid.blending import blend
from src.hybrid.ordering import OrderingState, compute_pi, rho_global, step_mode
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
    u_hist = np.zeros((steps, sys.N), dtype=float)
    pi_hist: list[list[int]] = []
    rho_hist = np.zeros(steps, dtype=float)
    dag_hist = np.zeros(steps, dtype=bool)
    topo_hist = np.zeros(steps, dtype=bool)
    edges_hist: list[list[tuple[int, int]]] = []
    switch_times: list[float] = []
    switch_steps: list[int] = []
    topo_failures: list[dict] = []

    x_hist[0] = x

    mode_state = OrderingState()
    u_prev = np.zeros(sys.N, dtype=float)

    for k in range(steps):
        t = k * dt

        x1, x2 = split_state(x, sys.N)
        s_true = s_metric(x, sys)
        s_hat = s_true + rng.uniform(-options.noise_delta, options.noise_delta, size=sys.N)

        lockout_sec = (
            float(options.lockout_sec_override)
            if options.lockout_sec_override is not None
            else float(ord_cfg.lockout_sec)
        )
        pi_mode, did_switch = step_mode(
            x=x,
            t=t,
            k=k,
            state=mode_state,
            epsilon=float(ord_cfg.epsilon),
            lockout_sec=lockout_sec,
            lockout_samples=int(ord_cfg.lockout_samples),
            s_override=s_hat,
        )
        if did_switch:
            switch_times.append(t)
            switch_steps.append(k)

        # Evaluate topological consistency against a single consistent snapshot.
        s_eval = s_hat.copy()
        pi_eval = compute_pi(s_eval)
        g = coupling_graph(
            x,
            sys,
            mode={"pi": pi_eval, "epsilon": float(ord_cfg.epsilon)},
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
        phi = compute_phi(zeta=zeta, x=x, pi=pi_mode, params=sys)
        u_nom = psi(phi=phi, params=sys)

        rho = rho_global(x1)
        if options.blending_on:
            u = blend(
                u_a=u_prev,
                u_b=u_nom,
                rho=rho,
                eps=float(ord_cfg.epsilon),
                eta=float(ord_cfg.eta),
            )
        else:
            u = u_nom

        # Euler integration with fixed step.
        xdot = f(x, u, sys)
        x = x + dt * xdot

        x_hist[k + 1] = x
        u_hist[k] = u
        u_prev = u
        pi_hist.append(pi_mode.copy())
        rho_hist[k] = rho
        dag_hist[k] = dag
        topo_hist[k] = topo_ok
        edges_hist.append(list(g.edges()))

    return {
        "t": t_hist,
        "x": x_hist,
        "u": u_hist,
        "pi": pi_hist,
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

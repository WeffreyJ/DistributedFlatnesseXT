"""Gate 1: Graph compatibility and switching well-posedness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.config import load_config
from src.control.closed_loop import SimOptions, simulate_closed_loop
from src.verify.utils import dump_json, make_results_dir, seed_rng


def _sample_x0(cfg, rng: np.random.Generator) -> np.ndarray:
    n = cfg.system.N
    x1 = rng.uniform(cfg.x0.x1[0], cfg.x0.x1[1], size=n)
    x2 = rng.uniform(cfg.x0.x2[0], cfg.x0.x2[1], size=n)
    return np.concatenate([x1, x2], axis=0)


def _gate1_opts(cfg) -> dict[str, object]:
    g1 = getattr(cfg, "gate1", object())
    vg1 = getattr(getattr(cfg, "verify", object()), "gate1", object())
    return {
        "mc_runs": int(getattr(vg1, "mc_runs", getattr(g1, "mc_runs", 10))),
        "export_negative_vignette": bool(
            getattr(vg1, "export_negative_vignette", getattr(g1, "export_negative_vignette", False))
        ),
        "vignette_out_dir": str(
            getattr(vg1, "vignette_out_dir", getattr(g1, "vignette_out_dir", "results/vignette_negative"))
        ),
        "vignette_prefer": str(getattr(vg1, "vignette_prefer", getattr(g1, "vignette_prefer", "cycle"))),
        "vignette_window": int(getattr(vg1, "vignette_window", getattr(g1, "vignette_window", 80))),
        "vignette_max_episodes": int(
            getattr(vg1, "vignette_max_episodes", getattr(g1, "vignette_max_episodes", 200))
        ),
        "vignette_graph_mode": str(
            getattr(vg1, "vignette_graph_mode", getattr(g1, "vignette_graph_mode", "sim"))
        ),
    }


def _select_failure_step(sim: dict, prefer: str) -> tuple[str, int] | None:
    dag_bool = np.asarray(sim.get("is_DAG", sim.get("dag", [])), dtype=bool)
    topo_bool = np.asarray(sim.get("topo_ok", sim.get("topo", [])), dtype=bool)

    k_cycle = None
    if dag_bool.size > 0:
        bad = np.where(~dag_bool)[0]
        if bad.size > 0:
            k_cycle = int(bad[0])

    k_topo = None
    if topo_bool.size > 0:
        bad = np.where(~topo_bool)[0]
        if bad.size > 0:
            k_topo = int(bad[0])

    if prefer == "cycle":
        if k_cycle is not None:
            return "cycle", k_cycle
        if k_topo is not None:
            return "topo", k_topo
        return None
    if prefer == "topo":
        if k_topo is not None:
            return "topo", k_topo
        if k_cycle is not None:
            return "cycle", k_cycle
        return None

    # either
    if k_cycle is not None:
        return "cycle", k_cycle
    if k_topo is not None:
        return "topo", k_topo
    return None


def _build_vignette_edges(x: np.ndarray, cfg, mode: str) -> list[tuple[int, int]]:
    """Build alternative graph edges for controlled negative vignette selection."""
    n = int(cfg.system.N)
    s = np.asarray(x[:n], dtype=float)
    gamma_edge = float(getattr(cfg.system, "gamma_edge", getattr(cfg.system, "gamma", 0.0)))
    wake_rx = float(getattr(cfg.system, "wake_Rx", getattr(cfg.system, "R_coup", 3.0)))
    edges: list[tuple[int, int]] = []

    if mode == "sim":
        return edges
    if mode == "physical_forward_only":
        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                dx = float(s[j] - s[i])
                if dx > gamma_edge and dx < wake_rx:
                    edges.append((j, i))
        return edges
    if mode == "physical_all_edges":
        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                dx_abs = abs(float(s[j] - s[i]))
                if dx_abs > gamma_edge and dx_abs < wake_rx:
                    edges.append((j, i))
        return edges
    return edges


def _recompute_failures(sim: dict, cfg, mode: str) -> dict[str, object]:
    """Recompute DAG/topo against an alternative physical graph from x-history."""
    pi_hist = sim.get("pi", [])
    x_hist = np.asarray(sim.get("x", []), dtype=float)
    steps = len(pi_hist)
    dag_alt = np.ones(steps, dtype=bool)
    topo_alt = np.ones(steps, dtype=bool)
    edges_alt: list[list[tuple[int, int]]] = []
    cycle_alt: list[dict] = []
    topo_alt_fail: list[dict] = []

    for k in range(steps):
        xk = x_hist[k] if k < len(x_hist) else x_hist[-1]
        edges_k = _build_vignette_edges(np.asarray(xk, dtype=float), cfg, mode=mode)
        edges_alt.append(edges_k)
        G = nx.DiGraph()
        G.add_nodes_from(range(int(cfg.system.N)))
        G.add_edges_from(edges_k)
        is_dag = nx.is_directed_acyclic_graph(G)
        dag_alt[k] = bool(is_dag)
        if not is_dag:
            cyc_edges = []
            try:
                cyc = nx.find_cycle(G, orientation="original")
                cyc_edges = [[int(a), int(b)] for (a, b, _) in cyc]
            except Exception:
                pass
            cycle_alt.append(
                {
                    "step": int(k),
                    "source": f"recomputed_{mode}",
                    "edges": [list(e) for e in edges_k],
                    "cycle_edges": cyc_edges,
                }
            )

        pi_k = [int(v) for v in (pi_hist[k] if k < len(pi_hist) else [])]
        pos = {a: idx for idx, a in enumerate(pi_k)}
        violating = [(int(u), int(v)) for (u, v) in edges_k if u in pos and v in pos and pos[u] >= pos[v]]
        topo_ok = len(violating) == 0
        topo_alt[k] = bool(topo_ok)
        if not topo_ok:
            topo_alt_fail.append(
                {
                    "step": int(k),
                    "source": f"recomputed_{mode}",
                    "pi": [int(v) for v in pi_k],
                    "edges": [list(e) for e in edges_k],
                    "violating_edges": [[int(a), int(b)] for (a, b) in violating[:10]],
                }
            )

    return {
        "dag": dag_alt,
        "topo": topo_alt,
        "edges": edges_alt,
        "cycle_snapshots": cycle_alt,
        "topo_failures": topo_alt_fail,
    }


def _export_negative_vignette(
    sim: dict,
    cfg,
    cfg_path: str,
    out_dir: Path,
    failure_type: str,
    run_index: int,
    seed: int,
    x0: np.ndarray,
    k_star: int,
    window: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Control-indexed timeline.
    t_control = np.asarray(sim.get("t_control", []), dtype=float)
    if t_control.size == 0:
        t = np.asarray(sim.get("t", []), dtype=float)
        if t.size >= 2:
            t_control = t[:-1]
        else:
            t_control = np.arange(len(sim.get("u_applied", [])), dtype=float) * float(sim.get("dt", 1.0))

    e_norm = np.asarray(sim.get("e_norm", []), dtype=float)
    if e_norm.size == 0 and "e" in sim:
        e = np.asarray(sim["e"], dtype=float)
        if e.ndim == 2 and e.shape[0] > 0:
            e_norm = np.linalg.norm(e, axis=1)

    J_raw = np.asarray(sim.get("J_raw", []), dtype=float)
    J = np.asarray(sim.get("J", []), dtype=float)
    jump_ratio = np.asarray(sim.get("jump_ratio", []), dtype=float)
    tie_gap = np.asarray(sim.get("tie_gap_min", []), dtype=float)
    dag_bool = np.asarray(sim.get("is_DAG", sim.get("dag", [])), dtype=bool)
    topo_bool = np.asarray(sim.get("topo_ok", sim.get("topo", [])), dtype=bool)

    edges_hist = sim.get("G_edges", sim.get("edges", []))
    edges_k = []
    if isinstance(edges_hist, list) and 0 <= k_star < len(edges_hist):
        edges_k = [(int(a), int(b)) for (a, b) in edges_hist[k_star]]

    pi_hist = sim.get("pi", [])
    pi_k = []
    if isinstance(pi_hist, list) and 0 <= k_star < len(pi_hist):
        pi_k = [int(v) for v in pi_hist[k_star]]

    tie_i = sim.get("tie_i", None)
    tie_j = sim.get("tie_j", None)
    tie_pair = (-1, -1)
    if tie_i is not None and tie_j is not None:
        tie_i_arr = np.asarray(tie_i, dtype=int)
        tie_j_arr = np.asarray(tie_j, dtype=int)
        if 0 <= k_star < len(tie_i_arr) and 0 <= k_star < len(tie_j_arr):
            tie_pair = (int(tie_i_arr[k_star]), int(tie_j_arr[k_star]))

    payload = {
        "failure_type": str(failure_type),
        "seed": int(seed),
        "episode_index": int(run_index),
        "k_star": int(k_star),
        "t_star": float(t_control[k_star]) if 0 <= k_star < len(t_control) else float(k_star),
        "pi": pi_k,
        "tie_gap_min": float(tie_gap[k_star]) if 0 <= k_star < len(tie_gap) else None,
        "tie_pair": [int(tie_pair[0]), int(tie_pair[1])],
        "J_raw": float(J_raw[k_star]) if 0 <= k_star < len(J_raw) else None,
        "J": float(J[k_star]) if 0 <= k_star < len(J) else None,
        "jump_ratio": float(jump_ratio[k_star]) if 0 <= k_star < len(jump_ratio) else None,
        "e_norm": float(e_norm[k_star]) if 0 <= k_star < len(e_norm) else None,
        "num_edges": int(len(edges_k)),
        "cfg_path": str(cfg_path),
        "x0": np.asarray(x0, dtype=float).tolist(),
    }
    dump_json(out_dir / "vignette_negative.json", payload)

    with (out_dir / "edges_kstar.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst"])
        for a, b in edges_k:
            writer.writerow([a, b])

    # Graph plot.
    G = nx.DiGraph()
    G.add_nodes_from(range(int(cfg.system.N)))
    G.add_edges_from(edges_k)
    highlight: set[tuple[int, int]] = set()

    if failure_type == "cycle":
        try:
            cyc = nx.find_cycle(G, orientation="original")
            for u, v, _ in cyc:
                highlight.add((int(u), int(v)))
        except Exception:
            pass
    else:
        pos = {agent: idx for idx, agent in enumerate(pi_k)}
        for u, v in edges_k:
            if u in pos and v in pos and pos[u] > pos[v]:
                highlight.add((int(u), int(v)))
                break

    pos2 = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    nx.draw_networkx_nodes(G, pos2, ax=ax, node_size=350)
    nx.draw_networkx_labels(G, pos2, ax=ax, font_size=8)
    normal_edges = [e for e in edges_k if e not in highlight]
    if normal_edges:
        nx.draw_networkx_edges(G, pos2, ax=ax, edgelist=normal_edges, arrows=True, width=1.0, alpha=0.4)
    if highlight:
        nx.draw_networkx_edges(G, pos2, ax=ax, edgelist=list(highlight), arrows=True, width=2.5, alpha=0.9)
    ax.set_title(f"Negative vignette: {failure_type} at k={k_star}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "graph_kstar.png", dpi=150)
    plt.close(fig)

    n_ctrl = int(
        max(
            len(J),
            len(J_raw),
            len(e_norm),
            len(tie_gap),
            len(dag_bool),
            len(topo_bool),
            len(t_control),
        )
    )
    if n_ctrl <= 0:
        return

    k0 = max(0, int(k_star) - int(window))
    k1 = min(n_ctrl - 1, int(k_star) + int(window))
    ks = np.arange(k0, k1 + 1, dtype=int)

    def _series(arr: np.ndarray, fill: float = np.nan) -> np.ndarray:
        out = np.full(n_ctrl, fill, dtype=float)
        m = min(len(arr), n_ctrl)
        if m > 0:
            out[:m] = arr[:m]
        return out

    t_ser = _series(t_control, fill=np.nan)
    if np.all(np.isnan(t_ser[: len(ks)])):
        t_ser = np.arange(n_ctrl, dtype=float)
    e_ser = _series(e_norm, fill=np.nan)
    jraw_ser = _series(J_raw, fill=np.nan)
    j_ser = _series(J, fill=np.nan)
    tie_ser = _series(tie_gap, fill=np.nan)
    dag_ser = _series(dag_bool.astype(float), fill=np.nan)
    topo_ser = _series(topo_bool.astype(float), fill=np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(ks, e_ser[ks], label="e_norm")
    ax.plot(ks, j_ser[ks], label="J")
    ax.plot(ks, jraw_ser[ks], label="J_raw")
    ax.axvline(int(k_star), linestyle="--")
    ax.set_xlabel("k (control step)")
    ax.set_title("Trace window around failure")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "trace_window.png", dpi=150)
    plt.close(fig)

    with (out_dir / "trace_window.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "t", "e_norm", "J_raw", "J", "tie_gap_min", "is_DAG", "topo_ok"])
        for k in ks:
            writer.writerow(
                [
                    int(k),
                    float(t_ser[k]),
                    float(e_ser[k]),
                    float(jraw_ser[k]),
                    float(j_ser[k]),
                    float(tie_ser[k]),
                    bool(dag_ser[k] > 0.5) if not np.isnan(dag_ser[k]) else "",
                    bool(topo_ser[k] > 0.5) if not np.isnan(topo_ser[k]) else "",
                ]
            )


def run_gate1(
    cfg_path: str,
    export_negative_vignette: bool | None = None,
    vignette_out_dir: str | None = None,
    vignette_prefer: str | None = None,
    vignette_window: int | None = None,
    vignette_max_episodes: int | None = None,
    vignette_graph_mode: str | None = None,
) -> Path:
    cfg = load_config(cfg_path)
    out_dir = make_results_dir("gate1")
    rng = seed_rng(int(cfg.seed) + 101)

    opts = _gate1_opts(cfg)
    mc_runs = int(opts["mc_runs"])
    export_vignette = bool(opts["export_negative_vignette"])
    out_vignette = Path(str(opts["vignette_out_dir"]))
    prefer = str(opts["vignette_prefer"]).lower()
    window = int(opts["vignette_window"])
    max_eps = int(opts["vignette_max_episodes"])
    graph_mode = str(opts["vignette_graph_mode"]).lower()

    # CLI overrides (optional)
    if export_negative_vignette is not None:
        export_vignette = bool(export_negative_vignette)
    if vignette_out_dir is not None:
        out_vignette = Path(vignette_out_dir)
    if vignette_prefer is not None:
        prefer = str(vignette_prefer).lower()
    if vignette_window is not None:
        window = int(vignette_window)
    if vignette_max_episodes is not None:
        max_eps = int(vignette_max_episodes)
    if vignette_graph_mode is not None:
        graph_mode = str(vignette_graph_mode).lower()

    if prefer not in {"cycle", "topo", "either"}:
        prefer = "either"
    if graph_mode not in {"sim", "physical_forward_only", "physical_all_edges"}:
        graph_mode = "sim"

    dag_vals: list[float] = []
    topo_vals: list[float] = []
    all_inter_switch: list[float] = []
    all_switch_rates: list[float] = []
    cycle_snapshots: list[dict] = []
    topo_failures: list[dict] = []
    vignette_written = False

    for run in range(mc_runs):
        x0 = _sample_x0(cfg, rng)
        sim = simulate_closed_loop(
            cfg,
            x0=x0,
            options=SimOptions(blending_on=False, seed=int(cfg.seed) + 1000 + run),
        )

        # Optional offline graph recompute for controlled negative examples.
        if export_vignette and graph_mode != "sim":
            alt = _recompute_failures(sim, cfg, mode=graph_mode)
            sim_eval = dict(sim)
            sim_eval["dag"] = alt["dag"]
            sim_eval["topo"] = alt["topo"]
            sim_eval["is_DAG"] = alt["dag"]
            sim_eval["topo_ok"] = alt["topo"]
            sim_eval["edges"] = alt["edges"]
            sim_eval["G_edges"] = alt["edges"]
            sim_eval["topo_failures"] = alt["topo_failures"]
        else:
            alt = None
            sim_eval = sim

        dag = np.asarray(sim_eval["dag"], dtype=float)
        topo = np.asarray(sim_eval["topo"], dtype=float)
        dag_vals.append(float(np.mean(dag)))
        topo_vals.append(float(np.mean(topo)))

        # TODO: richer cycle-state debug snapshots for larger models.
        for k, is_dag in enumerate(sim_eval["dag"]):
            if not bool(is_dag):
                cycle_snapshots.append(
                    {
                        "run": run,
                        "step": int(k),
                        "source": "sim" if graph_mode == "sim" else f"recomputed_{graph_mode}",
                        "edges": [list(e) for e in sim_eval["edges"][k]],
                    }
                )
        for fail in sim_eval.get("topo_failures", []):
            item = dict(fail)
            item["run"] = int(run)
            topo_failures.append(item)

        # Optional negative vignette export from first failure encountered.
        if export_vignette and (not vignette_written) and run < max_eps:
            chosen = _select_failure_step(sim_eval, prefer=prefer)
            if chosen is not None:
                failure_type, k_star = chosen
                _export_negative_vignette(
                    sim=sim_eval,
                    cfg=cfg,
                    cfg_path=cfg_path,
                    out_dir=out_vignette,
                    failure_type=failure_type,
                    run_index=run,
                    seed=int(cfg.seed) + 1000 + run,
                    x0=x0,
                    k_star=int(k_star),
                    window=window,
                )
                vignette_written = True

        switch_times = np.array(sim["switch_times"], dtype=float)
        if switch_times.size >= 2:
            dts = np.diff(switch_times)
            all_inter_switch.extend(dts.tolist())
        rate = float(switch_times.size / max(float(sim["horizon"]), 1e-8))
        all_switch_rates.append(rate)

    min_inter_switch = float(min(all_inter_switch)) if all_inter_switch else float("inf")
    payload = {
        "gate": "Gate 1",
        "mc_runs": mc_runs,
        "dag_rate_mean": float(np.mean(dag_vals)),
        "topo_pass_rate_mean": float(np.mean(topo_vals)),
        "min_inter_switch_time": min_inter_switch,
        "switches_per_second_mean": float(np.mean(all_switch_rates)),
        "num_cycle_snapshots": len(cycle_snapshots),
        "cycle_snapshots": cycle_snapshots[:25],
        "num_topo_failures": len(topo_failures),
        "negative_vignette_export_enabled": bool(export_vignette),
        "negative_vignette_written": bool(vignette_written),
        "negative_vignette_out_dir": str(out_vignette),
        "negative_vignette_graph_mode": str(graph_mode),
    }

    dump_json(out_dir / "gate1_summary.json", payload)
    topo_path = out_dir / "topo_failures.jsonl"
    with topo_path.open("w", encoding="utf-8") as f:
        for row in topo_failures:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["DAG rate", "Topo pass"], [payload["dag_rate_mean"], payload["topo_pass_rate_mean"]])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Gate 1 Compatibility Rates")
    fig.tight_layout()
    fig.savefig(out_dir / "gate1_rates.png", dpi=150)
    plt.close(fig)

    if all_inter_switch:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(all_inter_switch, bins=min(20, len(all_inter_switch)))
        ax.set_title("Inter-switch Times")
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "gate1_interswitch_hist.png", dpi=150)
        plt.close(fig)

    return out_dir / "gate1_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 1 verification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--export_negative_vignette", action="store_true")
    parser.add_argument("--vignette_out_dir", default=None)
    parser.add_argument("--vignette_prefer", default=None, choices=["cycle", "topo", "either"])
    parser.add_argument("--vignette_window", type=int, default=None)
    parser.add_argument("--vignette_max_episodes", type=int, default=None)
    parser.add_argument("--vignette_graph_mode", default=None, choices=["sim", "physical_forward_only", "physical_all_edges"])
    args = parser.parse_args()
    path = run_gate1(
        cfg_path=args.config,
        export_negative_vignette=args.export_negative_vignette if args.export_negative_vignette else None,
        vignette_out_dir=args.vignette_out_dir,
        vignette_prefer=args.vignette_prefer,
        vignette_window=args.vignette_window,
        vignette_max_episodes=args.vignette_max_episodes,
        vignette_graph_mode=args.vignette_graph_mode,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

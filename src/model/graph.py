"""Active coupling graph G(x) utilities."""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.model.coupling import active_edges


def coupling_graph(
    x: np.ndarray,
    params,
    mode: dict | None = None,
    s: np.ndarray | None = None,
) -> nx.DiGraph:
    """Build the directed active coupling graph for the selected plant family."""
    g = nx.DiGraph()
    g.add_nodes_from(range(params.N))
    g.add_edges_from(active_edges(x, params, mode=mode, s=s))
    return g

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def build_dependency_graph(
    matrix: pd.DataFrame,
    top_k_per_downstream: Optional[int] = None,
    threshold: Optional[float] = None,
    normalize_by_column: bool = True,
    quantile_per_origin: Optional[float] = None,
) -> nx.DiGraph:
    """
    Construct a directed weighted graph from an upstream x downstream matrix.

    Edges go from upstream (rows) to downstream (columns). We optionally
    normalise weights by column totals to interpret them as shares, then
    prune by keeping only the top-k edges into each downstream node and/or
    dropping edges below a threshold.
    """
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame")
    if not matrix.index.equals(matrix.columns):
        # We assume square (same sectors on rows/cols) for pruning symmetry
        pass

    mat = matrix.copy()
    if normalize_by_column:
        col_sums = mat.sum(axis=0).replace(0, np.nan)
        mat = mat.div(col_sums, axis=1)
        mat = mat.fillna(0)

    edges: List[Tuple[str, str, float]] = []
    for downstream in mat.columns:
        col = mat[downstream]
        series = col
        if quantile_per_origin is not None:
            qval = series.quantile(quantile_per_origin)
            series = series[series >= qval]
        if top_k_per_downstream:
            series = series.sort_values(ascending=False).head(top_k_per_downstream)
        for upstream, weight in series.items():
            if threshold is not None and weight < threshold:
                continue
            if weight == 0:
                continue
            edges.append((upstream, downstream, float(weight)))

    G = nx.DiGraph()
    G.add_nodes_from(mat.index)
    G.add_weighted_edges_from(edges)
    return G


def detect_communities(graph: nx.Graph, method: str = "louvain") -> Dict[str, int]:
    """
    Detect communities on an undirected projection of the dependency graph.
    Returns a node -> community_id mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}
    if method == "louvain" and hasattr(nx.algorithms.community, "louvain_communities"):
        comms = nx.algorithms.community.louvain_communities(graph, weight="weight")
    else:
        comms = nx.algorithms.community.greedy_modularity_communities(
            graph, weight="weight"
        )
    mapping: Dict[str, int] = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            mapping[n] = cid
    return mapping


def build_layers(graph: nx.DiGraph, base_node) -> Dict:
    """Assign layer depth from a base node (0 = base, larger = higher)."""
    layers: Dict = {base_node: 0}
    frontier = [base_node]
    while frontier:
        next_frontier = []
        for node in frontier:
            for succ in graph.successors(node):
                if succ not in layers:
                    layers[succ] = layers[node] + 1
                    next_frontier.append(succ)
        frontier = next_frontier
    return layers


def symmetrize(graph: nx.DiGraph, agg: str = "max") -> nx.Graph:
    """
    Convert a directed weighted graph to undirected by aggregating reciprocal edges.
    """
    G = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        if G.has_edge(u, v):
            existing = G[u][v]["weight"]
            if agg == "max":
                G[u][v]["weight"] = max(existing, w)
            elif agg == "sum":
                G[u][v]["weight"] = existing + w
            else:
                G[u][v]["weight"] = (existing + w) / 2
        else:
            G.add_edge(u, v, weight=w)
    return G


def plot_foundations(series: pd.Series, top_n: int = 15, title: str = "") -> plt.Axes:
    """
    Quick horizontal bar plot of top contributors.
    """
    top = series.sort_values(ascending=True).tail(top_n)
    ax = top.plot(kind="barh", figsize=(7, 6), color="#4c72b0")
    ax.set_xlabel("Share of total" if np.isclose(top.sum(), 1.0) else "Value")
    ax.set_ylabel("Sector")
    ax.set_title(title or "Top contributors")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return ax


def plot_inverted_pyramid(series: pd.Series, top_n: int = 15, title: str = "") -> plt.Axes:
    """Plot a simple inverted-pyramid style chart (stacked bars with largest base at bottom)."""
    top = series.sort_values(ascending=False).head(top_n)
    shares = top / top.sum() if top.sum() != 0 else top
    y_pos = np.arange(len(top))[::-1]  # largest at bottom

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(y_pos, shares.values, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top))))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()  # largest at bottom visually
    ax.set_xlabel("Share of total")
    ax.set_title(title or "Inverted pyramid (top contributors)")
    for y, v in zip(y_pos, shares.values):
        ax.text(v + 0.005, y, f"{v:.2%}", va="center", fontsize=8)
    ax.set_xlim(0, max(1.0, shares.max() * 1.1))
    plt.tight_layout()
    return ax


def plot_dependency_graph(graph: nx.DiGraph, community_map: Optional[Dict[str, int]] = None) -> plt.Axes:
    """
    Quick network plot; for more polished figures, replace with custom visuals.
    """
    pos = nx.spring_layout(graph, seed=0, weight="weight")
    node_colors = (
        [community_map.get(n, 0) for n in graph.nodes()] if community_map else "lightblue"
    )
    weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
    ax = plt.gca()
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_size=50,
        node_color=node_colors,
        edge_color="gray",
        width=[w * 5 for w in weights],
        alpha=0.7,
        ax=ax,
    )
    return ax


def plot_layered_pyramid(
    graph: nx.DiGraph,
    base_node,
    labels: Optional[Dict] = None,
    node_size: int = 500,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Inverted pyramid",
) -> plt.Axes:
    """Plot an inverted pyramid layout from a directed graph given a base node."""
    layers = build_layers(graph, base_node)
    reachable_nodes = layers.keys()
    sub = graph.subgraph(reachable_nodes).copy()
    # Organize nodes by layer
    layer_nodes: Dict[int, List] = {}
    for n, layer in layers.items():
        layer_nodes.setdefault(layer, []).append(n)
    # Assign positions: x spaced within each layer, y = -layer (base at bottom)
    pos = {}
    for layer, nodes in sorted(layer_nodes.items()):
        xs = np.linspace(-len(nodes), len(nodes), num=len(nodes)) if len(nodes) > 1 else [0]
        for x, n in zip(xs, nodes):
            pos[n] = (x, -layer)

    node_labels = {n: (labels.get(n, n) if labels else n) for n in sub.nodes()}
    weights = [sub[u][v].get("weight", 1.0) for u, v in sub.edges()]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        sub,
        pos=pos,
        labels=node_labels,
        with_labels=True,
        node_size=node_size,
        node_color="#e6f0ff",
        edge_color="gray",
        width=[w * 5 for w in weights],
        font_size=8,
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()
    return ax

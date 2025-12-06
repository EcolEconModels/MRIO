from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

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




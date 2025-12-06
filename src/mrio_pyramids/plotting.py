from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Set default seaborn theme
sns.set_theme(style="whitegrid", context="paper")


def plot_foundations(series: pd.Series, top_n: int = 15, title: str = "") -> plt.Axes:
    """
    Quick horizontal bar plot of top contributors using Seaborn.
    """
    top = series.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Convert index to string to avoid MultiIndex issues with seaborn
    y_labels = [str(x) for x in top.index]
    sns.barplot(x=top.values, y=y_labels, ax=ax, palette="viridis", orient="h")
    
    ax.set_xlabel("Share of total" if np.isclose(top.sum(), 1.0) else "Value")
    ax.set_ylabel("Sector")
    ax.set_title(title or "Top contributors")
    plt.tight_layout()
    return ax


def plot_inverted_pyramid(series: pd.Series, top_n: int = 15, title: str = "") -> plt.Axes:
    """Plot a simple inverted-pyramid style chart (stacked bars with largest base at bottom)."""
    top = series.sort_values(ascending=False).head(top_n)
    shares = top / top.sum() if top.sum() != 0 else top
    y_pos = np.arange(len(top))[::-1]  # largest at bottom

    fig, ax = plt.subplots(figsize=(6, 8))
    # Custom color palette for the pyramid effect
    colors = sns.color_palette("Blues", n_colors=len(top))
    
    ax.barh(y_pos, shares.values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()  # largest at bottom visually
    ax.set_xlabel("Share of total")
    ax.set_title(title or "Inverted pyramid (top contributors)")
    
    for y, v in zip(y_pos, shares.values):
        ax.text(v + 0.005, y, f"{v:.2%}", va="center", fontsize=8)
    
    ax.set_xlim(0, max(1.0, shares.max() * 1.1))
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    return ax


def plot_dependency_graph(graph: nx.DiGraph, community_map: Optional[Dict[str, int]] = None) -> plt.Axes:
    """
    Quick network plot using NetworkX with improved styling.
    """
    from mrio_pyramids.graph import build_layers # helper logic is tied to graph structure
    
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.15, iterations=50)
    
    node_colors = (
        [community_map.get(n, 0) for n in graph.nodes()] if community_map else "lightblue"
    )
    
    weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
    # Scale width for visibility
    max_w = max(weights) if weights else 1
    widths = [0.5 + (w / max_w) * 2 for w in weights]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        graph, pos, node_size=100, node_color=node_colors, cmap=plt.cm.tab20, alpha=0.9, ax=ax
    )
    nx.draw_networkx_edges(
        graph, pos, width=widths, edge_color="gray", alpha=0.3, ax=ax, arrows=True
    )
    ax.set_axis_off()
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
    from mrio_pyramids.graph import build_layers

    layers = build_layers(graph, base_node)
    reachable_nodes = layers.keys()
    sub = graph.subgraph(reachable_nodes).copy()
    
    # Organize nodes by layer
    layer_nodes: Dict[int, List] = {}
    for n, layer in layers.items():
        layer_nodes.setdefault(layer, []).append(n)
        
    pos = {}
    for layer, nodes in sorted(layer_nodes.items()):
        xs = np.linspace(-len(nodes), len(nodes), num=len(nodes)) if len(nodes) > 1 else [0]
        for x, n in zip(xs, nodes):
            pos[n] = (x, -layer)

    node_labels = {n: (labels.get(n, n) if labels else n) for n in sub.nodes()}
    weights = [sub[u][v].get("weight", 1.0) for u, v in sub.edges()]
    max_w = max(weights) if weights else 1
    widths = [0.5 + (w / max_w) * 3 for w in weights]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        sub,
        pos=pos,
        labels=node_labels,
        with_labels=True,
        node_size=node_size,
        node_color="#e6f0ff",
        edge_color="#444444",
        width=widths,
        font_size=8,
        ax=ax,
        connectionstyle="arc3,rad=0.1",
    )
    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout()
    return ax

"""
Network visualization functions for stranding cascade analysis.

Implements visualizations from Cahen-Fourot et al. (2021):
- Figure 3: Stranding cascade network (hierarchical tree)
- Figure 5: Country exposure scatter plot
- Figure 9: Country-specific exposure network
- Figure A.3: Global sector cascade (e.g., land transport)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def plot_cascade_network(
    rounds: Dict[int, Dict[str, float]],
    fossil_sector: str = "MINfos",
    title: str = "Stranding Cascade Network",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot hierarchical cascade network showing rounds of stranding effects.
    
    Replicates Figure 3 / Figure A.3 from Cahen-Fourot et al. (2021).
    
    Args:
        rounds: Dict mapping round_number → {sector: stranding_value}
        fossil_sector: Name of fossil fuel sector at top
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add fossil sector as root (layer 0)
    G.add_node(fossil_sector, layer=0, value=0, type='fossil')
    
    # Add nodes and edges by round
    prev_layer = [fossil_sector]
    
    for round_num, sector_dict in rounds.items():
        current_layer = []
        
        for sector, value in sector_dict.items():
            G.add_node(sector, layer=round_num, value=value, type='affected')
            current_layer.append(sector)
            
            # Add edges from previous layer
            # For simplicity, connect to all previous layer nodes
            # In reality, we'd track which nodes caused this one
            for prev_node in prev_layer:
                G.add_edge(prev_node, sector, weight=value)
        
        prev_layer = current_layer
    
    # Layout: hierarchical (top-to-bottom)
    pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
    
    # Invert y-axis to have fossil at top
    for node in pos:
        pos[node] = (pos[node][0], -pos[node][1])
    
    # Node sizes based on stranding value
    node_sizes = []
    for node in G.nodes():
        value = G.nodes[node].get('value', 1)
        size = 300 + value * 20  # Scale appropriately
        node_sizes.append(size)
    
    # Node colors by type
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'fossil':
            node_colors.append('#d62728')  # Red for fossil
        else:
            node_colors.append('#1f77b4')  # Blue for affected
    
    # Draw network
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Draw edges with varying transparency based on weight
    edges = G.edges(data=True)
    for (u, v, d) in edges:
        weight = d.get('weight', 1)
        alpha = min(0.8, weight / 50)  # Scale alpha
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            alpha=alpha,
            edge_color='gray',
            arrows=True,
            arrowsize=10,
            ax=ax
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cascade network to {save_path}")
    
    return fig


def plot_country_exposure_scatter(
    exposure_by_country: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top Countries by External Fossil Stranding Multiplier",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot country exposure scatter showing stranding in affected vs originating countries.
    
    Replicates Figure 5 from Cahen-Fourot et al. (2021).
    
    Args:
        exposure_by_country: DataFrame with columns like:
            - 'originating_country': Source of stranding
            - 'stranding_in_affected': Stranding in other countries
            - 'total_external': Total external stranding multiplier
        top_n: Number of top countries to label
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by total external exposure
    top_countries = exposure_by_country.nlargest(top_n, 'total_external')
    
    # Scatter plot
    ax.scatter(
        exposure_by_country['stranding_in_affected'],
        exposure_by_country['originating_country'],
        s=100,
        alpha=0.6,
        c='lightgray'
    )
    
    # Highlight top countries
    ax.scatter(
        top_countries['stranding_in_affected'],
        top_countries['originating_country'],
        s=200,
        alpha=0.8,
        c='#1f77b4'
    )
    
    # Annotate top countries
    for idx, row in top_countries.iterrows():
        ax.annotate(
            idx,  # Country code
            (row['stranding_in_affected'], row['originating_country']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )
    
    # Add total external multiplier as text on right
    for idx, row in top_countries.iterrows():
        ax.text(
            ax.get_xlim()[1] * 0.95,
            row['originating_country'],
            f"{row['total_external']:.2f}",
            ha='right',
            va='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    ax.set_xlabel('Stranding in Affected Countries', fontsize=12)
    ax.set_ylabel('Originating Country (fossil sector)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved country exposure scatter to {save_path}")
    
    return fig


def plot_country_exposure_network(
    exposure_links: pd.DataFrame,
    focal_country: str,
    focal_sectors: List[str],
    threshold: float = 0.05,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot country-specific exposure network showing main fossil exposure links.
    
    Replicates Figure 9 from Cahen-Fourot et al. (2021).
    
    Args:
        exposure_links: DataFrame with columns:
            - 'origin_country_sector': (country, sector) tuple
            - 'target_country_sector': (country, sector) tuple
            - 'stranding_value': Strength of link
        focal_country: Country to focus on (e.g., 'DEU')
        focal_sectors: List of focal sectors to show
        threshold: Minimum stranding value to show
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure
    """
    if title is None:
        title = f"Main Fossil Exposure Links for {focal_country}"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to relevant links
    relevant_links = exposure_links[
        (exposure_links['stranding_value'] >= threshold) &
        (exposure_links['target_country_sector'].apply(lambda x: x[0] == focal_country))
    ]
    
    # Create graph
    G = nx.DiGraph()
    
    for _, row in relevant_links.iterrows():
        origin = row['origin_country_sector']
        target = row['target_country_sector']
        value = row['stranding_value']
        
        # Add nodes
        G.add_node(origin, type='origin', country=origin[0], sector=origin[1])
        G.add_node(target, type='target', country=target[0], sector=target[1])
        
        # Add edge
        G.add_edge(origin, target, weight=value)
    
    # Layout: bipartite (origins on left, targets on right)
    origin_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'origin']
    target_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'target']
    
    pos = {}
    for i, node in enumerate(origin_nodes):
        pos[node] = (0, i)
    for i, node in enumerate(target_nodes):
        pos[node] = (1, i)
    
    # Node colors
    node_colors = {'origin': '#d62728', 'target': '#1f77b4'}
    colors = [node_colors[G.nodes[n]['type']] for n in G.nodes()]
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=colors, alpha=0.8, ax=ax)
    
    # Labels: show country_sector
    labels = {n: f"{d['country']}\n{d['sector']}" for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Edges with weights
    edges = G.edges(data=True)
    for (u, v, d) in edges:
        weight = d['weight']
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=weight * 2,  # Scale edge width
            alpha=0.6,
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            ax=ax
        )
        
        # Edge labels
        mid_point = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
        ax.text(
            mid_point[0], mid_point[1],
            f"{weight:.2f}",
            fontsize=7,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved country exposure network to {save_path}")
    
    return fig

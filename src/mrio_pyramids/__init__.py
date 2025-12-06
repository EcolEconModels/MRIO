"""
Reusable tools for MRIO-based inverted pyramid analysis.

This package keeps notebooks lean by centralising computations (Leontief,
embodied pressure allocation, ranking foundational sectors). Extend functions
here rather than inside notebooks to keep the workflow reproducible.
"""

from .pyramids import (
    dependency_contributions,
    embodied_pressure,
    rank_foundational_sectors,
)
from .graph import (
    build_dependency_graph,
    detect_communities,
    plot_dependency_graph,
    plot_foundations,
    plot_inverted_pyramid,
    plot_layered_pyramid,
    symmetrize,
)

from .clustering import sector_clustering

__all__ = [
    "dependency_contributions",
    "embodied_pressure",
    "rank_foundational_sectors",
    "build_dependency_graph",
    "detect_communities",
    "plot_dependency_graph",
    "plot_foundations",
    "plot_layered_pyramid",
    "symmetrize",

    "sector_clustering",
]

"""
Reusable tools for MRIO-based inverted pyramid analysis.

This package keeps notebooks lean by centralising computations (Leontief,
embodied pressure allocation, ranking foundational sectors). Extend functions
here rather than inside notebooks to keep the workflow reproducible.
"""

from .pyramids import (
    PyramidInputs,
    compute_leontief,
    compute_ghosh,
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
from .data import load_wiod2013, load_exiobase3, load_test_mrio, load_oecd_icio
from .clustering import sector_clustering

__all__ = [
    "PyramidInputs",
    "compute_leontief",
    "compute_ghosh",
    "dependency_contributions",
    "embodied_pressure",
    "rank_foundational_sectors",
    "build_dependency_graph",
    "detect_communities",
    "plot_dependency_graph",
    "plot_foundations",
    "plot_layered_pyramid",
    "symmetrize",
    "load_wiod2013",
    "load_exiobase3",
    "load_test_mrio",
    "load_oecd_icio",
    "sector_clustering",
]

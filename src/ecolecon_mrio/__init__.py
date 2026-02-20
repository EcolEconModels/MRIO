"""
Reusable tools for MRIO-based inverted pyramid analysis.

This package keeps notebooks lean by centralising computations (Leontief,
embodied pressure allocation, ranking foundational sectors). Extend functions
here rather than inside notebooks to keep the workflow reproducible.
"""

from .counterfactual import run_panel_scenarios
from .data import load_mrio
from .decomposition import decompose_slack_change
from .diagnostics import (
    SafeJustResult,
    aggregate_diagnostics,
    calc_boundary_slack,
    calc_pf_eigenvectors,
    calc_pf_sensitivities,
    calc_spectral_radius,
    calc_tech_slack,
    partial_sum_convergence,
    solve_safe_just_lp,
)
from .growth_direction import (
    PrincipalDirectionResult,
    principal_growth_direction,
    principal_growth_panel,
)
from .metrics import (
    dependency_contributions,
    embodied_pressure,
    rank_foundational_sectors,
    forward_linkages,
    demand_weighted_linkages,
)
from .stranding import (
    calc_capital_intensity,
    calc_stranding_multipliers,
    decompose_stranding_rounds,
    aggregate_stranding_by_rounds,
    calc_external_stranding_multiplier,
    calc_exposure_network,
)
from .network_viz import (
    plot_cascade_network,
    plot_country_exposure_scatter,
    plot_country_exposure_network,
)
from .graph import (
    build_dependency_graph,
    detect_communities,
    symmetrize,
    build_layers,
)
from .plotting import (
    plot_dependency_graph,
    plot_foundations,
    plot_inverted_pyramid,
    plot_layered_pyramid,
)
from .temporal import PanelDiagnostics, compute_viability_panel, load_mrio_panel
from .transition import classify_transition_years, principal_direction_index

from .clustering import sector_clustering

__all__ = [
    "load_mrio",
    "load_mrio_panel",
    "PanelDiagnostics",
    "compute_viability_panel",
    "decompose_slack_change",
    "PrincipalDirectionResult",
    "principal_growth_direction",
    "principal_growth_panel",
    "principal_direction_index",
    "classify_transition_years",
    "run_panel_scenarios",
    "SafeJustResult",
    "aggregate_diagnostics",
    "calc_boundary_slack",
    "calc_pf_eigenvectors",
    "calc_pf_sensitivities",
    "calc_spectral_radius",
    "calc_tech_slack",
    "partial_sum_convergence",
    "solve_safe_just_lp",
    "dependency_contributions",
    "embodied_pressure",
    "rank_foundational_sectors",
    "forward_linkages",
    "demand_weighted_linkages",
    "build_dependency_graph",
    "detect_communities",
    "plot_dependency_graph",
    "plot_foundations",
    "plot_inverted_pyramid",
    "plot_layered_pyramid",
    "symmetrize",
    "sector_clustering",
]

# ruff: noqa: F401
"""
Reusable tools for MRIO-based inverted pyramid analysis.

This package keeps notebooks lean by centralising computations (Leontief,
embodied pressure allocation, ranking foundational sectors). Extend functions
here rather than inside notebooks to keep the workflow reproducible.
"""

from .constraints import BetaResult, build_constraints, build_constraints_with_context, solve_beta
from .counterfactual import run_panel_scenarios, run_shock_suite
from .data import (
    load_mrio,
    load_mrio_matrices,
    load_mrio_matrix_result,
    MrioMatrixResult,
    ensure_exiobase_year_zip,
    resolve_exiobase_year_dir,
    resolve_exiobase_cache_paths,
)
from .data import exiobase3_monetary_unit
load_MRIO_matrices = load_mrio_matrices
from .decomposition import decompose_slack_change
from .diagnostics import (
    SafeJustResult,
    aggregate_diagnostics,
    calc_boundary_slack,
    calc_coefficient_diagnostics,
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
from .temporal import PanelDiagnostics, compute_output_change_decomposition, compute_viability_panel, load_mrio_panel
from .mrio_sda import (
    OutputSdaPairResult,
    MrioYearState,
    UnitMetadata,
    compute_output_sda_pair_result,
    compute_output_sda_pair_result_from_states,
    load_mrio_year_state,
    require_region_sector_multiindex,
    require_same_unit_string,
)
from .mrio_extension_sda import (
    ExtensionIntensityState,
    ExtensionSdaPairResult,
    compute_extension_intensity_state,
    compute_extension_sda_pair_result_from_states,
    load_mrio_extension_year_state,
)
from .runtime import (
    configure_low_memory_runtime,
    enforce_memory_limit_gb,
    memory_snapshot_text,
    tick,
)
from .transition import classify_transition_years, principal_direction_index
from .transition_support import (
    benchmark_step_for_emissions_cut,
    coefficient_change_bounds,
    emissions_cut_from_step,
    emissions_direction_slope,
    max_demand_nonnegative_emissions_cut,
    max_demand_nonnegative_step,
    max_output_nonnegative_emissions_cut,
    max_output_nonnegative_step,
    normalize_productive_certificate_vector,
    productive_certificate_row_rhs,
    productive_certificate_slack,
    satisfies_productive_certificate,
)
from .emissions_stabilization import (
    threshold_intensity,
    winsorize_intensity,
    shrink_intensity_to_group_mean,
    smooth_intensity_across_years,
)

from .download_utils import (
    DownloadProgressEvent,
    DownloadResult,
    DownloadSpec,
    download_file_resumable,
    download_urls_resumable,
    zip_contains_member,
)

from .panel_coverage import (
    CoverageManifest,
    YearAttemptRecord,
    attempt_mrio_year,
    build_year_coverage_manifest,
    select_contiguous_year_block,
    write_coverage_manifest,
    read_coverage_manifest,
)

from .inverse_primitives import (
    RowBoundsInterval,
    RowSolveDiagnostics,
    MatrixInverseDiagnostics,
    SpectralModeDiagnostics,
    compute_row_interval_bounds,
    row_min_norm_update,
    matrix_min_norm_inverse_step,
    dominant_mode_diagnostics,
    calibrate_relative_change_caps,
)

from .clustering import sector_clustering
from .exiobase3_extensions import (
    ExtensionRowResult,
    load_exiobase3_extension_for_year,
    load_exiobase3_extension_row_with_unit,
    load_exiobase3_extension_row_with_unit_for_year,
)
from .mapping_csv import (
    MAPPING_CSV_COLUMNS,
    flatten_exio_sectors_from_mapping,
    load_aggregation_mapping_frame,
    load_category_mapping_frame,
    load_split_mapping_frame,
    load_weighted_mapping_frame,
    read_aggregation_mapping,
    read_mapping_csv,
    read_split_mapping,
    read_weighted_mapping,
    validate_mapping_weights,
)

from .energy_balance_pxp import (
    compile_energy_balance_pxp_split_plan,
    load_energy_balance_dictionary_frame,
    load_energy_balance_pxp_input_technology_frame,
    load_energy_balance_pxp_prices_frame,
    load_energy_balance_pxp_reaggregation_frame,
    load_energy_balance_pxp_split_frame,
    run_energy_balance_pxp_split,
    write_energy_balance_pxp_diagnostics,
)

__all__ = [
    "load_mrio",
    "load_mrio_matrices",
    "load_mrio_matrix_result",
    "MrioMatrixResult",
    "load_MRIO_matrices",
    "ensure_exiobase_year_zip",
    "resolve_exiobase_year_dir",
    "resolve_exiobase_cache_paths",
    "load_mrio_panel",
    "PanelDiagnostics",
    "compute_viability_panel",
    "compute_output_change_decomposition",
    "load_mrio_year_state",
    "compute_output_sda_pair_result_from_states",
    "MrioYearState",
    "decompose_slack_change",
    "PrincipalDirectionResult",
    "principal_growth_direction",
    "principal_growth_panel",
    "principal_direction_index",
    "classify_transition_years",
    "benchmark_step_for_emissions_cut",
    "coefficient_change_bounds",
    "emissions_cut_from_step",
    "emissions_direction_slope",
    "max_demand_nonnegative_emissions_cut",
    "max_demand_nonnegative_step",
    "max_output_nonnegative_emissions_cut",
    "max_output_nonnegative_step",
    "normalize_productive_certificate_vector",
    "productive_certificate_row_rhs",
    "productive_certificate_slack",
    "satisfies_productive_certificate",
    "threshold_intensity",
    "winsorize_intensity",
    "shrink_intensity_to_group_mean",
    "smooth_intensity_across_years",
    "read_coverage_manifest",
    "write_coverage_manifest",
    "select_contiguous_year_block",
    "build_year_coverage_manifest",
    "attempt_mrio_year",
    "YearAttemptRecord",
    "CoverageManifest",
    "DownloadSpec",
    "DownloadResult",
    "DownloadProgressEvent",
    "download_file_resumable",
    "download_urls_resumable",
    "zip_contains_member",
    "run_panel_scenarios",
    "run_shock_suite",
    "BetaResult",
    "build_constraints",
    "build_constraints_with_context",
    "solve_beta",
    "SafeJustResult",
    "aggregate_diagnostics",
    "calc_boundary_slack",
    "calc_coefficient_diagnostics",
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
    "RowBoundsInterval",
    "RowSolveDiagnostics",
    "MatrixInverseDiagnostics",
    "SpectralModeDiagnostics",
    "compute_row_interval_bounds",
    "row_min_norm_update",
    "matrix_min_norm_inverse_step",
    "dominant_mode_diagnostics",
    "calibrate_relative_change_caps",
    "sector_clustering",
    "load_exiobase3_extension_for_year",
    "MAPPING_CSV_COLUMNS",
    "read_mapping_csv",
    "load_weighted_mapping_frame",
    "load_split_mapping_frame",
    "load_category_mapping_frame",
    "load_aggregation_mapping_frame",
    "read_weighted_mapping",
    "read_split_mapping",
    "read_aggregation_mapping",
    "validate_mapping_weights",
    "flatten_exio_sectors_from_mapping",
]

__all__.extend([
    "OutputSdaPairResult",
    "UnitMetadata",
    "compute_output_sda_pair_result",
    "require_region_sector_multiindex",
    "require_same_unit_string",
    "exiobase3_monetary_unit",
    "ExtensionRowResult",
    "load_exiobase3_extension_row_with_unit",
    "load_exiobase3_extension_row_with_unit_for_year",
])

__all__.extend([
    "ExtensionIntensityState",
    "ExtensionSdaPairResult",
    "compute_extension_intensity_state",
    "compute_extension_sda_pair_result_from_states",
    "load_mrio_extension_year_state",
    "configure_low_memory_runtime",
    "enforce_memory_limit_gb",
    "memory_snapshot_text",
    "tick",
])

__all__.extend([
    "compile_energy_balance_pxp_split_plan",
    "load_energy_balance_dictionary_frame",
    "load_energy_balance_pxp_input_technology_frame",
    "load_energy_balance_pxp_prices_frame",
    "load_energy_balance_pxp_reaggregation_frame",
    "load_energy_balance_pxp_split_frame",
    "run_energy_balance_pxp_split",
    "write_energy_balance_pxp_diagnostics",
])

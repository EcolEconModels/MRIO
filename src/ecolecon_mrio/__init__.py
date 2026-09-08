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
    load_mrio_iot_matrices,
    load_exiobase3_iot_matrices_in_price_basis,
    derive_A_from_Z_and_x,
    derive_nonexport_final_demand_from_Y,
    ensure_exiobase_year_zip,
    resolve_exiobase_year_dir,
    resolve_exiobase_cache_paths,
)
from .data import exiobase3_monetary_unit
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
    MrioExtensionYearState,
    compute_extension_intensity_state,
    compute_extension_sda_pair_result_from_extension_year_states,
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
    baseline_productivity_certificate,
    adjacent_coefficient_change_quantiles,
    benchmark_step_for_emissions_cut,
    certificate_safe_coefficient_capacity_products,
    certificate_safe_coefficient_headroom,
    complete_directions_with_fixed_final_demand,
    merge_panel_coefficient_envelope,
    coefficient_change_bounds,
    coefficient_edge_counts,
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
    "load_mrio_iot_matrices",
    "load_exiobase3_iot_matrices_in_price_basis",
    "derive_A_from_Z_and_x",
    "derive_nonexport_final_demand_from_Y",
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
    "adjacent_coefficient_change_quantiles",
    "baseline_productivity_certificate",
    "benchmark_step_for_emissions_cut",
    "coefficient_edge_counts",
    "certificate_safe_coefficient_capacity_products",
    "complete_directions_with_fixed_final_demand",
    "merge_panel_coefficient_envelope",
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
    "MrioExtensionYearState",
    "compute_extension_intensity_state",
    "compute_extension_sda_pair_result_from_extension_year_states",
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

from .region_aggregation import (
    MrioScopeSlice,
    RegionAggregationPlan,
    RegionScope,
    aggregate_external_import_extension_direct,
    aggregate_mrio_regions,
    aggregate_scope_final_demand_extension,
    aggregate_scope_production_extension,
    combine_total_supply_extension_direct,
    compile_region_aggregation,
    load_mrio_iot_scope_slice,
)

__all__.extend([
    "MrioScopeSlice",
    "RegionAggregationPlan",
    "RegionScope",
    "aggregate_external_import_extension_direct",
    "aggregate_mrio_regions",
    "aggregate_scope_final_demand_extension",
    "aggregate_scope_production_extension",
    "combine_total_supply_extension_direct",
    "compile_region_aggregation",
    "load_mrio_iot_scope_slice",
])

# ecolecon-price-index-api
# Price-index acquisition and primitive monetary price-adjustment API.
from .price_indices import (
    PriceIndexError,
    PriceIndexHTTPError,
    PriceIndexParseError,
    default_price_index_storage_folder,
    download_unsd_ama_gdp_deflators,
    load_unsd_ama_gdp_deflators,
    price_adjustment_factors,
    read_unsd_ama_gdp_deflators,
)
from .price_adjustment import adjust_mrio_iot_prices, expand_region_price_factors

__all__.extend([
    "PriceIndexError",
    "PriceIndexHTTPError",
    "PriceIndexParseError",
    "default_price_index_storage_folder",
    "download_unsd_ama_gdp_deflators",
    "load_unsd_ama_gdp_deflators",
    "price_adjustment_factors",
    "read_unsd_ama_gdp_deflators",
    "adjust_mrio_iot_prices",
    "expand_region_price_factors",
])
# /ecolecon-price-index-api

# ecolecon-exiobase-eur-price-factors
from .price_indices import (
    download_unsd_ama_gdp_usd_levels,
    load_unsd_ama_gdp_usd_levels,
    read_unsd_ama_gdp_usd_levels,
)
from .exiobase_price_factors import (
    build_exiobase3_eur_price_factors,
    common_eur_price_factors_from_usd,
    default_exiobase3_price_factor_storage_folder,
)

__all__.extend([
    "download_unsd_ama_gdp_usd_levels",
    "load_unsd_ama_gdp_usd_levels",
    "read_unsd_ama_gdp_usd_levels",
    "build_exiobase3_eur_price_factors",
    "common_eur_price_factors_from_usd",
    "default_exiobase3_price_factor_storage_folder",
])
# /ecolecon-exiobase-eur-price-factors

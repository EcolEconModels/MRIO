# pyright: reportMissingImports=false
r"""Strict production-side extension SDA for MRIO year pairs.

This module decomposes one direct production-side extension row, for example an
EXIOBASE ``air_emissions/F.txt`` CO2 row, using the economically correct identity

    f_t = q_t \odot x_t = q_t \odot L_t y_t.

The decomposition is over three factors: direct extension intensity ``q``,
production structure ``A`` through the Leontief system, and final demand ``y``.
It intentionally does not value monetary SDA components by an average extension
intensity, because that would not decompose the extension identity.

The implementation is sparse and source-zip friendly: it reuses ``MrioYearState``
and the existing sparse multi-RHS Leontief solve helper, never forms an explicit
Leontief inverse, never stores solver factorizations, and expects callers to use
the row-level extension loader for the selected stressor.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Literal
import gc

import numpy as np
import pandas as pd

from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI
from ecolecon_mrio.emissions_stabilization import threshold_intensity
from ecolecon_mrio.exiobase3_extensions import (
    ExtensionRowResult,
    load_exiobase3_extension_row_with_unit_for_year,
)
from ecolecon_mrio.mrio_sda import (
    MrioYearState,
    _solve_sparse_leontief_many,
    _strict_float_series,
    require_region_sector_multiindex,
    require_same_unit_string,
)

ProgressCallback = Callable[[str], None]
FactorName = Literal["intensity", "structure", "demand"]
_FACTOR_ORDER: tuple[FactorName, FactorName, FactorName] = ("intensity", "structure", "demand")
_SERIES_UNIT_ATTRS = ("unit", "unit_source")


@dataclass(frozen=True, slots=True)
class ExtensionIntensityState:
    """One year's direct extension row and stabilized production intensity.

    ``values`` stores the raw direct extension row ``f_t`` in its source unit,
    aligned to the MRIO ``MultiIndex(region, sector)``. ``raw_intensity`` is the
    direct coefficient ``f_t / x_t``. ``stabilized_intensity`` is the coefficient
    actually used in extension SDA, and ``stabilization_gap`` records the raw
    extension amount not represented by ``stabilized_intensity * x_t``.
    """

    year: int
    values: pd.Series
    raw_intensity: pd.Series
    stabilized_intensity: pd.Series
    stabilized_values: pd.Series
    stabilization_gap: pd.Series
    extension: str
    stressor: str
    kind: str
    unit: str
    unit_source: str
    intensity_unit: str
    stabilization: dict[str, object]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class MrioExtensionYearState:
    """Convenience bundle for one MRIO year plus the raw selected extension row.

    ``extension_row`` is kept because pair-level SDA must validate physical units,
    stressor identity, and row provenance at the pair boundary. Returning only an
    already-stabilized ``ExtensionIntensityState`` would force callers either to
    reload the raw row or to pass unit-weakened vectors downstream.
    """

    mrio_state: MrioYearState
    extension_row: ExtensionRowResult
    intensity_state: ExtensionIntensityState


@dataclass(frozen=True, slots=True)
class ExtensionSdaPairResult:
    """Strict three-factor extension SDA vectors for one MRIO year pair.

    All vector fields preserve the same canonical ``MultiIndex(region, sector)``.
    The SDA components decompose the stabilized Leontief identity
    ``q_tilde * L_t y_t``. ``sda_residual`` is only the numerical residual of that
    Leontief identity. Any endpoint difference between reported ``x`` and ``L y``
    is reported separately as ``reported_x_identity_gap_*``.
    """

    start_year: int
    end_year: int
    index: pd.MultiIndex
    observed_extension_delta: pd.Series
    stabilized_extension_delta: pd.Series
    leontief_extension_delta: pd.Series
    sda_intensity_delta: pd.Series
    sda_structure_delta: pd.Series
    sda_demand_delta: pd.Series
    sda_residual: pd.Series
    stabilization_gap_start: pd.Series
    stabilization_gap_end: pd.Series
    stabilization_gap_delta: pd.Series
    reported_x_identity_gap_start: pd.Series
    reported_x_identity_gap_end: pd.Series
    reported_x_identity_gap_delta: pd.Series
    identity_gap_start: pd.Series
    identity_gap_end: pd.Series
    identity_gap_delta: pd.Series
    unit: str
    unit_source: str
    intensity_unit: str
    residual_l2: float
    residual_linf: float
    residual_share_l2: float
    reported_x_identity_gap_l2: float
    metadata: dict[str, Any]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    """Call an optional progress callback at a scientifically meaningful boundary."""

    if progress is not None:
        progress(message)


def _required_series_attr(series: pd.Series, key: str) -> str:
    """Read required unit/provenance metadata from a plain pandas Series."""

    value = series.attrs.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Plain pandas Series extension rows require explicit metadata in "
            "Series.attrs: attrs['unit'] and attrs['unit_source'] are mandatory; "
            "prefer ExtensionRowResult for public API calls."
        )
    return value.strip()


def _series_attr(series: pd.Series, key: str, default: str) -> str:
    """Read optional string metadata from a Series without weakening unit checks."""

    value = series.attrs.get(key, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _extension_row_metadata(extension_row: ExtensionRowResult | pd.Series) -> dict[str, str]:
    """Return strict row metadata without loading a full extension object."""

    if isinstance(extension_row, ExtensionIntensityState):
        raise TypeError(
            "compute_extension_sda_pair_result_from_states expects raw ExtensionRowResult "
            "or a unit-annotated pandas Series, not ExtensionIntensityState. Use the "
            "extension_row field returned by load_mrio_extension_year_state(), or call "
            "compute_extension_sda_pair_result_from_extension_year_states(...)."
        )
    if isinstance(extension_row, ExtensionRowResult):
        return {
            "extension": str(extension_row.extension),
            "kind": str(extension_row.kind),
            "stressor": str(extension_row.stressor),
            "unit": str(extension_row.unit),
            "unit_source": str(extension_row.unit_source),
        }
    unit = _required_series_attr(extension_row, "unit")
    unit_source = _required_series_attr(extension_row, "unit_source")
    return {
        "extension": _series_attr(extension_row, "extension", "explicit_series"),
        "kind": _series_attr(extension_row, "kind", "F"),
        "stressor": _series_attr(extension_row, "stressor", str(extension_row.name or "unnamed_extension_row")),
        "unit": unit,
        "unit_source": unit_source,
    }


def _strict_extension_values(
    extension_row: ExtensionRowResult | pd.Series,
    *,
    index: pd.MultiIndex,
    name: str,
) -> pd.Series:
    """Return one production-side extension row aligned exactly to MRIO sectors."""

    metadata = _extension_row_metadata(extension_row)
    if metadata["kind"] != "F":
        raise ValueError(
            "Extension SDA currently supports only production-side kind='F' rows indexed by "
            "MultiIndex(region, sector); final-demand-side F_Y rows need a separate additive API."
        )
    values = extension_row.values if isinstance(extension_row, ExtensionRowResult) else extension_row
    row_index = values.index
    if not isinstance(row_index, pd.MultiIndex):
        raise TypeError(f"{name} must be indexed by MultiIndex(region, sector)")
    require_region_sector_multiindex(row_index)
    if not row_index.equals(index):
        raise ValueError(f"{name} index does not exactly match the MRIO year-state index")
    return _strict_float_series(values, index=index, name=name).rename(name)


def _reject_scientifically_empty_row(values: pd.Series, output: pd.Series, *, name: str, reject_all_zero: bool) -> None:
    """Fail on selected all-zero extension rows when the MRIO economy has output mass."""

    if not reject_all_zero:
        return
    value_array = values.to_numpy(dtype=float)
    output_array = output.to_numpy(dtype=float)
    # An exactly all-zero selected row with positive production is usually a wrong
    # stressor/extension selection or a failed source read, as seen for incomplete
    # EXIOBASE years. Do not let that produce a formally valid but meaningless SDA.
    if bool(np.any(output_array > 0.0)) and not bool(np.any(value_array != 0.0)):
        raise ValueError(
            f"{name} is an all-zero extension row while MRIO output has positive mass; "
            "check the selected extension/stressor/year or pass reject_all_zero=False explicitly."
        )


def _raw_intensity_from_values(values: pd.Series, output: pd.Series, *, name: str) -> pd.Series:
    """Compute ``f_t / x_t`` while making zero-output behavior explicit.

    A zero-output denominator cannot define a production intensity. The raw
    coefficient is set to zero only as an intermediate placeholder; the threshold
    stabilization step will replace tiny-output cells when requested.
    """

    f_values = values.to_numpy(dtype=float)
    x_values = output.to_numpy(dtype=float)
    intensity = np.zeros_like(f_values, dtype=float)
    valid = np.isfinite(f_values) & np.isfinite(x_values) & (x_values != 0.0)
    np.divide(f_values, x_values, out=intensity, where=valid)
    if not np.isfinite(intensity).all():
        raise ValueError(f"{name} contains non-finite extension intensities")
    return pd.Series(intensity, index=values.index, dtype=float, name=name)


def _stabilize_intensity(
    raw_intensity: pd.Series,
    output: pd.Series,
    *,
    stabilization: str,
    tau: float,
) -> pd.Series:
    """Return the direct extension coefficient used by the SDA identity."""

    method = str(stabilization).strip().lower()
    if method == "raw":
        return raw_intensity.astype(float).rename("stabilized_extension_intensity")
    if method == "threshold":
        return threshold_intensity(raw_intensity, output, tau=float(tau)).rename("stabilized_extension_intensity")
    raise ValueError("stabilization must be one of: 'threshold', 'raw'")


def compute_extension_intensity_state(
    *,
    mrio_state: MrioYearState,
    extension_row: ExtensionRowResult | pd.Series,
    stabilization: str = "threshold",
    tau: float = 1.0e-8,
    reject_all_zero: bool = True,
    progress: ProgressCallback | None = None,
) -> ExtensionIntensityState:
    """Build one year's stabilized production-side extension intensity state.

    The scientific estimand is the direct coefficient vector ``q_t`` in
    ``f_t = q_t * x_t``. Stabilization changes the coefficient vector used for
    SDA, so the raw/stabilized value gap is stored explicitly rather than hidden
    inside a residual.
    """

    idx = require_region_sector_multiindex(mrio_state.index)
    metadata = _extension_row_metadata(extension_row)
    output = _strict_float_series(mrio_state.x, index=idx, name=f"x_{mrio_state.year}")

    # The extension row must already be on the same production-sector basis as
    # the MRIO state. Reindexing would hide wrong EXIO system, row kind, or order.
    values = _strict_extension_values(
        extension_row,
        index=idx,
        name=f"extension_values_{mrio_state.year}",
    )
    _reject_scientifically_empty_row(
        values,
        output,
        name=f"extension_values_{mrio_state.year}",
        reject_all_zero=bool(reject_all_zero),
    )

    _emit(progress, f"compute raw direct extension intensity for {mrio_state.year}")
    raw_intensity = _raw_intensity_from_values(
        values,
        output,
        name=f"raw_extension_intensity_{mrio_state.year}",
    )

    _emit(progress, f"apply {stabilization} extension-intensity stabilization for {mrio_state.year}")
    stabilized_intensity = _stabilize_intensity(
        raw_intensity,
        output,
        stabilization=str(stabilization),
        tau=float(tau),
    )
    stabilized_values = (stabilized_intensity * output).astype(float).rename(
        f"stabilized_extension_values_{mrio_state.year}"
    )
    stabilization_gap = (values - stabilized_values).astype(float).rename(
        f"stabilization_gap_{mrio_state.year}"
    )
    _emit(progress, f"computed raw-vs-stabilized extension gap for {mrio_state.year}")

    unit = metadata["unit"]
    intensity_unit = f"{unit}/{mrio_state.monetary_unit}"
    return ExtensionIntensityState(
        year=int(mrio_state.year),
        values=values,
        raw_intensity=raw_intensity,
        stabilized_intensity=stabilized_intensity,
        stabilized_values=stabilized_values,
        stabilization_gap=stabilization_gap,
        extension=metadata["extension"],
        stressor=metadata["stressor"],
        kind=metadata["kind"],
        unit=unit,
        unit_source=metadata["unit_source"],
        intensity_unit=intensity_unit,
        stabilization={"method": str(stabilization), "tau": float(tau)},
        metadata={
            "year": int(mrio_state.year),
            "index_columns": ["region", "sector"],
            "stabilized_identity": "stabilized_intensity * output",
            "raw_gap": "extension_values - stabilized_intensity * output",
            "reject_all_zero": bool(reject_all_zero),
        },
    )


def _counterfactual_extension(
    *,
    intensity_index: int,
    structure_index: int,
    demand_index: int,
    q0: np.ndarray,
    q1: np.ndarray,
    outputs_by_structure_demand: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Evaluate ``q_a * L_b y_c`` for one binary counterfactual state."""

    q_values = q0 if int(intensity_index) == 0 else q1
    x_values = outputs_by_structure_demand[(int(structure_index), int(demand_index))]
    return q_values * x_values


def _six_polar_extension_components(
    *,
    q0: np.ndarray,
    q1: np.ndarray,
    outputs_by_structure_demand: dict[tuple[int, int], np.ndarray],
) -> dict[FactorName, np.ndarray]:
    """Average all six three-factor polar paths for ``q * L(A) * y``.

    Each path starts at ``(q0, A0, y0)`` and ends at ``(q1, A1, y1)``. The step
    contribution is assigned to the factor changed at that step. Averaging all
    six orders avoids privileging one arbitrary update sequence.
    """

    n = int(q0.shape[0])
    sums: dict[FactorName, np.ndarray] = {
        "intensity": np.zeros(n, dtype=float),
        "structure": np.zeros(n, dtype=float),
        "demand": np.zeros(n, dtype=float),
    }
    paths = list(permutations(_FACTOR_ORDER))
    for path in paths:
        # The current tuple records which endpoint is used for (q, A, y). Changing
        # exactly one element gives a scientifically attributable polar step.
        current = {"intensity": 0, "structure": 0, "demand": 0}
        previous = _counterfactual_extension(
            intensity_index=current["intensity"],
            structure_index=current["structure"],
            demand_index=current["demand"],
            q0=q0,
            q1=q1,
            outputs_by_structure_demand=outputs_by_structure_demand,
        )
        for factor in path:
            current[factor] = 1
            updated = _counterfactual_extension(
                intensity_index=current["intensity"],
                structure_index=current["structure"],
                demand_index=current["demand"],
                q0=q0,
                q1=q1,
                outputs_by_structure_demand=outputs_by_structure_demand,
            )
            sums[factor] += updated - previous
            previous = updated
    divisor = float(len(paths))
    return {factor: values / divisor for factor, values in sums.items()}


def _unit_source_pair(start_source: str, end_source: str) -> str:
    """Return compact provenance when start/end extension units match."""

    if str(start_source) == str(end_source):
        return str(start_source)
    return f"{start_source}; {end_source}"


def _state_stabilization_method(state: MrioExtensionYearState) -> str:
    """Return the method stored on a loaded convenience state."""

    return str(state.intensity_state.stabilization.get("method", "threshold"))


def _state_stabilization_tau(state: MrioExtensionYearState) -> float:
    """Return the numeric tau stored on a loaded convenience state."""

    raw_tau = state.intensity_state.stabilization.get("tau", 1.0e-8)
    if isinstance(raw_tau, bool) or not isinstance(raw_tau, (int, float)):
        raise TypeError("Loaded extension state stabilization tau must be numeric")
    return float(raw_tau)


def compute_extension_sda_pair_result_from_states(
    start: MrioYearState,
    end: MrioYearState,
    start_extension: ExtensionRowResult | pd.Series,
    end_extension: ExtensionRowResult | pd.Series,
    *,
    stabilization: str = "threshold",
    tau: float = 1.0e-8,
    reject_all_zero: bool = True,
    progress: ProgressCallback | None = None,
) -> ExtensionSdaPairResult:
    """Compute exact three-factor SDA for one production-side extension row.

    The components decompose ``q_tilde * L(A) * y`` using all six polar orderings.
    The result also reports raw observed extension change, stabilization gaps, and
    the difference between reported ``x`` and solved ``L y`` so callers can
    reconcile from the Leontief SDA identity back to source rows.
    """

    _emit(progress, f"validate MRIO and extension indexes for {start.year}-{end.year}")
    if not start.index.equals(end.index):
        raise ValueError("MRIO year-state indexes do not match")
    idx = require_region_sector_multiindex(start.index)
    require_same_unit_string(start.monetary_unit, end.monetary_unit, context="MRIO year-state monetary units")

    # Build year-specific intensity states before solving Leontief systems so any
    # unit, row-kind, or index mismatch fails before expensive sparse work starts.
    start_state = compute_extension_intensity_state(
        mrio_state=start,
        extension_row=start_extension,
        stabilization=stabilization,
        tau=float(tau),
        reject_all_zero=bool(reject_all_zero),
        progress=progress,
    )
    end_state = compute_extension_intensity_state(
        mrio_state=end,
        extension_row=end_extension,
        stabilization=stabilization,
        tau=float(tau),
        reject_all_zero=bool(reject_all_zero),
        progress=progress,
    )
    unit = require_same_unit_string(start_state.unit, end_state.unit, context="extension SDA row units")
    if start_state.stressor != end_state.stressor:
        raise ValueError(f"Extension stressor mismatch: {start_state.stressor!r} != {end_state.stressor!r}")
    if start_state.extension != end_state.extension:
        raise ValueError(f"Extension name mismatch: {start_state.extension!r} != {end_state.extension!r}")

    y0 = _strict_float_series(start.y, index=idx, name=f"y_{start.year}")
    y1 = _strict_float_series(end.y, index=idx, name=f"y_{end.year}")

    _emit(progress, f"solve start-year Leontief counterfactual outputs for {start.year}-{end.year}")
    start_outputs = _solve_sparse_leontief_many(
        "extension SDA start-year counterfactual outputs",
        start.A,
        np.column_stack((y0.to_numpy(dtype=float), y1.to_numpy(dtype=float))),
    )
    _emit(progress, f"solve end-year Leontief counterfactual outputs for {start.year}-{end.year}")
    end_outputs = _solve_sparse_leontief_many(
        "extension SDA end-year counterfactual outputs",
        end.A,
        np.column_stack((y0.to_numpy(dtype=float), y1.to_numpy(dtype=float))),
    )

    # Store only the four counterfactual output vectors needed by q * L(A) * y.
    # The sparse solver factorization and system matrix are not retained.
    outputs_by_structure_demand = {
        (0, 0): start_outputs[:, 0],
        (0, 1): start_outputs[:, 1],
        (1, 0): end_outputs[:, 0],
        (1, 1): end_outputs[:, 1],
    }
    q0 = start_state.stabilized_intensity.to_numpy(dtype=float)
    q1 = end_state.stabilized_intensity.to_numpy(dtype=float)

    _emit(progress, f"average six polar extension-SDA paths for {start.year}-{end.year}")
    component_arrays = _six_polar_extension_components(
        q0=q0,
        q1=q1,
        outputs_by_structure_demand=outputs_by_structure_demand,
    )

    observed_delta = (end_state.values - start_state.values).astype(float).rename("observed_extension_delta")
    stabilized_delta = (end_state.stabilized_values - start_state.stabilized_values).astype(float).rename(
        "stabilized_extension_delta"
    )
    leontief_start = pd.Series(q0 * outputs_by_structure_demand[(0, 0)], index=idx, dtype=float)
    leontief_end = pd.Series(q1 * outputs_by_structure_demand[(1, 1)], index=idx, dtype=float)
    leontief_delta = (leontief_end - leontief_start).astype(float).rename("leontief_extension_delta")

    intensity = pd.Series(component_arrays["intensity"], index=idx, dtype=float, name="sda_intensity_delta")
    structure = pd.Series(component_arrays["structure"], index=idx, dtype=float, name="sda_structure_delta")
    demand = pd.Series(component_arrays["demand"], index=idx, dtype=float, name="sda_demand_delta")

    # The six-polar components decompose q_tilde * L y. Keep the numerical SDA
    # residual separate from any reported-x-versus-Leontief-identity discrepancy.
    residual = (leontief_delta - intensity - structure - demand).astype(float).rename("sda_residual")
    reported_x_identity_gap_start = (start_state.stabilized_values - leontief_start).astype(float).rename(
        "reported_x_identity_gap_start"
    )
    reported_x_identity_gap_end = (end_state.stabilized_values - leontief_end).astype(float).rename(
        "reported_x_identity_gap_end"
    )
    reported_x_identity_gap_delta = (reported_x_identity_gap_end - reported_x_identity_gap_start).astype(float).rename(
        "reported_x_identity_gap_delta"
    )
    stabilization_gap_delta = (end_state.stabilization_gap - start_state.stabilization_gap).astype(float).rename(
        "stabilization_gap_delta"
    )

    residual_arr = residual.to_numpy(dtype=float)
    leontief_norm = float(np.linalg.norm(leontief_delta.to_numpy(dtype=float), ord=2))
    residual_l2 = float(np.linalg.norm(residual_arr, ord=2))
    residual_linf = float(np.linalg.norm(residual_arr, ord=np.inf)) if residual_arr.size else 0.0
    residual_share_l2 = float(residual_l2 / leontief_norm) if leontief_norm > 0.0 else 0.0
    reported_x_identity_gap_l2 = float(np.linalg.norm(reported_x_identity_gap_delta.to_numpy(dtype=float), ord=2))

    del start_outputs, end_outputs
    gc.collect()
    _emit(progress, f"built extension-SDA residual and reported-x identity-gap diagnostics for {start.year}-{end.year}")

    return ExtensionSdaPairResult(
        start_year=int(start.year),
        end_year=int(end.year),
        index=idx,
        observed_extension_delta=observed_delta,
        stabilized_extension_delta=stabilized_delta,
        leontief_extension_delta=leontief_delta,
        sda_intensity_delta=intensity,
        sda_structure_delta=structure,
        sda_demand_delta=demand,
        sda_residual=residual,
        stabilization_gap_start=start_state.stabilization_gap.rename("stabilization_gap_start"),
        stabilization_gap_end=end_state.stabilization_gap.rename("stabilization_gap_end"),
        stabilization_gap_delta=stabilization_gap_delta,
        reported_x_identity_gap_start=reported_x_identity_gap_start,
        reported_x_identity_gap_end=reported_x_identity_gap_end,
        reported_x_identity_gap_delta=reported_x_identity_gap_delta,
        identity_gap_start=reported_x_identity_gap_start.rename("identity_gap_start"),
        identity_gap_end=reported_x_identity_gap_end.rename("identity_gap_end"),
        identity_gap_delta=reported_x_identity_gap_delta.rename("identity_gap_delta"),
        unit=unit,
        unit_source=_unit_source_pair(start_state.unit_source, end_state.unit_source),
        intensity_unit=start_state.intensity_unit,
        residual_l2=residual_l2,
        residual_linf=residual_linf,
        residual_share_l2=residual_share_l2,
        reported_x_identity_gap_l2=reported_x_identity_gap_l2,
        metadata={
            "identity_mode": "six_polar_three_factor_extension_sda",
            "factors": list(_FACTOR_ORDER),
            "extension": start_state.extension,
            "stressor": start_state.stressor,
            "kind": start_state.kind,
            "stabilization": {"method": str(stabilization), "tau": float(tau)},
            "index_columns": ["region", "sector"],
            "forms_explicit_leontief_inverse": False,
            "stores_solver_factorization": False,
            "forms_sparse_delta_A": False,
            "rhs_columns_per_factorization": 2,
            "extension_loader": "row_level_zip_stream_expected",
            "cache_policy": "source_zip_only",
            "reject_all_zero": bool(reject_all_zero),
            "reconstruction": {
                "leontief_extension_delta": "sda_intensity_delta + sda_structure_delta + sda_demand_delta + sda_residual",
                "stabilized_extension_delta": "leontief_extension_delta + reported_x_identity_gap_delta",
                "observed_extension_delta": "stabilized_extension_delta + stabilization_gap_delta",
            },
            "returned_vectors": [
                "observed_extension_delta",
                "stabilized_extension_delta",
                "leontief_extension_delta",
                "sda_intensity_delta",
                "sda_structure_delta",
                "sda_demand_delta",
                "sda_residual",
                "stabilization_gap_delta",
                "reported_x_identity_gap_delta",
            ],
        },
    )


def compute_extension_sda_pair_result_from_extension_year_states(
    start: MrioExtensionYearState,
    end: MrioExtensionYearState,
    *,
    stabilization: str | None = None,
    tau: float | None = None,
    reject_all_zero: bool = True,
    progress: ProgressCallback | None = None,
) -> ExtensionSdaPairResult:
    """Compute pair SDA from convenience-loaded MRIO/extension year states."""

    method = _state_stabilization_method(start) if stabilization is None else str(stabilization)
    if stabilization is None and method != _state_stabilization_method(end):
        raise ValueError("Loaded extension year states used different stabilization methods")
    tau_value = _state_stabilization_tau(start) if tau is None else float(tau)
    if tau is None and tau_value != _state_stabilization_tau(end):
        raise ValueError("Loaded extension year states used different stabilization tau values")
    return compute_extension_sda_pair_result_from_states(
        start.mrio_state,
        end.mrio_state,
        start.extension_row,
        end.extension_row,
        stabilization=method,
        tau=tau_value,
        reject_all_zero=bool(reject_all_zero),
        progress=progress,
    )


def load_mrio_extension_year_state(
    year: int,
    *,
    extension: str,
    kind: Literal["F"] = "F",
    exact_priority: Sequence[str] = (),
    keywords: Sequence[str] = (),
    system: str = "pxp",
    data_root: str | Path | None = None,
    doi: str | None = None,
    allow_download_missing: bool = False,
    force_download: bool = False,
    unit_override: str | None = None,
    require_unit: bool = True,
    stabilization: str = "threshold",
    tau: float = 1.0e-8,
    reject_all_zero: bool = True,
    verbose: bool = False,
) -> MrioExtensionYearState:
    """Load one MRIO year plus the raw and stabilized production extension row.

    This convenience boundary keeps EXIOBASE source-zip discovery in MRIO while
    still reading only the sparse MRIO matrices and the selected extension row.
    It intentionally supports only ``kind='F'`` because ``F_Y`` is indexed by final
    demand categories and is not a production intensity over sectors.
    """

    if str(kind) != "F":
        raise ValueError("load_mrio_extension_year_state currently supports only production-side kind='F'")
    from ecolecon_mrio.mrio_sda import load_mrio_year_state

    doi_value = DEFAULT_EXIOBASE3_DOI if doi is None else str(doi)
    mrio_state = load_mrio_year_state(
        int(year),
        system=str(system),
        data_root=data_root,
        allow_download_missing=bool(allow_download_missing),
        force_download=bool(force_download),
        doi=doi_value,
        verbose=bool(verbose),
    )
    extension_row = load_exiobase3_extension_row_with_unit_for_year(
        int(year),
        extension=str(extension),
        kind="F",
        system=str(system),
        data_root=data_root,
        exact_priority=exact_priority,
        keywords=keywords,
        force_download=bool(force_download),
        allow_download_missing=bool(allow_download_missing),
        doi=doi_value,
        unit_override=unit_override,
        require_unit=bool(require_unit),
        verbose=bool(verbose),
    )
    intensity_state = compute_extension_intensity_state(
        mrio_state=mrio_state,
        extension_row=extension_row,
        stabilization=str(stabilization),
        tau=float(tau),
        reject_all_zero=bool(reject_all_zero),
    )
    return MrioExtensionYearState(
        mrio_state=mrio_state,
        extension_row=extension_row,
        intensity_state=intensity_state,
    )

# pyright: reportMissingImports=false
"""Strict, MultiIndex-preserving MRIO SDA pair utilities.

This module is intentionally small.  It does not load EXIOBASE, parse pymrio
objects, or manage caches.  Existing loaders such as ``load_mrio_matrices``
already provide sparse ``A`` matrices and aligned ``x``/``y`` vectors.  The role
of this module is to compute one scientifically auditable SDA pair result from
those inputs while preserving the canonical ``MultiIndex(region, sector)`` used
throughout the EcolEconModels MRIO workflow.

The legacy ``compute_output_change_decomposition`` function returns a long table
and silently skips/falls back when sparse solves fail.  The new API below is
stricter because paper-facing analysis should fail loudly when either polar SDA
identity cannot be solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import gc
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye as sparse_eye, issparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve


@dataclass(frozen=True)
class UnitMetadata:
    """Plain unit label plus provenance for reporting.

    This is deliberately not a unit-conversion system.  The current paper-facing
    pipeline only needs to carry labels such as ``MEUR`` or ``kg`` into CSV
    outputs.  Any conversion or harmonisation should be explicit in a later,
    separately tested API.
    """

    unit: str
    unit_source: str




@dataclass(frozen=True, slots=True)
class MrioYearState:
    """Loaded sparse MRIO state for one year.

    The object intentionally stores the sparse technical-coefficient matrix and
    the two monetary vectors needed for output SDA, but never stores a Leontief
    inverse or solver factorization.  Panel callers can keep a rolling window of
    these states so each EXIOBASE source zip is parsed once per run.
    """

    year: int
    A: csr_matrix
    x: pd.Series
    y: pd.Series
    index: pd.MultiIndex
    monetary_unit: str
    monetary_unit_source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OutputSdaPairResult:
    """Strict average-polar SDA vectors for one pair of MRIO years.

    All vector fields are indexed by the exact same canonical
    ``MultiIndex(region, sector)``.  The three monetary component vectors share
    ``monetary_unit`` and ``monetary_unit_source``.  Large inputs such as ``A``
    matrices are intentionally not stored on this result object; callers can
    release them after this compact result is created.
    """

    start_year: int
    end_year: int
    index: pd.MultiIndex
    observed_output_delta: pd.Series
    sda_intermediate_delta: pd.Series
    sda_demand_delta: pd.Series
    sda_residual: pd.Series
    monetary_unit: str
    monetary_unit_source: str
    residual_l2: float
    residual_linf: float
    residual_share_l2: float
    metadata: dict[str, Any]



def require_region_sector_multiindex(index: pd.Index) -> pd.MultiIndex:
    """Return ``index`` only if it is the canonical MRIO region-sector index.

    The transition-analysis repos align basis memberships, output vectors, SDA
    components, and extension rows by ``(region, sector)`` labels.  Accepting a
    flat index, duplicate labels, or stringified tuples would make it too easy to
    silently misalign scientific quantities.  Single-region datasets should still
    use a two-level index, for example ``region='TEST'`` and ``sector='s1'``.
    """

    if not isinstance(index, pd.MultiIndex):
        raise TypeError("MRIO index must be a pandas MultiIndex with levels ['region', 'sector'].")
    if list(index.names) != ["region", "sector"]:
        raise ValueError(f"MRIO MultiIndex names must be ['region', 'sector'], got {list(index.names)!r}.")
    if not index.is_unique:
        raise ValueError("MRIO MultiIndex must contain unique (region, sector) labels.")
    return index



def require_same_unit_string(unit_a: str, unit_b: str, *, context: str) -> str:
    """Return a common unit string or raise a clear metadata error.

    This helper is intentionally minimal.  It enforces equality for APIs that do
    combine dimensional values, but it does not attempt any conversion.  If a
    caller wants conversion, that conversion should happen before this strict
    SDA function is called and should be documented in ``unit_source`` metadata.
    """

    left = str(unit_a).strip()
    right = str(unit_b).strip()
    if not left or not right:
        raise ValueError(f"Missing unit metadata while checking {context}: {unit_a!r}, {unit_b!r}")
    if left != right:
        raise ValueError(f"Unit mismatch for {context}: {left!r} != {right!r}")
    return left




def _ensure_finite_array(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``values`` only if every numeric entry is finite.

    The strict SDA API is paper-facing: missing, NaN, infinite, or non-numeric
    data must fail loudly instead of being silently reinterpreted as zero output,
    zero demand, or zero technical coefficients.
    """

    arr = np.asarray(values, dtype=float)
    bad_mask = ~np.isfinite(arr)
    if bool(bad_mask.any()):
        bad_count = int(bad_mask.sum())
        raise ValueError(f"{name} contains {bad_count} non-finite or non-numeric value(s).")
    return arr





def _strict_float_series(values: pd.Series, *, index: pd.MultiIndex, name: str) -> pd.Series:
    """Validate and coerce one vector without hiding index or data errors.

    ``load_mrio_matrices`` returns pandas Series already indexed by the canonical
    EXIO basis.  Reindexing or filling missing values inside the strict SDA
    function would hide a scientific error, so this helper requires exact label
    equality and finite numeric entries before returning a float Series.
    """

    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values)!r}")
    if not values.index.equals(index):
        raise ValueError(f"{name} index does not match the supplied canonical MRIO index.")
    try:
        numeric = values.astype("float64", copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains non-numeric value(s).") from exc
    arr = np.asarray(numeric, dtype=float)
    bad_mask = ~np.isfinite(arr)
    if bool(bad_mask.any()):
        bad_positions = np.flatnonzero(bad_mask)
        bad_labels = [repr(values.index[int(pos)]) for pos in bad_positions[:5]]
        suffix = "" if int(bad_mask.sum()) <= 5 else "; first bad labels shown"
        raise ValueError(
            f"{name} contains {int(bad_mask.sum())} non-finite or non-numeric value(s)"
            f" at labels {bad_labels}{suffix}."
        )
    return pd.Series(arr, index=index, name=values.name, dtype=float)





def _strict_csr_matrix(values: Any, *, index: pd.MultiIndex, name: str) -> csr_matrix:
    """Return ``values`` as CSR while preserving sparse EXIO-scale behaviour.

    Sparse matrices are accepted by shape because SciPy CSR matrices do not carry
    pandas labels.  DataFrames are checked by row and column labels before being
    converted.  Non-finite matrix entries fail loudly; silently mapping malformed
    coefficients to zero would corrupt the SDA components.
    """

    n = len(index)
    if issparse(values):
        matrix = values if isinstance(values, csr_matrix) else values.tocsr()
        if matrix.dtype != np.dtype("float64"):
            matrix = matrix.astype(np.float64, copy=False)
        if matrix.shape != (n, n):
            raise ValueError(f"{name} shape {matrix.shape!r} does not match index length {n}.")
        if matrix.data.size:
            _ensure_finite_array(matrix.data, name=f"{name}.data")
        return matrix
    if isinstance(values, pd.DataFrame):
        if not values.index.equals(index) or not values.columns.equals(index):
            raise ValueError(f"{name} DataFrame row/column labels must match the canonical MRIO index.")
        numeric = values.apply(pd.to_numeric, errors="coerce")
        array = _ensure_finite_array(numeric.to_numpy(dtype=float), name=name)
        return csr_matrix(array)
    array = np.asarray(values, dtype=float)
    if array.shape != (n, n):
        raise ValueError(f"{name} array shape {array.shape!r} does not match index length {n}.")
    array = _ensure_finite_array(array, name=name)
    return csr_matrix(array)


def _solve_sparse_leontief_many(label: str, A: csr_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``(I - A) Z = RHS`` once for one or more right-hand sides.

    SciPy/SuperLU factorization is the memory-expensive step.  Passing both SDA
    right-hand sides for a given year in one dense ``(n, k)`` array lets SciPy
    reuse one factorization instead of refactorizing the same Leontief matrix for
    the intermediate-use and demand components separately.  This still does not
    form or store the explicit Leontief inverse, which would usually be dense.
    """

    rhs_array = np.asarray(rhs, dtype=float)
    if rhs_array.ndim == 1:
        rhs_matrix = rhs_array.reshape((-1, 1))
    elif rhs_array.ndim == 2:
        rhs_matrix = rhs_array
    else:
        raise ValueError(f"{label} RHS must be one- or two-dimensional, got shape {rhs_array.shape!r}")
    rhs_matrix = cast(np.ndarray, rhs_matrix)
    if rhs_matrix.shape[0] != A.shape[0]:
        raise ValueError(f"{label} RHS length {rhs_matrix.shape[0]} does not match matrix shape {A.shape!r}")
    if not np.isfinite(rhs_matrix).all():
        raise ValueError(f"{label} RHS contains non-finite values")

    system = sparse_eye(A.shape[0], format="csr", dtype=float) - A
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solution = spsolve(system, rhs_matrix)
    except Exception as exc:  # pragma: no cover - exact SciPy exception varies
        raise RuntimeError(f"{label} sparse Leontief solve failed") from exc
    finally:
        del system
        gc.collect()
    out = np.asarray(solution, dtype=float)
    if rhs_matrix.shape[1] == 1 and out.ndim == 1:
        out = out.reshape((-1, 1))
    if out.shape != rhs_matrix.shape or not np.isfinite(out).all():
        raise RuntimeError(f"{label} sparse Leontief solve returned non-finite values")
    return out


def _solve_sparse_leontief(label: str, A: csr_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve one Leontief RHS through the multi-RHS implementation."""

    return _solve_sparse_leontief_many(label, A, rhs)[:, 0]


def compute_output_sda_pair_result(
    *,
    start_year: int,
    end_year: int,
    A_start: Any,
    A_end: Any,
    x_start: pd.Series,
    x_end: pd.Series,
    y_start: pd.Series,
    y_end: pd.Series,
    index: pd.MultiIndex,
    monetary_unit: str,
    monetary_unit_source: str,
    average_identities: bool = True,
) -> OutputSdaPairResult:
    """Compute strict average-polar output SDA vectors for one year pair.

    The forward polar identity is
    ``L_end ΔA x_start + L_end Δy``.  When ``average_identities`` is true, the
    backward identity ``L_start ΔA x_end + L_start Δy`` is also solved and the
    two polar components are averaged.  Unlike the legacy table API, this function
    raises if a requested solve fails; it never skips a pair or silently falls back
    from average-polar to forward-only decomposition.
    """

    idx = require_region_sector_multiindex(index)
    unit = str(monetary_unit).strip()
    unit_source = str(monetary_unit_source).strip()
    if not unit:
        raise ValueError("monetary_unit must be a non-empty string")
    if not unit_source:
        raise ValueError("monetary_unit_source must be a non-empty string")

    # Keep all monetary vectors on the same explicit basis.  This avoids the old
    # long-table behaviour that stringified MultiIndex labels and made later basis
    # alignment fragile.
    x0 = _strict_float_series(x_start, index=idx, name="x_start")
    x1 = _strict_float_series(x_end, index=idx, name="x_end")
    y0 = _strict_float_series(y_start, index=idx, name="y_start")
    y1 = _strict_float_series(y_end, index=idx, name="y_end")

    # Convert matrices to CSR without densifying existing sparse loaders.  The
    # SDA formulas require A, not raw Z; callers that need Z should use a separate
    # loader API rather than this output-change decomposition function.
    A0 = _strict_csr_matrix(A_start, index=idx, name="A_start")
    A1 = _strict_csr_matrix(A_end, index=idx, name="A_end")
    x0_values = x0.to_numpy(dtype=float)
    x1_values = x1.to_numpy(dtype=float)
    delta_y = (y1 - y0).to_numpy(dtype=float)

    # Only the products ΔA x_t are needed.  Avoid materializing a third sparse
    # matrix ``delta_A = A1 - A0`` and solve both RHS columns for each Leontief
    # system together so SuperLU factorizes each year at most once.
    delta_A_x0 = A1.dot(x0_values) - A0.dot(x0_values)
    delta_A_x1 = A1.dot(x1_values) - A0.dot(x1_values)
    forward_solution = _solve_sparse_leontief_many(
        "forward components",
        A1,
        np.column_stack((delta_A_x0, delta_y)),
    )
    forward_intermediate = forward_solution[:, 0]
    forward_demand = forward_solution[:, 1]

    if average_identities:
        # Backward polar form: start-year Leontief inverse.  This must also solve
        # successfully; otherwise the average-polar SDA requested by the caller is
        # not available and the scientific result should not be silently degraded.
        backward_solution = _solve_sparse_leontief_many(
            "backward components",
            A0,
            np.column_stack((delta_A_x1, delta_y)),
        )
        backward_intermediate = backward_solution[:, 0]
        backward_demand = backward_solution[:, 1]
        intermediate_values = 0.5 * (forward_intermediate + backward_intermediate)
        demand_values = 0.5 * (forward_demand + backward_demand)
        identity_mode = "average_polar_forward_backward"
    else:
        intermediate_values = forward_intermediate
        demand_values = forward_demand
        identity_mode = "forward_only"

    observed = (x1 - x0).astype(float)
    intermediate = pd.Series(intermediate_values, index=idx, dtype=float, name="sda_intermediate_delta")
    demand = pd.Series(demand_values, index=idx, dtype=float, name="sda_demand_delta")
    residual = (observed - intermediate - demand).astype(float)
    residual.name = "sda_residual"

    residual_arr = residual.to_numpy(dtype=float)
    observed_norm = float(np.linalg.norm(observed.to_numpy(dtype=float), ord=2))
    residual_l2 = float(np.linalg.norm(residual_arr, ord=2))
    residual_linf = float(np.linalg.norm(residual_arr, ord=np.inf)) if residual_arr.size else 0.0
    residual_share_l2 = float(residual_l2 / observed_norm) if observed_norm > 0.0 else 0.0

    return OutputSdaPairResult(
        start_year=int(start_year),
        end_year=int(end_year),
        index=idx,
        observed_output_delta=observed.rename("observed_output_delta"),
        sda_intermediate_delta=intermediate,
        sda_demand_delta=demand,
        sda_residual=residual,
        monetary_unit=unit,
        monetary_unit_source=unit_source,
        residual_l2=residual_l2,
        residual_linf=residual_linf,
        residual_share_l2=residual_share_l2,
        metadata={
            "identity_mode": identity_mode,
            "average_identities": bool(average_identities),
            "index_columns": ["region", "sector"],
            "stores_large_matrices": False,
            "returned_vectors": [
                "observed_output_delta",
                "sda_intermediate_delta",
                "sda_demand_delta",
                "sda_residual",
            ],
        },
    )


def _load_sda_year_matrices(
    year: int,
    *,
    system: str,
    data_root: str | Path | None,
    allow_download_missing: bool,
    force_download: bool = False,
    doi: str | None = None,
    verbose: bool = False,
) -> tuple[csr_matrix, pd.Series, pd.Series, pd.MultiIndex, dict[str, Any]]:
    """Load one EXIOBASE year as sparse A plus checked x/y vectors for SDA.

    The helper still streams from the source zip through ``load_mrio_matrices``
    and keeps derived cache reads/writes disabled.
    """

    from ecolecon_mrio.data import load_mrio_matrices

    load_kwargs: dict[str, Any] = {
        "data_root": data_root,
        "allow_download_missing": bool(allow_download_missing),
        "force_download": bool(force_download),
        "exio_system": str(system),
        "verbose": bool(verbose),
        "cache_matrices": False,
        "read_existing_cache": False,
        "delete_source_zip_after_parse": False,
    }
    if doi is not None and str(doi).strip():
        load_kwargs["doi"] = str(doi).strip()
    A, x, y, index, metadata = load_mrio_matrices(int(year), **load_kwargs)
    idx = require_region_sector_multiindex(index)
    x_series = x if isinstance(x, pd.Series) else pd.Series(x, index=idx)
    y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=idx)
    return (
        _strict_csr_matrix(A, index=idx, name=f"A_{int(year)}"),
        _strict_float_series(x_series, index=idx, name=f"x_{int(year)}"),
        _strict_float_series(y_series, index=idx, name=f"y_{int(year)}"),
        idx,
        dict(metadata or {}),
    )


def _build_output_sda_result(
    *,
    start_year: int,
    end_year: int,
    index: pd.MultiIndex,
    x_start: pd.Series,
    x_end: pd.Series,
    intermediate_values: np.ndarray,
    demand_values: np.ndarray,
    monetary_unit: str,
    monetary_unit_source: str,
    identity_mode: str,
    average_identities: bool,
    metadata_extra: dict[str, Any] | None = None,
) -> OutputSdaPairResult:
    """Construct the compact SDA result after low-memory solves finish."""

    unit = str(monetary_unit).strip()
    unit_source = str(monetary_unit_source).strip()
    if not unit:
        raise ValueError("monetary_unit must be a non-empty string")
    if not unit_source:
        raise ValueError("monetary_unit_source must be a non-empty string")
    observed = (x_end - x_start).astype(float)
    intermediate = pd.Series(intermediate_values, index=index, dtype=float, name="sda_intermediate_delta")
    demand = pd.Series(demand_values, index=index, dtype=float, name="sda_demand_delta")
    residual = (observed - intermediate - demand).astype(float)
    residual.name = "sda_residual"
    residual_arr = residual.to_numpy(dtype=float)
    observed_norm = float(np.linalg.norm(observed.to_numpy(dtype=float), ord=2))
    residual_l2 = float(np.linalg.norm(residual_arr, ord=2))
    residual_linf = float(np.linalg.norm(residual_arr, ord=np.inf)) if residual_arr.size else 0.0
    residual_share_l2 = float(residual_l2 / observed_norm) if observed_norm > 0.0 else 0.0
    metadata: dict[str, Any] = {
        "identity_mode": identity_mode,
        "average_identities": bool(average_identities),
        "index_columns": ["region", "sector"],
        "stores_large_matrices": False,
        "low_memory_matrix_reload": True,
        "forms_explicit_leontief_inverse": False,
        "forms_sparse_delta_A": False,
        "rhs_columns_per_factorization": 2,
        "returned_vectors": [
            "observed_output_delta",
            "sda_intermediate_delta",
            "sda_demand_delta",
            "sda_residual",
        ],
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return OutputSdaPairResult(
        start_year=int(start_year),
        end_year=int(end_year),
        index=index,
        observed_output_delta=observed.rename("observed_output_delta"),
        sda_intermediate_delta=intermediate,
        sda_demand_delta=demand,
        sda_residual=residual,
        monetary_unit=unit,
        monetary_unit_source=unit_source,
        residual_l2=residual_l2,
        residual_linf=residual_linf,
        residual_share_l2=residual_share_l2,
        metadata=metadata,
    )



def load_mrio_year_state(
    year: int,
    *,
    system: str = "pxp",
    data_root: str | Path | None = None,
    allow_download_missing: bool = False,
    force_download: bool = False,
    doi: str | None = None,
    verbose: bool = False,
    monetary_unit: str | None = None,
    monetary_unit_source: str | None = None,
) -> MrioYearState:
    """Load one MRIO year once as sparse ``A`` plus aligned ``x``/``y`` vectors.

    This is the reusable state boundary for panel SDA runs.  It streams the
    source zip through the existing sparse loader, derives ``A`` from ``Z`` and
    ``x``, then returns a compact year state.  It does not form or store an
    explicit Leontief inverse, factorization, raw ``Z`` matrix, or derived cache.
    """

    from ecolecon_mrio.data import exiobase3_monetary_unit

    unit = str(monetary_unit).strip() if monetary_unit is not None else ""
    unit_source = str(monetary_unit_source).strip() if monetary_unit_source is not None else ""
    if not unit:
        unit_metadata = exiobase3_monetary_unit(system=str(system))
        unit = str(unit_metadata.unit).strip()
        if not unit_source:
            unit_source = str(unit_metadata.unit_source).strip()
    if not unit_source:
        unit_source = "explicit_monetary_unit_argument"

    A, x, y, index, metadata = _load_sda_year_matrices(
        int(year),
        system=str(system),
        data_root=data_root,
        allow_download_missing=bool(allow_download_missing),
        force_download=bool(force_download),
        doi=doi,
        verbose=bool(verbose),
    )
    return MrioYearState(
        year=int(year),
        A=A,
        x=x,
        y=y,
        index=index,
        monetary_unit=unit,
        monetary_unit_source=unit_source,
        metadata={
            **dict(metadata or {}),
            "year_state_api": "load_mrio_year_state",
            "forms_explicit_leontief_inverse": False,
            "stores_solver_factorization": False,
        },
    )


def compute_output_sda_pair_result_from_states(
    start: MrioYearState,
    end: MrioYearState,
    *,
    average_identities: bool = True,
) -> OutputSdaPairResult:
    """Compute strict average-polar SDA from two already-loaded year states."""

    if not start.index.equals(end.index):
        raise ValueError("MRIO year-state indexes do not match")
    unit = require_same_unit_string(
        start.monetary_unit,
        end.monetary_unit,
        context="MRIO year-state monetary units",
    )
    source = start.monetary_unit_source
    if end.monetary_unit_source != source:
        source = f"{source}; {end.monetary_unit_source}"
    result = compute_output_sda_pair_result(
        start_year=start.year,
        end_year=end.year,
        A_start=start.A,
        A_end=end.A,
        x_start=start.x,
        x_end=end.x,
        y_start=start.y,
        y_end=end.y,
        index=start.index,
        monetary_unit=unit,
        monetary_unit_source=source,
        average_identities=bool(average_identities),
    )
    result.metadata.update(
        {
            "input_api": "MrioYearState",
            "year_state_reuse": True,
            "low_memory_matrix_reload": False,
            "forms_explicit_leontief_inverse": False,
            "forms_sparse_delta_A": False,
            "start_metadata": start.metadata,
            "end_metadata": end.metadata,
        }
    )
    return result

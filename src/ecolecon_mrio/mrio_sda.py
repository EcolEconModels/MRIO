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
from typing import Any
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
        matrix = values.tocsr().astype(float)
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


def _solve_sparse_leontief(label: str, A: csr_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``(I - A) z = rhs`` and convert singular warnings into errors."""

    system = sparse_eye(A.shape[0], format="csr", dtype=float) - A
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solution = spsolve(system, rhs)
    except Exception as exc:  # pragma: no cover - exact SciPy exception varies
        raise RuntimeError(f"{label} sparse Leontief solve failed") from exc
    out = np.asarray(solution, dtype=float)
    if out.shape != rhs.shape or not np.isfinite(out).all():
        raise RuntimeError(f"{label} sparse Leontief solve returned non-finite values")
    return out


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
    delta_A = A1 - A0
    delta_y = (y1 - y0).to_numpy(dtype=float)

    # Forward polar form: end-year Leontief inverse premultiplies both ΔA and Δy.
    forward_intermediate = _solve_sparse_leontief(
        "forward intermediate-use component",
        A1,
        delta_A.dot(x0.to_numpy(dtype=float)),
    )
    forward_demand = _solve_sparse_leontief("forward demand component", A1, delta_y)

    if average_identities:
        # Backward polar form: start-year Leontief inverse.  This must also solve
        # successfully; otherwise the average-polar SDA requested by the caller is
        # not available and the scientific result should not be silently degraded.
        backward_intermediate = _solve_sparse_leontief(
            "backward intermediate-use component",
            A0,
            delta_A.dot(x1.to_numpy(dtype=float)),
        )
        backward_demand = _solve_sparse_leontief("backward demand component", A0, delta_y)
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

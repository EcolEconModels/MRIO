from __future__ import annotations

from typing import Any, Iterable, cast
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye as sparse_eye, issparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve


def normalize_productive_certificate_vector(
    q: pd.Series | np.ndarray | Iterable[float],
    index: pd.Index,
    *,
    floor: float = 1.0e-12,
) -> pd.Series:
    """Return a strictly positive certificate vector normalized to unit sum."""
    q_ser = cast(
        pd.Series,
        pd.to_numeric(pd.Series(q).reindex(index), errors="coerce"),
    ).fillna(0.0).astype(float)
    q_ser = q_ser.clip(lower=float(floor))
    total = float(q_ser.sum())
    if not np.isfinite(total) or total <= 0.0:
        n = max(len(index), 1)
        return pd.Series(np.full(len(index), 1.0 / n, dtype=float), index=index, dtype=float)
    return (q_ser / total).astype(float)


def coefficient_change_bounds(
    a_row: pd.Series | np.ndarray | Iterable[float],
    *,
    relative_change_cap: float | None,
    relative_change_cap_floor: float = 1.0e-4,
    nonnegativity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for coefficient changes around an observed row."""
    a = cast(
        pd.Series,
        pd.to_numeric(pd.Series(a_row), errors="coerce"),
    ).fillna(0.0).to_numpy(dtype=float)
    lower = np.full_like(a, -np.inf, dtype=float)
    upper = np.full_like(a, np.inf, dtype=float)
    if nonnegativity:
        lower = np.maximum(lower, -a)
    if relative_change_cap is not None and np.isfinite(float(relative_change_cap)):
        cap = float(relative_change_cap) * np.maximum(np.abs(a), float(relative_change_cap_floor))
        lower = np.maximum(lower, -cap)
        upper = np.minimum(upper, cap)
    return lower.astype(float), upper.astype(float)


def productive_certificate_row_rhs(
    a_row: pd.Series,
    productive_q: pd.Series | np.ndarray | Iterable[float] | None,
    productive_rho: float | None,
    *,
    row_label: Any,
    tolerance: float = 1.0e-9,
) -> float | None:
    r"""Return the available row slack under the certificate \(A' q \le \rho q\)."""
    if productive_q is None or productive_rho is None:
        return None
    q_ser = cast(
        pd.Series,
        pd.to_numeric(pd.Series(productive_q).reindex(a_row.index), errors="coerce"),
    ).fillna(0.0).astype(float)
    if not np.isfinite(q_ser.to_numpy(dtype=float)).all():
        return None
    lhs_base = float(
        np.dot(
            cast(
                pd.Series,
                pd.to_numeric(a_row.reindex(q_ser.index), errors="coerce"),
            ).fillna(0.0).to_numpy(dtype=float),
            q_ser.to_numpy(dtype=float),
        )
    )
    numeric_scalar = cast(
        Any,
        pd.to_numeric(pd.Series(productive_q).get(row_label, np.nan), errors="coerce"),
    )
    rhs_total = float(productive_rho) * float(numeric_scalar)
    return float(rhs_total - lhs_base + float(tolerance))


def productive_certificate_slack(
    A: pd.DataFrame,
    productive_q: pd.Series | np.ndarray | Iterable[float],
    productive_rho: float,
) -> pd.Series:
    r"""Return row-wise slack for the certificate \(A q \le \rho q\)."""
    A_df = pd.DataFrame(A).copy()
    q_ser = normalize_productive_certificate_vector(productive_q, pd.Index(A_df.columns, dtype=object))
    q_row = q_ser.reindex(A_df.index).fillna(0.0).astype(float)
    lhs = cast(
        pd.Series,
        pd.to_numeric(
            A_df.reindex(index=q_row.index, columns=q_ser.index).fillna(0.0).dot(q_ser),
            errors="coerce",
        ),
    ).fillna(0.0).astype(float)
    rhs = float(productive_rho) * q_row
    return (rhs - lhs).astype(float)


def satisfies_productive_certificate(
    A: pd.DataFrame,
    productive_q: pd.Series | np.ndarray | Iterable[float],
    productive_rho: float,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    r"""Check whether \(A q \le \rho q\) holds row by row up to tolerance."""
    slack = productive_certificate_slack(A, productive_q, productive_rho)
    return bool((slack >= -float(tolerance)).all())


def emissions_direction_slope(
    emissions_intensity: pd.Series | np.ndarray | Iterable[float],
    direction: pd.Series | np.ndarray | Iterable[float],
) -> float:
    r"""Return \(e^\top d\) after aligning indices when both inputs are Series."""
    e_ser = pd.Series(emissions_intensity)
    d_ser = pd.Series(direction)
    if isinstance(emissions_intensity, pd.Series) or isinstance(direction, pd.Series):
        d_ser = d_ser.reindex(e_ser.index)
    e_vec = cast(pd.Series, pd.to_numeric(e_ser, errors="coerce")).fillna(0.0).to_numpy(dtype=float)
    d_vec = cast(pd.Series, pd.to_numeric(d_ser, errors="coerce")).fillna(0.0).to_numpy(dtype=float)
    return float(np.dot(e_vec, d_vec))


def benchmark_step_for_emissions_cut(
    cut_share: float,
    *,
    baseline_emissions: float,
    emissions_slope: float,
) -> float:
    """Map a benchmark direct-emissions cut to the required signed-output step."""
    if float(cut_share) <= 0.0:
        return 0.0
    if not np.isfinite(float(baseline_emissions)) or float(baseline_emissions) <= 0.0:
        return np.nan
    if not np.isfinite(float(emissions_slope)) or float(emissions_slope) >= -1.0e-12:
        return np.nan
    return float(cut_share) * float(baseline_emissions) / float(-emissions_slope)


def emissions_cut_from_step(
    step_abs: float,
    *,
    baseline_emissions: float,
    emissions_slope: float,
) -> float:
    """Convert a signed-output step into the associated direct-emissions cut share."""
    if not np.isfinite(float(step_abs)):
        return np.nan
    if float(step_abs) <= 0.0:
        return 0.0
    if not np.isfinite(float(baseline_emissions)) or float(baseline_emissions) <= 0.0:
        return np.nan
    if not np.isfinite(float(emissions_slope)) or float(emissions_slope) >= -1.0e-12:
        return np.nan
    return float((-float(emissions_slope) * float(step_abs)) / float(baseline_emissions))


def max_output_nonnegative_step(
    x_obs: pd.Series,
    direction: pd.Series,
) -> float:
    r"""Largest absolute step such that \(x + s d \ge 0\) componentwise."""
    x_ser = cast(pd.Series, pd.to_numeric(pd.Series(x_obs), errors="coerce")).fillna(0.0).astype(float)
    d_ser = cast(
        pd.Series,
        pd.to_numeric(pd.Series(direction).reindex(x_ser.index), errors="coerce"),
    ).fillna(0.0).astype(float)
    neg_mask = d_ser < 0.0
    if not bool(neg_mask.any()):
        return np.inf
    return float((x_ser.loc[neg_mask] / (-d_ser.loc[neg_mask])).min())


def max_output_nonnegative_emissions_cut(
    x_obs: pd.Series,
    direction: pd.Series,
    *,
    baseline_emissions: float,
    emissions_slope: float,
) -> float:
    """Largest direct-emissions cut consistent with nonnegative gross output."""
    step_abs = max_output_nonnegative_step(x_obs, direction)
    if not np.isfinite(step_abs):
        return np.inf
    return max(0.0, float(emissions_cut_from_step(step_abs, baseline_emissions=baseline_emissions, emissions_slope=emissions_slope)))


def max_demand_nonnegative_step(
    A: pd.DataFrame,
    y_obs: pd.Series,
    direction: pd.Series,
) -> float:
    r"""Largest absolute step such that \(y + s(I-A)d \ge 0\) componentwise."""
    if issparse(A):
        A_sparse = cast(csr_matrix, csr_matrix(A, dtype=float))
        sparse_shape = cast(tuple[int, int], A_sparse.shape)
        idx = pd.RangeIndex(sparse_shape[0])
        if hasattr(A, "index"):
            idx = pd.Index(getattr(A, "index"), dtype=object)
        y_ser = cast(pd.Series, pd.to_numeric(pd.Series(y_obs).reindex(idx), errors="coerce")).fillna(0.0).astype(float)
        d_ser = cast(pd.Series, pd.to_numeric(pd.Series(direction).reindex(idx), errors="coerce")).fillna(0.0).astype(float)
        d_values = d_ser.to_numpy(dtype=float)
        delta_y_unit = pd.Series(d_values - A_sparse.dot(d_values), index=idx, dtype=float)
    elif isinstance(A, pd.DataFrame) and all(isinstance(dtype, pd.SparseDtype) for dtype in A.dtypes):
        A_sparse = A.sparse.to_coo().tocsr().astype(float)
        idx = pd.Index(A.index, dtype=object)
        y_ser = cast(pd.Series, pd.to_numeric(pd.Series(y_obs).reindex(idx), errors="coerce")).fillna(0.0).astype(float)
        d_ser = cast(pd.Series, pd.to_numeric(pd.Series(direction).reindex(idx), errors="coerce")).fillna(0.0).astype(float)
        d_values = d_ser.to_numpy(dtype=float)
        delta_y_unit = pd.Series(d_values - A_sparse.dot(d_values), index=idx, dtype=float)
    else:
        A_df = pd.DataFrame(A).copy()
        idx = pd.Index(A_df.index, dtype=object)
        A_df = A_df.reindex(index=idx, columns=idx).fillna(0.0).astype(float)
        y_ser = cast(pd.Series, pd.to_numeric(pd.Series(y_obs).reindex(idx), errors="coerce")).fillna(0.0).astype(float)
        d_ser = cast(pd.Series, pd.to_numeric(pd.Series(direction).reindex(idx), errors="coerce")).fillna(0.0).astype(float)
        I_minus_A = pd.DataFrame(np.eye(len(idx)), index=idx, columns=idx) - A_df
        delta_y_unit = cast(
            pd.Series,
            pd.to_numeric(I_minus_A.dot(d_ser), errors="coerce"),
        ).fillna(0.0).astype(float)
    neg_mask = delta_y_unit < 0.0
    if not bool(neg_mask.any()):
        return np.inf
    return float((y_ser.loc[neg_mask] / (-delta_y_unit.loc[neg_mask])).min())


def max_demand_nonnegative_emissions_cut(
    A: pd.DataFrame,
    y_obs: pd.Series,
    direction: pd.Series,
    *,
    baseline_emissions: float,
    emissions_slope: float,
) -> float:
    """Largest direct-emissions cut consistent with nonnegative final demand under fixed technology."""
    step_abs = max_demand_nonnegative_step(A, y_obs, direction)
    if not np.isfinite(step_abs):
        return np.inf
    return max(0.0, float(emissions_cut_from_step(step_abs, baseline_emissions=baseline_emissions, emissions_slope=emissions_slope)))

def _nonnegative_support_matrix(A: Any) -> csr_matrix:
    """Return a finite non-negative square CSR matrix for support diagnostics."""

    matrix = cast(csr_matrix, csr_matrix(A, dtype=np.float64).copy())
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix_shape = cast(tuple[int, int], matrix.shape)
    matrix_data = cast(np.ndarray, matrix.data)
    if matrix_shape[0] == 0 or matrix_shape[0] != matrix_shape[1]:
        raise ValueError(f"A must be non-empty and square, got {matrix_shape}")
    if matrix_data.size and not np.isfinite(matrix_data).all():
        raise ValueError("A contains non-finite stored coefficients")
    if matrix_data.size and bool((matrix_data < 0.0).any()):
        raise ValueError("A must be non-negative")
    return matrix


def baseline_productivity_certificate(
    A: Any,
    *,
    margin_fraction: float = 0.5,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, float, np.ndarray]:
    r"""Derive the fixed sufficient certificate ``A'v <= (1-epsilon)v``.

    For productive non-negative ``A``, ``v=(I-A)^{-1}1`` is positive and obeys
    ``Av=v-1``. Retaining a fraction of ``min(1/v)`` keeps the observed matrix
    inside the later coefficient-support domain. The fixed vector is sufficient,
    not necessary, for productivity of a counterfactual matrix.
    """

    fraction = float(margin_fraction)
    tol = float(tolerance)
    if not 0.0 < fraction <= 1.0 or tol <= 0.0:
        raise ValueError("margin_fraction must be in (0, 1] and tolerance positive")
    matrix = _nonnegative_support_matrix(A)
    matrix_shape = cast(tuple[int, int], matrix.shape)
    system = sparse_eye(matrix_shape[0], format="csr", dtype=np.float64) - matrix
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            vector = np.asarray(
                spsolve(system, np.ones(matrix_shape[0], dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
    except Exception as exc:  # pragma: no cover - SciPy failure varies by factorization
        raise RuntimeError("baseline productivity-certificate solve failed") from exc
    if vector.shape != (matrix_shape[0],) or not np.isfinite(vector).all():
        raise RuntimeError("baseline productivity-certificate solve returned invalid values")
    if bool((vector <= tol).any()):
        raise ValueError("baseline matrix has no strictly positive certificate")
    epsilon = fraction * float(np.min(1.0 / vector))
    slack = np.asarray((1.0 - epsilon) * vector - matrix.dot(vector), dtype=np.float64)
    if float(slack.min()) < -tol:
        raise RuntimeError("constructed productivity certificate has negative slack")
    return vector, epsilon, slack


def coefficient_edge_counts(A: Any, thresholds: Iterable[float]) -> dict[float, int]:
    """Count stored coefficients strictly above each declared dimensionless threshold."""

    matrix = _nonnegative_support_matrix(A)
    values = tuple(float(value) for value in thresholds)
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("coefficient thresholds must be finite and non-negative")
    return {value: int(np.count_nonzero(matrix.data > value)) for value in values}


def adjacent_coefficient_relative_changes(
    A_start: Any,
    A_end: Any,
    *,
    floor: float,
) -> tuple[np.ndarray, int, int]:
    r"""Return pair-active ``|A1-A0|/max(A0,floor)`` values and edge turnover.

    Pair-active links exceed ``floor`` in at least one endpoint. Unchanged active
    links remain as zeros in the returned distribution; structural zeros are not
    allowed to dominate the empirical cap calibration.
    """

    epsilon = float(floor)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("floor must be finite and positive")
    start = _nonnegative_support_matrix(A_start)
    end = _nonnegative_support_matrix(A_end)
    if start.shape != end.shape:
        raise ValueError(f"adjacent coefficient matrices differ in shape: {start.shape}, {end.shape}")
    active = (start.maximum(end) > epsilon).tocoo()
    if active.nnz == 0:
        return np.array([], dtype=np.float64), 0, 0
    start_values = np.asarray(start[active.row, active.col]).reshape(-1)
    end_values = np.asarray(end[active.row, active.col]).reshape(-1)
    relative = np.abs(end_values - start_values) / np.maximum(start_values, epsilon)
    appearing = int(np.count_nonzero((start_values <= epsilon) & (end_values > epsilon)))
    disappearing = int(np.count_nonzero((start_values > epsilon) & (end_values <= epsilon)))
    return np.asarray(relative, dtype=np.float64), appearing, disappearing


def certificate_safe_coefficient_delta_bounds(
    A: Any,
    panel_mask: Any,
    *,
    certificate_vector: np.ndarray,
    productivity_epsilon: float,
    relative_change_cap: float,
    coefficient_floor: float,
    tolerance: float = 1.0e-12,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    r"""Return exact sparse delta-A bounds that guarantee the fixed certificate.

    The raw local cap is ``kappa * max(A_ij, floor)`` on panel-observed links.
    Negative changes retain the full non-negativity-limited cap. Positive caps
    are multiplied by one row factor so their worst-case certificate use fits
    inside the baseline slack. Hence every delta within the returned interval
    satisfies ``(A + delta) v <= (1-epsilon) v`` without relying on offsetting
    coefficient decreases elsewhere in the row.
    """

    matrix = _nonnegative_support_matrix(A)
    matrix_shape = cast(tuple[int, int], matrix.shape)
    cap = float(relative_change_cap)
    floor = float(coefficient_floor)
    epsilon = float(productivity_epsilon)
    tol = float(tolerance)
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError("relative_change_cap must be finite and non-negative")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("coefficient_floor must be finite and positive")
    if not 0.0 < epsilon < 1.0 or tol <= 0.0:
        raise ValueError("productivity_epsilon must lie in (0, 1) and tolerance be positive")

    vector = np.asarray(certificate_vector, dtype=np.float64).reshape(-1)
    if (
        vector.shape != (matrix_shape[0],)
        or not np.isfinite(vector).all()
        or bool((vector <= 0.0).any())
    ):
        raise ValueError("certificate_vector must be finite, strictly positive, and aligned")

    raw_mask = cast(csr_matrix, csr_matrix(panel_mask, dtype=np.float64).copy())
    raw_mask.sum_duplicates()
    raw_mask.eliminate_zeros()
    raw_mask_shape = cast(tuple[int, int], raw_mask.shape)
    if raw_mask_shape != matrix_shape:
        raise ValueError(f"panel mask shape {raw_mask_shape} differs from A shape {matrix_shape}")
    mask_data = cast(np.ndarray, raw_mask.data)
    if mask_data.size and (
        not np.isfinite(mask_data).all() or bool((mask_data < 0.0).any())
    ):
        raise ValueError("panel mask must contain finite non-negative values")
    mask = cast(Any, (raw_mask > 0.0).tocoo())
    mask_rows = np.asarray(mask.row, dtype=np.int64)
    mask_columns = np.asarray(mask.col, dtype=np.int64)

    baseline_values = np.asarray(matrix[mask_rows, mask_columns]).reshape(-1)
    raw_cap = cap * np.maximum(baseline_values, floor)
    lower_data = -np.minimum(baseline_values, raw_cap)
    lower = cast(
        csr_matrix,
        csr_matrix((lower_data, (mask_rows, mask_columns)), shape=matrix_shape),
    )
    upper_raw = cast(
        csr_matrix,
        csr_matrix((raw_cap, (mask_rows, mask_columns)), shape=matrix_shape),
    )

    baseline_slack = np.asarray(
        (1.0 - epsilon) * vector - matrix.dot(vector), dtype=np.float64
    )
    if float(baseline_slack.min()) < -tol:
        raise ValueError("baseline matrix violates the supplied productivity certificate")
    positive_use = np.asarray(upper_raw.dot(vector), dtype=np.float64)
    row_scale = np.ones(matrix_shape[0], dtype=np.float64)
    active = positive_use > tol
    row_scale[active] = np.minimum(
        1.0,
        np.maximum(baseline_slack[active], 0.0) / positive_use[active],
    )
    upper = cast(csr_matrix, upper_raw.multiply(row_scale[:, None]).tocsr())
    guaranteed_margin = baseline_slack - np.asarray(upper.dot(vector), dtype=np.float64)
    if float(guaranteed_margin.min()) < -tol:
        raise RuntimeError("certificate-safe coefficient bounds have negative guaranteed margin")
    lower.eliminate_zeros()
    upper.eliminate_zeros()
    return lower, upper, row_scale, guaranteed_margin

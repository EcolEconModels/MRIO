from __future__ import annotations

from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, lgmres


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
    """Return a canonical finite non-negative CSR matrix without mutating callers."""

    matrix = cast(csr_matrix, csr_matrix(A, dtype=np.float64, copy=False))
    matrix_data = cast(np.ndarray, matrix.data)
    needs_cleanup = (not matrix.has_canonical_format) or bool(
        matrix_data.size and (matrix_data == 0.0).any()
    )
    if needs_cleanup:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix_data = cast(np.ndarray, matrix.data)
    matrix_shape = cast(tuple[int, int], matrix.shape)
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
    ``Av=v-1``. A restarted LGMRES solve avoids the fill-in and retained factors
    of sparse direct LU. Retaining a fraction of ``min(1/v)`` keeps the observed
    matrix inside the later coefficient-support domain. The fixed vector is
    sufficient, not necessary, for productivity of a counterfactual matrix.
    """

    fraction = float(margin_fraction)
    tol = float(tolerance)
    if not 0.0 < fraction <= 1.0 or tol <= 0.0:
        raise ValueError("margin_fraction must be in (0, 1] and tolerance positive")
    matrix = _nonnegative_support_matrix(A)
    matrix_shape = cast(tuple[int, int], matrix.shape)
    rhs = np.ones(matrix_shape[0], dtype=np.float64)
    linear_operator = cast(Any, LinearOperator)
    operator = linear_operator(
        matrix_shape,
        matvec=lambda values: np.asarray(values, dtype=np.float64) - matrix.dot(values),
        dtype=np.float64,
    )
    iterative_solve = cast(Any, lgmres)
    try:
        vector_raw, info = iterative_solve(
            operator,
            rhs,
            x0=rhs,
            rtol=1.0e-11,
            atol=0.0,
            maxiter=500,
            inner_m=30,
            outer_k=3,
        )
    except Exception as exc:  # pragma: no cover - SciPy failure varies by backend
        raise RuntimeError("baseline productivity-certificate iterative solve failed") from exc
    vector = np.asarray(vector_raw, dtype=np.float64).reshape(-1)
    if info != 0:
        raise RuntimeError(
            f"baseline productivity-certificate iterative solve did not converge: info={info}"
        )
    isfinite = cast(Any, np.isfinite)
    if vector.shape != (matrix_shape[0],) or not isfinite(vector).all():
        raise RuntimeError("baseline productivity-certificate solve returned invalid values")
    residual = np.asarray(operator.matvec(vector) - rhs, dtype=np.float64)
    residual_linf = float(np.max(np.abs(residual)))
    if residual_linf > 1.0e-9:
        raise RuntimeError(
            "baseline productivity-certificate solve residual exceeds 1e-9: "
            f"{residual_linf:.6g}"
        )
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


def _pair_active_relative_row(
    start: csr_matrix,
    end: csr_matrix,
    *,
    row: int,
    floor: float,
) -> tuple[np.ndarray, int, int]:
    """Return pair-active relative changes for one CSR row only."""

    start_left = int(start.indptr[row])
    start_right = int(start.indptr[row + 1])
    end_left = int(end.indptr[row])
    end_right = int(end.indptr[row + 1])
    start_columns = np.asarray(start.indices[start_left:start_right], dtype=np.intp)
    end_columns = np.asarray(end.indices[end_left:end_right], dtype=np.intp)
    if start_columns.size == 0 and end_columns.size == 0:
        return np.empty(0, dtype=np.float64), 0, 0

    columns = np.union1d(start_columns, end_columns)
    start_values = np.zeros(columns.size, dtype=np.float64)
    end_values = np.zeros(columns.size, dtype=np.float64)
    if start_columns.size:
        start_values[np.searchsorted(columns, start_columns)] = start.data[start_left:start_right]
    if end_columns.size:
        end_values[np.searchsorted(columns, end_columns)] = end.data[end_left:end_right]

    active = np.maximum(start_values, end_values) > floor
    if not bool(active.any()):
        return np.empty(0, dtype=np.float64), 0, 0
    start_active = start_values[active]
    end_active = end_values[active]
    appearing = int(np.count_nonzero((start_active <= floor) & (end_active > floor)))
    disappearing = int(np.count_nonzero((start_active > floor) & (end_active <= floor)))
    relative = np.abs(end_active - start_active) / np.maximum(start_active, floor)
    return np.asarray(relative, dtype=np.float64), appearing, disappearing


def adjacent_coefficient_change_quantiles(
    A_start: Any,
    A_end: Any,
    *,
    floor: float,
    quantiles: Iterable[float],
) -> tuple[int, int, int, dict[float, float]]:
    r"""Return exact pair-active coefficient-change quantiles with bounded scratch.

    The statistic remains ``|A1-A0|/max(A0,floor)`` on links exceeding ``floor``
    in at least one endpoint.  CSR rows are merged independently, so no full
    pair-active COO union or endpoint-value arrays are materialised.  After an
    initial counting pass, exactly one float64 vector is allocated for the full
    relative-change distribution; NumPy linear quantiles are then selected in
    place.  Callers should release both annual matrices after the pair result.
    """

    epsilon = float(floor)
    requested = tuple(float(value) for value in quantiles)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("floor must be finite and positive")
    if not requested or any(
        not np.isfinite(value) or value < 0.0 or value > 1.0 for value in requested
    ):
        raise ValueError("quantiles must be finite values in [0, 1]")

    start = _nonnegative_support_matrix(A_start)
    end = _nonnegative_support_matrix(A_end)
    if start.shape != end.shape:
        raise ValueError(
            f"adjacent coefficient matrices differ in shape: {start.shape}, {end.shape}"
        )
    shape = cast(tuple[int, int], start.shape)
    active_count = 0
    appearing = 0
    disappearing = 0
    for row in range(shape[0]):
        values, row_appearing, row_disappearing = _pair_active_relative_row(
            start, end, row=row, floor=epsilon
        )
        active_count += int(values.size)
        appearing += row_appearing
        disappearing += row_disappearing

    if active_count == 0:
        return 0, 0, 0, {value: np.nan for value in requested}

    changes = np.empty(active_count, dtype=np.float64)
    offset = 0
    for row in range(shape[0]):
        values, _, _ = _pair_active_relative_row(start, end, row=row, floor=epsilon)
        next_offset = offset + int(values.size)
        changes[offset:next_offset] = values
        offset = next_offset
    if offset != active_count or not np.isfinite(changes).all():
        raise RuntimeError("adjacent coefficient-change streaming count mismatch")

    selected = np.asarray(
        np.quantile(
            changes,
            np.asarray(requested, dtype=np.float64),
            method="linear",
            overwrite_input=True,
        ),
        dtype=np.float64,
    ).reshape(-1)
    return (
        active_count,
        appearing,
        disappearing,
        {quantile: float(value) for quantile, value in zip(requested, selected, strict=True)},
    )



def certificate_safe_coefficient_capacity_products(
    A: Any,
    panel_mask: Any,
    right_hand_sides: np.ndarray,
    *,
    certificate_vector: np.ndarray,
    productivity_epsilon: float,
    relative_change_cap: float,
    coefficient_floor: float,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Apply certificate-safe coefficient bounds directly to selected RHS vectors.

    Figure 6 needs ``lower_delta_A @ R`` and ``upper_delta_A @ R`` for baseline
    output plus six primitive axes, not the sparse bound matrices themselves.
    This routine traverses the panel-observed mask by CSR row, reconstructs only
    the current row's baseline coefficients, applies the same local cap and fixed
    productivity certificate as the declared inner domain, and returns the exact
    capacity products without a full-panel COO conversion or delta-A matrices.
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
    rhs = np.asarray(right_hand_sides, dtype=np.float64)
    if (
        vector.shape != (matrix_shape[0],)
        or not np.isfinite(vector).all()
        or bool((vector <= 0.0).any())
    ):
        raise ValueError("certificate_vector must be finite, strictly positive, and aligned")
    if rhs.ndim != 2 or rhs.shape[0] != matrix_shape[0] or not np.isfinite(rhs).all():
        raise ValueError("right_hand_sides must be a finite N x K matrix aligned to A")

    mask = cast(csr_matrix, csr_matrix(panel_mask, dtype=np.int8, copy=False))
    mask_data = cast(np.ndarray, mask.data)
    if (not mask.has_canonical_format) or bool(mask_data.size and (mask_data == 0).any()):
        mask = mask.copy()
        mask.sum_duplicates()
        mask.eliminate_zeros()
        mask.sort_indices()
        mask_data = cast(np.ndarray, mask.data)
    mask_shape = cast(tuple[int, int], mask.shape)
    if mask_shape != matrix_shape:
        raise ValueError(f"panel mask shape {mask_shape} differs from A shape {matrix_shape}")
    if mask_data.size and bool((mask_data < 0).any()):
        raise ValueError("panel mask must contain non-negative values")

    baseline_slack = np.asarray(
        (1.0 - epsilon) * vector - matrix.dot(vector), dtype=np.float64
    )
    if float(baseline_slack.min()) < -tol:
        raise ValueError("baseline matrix violates the supplied productivity certificate")

    lower_products = np.zeros((matrix_shape[0], rhs.shape[1]), dtype=np.float64)
    upper_products = np.zeros_like(lower_products)
    row_scale = np.ones(matrix_shape[0], dtype=np.float64)
    guaranteed_margin = baseline_slack.copy()

    for row in range(matrix_shape[0]):
        mask_left = int(mask.indptr[row])
        mask_right = int(mask.indptr[row + 1])
        columns = np.asarray(mask.indices[mask_left:mask_right], dtype=np.intp)
        if columns.size == 0:
            continue

        a_left = int(matrix.indptr[row])
        a_right = int(matrix.indptr[row + 1])
        a_columns = np.asarray(matrix.indices[a_left:a_right], dtype=np.intp)
        a_values = np.asarray(matrix.data[a_left:a_right], dtype=np.float64)
        baseline_values = np.zeros(columns.size, dtype=np.float64)
        if a_columns.size:
            positions = np.searchsorted(a_columns, columns)
            in_bounds = positions < a_columns.size
            matches = np.zeros(columns.size, dtype=bool)
            if bool(in_bounds.any()):
                bounded_positions = positions[in_bounds]
                matches[in_bounds] = a_columns[bounded_positions] == columns[in_bounds]
            if bool(matches.any()):
                baseline_values[matches] = a_values[positions[matches]]

        raw_cap = cap * np.maximum(baseline_values, floor)
        lower_data = -np.minimum(baseline_values, raw_cap)
        positive_use = float(np.dot(raw_cap, vector[columns]))
        scale = 1.0
        if positive_use > tol:
            scale = min(1.0, max(float(baseline_slack[row]), 0.0) / positive_use)
        row_scale[row] = scale
        selected_rhs = rhs[columns, :]
        lower_products[row, :] = lower_data @ selected_rhs
        upper_products[row, :] = (scale * raw_cap) @ selected_rhs
        guaranteed_margin[row] = float(baseline_slack[row]) - scale * positive_use

    if float(guaranteed_margin.min()) < -tol:
        raise RuntimeError("certificate-safe coefficient capacities have negative guaranteed margin")
    return lower_products, upper_products, row_scale, guaranteed_margin

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

def merge_panel_observed_coefficient_mask(
    current: Any | None,
    A: Any,
    *,
    threshold: float,
) -> csr_matrix:
    """Merge one annual ``A`` into the panel-observed coefficient mask.

    A link is present iff its observed coefficient is strictly greater than the
    declared threshold in at least one processed year.  The helper keeps the
    union sparse and canonical so audit and support builders cannot drift on
    threshold strictness, zero elimination, or CSR normalization.
    """

    cutoff = float(threshold)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("coefficient-mask threshold must be finite and non-negative")
    matrix = _nonnegative_support_matrix(A)
    annual = cast(csr_matrix, (matrix > cutoff).astype(np.int8).tocsr())
    annual.eliminate_zeros()
    annual.sort_indices()
    if current is None:
        return annual

    panel = cast(csr_matrix, csr_matrix(current, dtype=np.int8, copy=False))
    if panel.shape != matrix.shape:
        raise ValueError(
            f"panel mask shape {panel.shape} differs from annual A shape {matrix.shape}"
        )
    panel_data = cast(np.ndarray, panel.data)
    if panel_data.size and bool((panel_data < 0).any()):
        raise ValueError("panel mask must contain non-negative values")
    if (not panel.has_canonical_format) or bool(panel_data.size and (panel_data == 0).any()):
        panel = panel.copy()
        panel.sum_duplicates()
        panel.eliminate_zeros()
        panel.sort_indices()
    merged = cast(csr_matrix, panel.maximum(annual).tocsr())
    merged.eliminate_zeros()
    merged.sort_indices()
    return merged


def merge_panel_coefficient_envelope(
    active_year_counts: Any | None,
    thresholded_minimum: Any | None,
    thresholded_maximum: Any | None,
    A: Any,
    *,
    threshold: float,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Merge one annual ``A`` into sparse thresholded count/min/max summaries.

    Coefficients at or below ``threshold`` are treated as zero for this panel
    calibration. ``active_year_counts`` therefore records strict-threshold
    persistence, ``thresholded_minimum`` is zero whenever a link is absent in any
    processed year, and ``thresholded_maximum`` stores the largest observed
    above-threshold level. The routine retains only the three sparse summaries,
    never annual matrices.
    """

    cutoff = float(threshold)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("coefficient-envelope threshold must be finite and non-negative")

    matrix = _nonnegative_support_matrix(A)
    matrix_shape = cast(tuple[int, int], matrix.shape)
    matrix_indptr = cast(np.ndarray, matrix.indptr)
    matrix_indices = cast(np.ndarray, matrix.indices)
    matrix_data = cast(np.ndarray, matrix.data)

    missing = (
        active_year_counts is None,
        thresholded_minimum is None,
        thresholded_maximum is None,
    )
    if all(missing):
        # Build the first annual summaries row by row.  A whole-matrix boolean
        # selection would duplicate EXIO-scale index and coefficient arrays.
        row_counts = np.empty(matrix_shape[0], dtype=np.int64)
        for row in range(matrix_shape[0]):
            start, stop = matrix_indptr[row], matrix_indptr[row + 1]
            row_counts[row] = int(np.count_nonzero(matrix_data[start:stop] > cutoff))
        indptr = np.empty(matrix_shape[0] + 1, dtype=matrix_indptr.dtype)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        nnz = int(indptr[-1])
        indices = np.empty(nnz, dtype=matrix_indices.dtype)
        levels = np.empty(nnz, dtype=np.float64)
        for row in range(matrix_shape[0]):
            source_start, source_stop = matrix_indptr[row], matrix_indptr[row + 1]
            target_start, target_stop = indptr[row], indptr[row + 1]
            active = matrix_data[source_start:source_stop] > cutoff
            indices[target_start:target_stop] = matrix_indices[source_start:source_stop][active]
            levels[target_start:target_stop] = matrix_data[source_start:source_stop][active]
        counts = csr_matrix((np.ones(nnz, dtype=np.uint8), indices.copy(), indptr.copy()), shape=matrix_shape)
        minimum = csr_matrix((levels.copy(), indices.copy(), indptr.copy()), shape=matrix_shape)
        maximum = csr_matrix((levels, indices, indptr), shape=matrix_shape)
        return counts, minimum, maximum
    if any(missing):
        raise ValueError("panel coefficient envelope state must be all None or all populated")

    raw_counts = cast(csr_matrix, csr_matrix(active_year_counts, copy=False))
    raw_count_data = np.asarray(raw_counts.data)
    if raw_count_data.size and (
        not np.isfinite(raw_count_data).all()
        or bool((raw_count_data < 0).any())
        or not np.array_equal(raw_count_data, np.rint(raw_count_data))
    ):
        raise ValueError("panel active-year counts must be finite non-negative integers")
    if raw_count_data.size and int(raw_count_data.max()) >= np.iinfo(np.uint8).max:
        raise OverflowError("panel active-year count exceeds uint8 capacity")
    counts = cast(csr_matrix, raw_counts.astype(np.uint8, copy=False))
    minimum = _nonnegative_support_matrix(thresholded_minimum)
    maximum = _nonnegative_support_matrix(thresholded_maximum)
    if any(item.shape != matrix_shape for item in (counts, minimum, maximum)):
        raise ValueError("panel coefficient-envelope matrices must share A's shape")

    # First pass sizes the exact union support for counts/maxima and the exact
    # intersection support for minima.  It retains no EXIO-scale temporary COO or
    # sparse binary-operation result while one annual matrix is live.
    union_counts = np.empty(matrix_shape[0], dtype=np.int64)
    intersection_counts = np.empty(matrix_shape[0], dtype=np.int64)
    for row in range(matrix_shape[0]):
        start, stop = matrix_indptr[row], matrix_indptr[row + 1]
        annual_columns = matrix_indices[start:stop][matrix_data[start:stop] > cutoff]
        count_columns = counts.indices[counts.indptr[row]:counts.indptr[row + 1]]
        minimum_columns = minimum.indices[minimum.indptr[row]:minimum.indptr[row + 1]]
        union_counts[row] = np.union1d(count_columns, annual_columns).size
        intersection_counts[row] = np.intersect1d(minimum_columns, annual_columns, assume_unique=True).size

    union_indptr = np.empty(matrix_shape[0] + 1, dtype=matrix_indptr.dtype)
    minimum_indptr = np.empty(matrix_shape[0] + 1, dtype=matrix_indptr.dtype)
    union_indptr[0] = minimum_indptr[0] = 0
    np.cumsum(union_counts, out=union_indptr[1:])
    np.cumsum(intersection_counts, out=minimum_indptr[1:])
    union_nnz = int(union_indptr[-1])
    minimum_nnz = int(minimum_indptr[-1])
    merged_indices = np.empty(union_nnz, dtype=matrix_indices.dtype)
    merged_counts_data = np.empty(union_nnz, dtype=np.uint8)
    merged_maximum_data = np.empty(union_nnz, dtype=np.float64)
    merged_minimum_indices = np.empty(minimum_nnz, dtype=matrix.indices.dtype)
    merged_minimum_data = np.empty(minimum_nnz, dtype=np.float64)

    # Second pass fills the preallocated CSR arrays.  A missing annual link drops
    # from the minimum support, thereby encoding its thresholded zero permanently.
    for row in range(matrix_shape[0]):
        annual_start, annual_stop = matrix_indptr[row], matrix_indptr[row + 1]
        annual_active = matrix_data[annual_start:annual_stop] > cutoff
        annual_columns = matrix_indices[annual_start:annual_stop][annual_active]
        annual_values = matrix_data[annual_start:annual_stop][annual_active]
        count_start, count_stop = counts.indptr[row], counts.indptr[row + 1]
        count_columns = counts.indices[count_start:count_stop]
        count_values = counts.data[count_start:count_stop]
        maximum_start, maximum_stop = maximum.indptr[row], maximum.indptr[row + 1]
        maximum_columns = maximum.indices[maximum_start:maximum_stop]
        maximum_values = maximum.data[maximum_start:maximum_stop]
        minimum_start, minimum_stop = minimum.indptr[row], minimum.indptr[row + 1]
        minimum_columns = minimum.indices[minimum_start:minimum_stop]
        minimum_values = minimum.data[minimum_start:minimum_stop]

        output_start, output_stop = union_indptr[row], union_indptr[row + 1]
        columns = np.union1d(count_columns, annual_columns)
        merged_indices[output_start:output_stop] = columns
        annual_pos = np.searchsorted(annual_columns, columns)
        annual_present = (annual_pos < annual_columns.size) & (annual_columns[np.minimum(annual_pos, max(annual_columns.size - 1, 0))] == columns) if annual_columns.size else np.zeros(columns.size, dtype=bool)
        count_pos = np.searchsorted(count_columns, columns)
        count_present = (count_pos < count_columns.size) & (count_columns[np.minimum(count_pos, max(count_columns.size - 1, 0))] == columns) if count_columns.size else np.zeros(columns.size, dtype=bool)
        maximum_pos = np.searchsorted(maximum_columns, columns)
        maximum_present = (maximum_pos < maximum_columns.size) & (maximum_columns[np.minimum(maximum_pos, max(maximum_columns.size - 1, 0))] == columns) if maximum_columns.size else np.zeros(columns.size, dtype=bool)
        annual_output = np.zeros(columns.size, dtype=np.float64)
        if annual_columns.size:
            annual_output[annual_present] = annual_values[annual_pos[annual_present]]
        count_output = np.zeros(columns.size, dtype=np.uint8)
        if count_columns.size:
            count_output[count_present] = count_values[count_pos[count_present]]
        maximum_output = np.zeros(columns.size, dtype=np.float64)
        if maximum_columns.size:
            maximum_output[maximum_present] = maximum_values[maximum_pos[maximum_present]]
        merged_counts_data[output_start:output_stop] = annual_present.astype(np.uint8) + count_output
        merged_maximum_data[output_start:output_stop] = np.maximum(
            annual_output,
            maximum_output,
        )

        minimum_columns_out = np.intersect1d(minimum_columns, annual_columns, assume_unique=True)
        minimum_output_start, minimum_output_stop = minimum_indptr[row], minimum_indptr[row + 1]
        merged_minimum_indices[minimum_output_start:minimum_output_stop] = minimum_columns_out
        old_pos = np.searchsorted(minimum_columns, minimum_columns_out)
        new_pos = np.searchsorted(annual_columns, minimum_columns_out)
        merged_minimum_data[minimum_output_start:minimum_output_stop] = np.minimum(minimum_values[old_pos], annual_values[new_pos])

    return (
        csr_matrix((merged_counts_data, merged_indices.copy(), union_indptr.copy()), shape=matrix_shape),
        csr_matrix((merged_minimum_data, merged_minimum_indices, minimum_indptr), shape=matrix_shape),
        csr_matrix((merged_maximum_data, merged_indices, union_indptr), shape=matrix_shape),
    )


def complete_directions_with_fixed_final_demand(
    A: Any,
    prescribed_rows: np.ndarray,
    prescribed_directions: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    r"""Complete prescribed rows so ``[(I-A)D]_N = 0`` on complementary rows.

    The supplied rows ``S`` are preserved exactly. For ``N`` equal to their
    complement, each direction column solves
    ``(I-A_NN)D_N = A_NS D_S`` with the same restarted sparse LGMRES convention
    used by the productivity certificate. The returned full ``N_total x K``
    direction matrix is dense only in its small direction dimension; no Leontief
    inverse or dense coefficient matrix is formed.
    """

    matrix = _nonnegative_support_matrix(A)
    mask = np.asarray(prescribed_rows, dtype=bool).reshape(-1)
    directions = np.asarray(prescribed_directions, dtype=np.float64)
    matrix_shape = cast(tuple[int, int], matrix.shape)
    n = matrix_shape[0]
    if mask.shape != (n,) or not bool(mask.any()):
        raise ValueError("prescribed_rows must be a non-empty boolean mask aligned to A")
    if (
        directions.ndim != 2
        or directions.shape[0] != int(mask.sum())
        or directions.shape[1] == 0
        or not np.isfinite(directions).all()
    ):
        raise ValueError("prescribed_directions must be a finite |S| x K matrix")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("completion tolerance must be finite and positive")

    full = np.zeros((n, directions.shape[1]), dtype=np.float64)
    full[mask, :] = directions
    free = ~mask
    if not bool(free.any()):
        return full

    matrix_view = cast(Any, matrix)
    A_nn = cast(csr_matrix, csr_matrix(matrix_view[free][:, free], dtype=np.float64))
    A_ns = cast(csr_matrix, csr_matrix(matrix_view[free][:, mask], dtype=np.float64))
    rhs = np.asarray(A_ns.dot(directions), dtype=np.float64)
    linear_operator = cast(Any, LinearOperator)
    operator = linear_operator(
        A_nn.shape,
        matvec=lambda values: np.asarray(values, dtype=np.float64) - A_nn.dot(values),
        dtype=np.float64,
    )
    iterative_solve = cast(Any, lgmres)
    for column in range(directions.shape[1]):
        target = rhs[:, column]
        if not bool(np.any(target)):
            solution = np.zeros(target.shape, dtype=np.float64)
        else:
            solution_raw, info = iterative_solve(
                operator,
                target,
                x0=np.zeros_like(target),
                rtol=1.0e-11,
                atol=0.0,
                maxiter=500,
                inner_m=30,
                outer_k=3,
            )
            solution = np.asarray(solution_raw, dtype=np.float64).reshape(-1)
            if int(info) != 0:
                raise RuntimeError(
                    f"fixed-final-demand direction completion failed to converge; info={info}"
                )
        if solution.shape != target.shape or not np.isfinite(solution).all():
            raise RuntimeError("fixed-final-demand direction completion returned invalid values")
        full[free, column] = solution

    residual = np.asarray(full - matrix.dot(full), dtype=np.float64)[free, :]
    scale = max(1.0, float(np.max(np.abs(rhs))), float(np.max(np.abs(full))))
    relative_residual = float(np.max(np.abs(residual)) / scale) if residual.size else 0.0
    if relative_residual > tol:
        raise RuntimeError(
            "fixed-final-demand direction completion residual exceeds tolerance: "
            f"{relative_residual:.6g} > {tol:.6g}"
        )
    return full


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pair-active changes and observed magnitudes for threshold crossings."""

    start_left = int(start.indptr[row])
    start_right = int(start.indptr[row + 1])
    end_left = int(end.indptr[row])
    end_right = int(end.indptr[row + 1])
    start_columns = np.asarray(start.indices[start_left:start_right], dtype=np.intp)
    end_columns = np.asarray(end.indices[end_left:end_right], dtype=np.intp)
    empty = np.empty(0, dtype=np.float64)
    if start_columns.size == 0 and end_columns.size == 0:
        return empty, empty, empty

    # Merge one CSR row only: EXIO-scale pair diagnostics must not materialize a
    # full pair-active COO union or retain a floating panel-wide reference matrix.
    columns = np.union1d(start_columns, end_columns)
    start_values = np.zeros(columns.size, dtype=np.float64)
    end_values = np.zeros(columns.size, dtype=np.float64)
    if start_columns.size:
        start_values[np.searchsorted(columns, start_columns)] = start.data[start_left:start_right]
    if end_columns.size:
        end_values[np.searchsorted(columns, end_columns)] = end.data[end_left:end_right]

    active = np.maximum(start_values, end_values) > floor
    if not bool(active.any()):
        return empty, empty, empty
    start_active = start_values[active]
    end_active = end_values[active]
    appearing = (start_active <= floor) & (end_active > floor)
    disappearing = (start_active > floor) & (end_active <= floor)
    relative = np.abs(end_active - start_active) / np.maximum(start_active, floor)
    return relative, end_active[appearing], start_active[disappearing]


def adjacent_coefficient_change_diagnostics(
    A_start: Any,
    A_end: Any,
    *,
    floor: float,
    quantiles: Iterable[float],
) -> tuple[
    int,
    int,
    int,
    dict[float, float],
    dict[float, float],
    dict[float, float],
]:
    r"""Return relative-change and threshold-crossing coefficient quantiles.

    Besides ``|A1-A0|/max(A0,floor)``, this reports the actual positive end
    coefficient for appearing links and positive start coefficient for disappearing
    links. Those dimensionless magnitudes, rather than ``floor`` itself, are the
    empirical evidence relevant to a later dormant-link cap decision.
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

    # Count first so the only pair-wide scratch is three exactly sized float vectors.
    active_count = appearing_count = disappearing_count = 0
    for row in range(shape[0]):
        relative, appearing, disappearing = _pair_active_relative_row(
            start, end, row=row, floor=epsilon
        )
        active_count += int(relative.size)
        appearing_count += int(appearing.size)
        disappearing_count += int(disappearing.size)

    missing = {value: np.nan for value in requested}
    if active_count == 0:
        return 0, 0, 0, missing, missing.copy(), missing.copy()

    changes = np.empty(active_count, dtype=np.float64)
    appearing_values = np.empty(appearing_count, dtype=np.float64)
    disappearing_values = np.empty(disappearing_count, dtype=np.float64)
    offsets = [0, 0, 0]
    for row in range(shape[0]):
        values = _pair_active_relative_row(start, end, row=row, floor=epsilon)
        for slot, array in enumerate(values):
            next_offset = offsets[slot] + int(array.size)
            (changes, appearing_values, disappearing_values)[slot][offsets[slot]:next_offset] = array
            offsets[slot] = next_offset
    if offsets != [active_count, appearing_count, disappearing_count]:
        raise RuntimeError("adjacent coefficient diagnostic streaming count mismatch")
    if not all(np.isfinite(values).all() for values in (changes, appearing_values, disappearing_values)):
        raise RuntimeError("adjacent coefficient diagnostics contain non-finite values")

    def select(values: np.ndarray) -> dict[float, float]:
        if values.size == 0:
            return {value: np.nan for value in requested}
        selected = np.asarray(
            np.quantile(
                values, np.asarray(requested), method="linear", overwrite_input=True
            ),
            dtype=np.float64,
        ).reshape(-1)
        return dict(zip(requested, map(float, selected), strict=True))

    return (
        active_count,
        appearing_count,
        disappearing_count,
        select(changes),
        select(appearing_values),
        select(disappearing_values),
    )


def adjacent_coefficient_change_quantiles(
    A_start: Any,
    A_end: Any,
    *,
    floor: float,
    quantiles: Iterable[float],
) -> tuple[int, int, int, dict[float, float]]:
    """Return the established four-value pair-active relative-change API."""

    count, appearing, disappearing, selected, _, _ = adjacent_coefficient_change_diagnostics(
        A_start, A_end, floor=floor, quantiles=quantiles
    )
    return count, appearing, disappearing, selected


def _canonical_coefficient_allowed_mask(allowed_mask: Any, shape: tuple[int, int]) -> csr_matrix:
    """Return the canonical non-negative sparse support used by coefficient-bound APIs.

    Stored zeros are not allowed to turn into adjustable links merely because a
    caller supplied an already-canonical CSR matrix.  Keeping this normalization
    in one private helper ensures the compact fixed-RHS and LP-facing APIs use
    exactly the same eligible edge set.
    """

    mask = cast(csr_matrix, csr_matrix(allowed_mask, dtype=np.int8, copy=False))
    if mask.shape != shape:
        raise ValueError("coefficient allowed mask must share A's shape")
    mask_data = np.asarray(mask.data)
    if mask_data.size and bool((mask_data < 0).any()):
        raise ValueError("allowed coefficient mask must be non-negative")
    if (not mask.has_canonical_format) or bool(mask_data.size and (mask_data == 0).any()):
        mask = mask.copy()
        mask.sum_duplicates()
        mask.eliminate_zeros()
        mask.sort_indices()
    return mask


def _sparse_row_values(source: csr_matrix, *, row: int, columns: np.ndarray) -> np.ndarray:
    """Read one sorted CSR row on ``columns`` without a dense row allocation."""

    left = int(source.indptr[row])
    right = int(source.indptr[row + 1])
    source_columns = np.asarray(source.indices[left:right], dtype=np.intp)
    result = np.zeros(columns.size, dtype=np.float64)
    if source_columns.size == 0 or columns.size == 0:
        return result
    positions = np.searchsorted(source_columns, columns)
    in_bounds = positions < source_columns.size
    matches = np.zeros(columns.size, dtype=bool)
    if bool(in_bounds.any()):
        matches[in_bounds] = source_columns[positions[in_bounds]] == columns[in_bounds]
    if bool(matches.any()):
        row_values = np.asarray(source.data[left:right], dtype=np.float64)
        result[matches] = row_values[positions[matches]]
    return result


def _certificate_safe_row_headroom(
    *,
    matrix: csr_matrix,
    minimum: csr_matrix,
    maximum: csr_matrix,
    row: int,
    columns: np.ndarray,
    certificate_vector: np.ndarray,
    baseline_slack: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return one row's lower/upper change interval after certificate scaling.

    Historical lower levels bound coefficient reductions.  Positive historical
    headroom is scaled by one row factor so the simultaneous positive corner obeys
    the fixed sufficient productivity certificate.  Both public coefficient-bound
    APIs call this helper so they cannot drift on level lookup or certificate use.
    """

    baseline = _sparse_row_values(matrix, row=row, columns=columns)
    lower_level = _sparse_row_values(minimum, row=row, columns=columns)
    upper_level = _sparse_row_values(maximum, row=row, columns=columns)
    raw_lower = lower_level - baseline
    raw_upper = upper_level - baseline
    tol = float(tolerance)
    if float(raw_lower.max(initial=0.0)) > tol or float(raw_upper.min(initial=0.0)) < -tol:
        raise ValueError("baseline coefficient lies outside supplied historical level bounds")
    raw_lower = np.minimum(raw_lower, 0.0)
    raw_upper = np.maximum(raw_upper, 0.0)
    positive_use = float(np.dot(raw_upper, certificate_vector[columns]))
    scale = 1.0
    if positive_use > tol:
        scale = min(1.0, max(float(baseline_slack[row]), 0.0) / positive_use)
    safe_upper = scale * raw_upper
    margin = float(baseline_slack[row]) - scale * positive_use
    return raw_lower, safe_upper, scale, margin


def _coefficient_headroom_inputs(
    A: Any,
    allowed_mask: Any,
    lower_levels: Any,
    upper_levels: Any,
    *,
    certificate_vector: np.ndarray,
    productivity_epsilon: float,
    tolerance: float,
) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """Validate and align the shared sparse inputs for certificate-safe headroom."""

    matrix = _nonnegative_support_matrix(A)
    matrix_shape = cast(tuple[int, int], matrix.shape)
    n = matrix_shape[0]
    epsilon = float(productivity_epsilon)
    tol = float(tolerance)
    if not 0.0 < epsilon < 1.0 or tol <= 0.0:
        raise ValueError("productivity_epsilon must lie in (0, 1) and tolerance be positive")
    mask = _canonical_coefficient_allowed_mask(allowed_mask, matrix_shape)
    minimum = _nonnegative_support_matrix(lower_levels)
    maximum = _nonnegative_support_matrix(upper_levels)
    if minimum.shape != matrix_shape or maximum.shape != matrix_shape:
        raise ValueError("coefficient level bounds must share A's shape")
    vector = np.asarray(certificate_vector, dtype=np.float64).reshape(-1)
    if vector.shape != (n,) or not np.isfinite(vector).all() or bool((vector <= 0.0).any()):
        raise ValueError("certificate_vector must be finite, strictly positive, and aligned")
    baseline_slack = np.asarray((1.0 - epsilon) * vector - matrix.dot(vector), dtype=np.float64)
    if float(baseline_slack.min()) < -tol:
        raise ValueError("baseline matrix violates the supplied productivity certificate")
    return matrix, mask, minimum, maximum, vector, baseline_slack


def certificate_safe_coefficient_capacity_products(
    A: Any,
    allowed_mask: Any,
    lower_levels: Any,
    upper_levels: Any,
    right_hand_sides: np.ndarray,
    *,
    certificate_vector: np.ndarray,
    productivity_epsilon: float,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Apply a coefficient-level box and certificate to compact RHS vectors.

    ``allowed_mask`` declares which coefficient links may change. On those links
    the raw change interval is ``lower_levels-A <= Delta A <= upper_levels-A``.
    Negative headroom is limited only by the supplied lower level; positive
    headroom is scaled row-wise so the simultaneous upper corner satisfies the
    fixed sufficient certificate ``A'v <= (1-epsilon)v``. Only products of these
    bounds with the supplied small RHS matrix are returned, avoiding EXIO-scale
    stored ``Delta A`` bound matrices.
    """

    tol = float(tolerance)
    matrix, mask, minimum, maximum, vector, baseline_slack = _coefficient_headroom_inputs(
        A,
        allowed_mask,
        lower_levels,
        upper_levels,
        certificate_vector=certificate_vector,
        productivity_epsilon=productivity_epsilon,
        tolerance=tol,
    )
    n = cast(tuple[int, int], matrix.shape)[0]
    rhs = np.asarray(right_hand_sides, dtype=np.float64)
    if rhs.ndim != 2 or rhs.shape[0] != n or not np.isfinite(rhs).all():
        raise ValueError("right_hand_sides must be a finite N x K matrix aligned to A")

    lower_products = np.zeros((n, rhs.shape[1]), dtype=np.float64)
    upper_products = np.zeros_like(lower_products)
    row_scale = np.ones(n, dtype=np.float64)
    guaranteed_margin = baseline_slack.copy()
    for row in range(n):
        left = int(mask.indptr[row])
        right = int(mask.indptr[row + 1])
        columns = np.asarray(mask.indices[left:right], dtype=np.intp)
        if columns.size == 0:
            continue
        raw_lower, safe_upper, scale, margin = _certificate_safe_row_headroom(
            matrix=matrix,
            minimum=minimum,
            maximum=maximum,
            row=row,
            columns=columns,
            certificate_vector=vector,
            baseline_slack=baseline_slack,
            tolerance=tol,
        )
        selected_rhs = rhs[columns, :]
        lower_products[row, :] = raw_lower @ selected_rhs
        upper_products[row, :] = safe_upper @ selected_rhs
        row_scale[row] = scale
        guaranteed_margin[row] = margin

    if float(guaranteed_margin.min()) < -tol:
        raise RuntimeError("certificate-safe coefficient capacities have negative guaranteed margin")
    return lower_products, upper_products, row_scale, guaranteed_margin


def certificate_safe_coefficient_headroom(
    A: Any,
    allowed_mask: Any,
    lower_levels: Any,
    upper_levels: Any,
    *,
    certificate_vector: np.ndarray,
    productivity_epsilon: float,
    tolerance: float = 1.0e-12,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    r"""Return sparse certificate-safe coefficient-change headroom matrices.

    This LP-facing form has exactly the same allowed-edge, historical-level, and
    productivity-certificate semantics as
    ``certificate_safe_coefficient_capacity_products``.  It is used only when the
    output vector is itself an LP variable.  A count-and-fill two-pass CSR construction allocates exactly the retained union
    support, so EXIO-scale Python object lists, overallocated slice backing arrays,
    and conversion copies are never created.
    """

    tol = float(tolerance)
    matrix, mask, minimum, maximum, vector, baseline_slack = _coefficient_headroom_inputs(
        A,
        allowed_mask,
        lower_levels,
        upper_levels,
        certificate_vector=certificate_vector,
        productivity_epsilon=productivity_epsilon,
        tolerance=tol,
    )
    n = cast(tuple[int, int], matrix.shape)[0]
    row_counts = np.zeros(n, dtype=np.int64)
    row_scale = np.ones(n, dtype=np.float64)
    guaranteed_margin = baseline_slack.copy()

    # First pass determines the exact union support needed by lower/upper
    # headroom.  This avoids both Python object lists and an overallocated
    # ``mask.nnz`` buffer being retained through NumPy slice views.
    for row in range(n):
        left = int(mask.indptr[row])
        right = int(mask.indptr[row + 1])
        columns = np.asarray(mask.indices[left:right], dtype=np.intp)
        if columns.size:
            raw_lower, safe_upper, scale, margin = _certificate_safe_row_headroom(
                matrix=matrix,
                minimum=minimum,
                maximum=maximum,
                row=row,
                columns=columns,
                certificate_vector=vector,
                baseline_slack=baseline_slack,
                tolerance=tol,
            )
            row_counts[row] = int(np.count_nonzero((raw_lower != 0.0) | (safe_upper != 0.0)))
            row_scale[row] = scale
            guaranteed_margin[row] = margin

    indptr = np.empty(n + 1, dtype=mask.indptr.dtype)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    nnz = int(indptr[-1])
    lower_data = np.empty(nnz, dtype=np.float64)
    upper_data = np.empty(nnz, dtype=np.float64)
    indices = np.empty(nnz, dtype=mask.indices.dtype)

    # Second pass fills exactly sized CSR arrays.  Recomputing one sparse row is
    # cheaper than retaining EXIO-scale temporary headroom arrays between passes.
    for row in range(n):
        output_left = int(indptr[row])
        output_right = int(indptr[row + 1])
        if output_left == output_right:
            continue
        left = int(mask.indptr[row])
        right = int(mask.indptr[row + 1])
        columns = np.asarray(mask.indices[left:right], dtype=np.intp)
        raw_lower, safe_upper, _scale, _margin = _certificate_safe_row_headroom(
            matrix=matrix,
            minimum=minimum,
            maximum=maximum,
            row=row,
            columns=columns,
            certificate_vector=vector,
            baseline_slack=baseline_slack,
            tolerance=tol,
        )
        keep = (raw_lower != 0.0) | (safe_upper != 0.0)
        if int(np.count_nonzero(keep)) != output_right - output_left:
            raise RuntimeError("certificate-safe headroom two-pass support count changed")
        indices[output_left:output_right] = columns[keep]
        lower_data[output_left:output_right] = raw_lower[keep]
        upper_data[output_left:output_right] = safe_upper[keep]

    if float(guaranteed_margin.min()) < -tol:
        raise RuntimeError("certificate-safe coefficient headroom has negative guaranteed margin")
    lower = csr_matrix((lower_data, indices.copy(), indptr.copy()), shape=matrix.shape)
    upper = csr_matrix((upper_data, indices, indptr), shape=matrix.shape)
    lower.eliminate_zeros()
    lower.sort_indices()
    upper.eliminate_zeros()
    upper.sort_indices()
    return lower, upper, row_scale, guaranteed_margin

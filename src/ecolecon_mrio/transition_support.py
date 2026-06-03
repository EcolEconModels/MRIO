from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse


def normalize_productive_certificate_vector(
    q: pd.Series | np.ndarray | Iterable[float],
    index: pd.Index,
    *,
    floor: float = 1.0e-12,
) -> pd.Series:
    """Return a strictly positive certificate vector normalized to unit sum."""
    q_ser = pd.to_numeric(pd.Series(q).reindex(index), errors="coerce").fillna(0.0).astype(float)
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
    a = pd.to_numeric(pd.Series(a_row), errors="coerce").fillna(0.0).to_numpy(dtype=float)
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
    q_ser = pd.to_numeric(pd.Series(productive_q).reindex(a_row.index), errors="coerce").fillna(0.0).astype(float)
    if not np.isfinite(q_ser.to_numpy(dtype=float)).all():
        return None
    lhs_base = float(
        np.dot(
            pd.to_numeric(a_row.reindex(q_ser.index), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            q_ser.to_numpy(dtype=float),
        )
    )
    rhs_total = float(productive_rho) * float(pd.to_numeric(pd.Series(productive_q).get(row_label, np.nan), errors="coerce"))
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
    lhs = pd.to_numeric(A_df.reindex(index=q_row.index, columns=q_ser.index).fillna(0.0).dot(q_ser), errors="coerce").fillna(0.0).astype(float)
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
    e_vec = pd.to_numeric(e_ser, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    d_vec = pd.to_numeric(d_ser, errors="coerce").fillna(0.0).to_numpy(dtype=float)
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
    x_ser = pd.to_numeric(pd.Series(x_obs), errors="coerce").fillna(0.0).astype(float)
    d_ser = pd.to_numeric(pd.Series(direction).reindex(x_ser.index), errors="coerce").fillna(0.0).astype(float)
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
        A_sparse = csr_matrix(A, dtype=float)
        idx = pd.RangeIndex(A_sparse.shape[0])
        if hasattr(A, "index"):
            idx = pd.Index(getattr(A, "index"), dtype=object)
        y_ser = pd.to_numeric(pd.Series(y_obs).reindex(idx), errors="coerce").fillna(0.0).astype(float)
        d_ser = pd.to_numeric(pd.Series(direction).reindex(idx), errors="coerce").fillna(0.0).astype(float)
        d_values = d_ser.to_numpy(dtype=float)
        delta_y_unit = pd.Series(d_values - A_sparse.dot(d_values), index=idx, dtype=float)
    elif isinstance(A, pd.DataFrame) and all(isinstance(dtype, pd.SparseDtype) for dtype in A.dtypes):
        A_sparse = A.sparse.to_coo().tocsr().astype(float)
        idx = pd.Index(A.index, dtype=object)
        y_ser = pd.to_numeric(pd.Series(y_obs).reindex(idx), errors="coerce").fillna(0.0).astype(float)
        d_ser = pd.to_numeric(pd.Series(direction).reindex(idx), errors="coerce").fillna(0.0).astype(float)
        d_values = d_ser.to_numpy(dtype=float)
        delta_y_unit = pd.Series(d_values - A_sparse.dot(d_values), index=idx, dtype=float)
    else:
        A_df = pd.DataFrame(A).copy()
        idx = pd.Index(A_df.index, dtype=object)
        A_df = A_df.reindex(index=idx, columns=idx).fillna(0.0).astype(float)
        y_ser = pd.to_numeric(pd.Series(y_obs).reindex(idx), errors="coerce").fillna(0.0).astype(float)
        d_ser = pd.to_numeric(pd.Series(direction).reindex(idx), errors="coerce").fillna(0.0).astype(float)
        I_minus_A = pd.DataFrame(np.eye(len(idx)), index=idx, columns=idx) - A_df
        delta_y_unit = pd.to_numeric(I_minus_A.dot(d_ser), errors="coerce").fillna(0.0).astype(float)
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

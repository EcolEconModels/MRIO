from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs


@dataclass
class RowBoundsInterval:
    row_index: int
    rhs: float
    lower_sum: float
    upper_sum: float
    feasible: bool


@dataclass
class RowSolveDiagnostics:
    row_index: int
    success: bool
    method: str
    status: int | None
    message: str
    objective: float
    residual_abs: float
    stationarity_norm: float
    max_bound_violation: float


@dataclass
class MatrixInverseDiagnostics:
    success: bool
    status: str
    reason: str
    row_success_share: float
    infeasible_bounds_share: float
    residual_abs_max: float
    residual_abs_median: float
    failure_reason_counts: dict[str, int]


@dataclass
class SpectralModeDiagnostics:
    lambda1: float
    lambda2_abs: float
    spectral_gap: float
    cosine_target: float


def _normalize_target_nonnegative(target: pd.Series | np.ndarray | Iterable[float], index: pd.Index | None = None) -> tuple[np.ndarray | None, str]:
    s = pd.Series(target)
    if index is not None:
        s = s.reindex(index)
    d = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)
    denom = float(d.sum())
    if not np.isfinite(denom) or denom <= 0:
        return None, "nonpositive_target_support"
    return (d / denom).to_numpy(dtype=float), "ok"


def compute_row_interval_bounds(
    *,
    a_row: np.ndarray,
    d: np.ndarray,
    lambda_target: float,
    d_i: float,
    relative_change_cap: float | None,
    relative_change_cap_floor: float = 1e-4,
    nonnegativity: bool = True,
) -> tuple[RowBoundsInterval, np.ndarray, np.ndarray]:
    n = len(a_row)
    lower = np.full(n, -np.inf, dtype=float)
    upper = np.full(n, np.inf, dtype=float)
    if nonnegativity:
        lower = np.maximum(lower, -a_row)
    if relative_change_cap is not None and relative_change_cap >= 0:
        cap = float(relative_change_cap) * np.maximum(a_row, float(relative_change_cap_floor))
        lower = np.maximum(lower, -cap)
        upper = np.minimum(upper, cap)

    rhs = float(lambda_target * d_i - np.dot(a_row, d))
    lower_sum = float(np.dot(d, lower))
    upper_pos = upper[d > 0]
    if upper_pos.size and np.isinf(upper_pos).any():
        upper_sum = np.inf
    else:
        upper_sum = float(np.dot(d, upper))

    feasible = bool((rhs >= lower_sum - 1e-12) and (rhs <= upper_sum + 1e-12))
    return (
        RowBoundsInterval(
            row_index=-1,
            rhs=rhs,
            lower_sum=lower_sum,
            upper_sum=upper_sum,
            feasible=feasible,
        ),
        lower,
        upper,
    )


def _closed_form_row_solution(
    *,
    d: np.ndarray,
    rhs: float,
    lower: np.ndarray,
    upper: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, RowSolveDiagnostics]:
    n = len(d)
    x = np.zeros(n, dtype=float)
    d_pos = d.copy()
    d_pos[np.abs(d_pos) < 1e-15] = 0.0
    zero_mask = d_pos <= 0
    if zero_mask.any():
        x[zero_mask] = np.clip(0.0, lower[zero_mask], upper[zero_mask])
    pos_mask = d_pos > 0

    min_sum = float(np.dot(d_pos, lower))
    upper_pos = upper[pos_mask]
    if upper_pos.size and np.isinf(upper_pos).any():
        max_sum = np.inf
    else:
        max_sum = float(np.dot(d_pos, upper))

    if rhs < min_sum - 1e-10 or rhs > max_sum + 1e-10:
        if pos_mask.any():
            if rhs < min_sum:
                x[pos_mask] = lower[pos_mask]
            else:
                x[pos_mask] = upper[pos_mask]
        eq_res = float(abs(np.dot(d, x) - rhs))
        return x, RowSolveDiagnostics(
            row_index=-1,
            success=False,
            method="closed_form_infeasible_bounds",
            status=2,
            message=f"rhs_outside_bounds_interval[{min_sum:.3e},{max_sum:.3e}]",
            objective=0.5 * float(np.dot(x, x)),
            residual_abs=eq_res,
            stationarity_norm=float(np.linalg.norm(x)),
            max_bound_violation=0.0,
        )

    if pos_mask.any():
        ld = lower[pos_mask] / d_pos[pos_mask]
        lam_lo = float(np.min(ld)) - 1.0
        d2 = float(np.dot(d_pos[pos_mask], d_pos[pos_mask]))
        lam_hi = max(lam_lo + 1.0, rhs / d2 if d2 > 0 else lam_lo + 1.0)

        def _sum_at(lam: float) -> float:
            x_mid = np.clip(lam * d_pos[pos_mask], lower[pos_mask], upper[pos_mask])
            return float(np.dot(d_pos[pos_mask], x_mid) + np.dot(d_pos[zero_mask], x[zero_mask]))

        s_lo = _sum_at(lam_lo)
        s_hi = _sum_at(lam_hi)
        g = 0
        while s_hi < rhs and g < 100:
            lam_hi = (2.0 * lam_hi) + 1.0
            s_hi = _sum_at(lam_hi)
            g += 1
        g = 0
        while s_lo > rhs and g < 100:
            lam_lo = (2.0 * lam_lo) - 1.0
            s_lo = _sum_at(lam_lo)
            g += 1
        for _ in range(80):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            s_mid = _sum_at(lam_mid)
            if s_mid < rhs:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        lam = 0.5 * (lam_lo + lam_hi)
        x[pos_mask] = np.clip(lam * d_pos[pos_mask], lower[pos_mask], upper[pos_mask])

    eq_res = float(abs(np.dot(d, x) - rhs))
    return x, RowSolveDiagnostics(
        row_index=-1,
        success=bool(eq_res <= max(1e-9, tol * 10)),
        method="closed_form_bisection",
        status=0,
        message="ok",
        objective=0.5 * float(np.dot(x, x)),
        residual_abs=eq_res,
        stationarity_norm=float(np.linalg.norm(x)),
        max_bound_violation=0.0,
    )


def row_min_norm_update(
    *,
    a_row: np.ndarray,
    d: np.ndarray,
    lambda_target: float,
    d_i: float,
    relative_change_cap: float | None,
    relative_change_cap_floor: float = 1e-4,
    nonnegativity: bool = True,
    row_sum_cap: float | None = None,
    method_primary: str = "trust-constr",
    method_fallback: str = "SLSQP",
    tol: float = 1e-9,
    maxiter: int = 1000,
) -> tuple[np.ndarray, RowSolveDiagnostics, RowBoundsInterval]:
    interval, lower, upper = compute_row_interval_bounds(
        a_row=a_row,
        d=d,
        lambda_target=lambda_target,
        d_i=d_i,
        relative_change_cap=relative_change_cap,
        relative_change_cap_floor=relative_change_cap_floor,
        nonnegativity=nonnegativity,
    )
    rhs = interval.rhs

    if row_sum_cap is None and np.all(d >= -1e-14):
        x, diag = _closed_form_row_solution(d=d, rhs=rhs, lower=lower, upper=upper, tol=tol)
        return x, diag, interval

    n = len(a_row)

    def objective(x: np.ndarray) -> float:
        return 0.5 * float(np.dot(x, x))

    def grad(x: np.ndarray) -> np.ndarray:
        return x

    def hess(_x: np.ndarray) -> np.ndarray:
        return np.eye(n)

    constraints: list[Any] = [LinearConstraint(d.reshape(1, -1), np.array([rhs]), np.array([rhs]))]
    if row_sum_cap is not None and np.isfinite(row_sum_cap):
        rhs_sum = float(row_sum_cap - np.sum(a_row))
        constraints.append(LinearConstraint(np.ones((1, n), dtype=float), -np.inf * np.ones(1), np.array([rhs_sum])))

    bounds = Bounds(lower, upper)
    d_norm_sq = float(np.dot(d, d))
    x0 = d * (rhs / d_norm_sq) if d_norm_sq > 0 else np.zeros(n, dtype=float)
    x0 = np.clip(x0, lower, upper)

    def _run(method: str, *, maxiter_eff: int, tol_eff: float):
        kw = {
            "fun": objective,
            "x0": x0,
            "jac": grad,
            "bounds": bounds,
            "constraints": constraints,
            "method": method,
            "options": {"maxiter": int(maxiter_eff)},
        }
        if method == "trust-constr":
            kw["hess"] = hess
            kw["options"].update({"gtol": float(tol_eff), "xtol": float(tol_eff), "barrier_tol": float(tol_eff)})
        else:
            kw["options"].update({"ftol": float(tol_eff)})
        return minimize(**kw)

    effective_primary = method_primary
    if n >= 40 and method_primary == "trust-constr":
        effective_primary = "SLSQP"
    maxiter_eff = int(maxiter)
    tol_eff = float(tol)
    if n >= 40 and effective_primary == "SLSQP":
        maxiter_eff = min(maxiter_eff, 150)
        tol_eff = max(tol_eff, 1e-7)

    res = _run(effective_primary, maxiter_eff=maxiter_eff, tol_eff=tol_eff)
    method_used = effective_primary
    if (not bool(res.success)) and method_fallback and method_fallback != method_used:
        res = _run(method_fallback, maxiter_eff=maxiter_eff, tol_eff=tol_eff)
        method_used = method_fallback

    x = np.array(res.x, dtype=float) if res.x is not None else np.zeros(n, dtype=float)
    eq_res = float(abs(np.dot(d, x) - rhs))
    stationarity = float(np.linalg.norm(x))
    lb_v = np.maximum(lower - x, 0.0)
    ub_v = np.maximum(x - upper, 0.0)
    max_bv = float(max(lb_v.max(initial=0.0), ub_v.max(initial=0.0)))

    diag = RowSolveDiagnostics(
        row_index=-1,
        success=bool(res.success),
        method=method_used,
        status=int(getattr(res, "status", -1)) if getattr(res, "status", None) is not None else None,
        message=str(getattr(res, "message", "")),
        objective=float(res.fun) if getattr(res, "fun", None) is not None else float("nan"),
        residual_abs=eq_res,
        stationarity_norm=stationarity,
        max_bound_violation=max_bv,
    )
    return x, diag, interval


def matrix_min_norm_inverse_step(
    *,
    A: pd.DataFrame,
    target_d: pd.Series,
    lambda_target: float,
    solver_cfg: Mapping[str, Any] | None = None,
    constraints_cfg: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[RowSolveDiagnostics], list[RowBoundsInterval], MatrixInverseDiagnostics]:
    solver_cfg = dict(solver_cfg or {})
    constraints_cfg = dict(constraints_cfg or {})

    A_df = pd.DataFrame(A).copy()
    n = int(len(A_df.index))
    if n == 0 or A_df.shape[0] != A_df.shape[1]:
        empty = pd.DataFrame(index=A_df.index, columns=A_df.columns, dtype=float).fillna(0.0)
        diag = MatrixInverseDiagnostics(
            success=False,
            status="invalid_input",
            reason="A must be non-empty square",
            row_success_share=0.0,
            infeasible_bounds_share=0.0,
            residual_abs_max=np.nan,
            residual_abs_median=np.nan,
            failure_reason_counts={"invalid_input": 1},
        )
        return A_df, empty, [], [], diag

    A_df = A_df.reindex(index=A_df.index, columns=A_df.index).fillna(0.0)
    d, d_reason = _normalize_target_nonnegative(target_d, index=A_df.index)
    if d is None:
        empty = pd.DataFrame(np.zeros_like(A_df.to_numpy(dtype=float)), index=A_df.index, columns=A_df.columns)
        diag = MatrixInverseDiagnostics(
            success=False,
            status="invalid_target",
            reason=d_reason,
            row_success_share=0.0,
            infeasible_bounds_share=0.0,
            residual_abs_max=np.nan,
            residual_abs_median=np.nan,
            failure_reason_counts={"invalid_target": 1},
        )
        return A_df, empty, [], [], diag

    method_primary = str(solver_cfg.get("method_primary", "trust-constr"))
    method_fallback = str(solver_cfg.get("method_fallback", "SLSQP"))
    maxiter = int(solver_cfg.get("maxiter", 1000))
    tol = float(solver_cfg.get("tol", 1e-9))

    rel_cap = constraints_cfg.get("relative_change_cap", None)
    rel_cap = None if rel_cap is None else float(rel_cap)
    rel_cap_floor = float(constraints_cfg.get("relative_change_cap_floor", 1e-4))
    nonnegativity = bool(constraints_cfg.get("nonnegativity", True))
    row_sum_guard = bool(constraints_cfg.get("row_sum_guard", False))
    row_sum_relax = float(constraints_cfg.get("row_sum_relax", 0.0))

    A_val = A_df.to_numpy(dtype=float)
    delta = np.zeros_like(A_val)
    row_diags: list[RowSolveDiagnostics] = []
    row_intervals: list[RowBoundsInterval] = []

    for i in range(n):
        a_row = A_val[i, :].copy()
        row_cap = float(np.sum(a_row) * (1.0 + row_sum_relax)) if row_sum_guard else None
        x_opt, row_diag, interval = row_min_norm_update(
            a_row=a_row,
            d=d,
            lambda_target=float(lambda_target),
            d_i=float(d[i]),
            relative_change_cap=rel_cap,
            relative_change_cap_floor=rel_cap_floor,
            nonnegativity=nonnegativity,
            row_sum_cap=row_cap,
            method_primary=method_primary,
            method_fallback=method_fallback,
            tol=tol,
            maxiter=maxiter,
        )
        row_diag.row_index = i
        interval.row_index = i
        delta[i, :] = x_opt
        row_diags.append(row_diag)
        row_intervals.append(interval)

    A_new_val = A_val + delta
    if nonnegativity:
        A_new_val = np.maximum(A_new_val, 0.0)

    A_new = pd.DataFrame(A_new_val, index=A_df.index, columns=A_df.columns)
    delta_A = pd.DataFrame(A_new_val - A_val, index=A_df.index, columns=A_df.columns)

    success_share = float(np.mean([1.0 if r.success else 0.0 for r in row_diags])) if row_diags else 0.0
    infeasible_share = float(np.mean([1.0 if (not i.feasible) else 0.0 for i in row_intervals])) if row_intervals else 0.0
    failure_counts: dict[str, int] = {}
    for r in row_diags:
        if r.success:
            continue
        key = r.method if r.method else "unknown"
        failure_counts[key] = failure_counts.get(key, 0) + 1
    residuals = [float(r.residual_abs) for r in row_diags]
    status = "ok" if success_share >= 0.99 else "partial_failure"
    reason = "ok" if status == "ok" else f"row_success_share={success_share:.3f}"

    diag = MatrixInverseDiagnostics(
        success=bool(status == "ok"),
        status=status,
        reason=reason,
        row_success_share=success_share,
        infeasible_bounds_share=infeasible_share,
        residual_abs_max=float(np.max(residuals)) if residuals else np.nan,
        residual_abs_median=float(np.median(residuals)) if residuals else np.nan,
        failure_reason_counts=failure_counts,
    )

    return A_new, delta_A, row_diags, row_intervals, diag


def dominant_mode_diagnostics(A: pd.DataFrame, target_d: pd.Series) -> SpectralModeDiagnostics:
    A_df = pd.DataFrame(A)
    d_arr, _ = _normalize_target_nonnegative(target_d, index=A_df.index)
    if d_arr is None:
        return SpectralModeDiagnostics(np.nan, np.nan, np.nan, np.nan)

    A_val = A_df.to_numpy(dtype=float)
    n = int(A_val.shape[0])
    vals: np.ndarray
    vecs: np.ndarray
    try:
        if n >= 200:
            vals, vecs = eigs(csr_matrix(A_val), k=2 if n >= 3 else 1, which="LR", tol=1e-6, maxiter=4000)
        else:
            vals, vecs = np.linalg.eig(A_val)
    except ArpackNoConvergence as exc:
        vals = np.asarray(exc.eigenvalues) if exc.eigenvalues is not None else np.array([])
        vecs = np.asarray(exc.eigenvectors) if exc.eigenvectors is not None else np.empty((n, 0))
        if vals.size == 0 or vecs.size == 0:
            vals, vecs = np.linalg.eig(A_val)
    except Exception:
        vals, vecs = np.linalg.eig(A_val)

    mags = np.abs(vals)
    if mags.size == 0 or vecs.size == 0:
        return SpectralModeDiagnostics(np.nan, np.nan, np.nan, np.nan)

    i1 = int(np.argmax(mags))
    lambda1 = float(np.real(vals[i1]))
    lambda2_abs = float(np.partition(mags, -2)[-2]) if mags.size >= 2 else np.nan
    gap = float(mags[i1] - lambda2_abs) if np.isfinite(lambda2_abs) else np.nan

    q = np.abs(np.real(vecs[:, i1]))
    q_sum = float(np.sum(q))
    if q_sum > 0:
        q = q / q_sum
    denom = float(np.linalg.norm(q) * np.linalg.norm(d_arr))
    cosine = float(np.dot(q, d_arr) / denom) if denom > 0 else np.nan

    return SpectralModeDiagnostics(lambda1=lambda1, lambda2_abs=lambda2_abs, spectral_gap=gap, cosine_target=cosine)


def calibrate_relative_change_caps(
    A_by_year: Mapping[int, pd.DataFrame],
    *,
    floor: float = 1e-8,
    quantiles: Iterable[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> dict[str, Any]:
    years = sorted(int(y) for y in A_by_year.keys())
    if len(years) < 2:
        return {
            "n_pairs": 0,
            "n_obs": 0,
            "quantiles": {f"P{int(round(q * 100))}": np.nan for q in quantiles},
            "values": np.array([], dtype=float),
        }

    vals: list[np.ndarray] = []
    for y0, y1 in zip(years[:-1], years[1:]):
        A0 = pd.DataFrame(A_by_year[y0])
        A1 = pd.DataFrame(A_by_year[y1]).reindex(index=A0.index, columns=A0.columns).fillna(0.0)
        a0 = A0.to_numpy(dtype=float)
        a1 = A1.to_numpy(dtype=float)
        denom = np.maximum(np.abs(a0), float(floor))
        rel = np.abs(a1 - a0) / denom
        vals.append(rel.reshape(-1))

    if not vals:
        arr = np.array([], dtype=float)
    else:
        arr = np.concatenate(vals)
        arr = arr[np.isfinite(arr)]

    q_map: dict[str, float] = {}
    for q in quantiles:
        label = f"P{int(round(float(q) * 100))}"
        q_map[label] = float(np.quantile(arr, float(q))) if arr.size else np.nan

    return {
        "n_pairs": int(len(years) - 1),
        "n_obs": int(arr.size),
        "quantiles": q_map,
        "values": arr,
    }

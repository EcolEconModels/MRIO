"""Low-level diagnostics to support the viability-margins analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class SafeJustResult:
    x_opt: pd.Series
    delta_plus: pd.Series
    delta_minus: pd.Series
    objective: float
    inequality_duals: np.ndarray | None
    equality_duals: np.ndarray | None
    eco_violation: pd.Series | None = None
    soc_violation: pd.Series | None = None
    status: str | None = None
    message: str | None = None


def calc_spectral_radius(matrix: pd.DataFrame) -> float:
    eigvals = linalg.eigvals(matrix.values)
    return float(np.max(np.abs(eigvals)))


def calc_tech_slack(matrix: pd.DataFrame) -> tuple[float, float]:
    rho = calc_spectral_radius(matrix)
    return rho, max(0.0, 1.0 - rho)


def _orient_nonnegative(vec: np.ndarray, tol: float) -> tuple[np.ndarray, bool]:
    out = np.real(np.asarray(vec)).astype(float, copy=False)
    if np.all(out >= -tol):
        out = np.clip(out, 0.0, None)
        return out, False
    if np.all(out <= tol):
        out = np.clip(-out, 0.0, None)
        return out, False
    return np.abs(out), True


def _dominant_block_indices(values: np.ndarray, tol: float) -> tuple[np.ndarray, dict[str, Any]]:
    adjacency = (values > tol).astype(int)
    n_comp, labels = connected_components(csr_matrix(adjacency), directed=True, connection="strong")
    meta: dict[str, Any] = {
        "reducible": bool(n_comp > 1),
        "n_strong_components": int(n_comp),
        "dominant_block_size": int(values.shape[0]),
        "dominant_block_id": 0,
    }
    if n_comp <= 1:
        return np.arange(values.shape[0]), meta

    comp_ids = np.unique(labels)
    comp_rhos: list[tuple[int, float, int]] = []
    for comp in comp_ids:
        idx = np.where(labels == comp)[0]
        block = values[np.ix_(idx, idx)]
        rho = float(np.max(np.abs(linalg.eigvals(block))))
        comp_rhos.append((int(comp), rho, int(len(idx))))

    max_rho = max(r for _, r, _ in comp_rhos)
    dominant = [c for c in comp_rhos if np.isclose(c[1], max_rho, atol=tol, rtol=1e-8)]
    dominant.sort(key=lambda x: (-x[2], x[0]))
    block_id = dominant[0][0]
    block_idx = np.where(labels == block_id)[0]

    meta["dominant_block_id"] = int(block_id)
    meta["dominant_block_size"] = int(len(block_idx))
    meta["dominant_block_tie"] = bool(len(dominant) > 1)
    return block_idx, meta


def calc_pf_eigenvectors(
    matrix: pd.DataFrame,
    *,
    return_metadata: bool = False,
    tol: float = 1e-10,
) -> tuple[pd.Series, pd.Series, float] | tuple[pd.Series, pd.Series, float, dict[str, Any]]:
    values = np.asarray(matrix.values, dtype=float)
    if np.any(values < -tol):
        raise ValueError("PF diagnostics require a nonnegative matrix.")

    block_idx, metadata = _dominant_block_indices(values, tol=tol)
    block = values[np.ix_(block_idx, block_idx)]
    eigvals, left_vecs, right_vecs = linalg.eig(block, left=True, right=True)

    absvals = np.abs(eigvals)
    dom_abs = float(np.max(absvals))
    candidate = np.where(np.isclose(absvals, dom_abs, atol=tol, rtol=1e-8))[0]
    dom_idx = int(candidate[np.argmax(np.real(eigvals[candidate]))])

    rho_raw = eigvals[dom_idx]
    complex_dom = bool(abs(np.imag(rho_raw)) > 1e-8)
    degenerate = bool(len(candidate) > 1)

    u_block, used_abs_u = _orient_nonnegative(left_vecs[:, dom_idx], tol=tol)
    v_block, used_abs_v = _orient_nonnegative(right_vecs[:, dom_idx], tol=tol)
    used_abs = bool(used_abs_u or used_abs_v)

    norm = float(np.dot(u_block, v_block))
    if not np.isfinite(norm) or norm <= tol:
        raise ValueError("PF eigenvectors cannot be normalized (nonpositive inner product).")
    u_block = u_block / norm

    u_full = pd.Series(0.0, index=matrix.index, dtype=float)
    v_full = pd.Series(0.0, index=matrix.columns, dtype=float)
    u_full.iloc[block_idx] = u_block
    v_full.iloc[block_idx] = v_block

    rho = float(np.real(rho_raw))
    metadata.update(
        {
            "degenerate_dominant_eigenvalue": degenerate,
            "complex_dominant": complex_dom,
            "used_absolute_projection": used_abs,
            "rho": rho,
        }
    )

    if (degenerate or complex_dom or metadata.get("reducible", False) or used_abs) and not return_metadata:
        warnings.warn(
            f"PF diagnostics warning metadata={metadata}",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_metadata:
        return u_full, v_full, rho, metadata
    return u_full, v_full, rho


def calc_pf_sensitivities(matrix: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    u, v, rho, meta = calc_pf_eigenvectors(matrix, return_metadata=True)
    if strict and (meta["degenerate_dominant_eigenvalue"] or meta["complex_dominant"]):
        raise ValueError(f"PF sensitivities ill-defined for degenerate/complex dominant eigenvalue: {meta}")
    outer = np.outer(u.values, v.values)
    sensitivities = pd.DataFrame(outer, index=matrix.index, columns=matrix.columns)
    return sensitivities


def partial_sum_convergence(matrix: pd.DataFrame, max_iter: int = 12) -> pd.DataFrame:
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for partial sums.")
    sums = []
    S = np.zeros((n, n))
    for k in range(max_iter):
        term = np.linalg.matrix_power(matrix.values, k) if k > 0 else np.eye(n)
        S = S + term
        sums.append(
            {
                "iteration": k,
                "max_entry": float(np.max(S)),
                "l1_norm": float(np.linalg.norm(S, ord=1)),
                "term_norm": float(np.linalg.norm(term, ord=1)),
            }
        )
    return pd.DataFrame(sums)


def solve_safe_just_lp(
    x: pd.Series,
    y: pd.Series,
    A: pd.DataFrame,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
    *,
    weight_delta_plus: float = 1.0,
    weight_delta_minus: float = 1.0,
    allow_violation: bool = False,
    violation_penalty: float = 1e6,
) -> SafeJustResult:
    n = len(x)
    if not (A.shape[0] == A.shape[1] == n):
        raise ValueError("Technical coefficient matrix must be square and match x.")
    if not all(index == x.index[i] for i, index in enumerate(A.index)):
        A = A.reindex(index=x.index, columns=x.index)
    I = np.eye(n)
    A_matrix = A.values
    me = int(E.shape[0])
    ms = int(S.shape[0])
    base_objective = np.concatenate(
        [np.zeros(n), np.full(n, float(weight_delta_plus)), np.full(n, float(weight_delta_minus))]
    )
    if allow_violation:
        objective = np.concatenate(
            [base_objective, np.full(me, float(violation_penalty)), np.full(ms, float(violation_penalty))]
        )
        n_vars = 3 * n + me + ms
    else:
        objective = base_objective
        n_vars = 3 * n
    # Inequality constraints
    # 1) x_new - A x_new >= y  <=>  (A - I) x_new <= -y
    ineq1 = np.hstack([A_matrix - I, np.zeros((n, n_vars - n))])
    b1 = -y.reindex(x.index).values
    # 2) Ecological ceilings: E x_new <= e_bar
    eco_block = E.reindex(columns=x.index).values
    if allow_violation:
        eco_slack = -np.eye(me)
        eco_tail = np.zeros((me, ms))
        ineq2 = np.hstack([eco_block, np.zeros((me, 2 * n)), eco_slack, eco_tail])
    else:
        ineq2 = np.hstack([eco_block, np.zeros((me, n_vars - n))])
    b2 = e_bar.reindex(E.index).values
    # 3) Social floors: -S x_new <= -s_floor
    soc_block = -S.reindex(columns=x.index).values
    if allow_violation:
        soc_tail = -np.eye(ms)
        ineq3 = np.hstack([soc_block, np.zeros((ms, 2 * n + me)), soc_tail])
    else:
        ineq3 = np.hstack([soc_block, np.zeros((ms, n_vars - n))])
    b3 = -s_floor.reindex(S.index).values
    A_ub = np.vstack([ineq1, ineq2, ineq3])
    b_ub = np.concatenate([b1, b2, b3])
    # Equality constraints to link delta vars
    eq = np.hstack([np.eye(n), -np.eye(n), np.eye(n), np.zeros((n, n_vars - 3 * n))])
    A_eq = eq
    b_eq = x.values
    bounds = [(0, None)] * n_vars
    res = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if res.status != 0 or not res.success:
        raise RuntimeError(f"Safe-and-just LP failed: {res.message}")
    x_new = res.x[:n]
    delta_plus = res.x[n:2 * n]
    delta_minus = res.x[2 * n : 3 * n]
    eco_violation = None
    soc_violation = None
    if allow_violation:
        eco_violation = pd.Series(res.x[3 * n : 3 * n + me], index=E.index, dtype=float)
        soc_violation = pd.Series(res.x[3 * n + me : 3 * n + me + ms], index=S.index, dtype=float)
    return SafeJustResult(
        x_opt=pd.Series(x_new, index=x.index),
        delta_plus=pd.Series(delta_plus, index=x.index),
        delta_minus=pd.Series(delta_minus, index=x.index),
        objective=res.fun,
        inequality_duals=(
            np.asarray(res.ineqlin.marginals, dtype=float)
            if hasattr(res, "ineqlin") and hasattr(res.ineqlin, "marginals")
            else None
        ),
        equality_duals=(
            np.asarray(res.eqlin.marginals, dtype=float)
            if hasattr(res, "eqlin") and hasattr(res.eqlin, "marginals")
            else None
        ),
        eco_violation=eco_violation,
        soc_violation=soc_violation,
        status=str(res.status),
        message=str(res.message),
    )


def calc_boundary_slack(
    E: pd.DataFrame, S: pd.DataFrame, x_new: pd.Series, e_bar: pd.Series, s_floor: pd.Series
) -> tuple[pd.Series, pd.Series]:
    eco_slack = e_bar.reindex(E.index) - E.reindex(columns=x_new.index).dot(x_new)
    soc_slack = S.reindex(columns=x_new.index).dot(x_new) - s_floor.reindex(S.index)
    return eco_slack, soc_slack


def aggregate_diagnostics(
    pf_sensitivities: pd.DataFrame,
    E: pd.DataFrame,
    S: pd.DataFrame,
    eco_slack: pd.Series,
    soc_slack: pd.Series,
    epsilon: float = 1e-6,
) -> pd.Series:
    column_sensitivity = pf_sensitivities.sum(axis=0)
    pressure = pd.Series(0.0, index=column_sensitivity.index)
    if not eco_slack.empty:
        eco_contrib = E.div(eco_slack + epsilon, axis=0).sum(axis=0)
        pressure = pressure.add(eco_contrib, fill_value=0.0)
    if not soc_slack.empty:
        soc_contrib = S.div(soc_slack + epsilon, axis=0).sum(axis=0)
        pressure = pressure.add(soc_contrib, fill_value=0.0)
    return column_sensitivity + pressure

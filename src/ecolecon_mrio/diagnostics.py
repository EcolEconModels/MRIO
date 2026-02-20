"""Low-level diagnostics to support the viability-margins analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import linprog


@dataclass
class SafeJustResult:
    x_opt: pd.Series
    delta_plus: pd.Series
    delta_minus: pd.Series
    objective: float
    inequality_duals: np.ndarray | None
    equality_duals: np.ndarray | None


def calc_spectral_radius(matrix: pd.DataFrame) -> float:
    eigvals = linalg.eigvals(matrix.values)
    return float(np.max(np.abs(eigvals)))


def calc_tech_slack(matrix: pd.DataFrame) -> tuple[float, float]:
    rho = calc_spectral_radius(matrix)
    return rho, max(0.0, 1.0 - rho)


def calc_pf_eigenvectors(matrix: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    eigvals, left_vecs, right_vecs = linalg.eig(matrix.values, left=True, right=True)
    idx = int(np.argmax(np.abs(eigvals)))
    rho = eigvals[idx]
    u_raw = np.real(left_vecs[:, idx])
    v_raw = np.real(right_vecs[:, idx])
    scale = float(np.dot(u_raw, v_raw))
    if scale == 0 or np.isnan(scale):
        raise ValueError("PF eigenvectors cannot be normalized (zero inner product).")
    u = u_raw / scale
    v = v_raw
    u_series = pd.Series(u, index=matrix.index)
    v_series = pd.Series(v, index=matrix.columns)
    return u_series, v_series, float(np.real(rho))


def calc_pf_sensitivities(matrix: pd.DataFrame) -> pd.DataFrame:
    u, v, rho = calc_pf_eigenvectors(matrix)
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
) -> SafeJustResult:
    n = len(x)
    if not (A.shape[0] == A.shape[1] == n):
        raise ValueError("Technical coefficient matrix must be square and match x.")
    if not all(index == x.index[i] for i, index in enumerate(A.index)):
        A = A.reindex(index=x.index, columns=x.index)
    I = np.eye(n)
    A_matrix = A.values
    objective = np.concatenate([np.zeros(n), np.ones(n), np.ones(n)])
    # Inequality constraints
    # 1) x_new - A x_new >= y  <=>  (A - I) x_new <= -y
    ineq1 = np.hstack([A_matrix - I, np.zeros((n, 2 * n))])
    b1 = -y.reindex(x.index).values
    # 2) Ecological ceilings: E x_new <= e_bar
    ineq2 = np.hstack([E.reindex(columns=x.index).values, np.zeros((E.shape[0], 2 * n))])
    b2 = e_bar.reindex(E.index).values
    # 3) Social floors: -S x_new <= -s_floor
    ineq3 = np.hstack([-S.reindex(columns=x.index).values, np.zeros((S.shape[0], 2 * n))])
    b3 = -s_floor.reindex(S.index).values
    A_ub = np.vstack([ineq1, ineq2, ineq3])
    b_ub = np.concatenate([b1, b2, b3])
    # Equality constraints to link delta vars
    eq = np.hstack([np.eye(n), -np.eye(n), np.eye(n)])
    A_eq = eq
    b_eq = x.values
    bounds = [(0, None)] * (3 * n)
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

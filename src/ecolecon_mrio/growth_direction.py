"""Principal growth-direction diagnostics linked to von Neumann growth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix, issparse

from .diagnostics import calc_pf_eigenvectors


@dataclass
class PrincipalDirectionResult:
    year: int | None
    alpha: float
    q: pd.Series
    p: pd.Series
    method: str
    fallback_used: bool
    pf_metadata: dict[str, Any] | None = None


def _orient_and_normalize(vec: np.ndarray, index: pd.Index, tol: float = 1e-12) -> tuple[pd.Series, bool]:
    out = np.real(np.asarray(vec)).astype(float, copy=False)
    used_abs = False
    if np.all(out >= -tol):
        out = np.clip(out, 0.0, None)
    elif np.all(out <= tol):
        out = np.clip(-out, 0.0, None)
    else:
        out = np.abs(out)
        used_abs = True
    total = float(out.sum())
    if total <= tol:
        raise ValueError("Principal direction vector could not be normalized.")
    return pd.Series(out / total, index=index, dtype=float), used_abs


def _is_sign_consistent(vec: np.ndarray, tol: float = 1e-12) -> bool:
    arr = np.real(np.asarray(vec)).astype(float, copy=False)
    return bool(np.all(arr >= -tol) or np.all(arr <= tol))


def principal_growth_direction(
    A: pd.DataFrame,
    B: pd.DataFrame | None = None,
    *,
    require_generalized: bool = False,
    year: int | None = None,
    index: pd.Index | None = None,
) -> PrincipalDirectionResult:
    """Compute principal growth direction from generalized von Neumann eigenproblem.

    For a generalized activity formulation, solve B q = alpha A q with q >= 0.
    If B is omitted, fallback is B = I, equivalent to alpha = 1 / rho(A).
    """
    if issparse(A):
        A_sparse = csr_matrix(A, dtype=float)
        if A_sparse.shape[0] != A_sparse.shape[1]:
            raise ValueError("A must be square.")
        basis = index if index is not None else getattr(A, "index", pd.RangeIndex(A_sparse.shape[0]))
        if len(basis) != A_sparse.shape[0] or not basis.is_unique:
            raise ValueError("Sparse A index must be unique and match the matrix dimension.")
        for attribute in ("index", "columns"):
            labels = getattr(A, attribute, None)
            if labels is not None and index is not None and not pd.Index(labels).equals(index):
                raise ValueError("Explicit sparse A index must exactly match matrix label metadata.")
        if A_sparse.nnz and float(A_sparse.data.min()) < -1e-12:
            raise ValueError("A must be nonnegative for von Neumann/PF diagnostics.")
        A_mat = None
    else:
        A_mat = pd.DataFrame(A, copy=True)
        if A_mat.shape[0] != A_mat.shape[1]:
            raise ValueError("A must be square.")
        if not A_mat.index.equals(A_mat.columns):
            raise ValueError("Dense A must have identical row and column axes.")
        if index is not None and not A_mat.index.equals(index):
            raise ValueError("Explicit index must exactly equal the dense A basis.")
        basis = A_mat.index
        if np.any(A_mat.values < -1e-12):
            raise ValueError("A must be nonnegative for von Neumann/PF diagnostics.")

    if B is None:
        if require_generalized:
            raise ValueError("Generalized mode requested but B is missing.")
        u, v, rho, pf_meta = calc_pf_eigenvectors(A_sparse if issparse(A) else A_mat, return_metadata=True, index=basis)
        alpha = float(1.0 / rho) if rho > 0 else np.inf
        q = v / float(v.sum())
        p = u / float(u.sum()) if float(u.sum()) > 0 else u
        return PrincipalDirectionResult(
            year=year,
            alpha=alpha,
            q=q,
            p=p,
            method="pf_fallback",
            fallback_used=True,
            pf_metadata=pf_meta,
        )

    if A_mat is None:
        raise ValueError("Generalized B/A eigensolve is not supported for sparse A; use B=None/PF fallback.")
    B_mat = pd.DataFrame(B, copy=True)
    if not B_mat.index.equals(A_mat.index) or not B_mat.columns.equals(A_mat.columns):
        raise ValueError("B must align with A indices/columns.")
    if np.any(B_mat.values < -1e-12):
        raise ValueError("B must be nonnegative for von Neumann diagnostics.")

    eigvals, left_vecs, right_vecs = linalg.eig(B_mat.values, A_mat.values, left=True, right=True)
    finite = np.isfinite(eigvals)
    eigvals = eigvals[finite]
    left_vecs = left_vecs[:, finite]
    right_vecs = right_vecs[:, finite]
    if eigvals.size == 0:
        raise ValueError("No finite generalized eigenvalues found for (B, A).")

    candidate = np.where((np.real(eigvals) > 0) & (np.abs(np.imag(eigvals)) <= 1e-8))[0]
    if candidate.size == 0:
        raise ValueError("No positive real generalized eigenvalue available for von Neumann direction.")

    admissible: list[int] = []
    for i in candidate:
        if _is_sign_consistent(right_vecs[:, i]) and _is_sign_consistent(left_vecs[:, i]):
            admissible.append(int(i))
    if admissible:
        idx = int(admissible[np.argmax(np.real(eigvals[admissible]))])
        q, q_abs = _orient_and_normalize(right_vecs[:, idx], index=A_mat.columns)
        p, p_abs = _orient_and_normalize(left_vecs[:, idx], index=A_mat.index)
        fallback = bool(q_abs or p_abs)
        consistent = not fallback
    else:
        # Fallback to the largest real positive eigenvalue regardless of sign consistency
        idx = int(candidate[np.argmax(np.real(eigvals[candidate]))])
        q, q_abs = _orient_and_normalize(right_vecs[:, idx], index=A_mat.columns)
        p, p_abs = _orient_and_normalize(left_vecs[:, idx], index=A_mat.index)
        fallback = True
        consistent = False

    alpha = float(np.real(eigvals[idx]))
    pf_meta = {"mathematically_consistent": consistent}
    return PrincipalDirectionResult(
        year=year,
        alpha=alpha,
        q=q,
        p=p,
        method="von_neumann_generalized",
        fallback_used=fallback,
        pf_metadata=pf_meta,
    )


def principal_growth_panel(
    x_by_year: Mapping[int, pd.Series],
    A_by_year: Mapping[int, pd.DataFrame],
    B_by_year: Mapping[int, pd.DataFrame] | None = None,
    *,
    require_generalized: bool = False,
    year_result_callback: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, PrincipalDirectionResult]]:
    """Compute annual principal directions and explicit pair-indexed diagnostics.

    Returns
    -------
    year_summary : DataFrame
        One row per year with year-local growth diagnostics (`alpha`, method, fallback flags).
    pair_summary : DataFrame
        One row per forward pair (`year_from`, `year_to`) with pairwise diagnostics computed
        from q_year_from and observed output change x_year_to - x_year_from.
    results : dict[int, PrincipalDirectionResult]
        Year-keyed detailed results.
    """
    years = sorted(set(x_by_year.keys()) & set(A_by_year.keys()))
    if not years:
        raise ValueError("No overlapping years between x_by_year and A_by_year.")

    results: dict[int, PrincipalDirectionResult] = {}
    for year in years:
        B = None if B_by_year is None else B_by_year.get(year)
        res = principal_growth_direction(
            A_by_year[year],
            B=B,
            require_generalized=require_generalized,
            year=year,
            index=x_by_year[year].index,
        )
        basis_index = x_by_year[year].index
        if not res.q.index.equals(basis_index) or not res.p.index.equals(basis_index):
            raise ValueError(f"PF basis does not match x basis for year {year}.")
        if year_result_callback is not None:
            year_result_callback(int(year), res)
        results[year] = res

    year_rows: list[dict[str, float | int | str | bool]] = []
    for year in years:
        res = results[year]
        year_rows.append(
            {
                "year": int(year),
                "alpha": float(res.alpha),
                "method": res.method,
                "fallback_used": bool(res.fallback_used),
            }
        )

    pair_rows: list[dict[str, float | int]] = []
    for i, year_from in enumerate(years[:-1]):
        year_to = years[i + 1]
        res0 = results[year_from]
        res1 = results[year_to]

        q0 = res0.q
        q1 = res1.q
        if not q1.index.equals(q0.index) or not x_by_year[year_to].index.equals(x_by_year[year_from].index):
            raise ValueError(f"Pair basis mismatch between years {year_from} and {year_to}.")
        q0v = q0.to_numpy(dtype=float)
        q1v = q1.to_numpy(dtype=float)
        theta = np.nan
        q_denom = float(np.linalg.norm(q0v) * np.linalg.norm(q1v))
        if q_denom > 0:
            theta = float(np.arccos(np.clip(float(np.dot(q0v, q1v)) / q_denom, -1.0, 1.0)))

        x0 = x_by_year[year_from]
        x1 = x_by_year[year_to]
        dx = (x1 - x0).to_numpy(dtype=float)
        qv = res0.q.to_numpy(dtype=float)
        align = np.nan
        denom = float(np.linalg.norm(dx) * np.linalg.norm(qv))
        if denom > 0:
            align = float(np.dot(dx, qv) / denom)

        pair_rows.append(
            {
                "year_from": int(year_from),
                "year_to": int(year_to),
                "theta_forward": theta,
                "alignment_with_dx_forward": align,
            }
        )

    year_summary = pd.DataFrame(year_rows, columns=["year", "alpha", "method", "fallback_used"])
    pair_summary = pd.DataFrame(
        pair_rows,
        columns=["year_from", "year_to", "theta_forward", "alignment_with_dx_forward"],
    )
    if not year_summary.empty:
        year_summary = year_summary.sort_values("year").reset_index(drop=True)
    if not pair_summary.empty:
        pair_summary = pair_summary.sort_values(["year_from", "year_to"]).reset_index(drop=True)
    return year_summary, pair_summary, results

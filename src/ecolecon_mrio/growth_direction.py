"""Principal growth-direction diagnostics linked to von Neumann growth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import linalg

from .diagnostics import calc_pf_eigenvectors


@dataclass
class PrincipalDirectionResult:
    year: int | None
    alpha: float
    q: pd.Series
    p: pd.Series
    method: str
    fallback_used: bool
    rotation_to_prev: float | None
    alignment_with_dx: float | None
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
) -> PrincipalDirectionResult:
    """Compute principal growth direction from generalized von Neumann eigenproblem.

    For a generalized activity formulation, solve B q = alpha A q with q >= 0.
    If B is omitted, fallback is B = I, equivalent to alpha = 1 / rho(A).
    """
    A_mat = pd.DataFrame(A, copy=True)
    if A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError("A must be square.")
    if np.any(A_mat.values < -1e-12):
        raise ValueError("A must be nonnegative for von Neumann/PF diagnostics.")

    if B is None:
        if require_generalized:
            raise ValueError("Generalized mode requested but B is missing.")
        u, v, rho, pf_meta = calc_pf_eigenvectors(A_mat, return_metadata=True)
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
            rotation_to_prev=None,
            alignment_with_dx=None,
            pf_metadata=pf_meta,
        )

    B_mat = pd.DataFrame(B, copy=True).reindex(index=A_mat.index, columns=A_mat.columns)
    if B_mat.isna().values.any():
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
    else:
        idx = int(candidate[np.argmax(np.real(eigvals[candidate]))])
        q, q_abs = _orient_and_normalize(right_vecs[:, idx], index=A_mat.columns)
        p, p_abs = _orient_and_normalize(left_vecs[:, idx], index=A_mat.index)
        fallback = True

    alpha = float(np.real(eigvals[idx]))
    return PrincipalDirectionResult(
        year=year,
        alpha=alpha,
        q=q,
        p=p,
        method="von_neumann_generalized",
        fallback_used=fallback,
        rotation_to_prev=None,
        alignment_with_dx=None,
        pf_metadata=None,
    )


def principal_growth_panel(
    x_by_year: Mapping[int, pd.Series],
    A_by_year: Mapping[int, pd.DataFrame],
    B_by_year: Mapping[int, pd.DataFrame] | None = None,
    *,
    require_generalized: bool = False,
) -> tuple[pd.DataFrame, dict[int, PrincipalDirectionResult]]:
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
        )
        results[year] = res

    year_rows: list[dict[str, float | int | str | bool | None]] = []
    for i, year in enumerate(years):
        res = results[year]
        rotation = np.nan
        if i > 0:
            prev = results[years[i - 1]]
            dot = float(np.dot(prev.q.values, res.q.values))
            denom = float(np.linalg.norm(prev.q.values) * np.linalg.norm(res.q.values))
            if denom > 0:
                rotation = float(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
        align = np.nan
        if i < len(years) - 1:
            next_year = years[i + 1]
            x0 = x_by_year[year]
            x1 = x_by_year[next_year].reindex(x0.index).fillna(0.0)
            dx = (x1 - x0).to_numpy(dtype=float)
            qv = res.q.reindex(x0.index).fillna(0.0).to_numpy(dtype=float)
            denom = float(np.linalg.norm(dx) * np.linalg.norm(qv))
            if denom > 0:
                align = float(np.dot(dx, qv) / denom)
        results[year].rotation_to_prev = None if np.isnan(rotation) else rotation
        results[year].alignment_with_dx = None if np.isnan(align) else align
        year_rows.append(
            {
                "year": int(year),
                "alpha": float(res.alpha),
                "method": res.method,
                "fallback_used": bool(res.fallback_used),
                "rotation_to_prev": rotation,
                "alignment_with_dx": align,
            }
        )

    return pd.DataFrame(year_rows).sort_values("year").reset_index(drop=True), results

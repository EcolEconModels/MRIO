"""Principal growth-direction diagnostics linked to von Neumann growth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

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
    A: Any,
    B: pd.DataFrame | None = None,
    *,
    require_generalized: bool = False,
    year: int | None = None,
    index: pd.Index | None = None,
) -> PrincipalDirectionResult:
    """Compute the labelled principal growth direction on one exact economic basis.

    Sparse ``A`` may be structurally unlabeled, in which case ``index`` is the
    authoritative economic basis supplied by the caller.  Dense labelled inputs
    must already use identical row/column axes; no reindexing or zero filling is
    permitted at this scientific boundary.
    """
    A_sparse: csr_matrix | None = None
    if issparse(A):
        A_sparse = csr_matrix(A, dtype=float)
        n_rows, n_cols = cast(tuple[int, int], A_sparse.shape)
        if n_rows != n_cols:
            raise ValueError("A must be square.")
        if A_sparse.nnz and not np.isfinite(A_sparse.data).all():
            raise ValueError("A must contain only finite technical coefficients.")
        basis_raw = index if index is not None else getattr(A, "index", pd.RangeIndex(n_rows))
        basis = basis_raw.copy() if isinstance(basis_raw, pd.Index) else pd.Index(basis_raw)
        if len(basis) != n_rows or not basis.is_unique:
            raise ValueError("Sparse A index must be unique and match the matrix dimension.")
        for attribute in ("index", "columns"):
            labels = getattr(A, attribute, None)
            if labels is not None and index is not None:
                labels_index = labels.copy() if isinstance(labels, pd.Index) else pd.Index(labels)
                if not labels_index.equals(index):
                    raise ValueError("Explicit sparse A index must exactly match matrix label metadata.")
        if A_sparse.nnz and float(A_sparse.data.min()) < -1e-12:
            raise ValueError("A must be nonnegative for von Neumann/PF diagnostics.")
        A_mat = None
    else:
        try:
            A_mat = pd.DataFrame(A, copy=True).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("A must be numeric for von Neumann/PF diagnostics.") from exc
        if A_mat.shape[0] != A_mat.shape[1]:
            raise ValueError("A must be square.")
        if not A_mat.index.equals(A_mat.columns):
            raise ValueError("Dense A must have identical row and column axes.")
        if not A_mat.index.is_unique:
            raise ValueError("Dense A basis must be unique.")
        if index is not None and not A_mat.index.equals(index):
            raise ValueError("Explicit index must exactly equal the dense A basis.")
        a_values = A_mat.to_numpy(dtype=float, copy=False)
        if not np.isfinite(a_values).all():
            raise ValueError("A must contain only finite technical coefficients.")
        basis = A_mat.index
        if np.any(a_values < -1e-12):
            raise ValueError("A must be nonnegative for von Neumann/PF diagnostics.")

    if B is None:
        if require_generalized:
            raise ValueError("Generalized mode requested but B is missing.")
        pf_matrix = A_sparse if A_sparse is not None else A_mat
        pf_result = calc_pf_eigenvectors(
            pf_matrix,
            return_metadata=True,
            index=basis,
        )
        u, v, rho, pf_meta = cast(tuple[pd.Series, pd.Series, float, dict[str, Any]], pf_result)
        alpha = float(1.0 / rho) if rho > 0 else np.inf
        q_sum = float(v.sum())
        p_sum = float(u.sum())
        if not np.isfinite(q_sum) or q_sum <= 0.0 or not np.isfinite(p_sum) or p_sum <= 0.0:
            raise ValueError("PF vectors must have finite positive normalization mass.")
        q = v / q_sum
        p = u / p_sum
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
    try:
        B_mat = pd.DataFrame(B, copy=True).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("B must be numeric for von Neumann diagnostics.") from exc
    if not B_mat.index.equals(A_mat.index) or not B_mat.columns.equals(A_mat.columns):
        raise ValueError("B must align with A indices/columns.")
    b_values = B_mat.to_numpy(dtype=float, copy=False)
    if not np.isfinite(b_values).all():
        raise ValueError("B must contain only finite coefficients.")
    if np.any(b_values < -1e-12):
        raise ValueError("B must be nonnegative for von Neumann diagnostics.")

    eig_result = linalg.eig(
        b_values,
        A_mat.to_numpy(dtype=float, copy=False),
        left=True,
        right=True,
    )
    eigvals, left_vecs, right_vecs = cast(tuple[np.ndarray, np.ndarray, np.ndarray], eig_result)
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
    A_by_year: Mapping[int, Any],
    B_by_year: Mapping[int, pd.DataFrame] | None = None,
    *,
    require_generalized: bool = False,
    year_result_callback: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, PrincipalDirectionResult]]:
    """Compute annual PF directions and pair diagnostics on one exact yearly basis."""
    x_years = set(x_by_year)
    a_years = set(A_by_year)
    if x_years != a_years:
        raise ValueError(
            "x_by_year and A_by_year must contain exactly the same years; "
            f"x_only={sorted(x_years - a_years)!r}, A_only={sorted(a_years - x_years)!r}."
        )
    if not x_years:
        raise ValueError("No years supplied to principal_growth_panel.")
    if B_by_year is not None:
        extra_b = set(B_by_year) - x_years
        if extra_b:
            raise ValueError(f"B_by_year contains years absent from the x/A panel: {sorted(extra_b)!r}.")
        if require_generalized:
            missing_b = x_years - set(B_by_year)
            if missing_b:
                raise ValueError(f"Generalized mode requires B for every panel year: {sorted(missing_b)!r}.")

    years = sorted(x_years)
    validated_x: dict[int, pd.Series] = {}
    results: dict[int, PrincipalDirectionResult] = {}
    for year in years:
        raw_x = x_by_year[year]
        if not isinstance(raw_x, pd.Series):
            raise TypeError(f"Gross output x must be a pandas Series for year {year}.")
        if not raw_x.index.is_unique:
            raise ValueError(f"Gross output x basis must be unique for year {year}.")
        try:
            numeric_x = cast(pd.Series, pd.to_numeric(raw_x, errors="raise")).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Gross output x must be numeric for year {year}.") from exc
        x_values = numeric_x.to_numpy(dtype=float)
        if not np.isfinite(x_values).all():
            raise ValueError(f"Gross output x must be finite for year {year}.")
        if bool((x_values < 0.0).any()):
            raise ValueError(f"Gross output x must be nonnegative for year {year}.")
        validated_x[year] = pd.Series(x_values, index=raw_x.index.copy(), dtype=float, name=raw_x.name)

        B = None if B_by_year is None else B_by_year.get(year)
        res = principal_growth_direction(
            A_by_year[year],
            B=B,
            require_generalized=require_generalized,
            year=year,
            index=validated_x[year].index,
        )
        basis_index = validated_x[year].index
        if not res.q.index.equals(basis_index) or not res.p.index.equals(basis_index):
            raise ValueError(f"PF basis does not match x basis for year {year}.")
        for label, vector in (("q", res.q), ("p", res.p)):
            vector_values = vector.to_numpy(dtype=float)
            if not np.isfinite(vector_values).all() or float(vector.sum()) <= 0.0:
                raise ValueError(f"PF {label} must be finite and nonzero for year {year}.")
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
    for year_from, year_to in zip(years, years[1:]):
        res0 = results[year_from]
        res1 = results[year_to]
        x0 = validated_x[year_from]
        x1 = validated_x[year_to]
        if not x1.index.equals(x0.index):
            raise ValueError(f"Pair x basis mismatch between years {year_from} and {year_to}.")
        if not res0.q.index.equals(x0.index) or not res1.q.index.equals(x0.index):
            raise ValueError(f"Pair PF basis mismatch between years {year_from} and {year_to}.")

        q0v = res0.q.to_numpy(dtype=float)
        q1v = res1.q.to_numpy(dtype=float)
        theta = np.nan
        q_denom = float(np.linalg.norm(q0v) * np.linalg.norm(q1v))
        if q_denom > 0:
            theta = float(np.arccos(np.clip(float(np.dot(q0v, q1v)) / q_denom, -1.0, 1.0)))

        dx = (x1 - x0).to_numpy(dtype=float)
        align = np.nan
        denom = float(np.linalg.norm(dx) * np.linalg.norm(q0v))
        if denom > 0:
            align = float(np.dot(dx, q0v) / denom)

        pair_rows.append(
            {
                "year_from": int(year_from),
                "year_to": int(year_to),
                "theta_forward": theta,
                "alignment_with_dx_forward": align,
            }
        )

    year_summary = pd.DataFrame(year_rows, columns=pd.Index(["year", "alpha", "method", "fallback_used"]))
    pair_summary = pd.DataFrame(
        pair_rows,
        columns=pd.Index(["year_from", "year_to", "theta_forward", "alignment_with_dx_forward"]),
    )
    return year_summary.sort_values("year").reset_index(drop=True), pair_summary.sort_values(
        ["year_from", "year_to"]
    ).reset_index(drop=True), results

"""Low-level diagnostics to support the viability-margins analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import gc
import warnings
import time

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack, issparse
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


def _rss_mb() -> float:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX fallback
        return float("nan")
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    import sys

    if sys.platform == "darwin":
        rss /= 1024.0 * 1024.0
    else:
        rss /= 1024.0
    return float(rss)


def _diag_print(message: str) -> None:
    print(f"[MRIO] {message}", flush=True)


def _matrix_nonzero_stats(values: np.ndarray) -> tuple[int, float]:
    nnz = int(np.count_nonzero(values))
    density = float(nnz / values.size) if values.size else float("nan")
    return nnz, density



def _matrix_to_csr_and_labels(matrix: Any, index: pd.Index | None = None) -> tuple[csr_matrix, pd.Index, pd.Index]:
    if issparse(matrix):
        values = csr_matrix(matrix, dtype=float)
        row_index = index if index is not None else getattr(matrix, "index", pd.RangeIndex(values.shape[0]))
        col_index = index if index is not None else getattr(matrix, "columns", pd.RangeIndex(values.shape[1]))
        return values, pd.Index(row_index), pd.Index(col_index)
    frame = pd.DataFrame(matrix, copy=False)
    return csr_matrix(frame.to_numpy(dtype=float, copy=False)), pd.Index(frame.index), pd.Index(frame.columns)

def _spectral_radius_method_hint(values: np.ndarray) -> str:
    n = int(values.shape[0])
    nnz, density = _matrix_nonzero_stats(values)
    if n >= 256:
        return f"sparse_eigs nnz={nnz} density={density:.4f}"
    return f"dense_eigvals nnz={nnz} density={density:.4f}"


def _spectral_radius_from_values(values: np.ndarray, *, tol: float = 1e-10) -> tuple[float, str]:
    n = int(values.shape[0])
    if n == 0:
        return float("nan"), "empty"
    nnz, density = _matrix_nonzero_stats(values)
    sparse_threshold = 256
    if n >= sparse_threshold:
        try:
            vals = eigs(
                csr_matrix(values),
                k=1,
                which="LM",
                tol=1e-6,
                maxiter=max(4000, 10 * n),
                return_eigenvectors=False,
            )
            rho = float(np.max(np.abs(vals)))
            return rho, f"sparse_eigs nnz={nnz} density={density:.4f}"
        except ArpackNoConvergence as exc:
            vals = np.asarray(exc.eigenvalues) if exc.eigenvalues is not None else np.array([])
            if vals.size:
                rho = float(np.max(np.abs(vals)))
                return rho, f"sparse_eigs_partial nnz={nnz} density={density:.4f}"
        except Exception:
            pass
    vals = linalg.eigvals(values)
    rho = float(np.max(np.abs(vals))) if vals.size else float("nan")
    return rho, f"dense_eigvals nnz={nnz} density={density:.4f}"


def calc_spectral_radius(matrix: Any) -> float:
    values, _row_index, _col_index = _matrix_to_csr_and_labels(matrix)
    nnz = int(values.nnz)
    density = float(nnz / (values.shape[0] * values.shape[1])) if values.shape[0] and values.shape[1] else float("nan")
    method = f"sparse_csr nnz={nnz} density={density:.4f}"
    _diag_print(f"spectral radius start shape={values.shape} method={method} rss={_rss_mb():.1f}MB")
    t0 = time.time()
    rho = _spectral_radius_from_sparse(values)
    _diag_print(
        f"spectral radius done shape={values.shape} rho={rho:.6g} method={method} "
        f"elapsed={time.time()-t0:.1f}s rss={_rss_mb():.1f}MB"
    )
    return rho


def calc_tech_slack(matrix: pd.DataFrame) -> tuple[float, float]:
    _diag_print(f"tech slack start shape={matrix.shape} rss={_rss_mb():.1f}MB")
    t0 = time.time()
    rho = calc_spectral_radius(matrix)
    slack = max(0.0, 1.0 - rho)
    _diag_print(
        f"tech slack done shape={matrix.shape} rho={rho:.6g} slack={slack:.6g} "
        f"elapsed={time.time()-t0:.1f}s rss={_rss_mb():.1f}MB"
    )
    return rho, slack


def calc_coefficient_diagnostics(
    matrix: Any,
    *,
    tol: float = 1e-12,
    upper_bound: float = 1.0,
) -> dict[str, float]:
    """Compute local coefficient diagnostics without densifying sparse EXIO matrices."""
    if issparse(matrix):
        A = csr_matrix(matrix, dtype=float)
        n_total = int(A.shape[0] * A.shape[1])
        if n_total == 0:
            return {
                "a_min": np.nan,
                "a_max": np.nan,
                "share_neg": np.nan,
                "share_gt_upper": np.nan,
                "share_zero_or_near_zero": np.nan,
                "upper_margin_min": np.nan,
                "col_sum_max": np.nan,
                "row_sum_max": np.nan,
            }
        data = A.data.astype(float, copy=False)
        explicit_nnz = int(data.size)
        implicit_zeros = int(n_total - explicit_nnz)
        a_min = float(min(0.0, float(data.min()) if data.size else 0.0))
        a_max = float(max(0.0, float(data.max()) if data.size else 0.0))
        near_zero_data = int(np.count_nonzero(data <= tol)) if data.size else 0
        return {
            "a_min": a_min,
            "a_max": a_max,
            "share_neg": float(np.count_nonzero(data < -tol) / n_total) if n_total else np.nan,
            "share_gt_upper": float(np.count_nonzero(data > (float(upper_bound) + tol)) / n_total) if n_total else np.nan,
            "share_zero_or_near_zero": float((implicit_zeros + near_zero_data) / n_total) if n_total else np.nan,
            "upper_margin_min": float(float(upper_bound) - a_max),
            "col_sum_max": float(np.asarray(A.sum(axis=0)).reshape(-1).max()) if A.shape[1] else np.nan,
            "row_sum_max": float(np.asarray(A.sum(axis=1)).reshape(-1).max()) if A.shape[0] else np.nan,
        }
    A = pd.DataFrame(matrix, copy=True).astype(float)
    values = A.to_numpy(dtype=float)
    if values.size == 0:
        return {
            "a_min": np.nan,
            "a_max": np.nan,
            "share_neg": np.nan,
            "share_gt_upper": np.nan,
            "share_zero_or_near_zero": np.nan,
            "upper_margin_min": np.nan,
            "col_sum_max": np.nan,
            "row_sum_max": np.nan,
        }
    a_min = float(np.min(values))
    a_max = float(np.max(values))
    return {
        "a_min": a_min,
        "a_max": a_max,
        "share_neg": float(np.mean(values < -tol)),
        "share_gt_upper": float(np.mean(values > (float(upper_bound) + tol))),
        "share_zero_or_near_zero": float(np.mean(values <= tol)),
        "upper_margin_min": float(float(upper_bound) - a_max),
        "col_sum_max": float(np.max(values.sum(axis=0))) if values.shape[1] else np.nan,
        "row_sum_max": float(np.max(values.sum(axis=1))) if values.shape[0] else np.nan,
    }


def _orient_nonnegative(vec: np.ndarray, tol: float) -> tuple[np.ndarray, bool]:
    out = np.real(np.asarray(vec)).astype(float, copy=False)
    if np.all(out >= -tol):
        out = np.clip(out, 0.0, None)
        return out, False
    if np.all(out <= tol):
        out = np.clip(-out, 0.0, None)
        return out, False
    return np.abs(out), True


def _allocate_role_quotas(working_set_n: int, role_mix: Mapping[str, float]) -> dict[str, int]:
    roles = ["expand", "contract", "neutral"]
    weights = {role: float(role_mix.get(role, 0.0)) for role in roles}
    total = float(sum(max(0.0, w) for w in weights.values()))
    if total <= 0:
        weights = {"expand": 0.3, "contract": 0.3, "neutral": 0.4}
        total = 1.0
    raw = {role: max(0.0, weights[role]) / total * float(working_set_n) for role in roles}
    quotas = {role: int(np.floor(raw[role])) for role in roles}
    remainder = int(working_set_n - sum(quotas.values()))
    if remainder > 0:
        order = sorted(roles, key=lambda role: (raw[role] - quotas[role], weights[role]), reverse=True)
        for role in order:
            if remainder <= 0:
                break
            quotas[role] += 1
            remainder -= 1
    return quotas


def _select_lp_working_set_index(
    x: pd.Series,
    *,
    working_set_n: int | None,
    mode: str,
    role_masks: Mapping[str, pd.Series] | None = None,
    role_mix: Mapping[str, float] | None = None,
) -> pd.Index:
    index = pd.Index(x.index)
    if working_set_n is None:
        return index
    n_work = int(working_set_n)
    if n_work <= 0 or n_work >= len(index):
        return index
    mode_norm = str(mode or "output_weighted").strip().lower()
    x_vals = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0).astype(float).reindex(index).fillna(0.0)
    if mode_norm in {"stratified_transition_roles", "transition_role_mix", "stratified_roles"} and role_masks:
        mix = role_mix or {"expand": 0.3, "contract": 0.3, "neutral": 0.4}
        quotas = _allocate_role_quotas(n_work, mix)
        used = pd.Series(False, index=index, dtype=bool)
        selected_parts: list[pd.Index] = []
        role_priority = ["expand", "contract", "neutral"]
        score = x_vals.abs()
        for role in role_priority:
            mask = pd.Series(role_masks.get(role, pd.Series(False, index=index)), index=index).fillna(False).astype(bool)
            mask = mask & (~used)
            if not mask.any():
                continue
            take_n = min(int(quotas.get(role, 0)), int(mask.sum()))
            if take_n <= 0:
                continue
            chosen = score.loc[mask].sort_values(ascending=False, kind="mergesort").head(take_n).index
            selected_parts.append(pd.Index(chosen, dtype=object))
            used.loc[chosen] = True
        selected = pd.Index([], dtype=object)
        if selected_parts:
            selected = pd.Index(np.concatenate([part.to_numpy(dtype=object) for part in selected_parts]), dtype=object)
        if len(selected) < n_work:
            leftover = score.loc[~used].sort_values(ascending=False, kind="mergesort").head(n_work - len(selected)).index
            selected = pd.Index(np.concatenate([selected.to_numpy(dtype=object), leftover.to_numpy(dtype=object)]), dtype=object)
        if len(selected) > n_work:
            selected = selected[:n_work]
        return pd.Index(index[index.isin(selected)], dtype=object)
    if mode_norm in {"output_weighted", "output_share", "output"}:
        scores = x_vals.abs()
    elif mode_norm in {"positive_output", "output_positive"}:
        scores = x_vals.clip(lower=0.0)
    elif mode_norm in {"absolute_output", "abs_output"}:
        scores = x_vals.abs()
    else:
        scores = pd.Series(np.arange(len(index), 0, -1, dtype=float), index=index)
    selected = scores.sort_values(ascending=False, kind="mergesort").head(n_work).index
    return pd.Index(selected, dtype=object)


def _spectral_radius_from_sparse(values: csr_matrix, *, tol: float = 1e-10) -> float:
    n = int(values.shape[0])
    if n == 0:
        return float("nan")
    nnz = int(values.nnz)
    density = float(nnz / (n * n)) if n else float("nan")
    sparse_threshold = 256
    if n >= sparse_threshold:
        try:
            vals = eigs(
                values,
                k=1,
                which="LM",
                tol=1e-6,
                maxiter=max(4000, 10 * n),
                return_eigenvectors=False,
            )
            rho = float(np.max(np.abs(vals)))
            return rho
        except ArpackNoConvergence as exc:
            vals = np.asarray(exc.eigenvalues) if exc.eigenvalues is not None else np.array([])
            if vals.size:
                rho = float(np.max(np.abs(vals)))
                return rho
        except Exception:
            pass
    vals = linalg.eigvals(values.toarray())
    rho = float(np.max(np.abs(vals))) if vals.size else float("nan")
    return rho


def _linear_operator_from_sparse(values: csr_matrix) -> LinearOperator:
    n = int(values.shape[0])
    return LinearOperator(
        shape=(n, n),
        matvec=lambda x: values.dot(x),
        rmatvec=lambda x: values.T.dot(x),
        dtype=values.dtype,
    )


def _dominant_block_indices(values: csr_matrix, tol: float) -> tuple[np.ndarray, dict[str, Any]]:
    adjacency = values.copy().tocsr()
    if adjacency.nnz:
        adjacency.data = (adjacency.data > tol).astype(np.int8, copy=False)
        adjacency.eliminate_zeros()
    n_comp, labels = connected_components(adjacency, directed=True, connection="strong")
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
        block = values[idx][:, idx]
        rho = _spectral_radius_from_sparse(block)
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
    matrix: Any,
    *,
    return_metadata: bool = False,
    tol: float = 1e-10,
    index: pd.Index | None = None,
) -> tuple[pd.Series, pd.Series, float] | tuple[pd.Series, pd.Series, float, dict[str, Any]]:
    values, row_index, col_index = _matrix_to_csr_and_labels(matrix, index=index)
    if values.nnz and np.min(values.data) < -tol:
        raise ValueError("PF diagnostics require a nonnegative matrix.")

    block_idx, metadata = _dominant_block_indices(values, tol=tol)
    block = values[block_idx][:, block_idx]
    n_block = int(block.shape[0])
    if n_block >= 256:
        try:
            _diag_print(
                f"PF eigensolve start shape={block.shape} method=sparse_eigs_matrix_free rss={_rss_mb():.1f}MB"
            )
            t0 = time.time()
            op = _linear_operator_from_sparse(block)
            op_t = _linear_operator_from_sparse(block.T.tocsr())
            eigvals_r, right_vecs = eigs(
                op,
                k=1,
                which="LM",
                tol=1e-6,
                maxiter=max(4000, 10 * n_block),
            )
            eigvals_l, left_vecs = eigs(
                op_t,
                k=1,
                which="LM",
                tol=1e-6,
                maxiter=max(4000, 10 * n_block),
            )
            eigvals = np.asarray(eigvals_r)
            _diag_print(
                f"PF eigensolve done shape={block.shape} method=sparse_eigs_matrix_free "
                f"elapsed={time.time()-t0:.1f}s rss={_rss_mb():.1f}MB"
            )
        except ArpackNoConvergence as exc:
            _diag_print(
                f"PF eigensolve partial convergence shape={block.shape} method=sparse_eigs_matrix_free "
                f"rss={_rss_mb():.1f}MB"
            )
            eigvals = np.asarray(exc.eigenvalues) if exc.eigenvalues is not None else np.array([])
            right_vecs = (
                np.asarray(exc.eigenvectors)
                if exc.eigenvectors is not None
                else np.empty((n_block, 0))
            )
            left_vecs = np.empty((n_block, 0))
            if eigvals.size == 0 or right_vecs.size == 0:
                eigvals, left_vecs, right_vecs = linalg.eig(block.toarray(), left=True, right=True)
        except Exception:
            eigvals, left_vecs, right_vecs = linalg.eig(block.toarray(), left=True, right=True)
    else:
        eigvals, left_vecs, right_vecs = linalg.eig(block.toarray(), left=True, right=True)

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

    u_full = pd.Series(0.0, index=row_index, dtype=float)
    v_full = pd.Series(0.0, index=col_index, dtype=float)
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
    _diag_print(f"pf sensitivities start shape={matrix.shape} rss={_rss_mb():.1f}MB")
    t0 = time.time()
    u, v, rho, meta = calc_pf_eigenvectors(matrix, return_metadata=True)
    if strict and (meta["degenerate_dominant_eigenvalue"] or meta["complex_dominant"]):
        raise ValueError(f"PF sensitivities ill-defined for degenerate/complex dominant eigenvalue: {meta}")
    outer = np.outer(u.values, v.values)
    sensitivities = pd.DataFrame(outer, index=matrix.index, columns=matrix.columns)
    _diag_print(
        f"pf sensitivities done shape={matrix.shape} rho={rho:.6g} "
        f"elapsed={time.time()-t0:.1f}s rss={_rss_mb():.1f}MB"
    )
    return sensitivities


def aggregate_diagnostics_from_pf_vectors(
    pf_left: pd.Series,
    pf_right: pd.Series,
    E: pd.DataFrame,
    S: pd.DataFrame,
    eco_slack: pd.Series,
    soc_slack: pd.Series,
    epsilon: float = 1e-6,
) -> pd.Series:
    """Compute the composite diagnostic score without materializing the PF outer product."""
    right = pd.to_numeric(pf_right.reindex(pf_right.index), errors="coerce").fillna(0.0).astype(float)
    left_sum = float(pd.to_numeric(pf_left.reindex(pf_left.index), errors="coerce").fillna(0.0).sum())
    column_sensitivity = pd.Series(right.to_numpy(dtype=float) * left_sum, index=right.index, dtype=float)
    pressure = pd.Series(0.0, index=column_sensitivity.index)
    if not eco_slack.empty:
        eco_denom = eco_slack.copy().astype(float)
        eco_denom = eco_denom.where(eco_denom > epsilon, np.nan)
        eco_contrib = E.div(eco_denom, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0)
        pressure = pressure.add(eco_contrib, fill_value=0.0)
    if not soc_slack.empty:
        soc_denom = soc_slack.copy().astype(float)
        soc_denom = soc_denom.where(soc_denom > epsilon, np.nan)
        soc_contrib = S.div(soc_denom, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0)
        pressure = pressure.add(soc_contrib, fill_value=0.0)
    return column_sensitivity.add(pressure, fill_value=0.0)


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



def _align_lp_A_sparse(A: Any, x_index: pd.Index) -> csr_matrix:
    """Return technical matrix as CSR in exactly ``x_index`` order.

    ``load_mrio_panel(dataset='exiobase')`` now keeps EXIOBASE ``A`` sparse to
    avoid materializing a 9800 x 9800 dense DataFrame.  The LP code therefore
    must not assume pandas ``.index``/``.values`` attributes.  If sparse matrix
    labels are present, they must already match ``x_index``; if labels are
    absent, the matrix order is treated as the caller-provided ``x`` order.
    Dense/DataFrame inputs retain the old reindex-and-fill behavior.
    """

    idx = pd.Index(x_index)
    n = int(len(idx))
    if issparse(A):
        mat = csr_matrix(A, dtype=float)
        if mat.shape != (n, n):
            raise ValueError(
                "Technical coefficient matrix must be square and match x: "
                f"A_shape={mat.shape}, len_x={n}."
            )
        for attr_name, axis_name in [("index", "row"), ("columns", "column")]:
            labels = getattr(A, attr_name, None)
            if labels is None:
                continue
            labels_idx = pd.Index(labels)
            if len(labels_idx) != n:
                raise ValueError(
                    f"Sparse A {axis_name} labels have length {len(labels_idx)}, expected {n}."
                )
            if not labels_idx.equals(idx):
                raise ValueError(
                    f"Sparse A {axis_name} labels do not match x.index. "
                    "Reindex the sparse matrix before calling solve_safe_just_lp."
                )
        mat.eliminate_zeros()
        return mat

    frame = pd.DataFrame(A, copy=False)
    if frame.shape != (n, n):
        raise ValueError(
            "Technical coefficient matrix must be square and match x: "
            f"A_shape={frame.shape}, len_x={n}."
        )
    if not pd.Index(frame.index).equals(idx) or not pd.Index(frame.columns).equals(idx):
        frame = frame.reindex(index=idx, columns=idx).fillna(0.0)
    return csr_matrix(frame.to_numpy(dtype=float, copy=False))

def solve_safe_just_lp(
    x: pd.Series,
    y: pd.Series,
    A: Any,
    E: pd.DataFrame,
    S: pd.DataFrame,
    e_bar: pd.Series,
    s_floor: pd.Series,
    *,
    weight_delta_plus: float = 1.0,
    weight_delta_minus: float = 1.0,
    allow_violation: bool = False,
    violation_penalty: float = 1e6,
    working_set_n: int | None = None,
    working_set_mode: str = "output_weighted",
    working_set_role_masks: Mapping[str, pd.Series] | None = None,
    working_set_role_mix: Mapping[str, float] | None = None,
) -> SafeJustResult:
    x_index = pd.Index(x.index)
    n = int(len(x_index))
    A_sparse_full = _align_lp_A_sparse(A, x_index)
    x = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0).astype(float).reindex(x_index).fillna(0.0)
    y = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0.0).astype(float).reindex(x_index).fillna(0.0)
    x_base = x.copy()
    active_index = _select_lp_working_set_index(
        x_base,
        working_set_n=working_set_n,
        mode=working_set_mode,
        role_masks=working_set_role_masks,
        role_mix=working_set_role_mix,
    )
    active_pos = np.asarray(x.index.get_indexer(active_index), dtype=int)
    use_working_set = len(active_index) < n
    _diag_print(
        f"safe-just LP start n={n} me={int(E.shape[0])} ms={int(S.shape[0])} "
        f"allow_violation={bool(allow_violation)} working_set={len(active_index) if use_working_set else n} "
        f"mode={str(working_set_mode)} rss={_rss_mb():.1f}MB"
    )
    t0 = time.time()
    if not use_working_set:
        A_sparse = A_sparse_full
        A_sparse_full = None
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
        _diag_print(f"safe-just LP building sparse matrices n_vars={n_vars} rss={_rss_mb():.1f}MB")
        # Inequality constraints
        # 1) x_new - A x_new >= y  <=>  (A - I) x_new <= -y
        ineq1 = hstack([A_sparse - eye(n, format="csr"), csr_matrix((n, n_vars - n))], format="csr")
        b1 = -y.reindex(x.index).values
        # 2) Ecological ceilings: E x_new <= e_bar
        eco_block = csr_matrix(E.reindex(columns=x.index).values)
        if allow_violation:
            eco_slack = -eye(me, format="csr")
            eco_tail = csr_matrix((me, ms))
            ineq2 = hstack([eco_block, csr_matrix((me, 2 * n)), eco_slack, eco_tail], format="csr")
        else:
            ineq2 = hstack([eco_block, csr_matrix((me, n_vars - n))], format="csr")
        b2 = e_bar.reindex(E.index).values
        # 3) Social floors: -S x_new <= -s_floor
        soc_block = csr_matrix(-S.reindex(columns=x.index).values)
        if allow_violation:
            soc_tail = -eye(ms, format="csr")
            ineq3 = hstack([soc_block, csr_matrix((ms, 2 * n + me)), soc_tail], format="csr")
        else:
            ineq3 = hstack([soc_block, csr_matrix((ms, n_vars - n))], format="csr")
        b3 = -s_floor.reindex(S.index).values
        A_ub = vstack([ineq1, ineq2, ineq3], format="csr")
        b_ub = np.concatenate([b1, b2, b3])
        # Equality constraints to link delta vars
        A_eq = hstack(
            [eye(n, format="csr"), -eye(n, format="csr"), eye(n, format="csr"), csr_matrix((n, n_vars - 3 * n))],
            format="csr",
        )
        b_eq = x.values
        bounds = [(0, None)] * n_vars
        # Release construction intermediates before the solve.
        del A_sparse, eco_block, soc_block, ineq1, ineq2, ineq3
        gc.collect()
        t_lp = time.time()
        _diag_print(
            f"safe-just LP calling linprog A_ub={A_ub.shape} A_eq={A_eq.shape} "
            f"nnz_ub={int(A_ub.nnz)} nnz_eq={int(A_eq.nnz)} rss={_rss_mb():.1f}MB"
        )
        res = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        _diag_print(
            f"safe-just LP linprog returned status={res.status} success={bool(res.success)} "
            f"elapsed={time.time()-t_lp:.1f}s total_elapsed={time.time()-t0:.1f}s rss={_rss_mb():.1f}MB"
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
    else:
        # Working-set approximation: keep the full basis fixed, but let only a screened subset
        # of sectors move. This preserves full id order while shrinking the LP substantially.
        active_pos = np.asarray(active_pos, dtype=int)
        k = int(len(active_pos))
        if k <= 0:
            raise ValueError("Working-set LP requires at least one active sector.")
        A_sparse = A_sparse_full
        A_sparse_full = None
        E_sparse = csr_matrix(E.reindex(columns=x.index).values)
        S_sparse = csr_matrix(S.reindex(columns=x.index).values)
        base_prod_rhs = -y.reindex(x.index).fillna(0.0).to_numpy(dtype=float) - (
            A_sparse - eye(n, format="csr")
        ).dot(x_base.to_numpy(dtype=float))
        base_eco_rhs = e_bar.reindex(E.index).fillna(0.0).to_numpy(dtype=float) - E_sparse.dot(
            x_base.to_numpy(dtype=float)
        )
        base_soc_rhs = -s_floor.reindex(S.index).fillna(0.0).to_numpy(dtype=float) + S_sparse.dot(
            x_base.to_numpy(dtype=float)
        )
        active_A = A_sparse[:, active_pos] - eye(n, format="csr")[:, active_pos]
        active_E = E_sparse[:, active_pos]
        active_S = S_sparse[:, active_pos]
        base_objective = np.concatenate(
            [np.full(k, float(weight_delta_plus)), np.full(k, float(weight_delta_minus))]
        )
        if allow_violation:
            objective = np.concatenate(
                [base_objective, np.full(int(E.shape[0]), float(violation_penalty)), np.full(int(S.shape[0]), float(violation_penalty))]
            )
            n_vars = 2 * k + int(E.shape[0]) + int(S.shape[0])
        else:
            objective = base_objective
            n_vars = 2 * k
        _diag_print(
            f"safe-just LP working-set build k={k} n_vars={n_vars} "
            f"mode={str(working_set_mode)} rss={_rss_mb():.1f}MB"
        )
        prod_block = hstack([active_A, -active_A], format="csr")
        if allow_violation:
            prod = hstack([prod_block, csr_matrix((n, n_vars - 2 * k))], format="csr")
            eco = hstack(
                [
                    active_E,
                    -active_E,
                    -eye(int(E.shape[0]), format="csr"),
                    csr_matrix((int(E.shape[0]), int(S.shape[0]))),
                ],
                format="csr",
            )
            soc = hstack(
                [
                    -active_S,
                    active_S,
                    csr_matrix((int(S.shape[0]), int(E.shape[0]))),
                    -eye(int(S.shape[0]), format="csr"),
                ],
                format="csr",
            )
            A_ub = vstack([prod, eco, soc], format="csr")
        else:
            prod = prod_block
            eco = hstack([active_E, -active_E], format="csr")
            soc = hstack([-active_S, active_S], format="csr")
            A_ub = vstack([prod, eco, soc], format="csr")
        b_ub = np.concatenate([base_prod_rhs, base_eco_rhs, base_soc_rhs])
        bounds = [(0, None)] * n_vars
        del A_sparse, E_sparse, S_sparse, active_A, active_E, active_S, prod_block, prod, eco, soc
        gc.collect()
        t_lp = time.time()
        _diag_print(
            f"safe-just LP calling linprog working-set A_ub={A_ub.shape} nnz_ub={int(A_ub.nnz)} "
            f"rss={_rss_mb():.1f}MB"
        )
        res = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        _diag_print(
            f"safe-just LP working-set returned status={res.status} success={bool(res.success)} "
            f"elapsed={time.time()-t_lp:.1f}s total_elapsed={time.time()-t0:.1f}s rss={_rss_mb():.1f}MB"
        )
        if res.status != 0 or not res.success:
            raise RuntimeError(f"Safe-and-just LP failed: {res.message}")
        delta_plus = np.zeros(n, dtype=float)
        delta_minus = np.zeros(n, dtype=float)
        adj_plus = res.x[:k]
        adj_minus = res.x[k:2 * k]
        delta_plus[active_pos] = adj_plus
        delta_minus[active_pos] = adj_minus
        x_new = x_base.to_numpy(dtype=float).copy()
        x_new[active_pos] += adj_plus - adj_minus
        eco_violation = None
        soc_violation = None
        if allow_violation:
            eco_violation = pd.Series(
                res.x[2 * k : 2 * k + int(E.shape[0])], index=E.index, dtype=float
            )
            soc_violation = pd.Series(
                res.x[2 * k + int(E.shape[0]) : 2 * k + int(E.shape[0]) + int(S.shape[0])],
                index=S.index,
                dtype=float,
            )
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
        eco_denom = eco_slack.copy().astype(float)
        eco_denom = eco_denom.where(eco_denom > epsilon, np.nan)
        eco_contrib = E.div(eco_denom, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0)
        pressure = pressure.add(eco_contrib, fill_value=0.0)
    if not soc_slack.empty:
        soc_denom = soc_slack.copy().astype(float)
        soc_denom = soc_denom.where(soc_denom > epsilon, np.nan)
        soc_contrib = S.div(soc_denom, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=0)
        pressure = pressure.add(soc_contrib, fill_value=0.0)
    return column_sensitivity + pressure

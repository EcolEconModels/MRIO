from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BalanceStats:
    iterations: int
    max_abs_row_gap: float
    max_abs_col_gap: float


def ras_balance(
    M0: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, BalanceStats]:
    """
    Standard (unconstrained) RAS / IPFP for nonnegative matrices.

    Returns balanced matrix and balance stats.
    """
    M0 = np.asarray(M0, dtype="float64")
    row_targets = np.asarray(row_targets, dtype="float64")
    col_targets = np.asarray(col_targets, dtype="float64")

    if (M0 < 0).any():
        raise ValueError("RAS requires a nonnegative initial matrix.")
    if (row_targets < 0).any() or (col_targets < 0).any():
        raise ValueError("RAS requires nonnegative targets.")

    if abs(row_targets.sum() - col_targets.sum()) > 1e-8:
        raise ValueError("Row and column targets are inconsistent (different totals).")

    M = M0.copy()
    r = row_targets
    c = col_targets

    for it in range(1, max_iter + 1):
        row_sum = M.sum(axis=1)
        row_factors = np.divide(r, row_sum, out=np.ones_like(r), where=row_sum != 0)
        M *= row_factors[:, None]

        col_sum = M.sum(axis=0)
        col_factors = np.divide(c, col_sum, out=np.ones_like(c), where=col_sum != 0)
        M *= col_factors[None, :]

        row_gap = float(np.max(np.abs(M.sum(axis=1) - r)))
        col_gap = float(np.max(np.abs(M.sum(axis=0) - c)))
        if row_gap <= tol and col_gap <= tol:
            return M, BalanceStats(it, row_gap, col_gap)

    row_gap = float(np.max(np.abs(M.sum(axis=1) - r)))
    col_gap = float(np.max(np.abs(M.sum(axis=0) - c)))
    return M, BalanceStats(max_iter, row_gap, col_gap)


def gras_balance(
    M0: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, BalanceStats]:
    """
    Generalized RAS (GRAS) for matrices with possible negative entries.

    Implements the iterative scheme based on a quadratic update for row/column
    multipliers with the decomposition M0 = P - N, where P=max(M0,0), N=max(-M0,0).

    Returns balanced matrix and balance stats.
    """
    M0 = np.asarray(M0, dtype="float64")
    row_targets = np.asarray(row_targets, dtype="float64")
    col_targets = np.asarray(col_targets, dtype="float64")

    if abs(row_targets.sum() - col_targets.sum()) > 1e-8:
        raise ValueError("Row and column targets are inconsistent (different totals).")

    P = np.clip(M0, 0.0, None)
    N = np.clip(-M0, 0.0, None)

    a = np.ones(P.shape[0], dtype="float64")
    b = np.ones(P.shape[1], dtype="float64")
    r = row_targets
    c = col_targets

    def _update_multiplier(A: np.ndarray, B: np.ndarray, target: np.ndarray) -> np.ndarray:
        out = np.ones_like(target)

        # Case 1: A > 0: quadratic solution
        mask_A = A > 0
        if mask_A.any():
            disc = target[mask_A] ** 2 + 4.0 * A[mask_A] * B[mask_A]
            disc = np.maximum(disc, 0.0)
            out[mask_A] = (target[mask_A] + np.sqrt(disc)) / (2.0 * A[mask_A])

        # Case 2: A == 0, B > 0: target must be negative
        mask_A0_B = (A == 0) & (B > 0)
        if mask_A0_B.any():
            if (target[mask_A0_B] >= 0).any():
                raise ValueError("GRAS infeasible: A=0,B>0 requires negative target.")
            out[mask_A0_B] = B[mask_A0_B] / (-target[mask_A0_B])

        # Case 3: A > 0, B == 0: linear scaling requires nonnegative target
        mask_B0_A = (B == 0) & (A > 0)
        if mask_B0_A.any():
            if (target[mask_B0_A] < 0).any():
                raise ValueError("GRAS infeasible: B=0,A>0 requires nonnegative target.")
            out[mask_B0_A] = np.divide(target[mask_B0_A], A[mask_B0_A], out=np.ones_like(target[mask_B0_A]), where=A[mask_B0_A] != 0)

        # Case 4: A==0,B==0: must have target==0
        mask_A0_B0 = (A == 0) & (B == 0)
        if mask_A0_B0.any():
            if (np.abs(target[mask_A0_B0]) > 1e-12).any():
                raise ValueError("GRAS infeasible: A=0,B=0 requires zero target.")
            out[mask_A0_B0] = 1.0

        out = np.clip(out, 1e-30, 1e30)
        return out

    for it in range(1, max_iter + 1):
        # Update a given b
        Pb = P * b[None, :]
        N_over_b = np.divide(N, b[None, :], out=np.zeros_like(N), where=b[None, :] != 0)
        A = Pb.sum(axis=1)
        B = N_over_b.sum(axis=1)
        a = _update_multiplier(A, B, r)

        # Update b given a
        aP = a[:, None] * P
        N_over_a = np.divide(N, a[:, None], out=np.zeros_like(N), where=a[:, None] != 0)
        A2 = aP.sum(axis=0)
        B2 = N_over_a.sum(axis=0)
        b = _update_multiplier(A2, B2, c)

        ab = a[:, None] * b[None, :]
        M = ab * P - np.divide(N, ab, out=np.zeros_like(N), where=ab != 0)

        row_gap = float(np.max(np.abs(M.sum(axis=1) - r)))
        col_gap = float(np.max(np.abs(M.sum(axis=0) - c)))
        if row_gap <= tol and col_gap <= tol:
            return M, BalanceStats(it, row_gap, col_gap)

    row_gap = float(np.max(np.abs(M.sum(axis=1) - r)))
    col_gap = float(np.max(np.abs(M.sum(axis=0) - c)))
    return M, BalanceStats(max_iter, row_gap, col_gap)


def balance_biproportional(
    M0: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    max_iter: int = 50_000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, BalanceStats, str]:
    """
    Choose RAS if feasible, else GRAS, based on the sign pattern of M0 and targets.
    """
    M0 = np.asarray(M0, dtype="float64")
    row_targets = np.asarray(row_targets, dtype="float64")
    col_targets = np.asarray(col_targets, dtype="float64")

    use_ras = (M0 >= 0).all() and (row_targets >= 0).all() and (col_targets >= 0).all()
    if use_ras:
        M, stats = ras_balance(M0, row_targets, col_targets, max_iter=max_iter, tol=tol)
        return M, stats, "RAS"
    M, stats = gras_balance(M0, row_targets, col_targets, max_iter=max_iter, tol=tol)
    return M, stats, "GRAS"


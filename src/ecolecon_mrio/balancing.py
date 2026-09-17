"""Generic matrix-balancing algorithms for MRIO disaggregation and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

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



# MULTI_MARGIN_IPFP_V1
@dataclass(frozen=True)
class IPFPMargin:
    """One partition-margin family for non-negative iterative proportional fitting.

    ``groups`` is an integer array with the same shape as the balanced matrix.
    Every matrix cell belongs to exactly one group in this family; ``targets[g]``
    is the required sum of all cells whose group id equals ``g``.
    """

    name: str
    groups: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class IPFPStats:
    """Convergence diagnostics for multi-margin IPFP."""

    iterations: int
    max_normalized_gap: float
    max_abs_gap_by_family: dict[str, float]


def ipfp_balance_partitions(
    M0: np.ndarray,
    margin_families: Sequence[IPFPMargin],
    *,
    max_iter: int = 50_000,
    rtol: float = 1e-12,
    atol: float = 1e-10,
) -> tuple[np.ndarray, IPFPStats]:
    """KL/IPFP-balance a non-negative matrix to several partition margins.

    Classical two-margin RAS is the special case with one row partition and one
    column partition.  Each projection rescales every cell in a group by the ratio
    of target to current group mass; cycling the families converges to the
    information projection when the declared margins are feasible on the initial
    support.  Zero cells therefore remain structural zeros.
    """
    M = np.asarray(M0, dtype="float64").copy()
    if M.ndim != 2 or 0 in M.shape:
        raise ValueError("IPFP requires a non-empty two-dimensional matrix.")
    if not np.isfinite(M).all() or bool((M < 0.0).any()):
        raise ValueError("IPFP requires a finite non-negative initial matrix.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if (
        not np.isfinite(rtol)
        or not np.isfinite(atol)
        or rtol < 0.0
        or atol < 0.0
        or (rtol == 0.0 and atol == 0.0)
    ):
        raise ValueError("IPFP tolerances must be finite, non-negative, and not both zero.")
    if not margin_families:
        raise ValueError("IPFP requires at least one margin family.")

    # Validate every partition independently, then require all families to conserve
    # the same total mass before any iterative projection is attempted.
    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen_names: set[str] = set()
    common_total: float | None = None
    for family in margin_families:
        name = str(family.name).strip()
        if not name or name in seen_names:
            raise ValueError(f"IPFP margin names must be non-empty and unique: {name!r}.")
        seen_names.add(name)
        groups = np.asarray(family.groups)
        targets = np.asarray(family.targets, dtype="float64").reshape(-1)
        if groups.shape != M.shape or not np.issubdtype(groups.dtype, np.integer):
            raise ValueError(
                f"IPFP margin {name!r} groups must be an integer array with shape {M.shape}."
            )
        groups = groups.astype(np.int64, copy=False)
        if targets.size == 0 or not np.isfinite(targets).all() or bool((targets < 0.0).any()):
            raise ValueError(f"IPFP margin {name!r} targets must be finite and non-negative.")
        if int(groups.min()) < 0 or int(groups.max()) >= targets.size:
            raise ValueError(
                f"IPFP margin {name!r} group ids must lie in [0, {targets.size - 1}]."
            )
        total = float(targets.sum())
        if common_total is None:
            common_total = total
        else:
            scale = max(1.0, abs(common_total), abs(total))
            if abs(total - common_total) > atol + rtol * scale:
                raise ValueError(
                    "IPFP margin totals are inconsistent: "
                    f"reference={common_total:.12g}, {name}={total:.12g}."
                )
        prepared.append((name, groups, targets))

    def family_gap(
        groups: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        current = np.bincount(
            groups.ravel(), weights=M.ravel(), minlength=targets.size
        ).astype("float64")
        abs_gap = np.abs(current - targets)
        tolerance = atol + rtol * np.maximum(np.abs(current), np.abs(targets))
        normalized = np.divide(
            abs_gap,
            tolerance,
            out=np.full_like(abs_gap, np.inf),
            where=tolerance > 0.0,
        )
        normalized[(tolerance == 0.0) & (abs_gap == 0.0)] = 0.0
        return current, float(np.max(abs_gap, initial=0.0)), float(
            np.max(normalized, initial=0.0)
        )

    worst_normalized = float("inf")
    abs_by_family: dict[str, float] = {}
    # Each family update is one KL/I-projection: proportional rescaling preserves
    # relative cell structure inside the group while imposing that group's margin.
    for iteration in range(1, max_iter + 1):
        for name, groups, targets in prepared:
            current, _, _ = family_gap(groups, targets)
            unsupported = (targets > 0.0) & (current == 0.0)
            if bool(np.any(unsupported)):
                group = int(np.flatnonzero(unsupported)[0])
                raise ValueError(
                    "IPFP infeasible on initial/current structural support: "
                    f"margin={name!r}, group={group}, target={targets[group]:.12g}."
                )
            factors = np.divide(
                targets,
                current,
                out=np.ones_like(targets),
                where=current != 0.0,
            )
            factors[(targets == 0.0) & (current != 0.0)] = 0.0
            M *= factors[groups]

        # A full cycle is accepted only when every family is simultaneously within
        # its elementwise generated-identity tolerance.
        worst_normalized = 0.0
        abs_by_family = {}
        for name, groups, targets in prepared:
            _, max_abs, max_norm = family_gap(groups, targets)
            abs_by_family[name] = max_abs
            worst_normalized = max(worst_normalized, max_norm)
        if worst_normalized <= 1.0:
            return M, IPFPStats(
                iterations=iteration,
                max_normalized_gap=worst_normalized,
                max_abs_gap_by_family=dict(abs_by_family),
            )

    raise ValueError(
        "IPFP did not converge within max_iter: "
        f"iterations={max_iter}, max_normalized_gap={worst_normalized:.6g}, "
        f"max_abs_gap_by_family={abs_by_family!r}. The margin families or "
        "structural support may be incompatible."
    )


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
    eps = max(float(tol), 1e-12)
    M = M0.copy()

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
            if (target[mask_A0_B] > eps).any():
                raise ValueError("GRAS infeasible: A=0,B>0 requires negative target.")
            out[mask_A0_B] = B[mask_A0_B] / (-target[mask_A0_B])

        # Case 3: A > 0, B == 0: linear scaling requires nonnegative target
        mask_B0_A = (B == 0) & (A > 0)
        if mask_B0_A.any():
            if (target[mask_B0_A] < -eps).any():
                raise ValueError("GRAS infeasible: B=0,A>0 requires nonnegative target.")
            out[mask_B0_A] = np.divide(target[mask_B0_A], A[mask_B0_A], out=np.ones_like(target[mask_B0_A]), where=A[mask_B0_A] != 0)

        # Case 4: A==0,B==0: must have target==0
        mask_A0_B0 = (A == 0) & (B == 0)
        if mask_A0_B0.any():
            if (np.abs(target[mask_A0_B0]) > eps).any():
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


def balance_biproportional_masked(
    M0: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    adjustable: np.ndarray,
    row_labels: list[str] | tuple[str, ...] | None = None,
    col_labels: list[str] | tuple[str, ...] | None = None,
    max_iter: int = 50_000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, BalanceStats, str]:
    """
    Constrained biproportional balancing with a fixed-cell mask.

    Parameters
    ----------
    M0
        Initial matrix.
    row_targets, col_targets
        Desired row/column sums for the full matrix (fixed + adjustable).
    adjustable
        Boolean mask of the same shape as M0; True entries are allowed to change.
        False entries remain fixed at their initial values in M0.

    Notes
    -----
    This is implemented by decomposing M0 into:
      M0 = M_fixed + M_adj

    and balancing M_adj to residual targets:
      row_targets_adj = row_targets - M_fixed.sum(axis=1)
      col_targets_adj = col_targets - M_fixed.sum(axis=0)

    The final balanced matrix is:
      M = M_fixed + M_adj_balanced
    """
    M0 = np.asarray(M0, dtype="float64")
    row_targets = np.asarray(row_targets, dtype="float64")
    col_targets = np.asarray(col_targets, dtype="float64")
    adjustable = np.asarray(adjustable, dtype=bool)
    eps = max(float(tol), 1e-12)

    if M0.shape != adjustable.shape:
        raise ValueError(f"adjustable mask must have same shape as M0, got {adjustable.shape} vs {M0.shape}.")

    fixed = np.where(adjustable, 0.0, M0)
    adj0 = np.where(adjustable, M0, 0.0)

    row_resid = row_targets - fixed.sum(axis=1)
    col_resid = col_targets - fixed.sum(axis=0)
    row_noise = np.maximum(eps, 1e-12 * np.maximum(1.0, np.abs(row_targets)))
    col_noise = np.maximum(eps, 1e-12 * np.maximum(1.0, np.abs(col_targets)))

    # Treat machine-noise residuals as exact zeros before feasibility checks.
    row_resid = np.where(np.abs(row_resid) <= row_noise, 0.0, row_resid)
    col_resid = np.where(np.abs(col_resid) <= col_noise, 0.0, col_resid)

    if abs(row_resid.sum() - col_resid.sum()) > 1e-8:
        raise ValueError("Row/column residual targets are inconsistent (different totals).")

    # If the adjustable block is nonnegative, residual targets must be nonnegative (within tolerance).
    if (adj0 >= 0).all() and ((row_resid < -row_noise).any() or (col_resid < -col_noise).any()):
        details: list[str] = []
        if (row_resid < -row_noise).any():
            row_idx = int(np.argmin(row_resid))
            row_label = row_labels[row_idx] if row_labels is not None else str(row_idx)
            details.append(
                "row="
                f"{row_label} target={float(row_targets[row_idx]):.10f} "
                f"fixed={float(fixed.sum(axis=1)[row_idx]):.10f} resid={float(row_resid[row_idx]):.10f}"
            )
        if (col_resid < -col_noise).any():
            col_idx = int(np.argmin(col_resid))
            col_label = col_labels[col_idx] if col_labels is not None else str(col_idx)
            details.append(
                "col="
                f"{col_label} target={float(col_targets[col_idx]):.10f} "
                f"fixed={float(fixed.sum(axis=0)[col_idx]):.10f} resid={float(col_resid[col_idx]):.10f}"
            )
        raise ValueError(
            "Masked balancing infeasible: fixed cells exceed targets, producing negative residual targets. "
            + "; ".join(details)
        )

    M_adj_bal, stats, method = balance_biproportional(
        adj0, row_targets=row_resid, col_targets=col_resid, max_iter=max_iter, tol=tol
    )

    M = fixed + M_adj_bal
    return M, stats, method

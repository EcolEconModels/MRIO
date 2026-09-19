"""Generic matrix-balancing algorithms for MRIO disaggregation and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy import sparse
from scipy.optimize import minimize, root


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



# IPFP_PARTITION_BOUNDS_V2
@dataclass(frozen=True)
class IPFPMargin:
    """One partition constraint for non-negative KL reconciliation.

    ``groups`` has the same shape as the balanced matrix and assigns every cell to
    exactly one integer group. ``targets[g]`` is either the exact group mass
    (``relation='equal'``) or an upper bound (``relation='upper'``).  The default
    keeps the pre-existing equality-only API fully backward compatible.
    """

    name: str
    groups: np.ndarray
    targets: np.ndarray
    relation: str = "equal"


@dataclass(frozen=True)
class IPFPStats:
    """Convergence diagnostics for partition-constrained KL reconciliation."""

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
    """KL-project a non-negative matrix onto partition equalities/upper bounds.

    Equality-only calls reduce to the repository's established multi-margin IPFP
    contract; ordinary row/column RAS is the two-equality-family special case.
    Upper-bound families are true convex KL constraints, not post-hoc clipping.
    Multiplicative Dykstra correction factors are retained per partition family so
    cycling equality and inequality projections converges to the joint information
    projection.  Structural zeros in ``M0`` remain structural zeros throughout.
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

    prepared: list[tuple[str, np.ndarray, np.ndarray, str, np.ndarray]] = []
    seen_names: set[str] = set()
    common_equal_total: float | None = None
    for family in margin_families:
        name = str(family.name).strip()
        relation = str(family.relation).strip().lower()
        if not name or name in seen_names:
            raise ValueError(f"IPFP margin names must be non-empty and unique: {name!r}.")
        if relation not in {"equal", "upper"}:
            raise ValueError(
                f"IPFP margin {name!r} relation must be 'equal' or 'upper', got {relation!r}."
            )
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
        if relation == "equal":
            total = float(targets.sum())
            if common_equal_total is None:
                common_equal_total = total
            else:
                scale = max(1.0, abs(common_equal_total), abs(total))
                if abs(total - common_equal_total) > atol + rtol * scale:
                    raise ValueError(
                        "IPFP equality-margin totals are inconsistent: "
                        f"reference={common_equal_total:.12g}, {name}={total:.12g}."
                    )
        support_mass = np.bincount(
            groups.ravel(), weights=M.ravel(), minlength=targets.size
        ).astype("float64")
        unsupported = relation == "equal" and bool(
            np.any((targets > 0.0) & (support_mass == 0.0))
        )
        if unsupported:
            group = int(np.flatnonzero((targets > 0.0) & (support_mass == 0.0))[0])
            raise ValueError(
                "IPFP infeasible on initial structural support: "
                f"margin={name!r}, group={group}, target={targets[group]:.12g}."
            )
        prepared.append((name, groups, targets, relation, np.ones_like(targets)))

    def family_gap(
        groups: np.ndarray, targets: np.ndarray, relation: str
    ) -> tuple[np.ndarray, float, float]:
        current = np.bincount(
            groups.ravel(), weights=M.ravel(), minlength=targets.size
        ).astype("float64")
        gap = (
            np.abs(current - targets)
            if relation == "equal"
            else np.maximum(current - targets, 0.0)
        )
        tolerance = atol + rtol * np.maximum(np.abs(current), np.abs(targets))
        normalized = np.divide(
            gap,
            tolerance,
            out=np.full_like(gap, np.inf),
            where=tolerance > 0.0,
        )
        normalized[(tolerance == 0.0) & (gap == 0.0)] = 0.0
        return current, float(np.max(gap, initial=0.0)), float(
            np.max(normalized, initial=0.0)
        )

    worst_normalized = float("inf")
    abs_by_family: dict[str, float] = {}
    for iteration in range(1, max_iter + 1):
        for position, (name, groups, targets, relation, correction) in enumerate(prepared):
            corrected = M * correction[groups]
            current = np.bincount(
                groups.ravel(), weights=corrected.ravel(), minlength=targets.size
            ).astype("float64")
            if relation == "equal":
                unsupported = (targets > 0.0) & (current == 0.0)
                if bool(np.any(unsupported)):
                    group = int(np.flatnonzero(unsupported)[0])
                    raise ValueError(
                        "IPFP infeasible on current structural support: "
                        f"margin={name!r}, group={group}, target={targets[group]:.12g}."
                    )
                factors = np.divide(
                    targets,
                    current,
                    out=np.ones_like(targets),
                    where=current != 0.0,
                )
                factors[(targets == 0.0) & (current != 0.0)] = 0.0
            else:
                factors = np.ones_like(targets)
                active = current > targets
                factors[active] = np.divide(
                    targets[active],
                    current[active],
                    out=np.zeros_like(targets[active]),
                    where=current[active] != 0.0,
                )
            M = corrected * factors[groups]
            next_correction = np.ones_like(correction)
            positive = factors > 0.0
            next_correction[positive] = 1.0 / factors[positive]
            # A zero target permanently removes that partition's support. If a
            # later equality needs positive mass there, the support check fails.
            prepared[position] = (
                name,
                groups,
                targets,
                relation,
                next_correction,
            )

        worst_normalized = 0.0
        abs_by_family = {}
        for name, groups, targets, relation, _correction in prepared:
            _, max_abs, max_norm = family_gap(groups, targets, relation)
            abs_by_family[name] = max_abs
            worst_normalized = max(worst_normalized, max_norm)
        if worst_normalized <= 1.0:
            return M, IPFPStats(
                iterations=iteration,
                max_normalized_gap=worst_normalized,
                max_abs_gap_by_family=dict(abs_by_family),
            )

    raise ValueError(
        "IPFP/KL-Dykstra did not converge within max_iter: "
        f"iterations={max_iter}, max_normalized_gap={worst_normalized:.6g}, "
        f"max_abs_gap_by_family={abs_by_family!r}. Equality margins, upper "
        "bounds, or structural support may be incompatible."
    )



# KL_LINEAR_EQUALITY_RECONCILIATION_V1
@dataclass(frozen=True)
class KLLinearStats:
    """Diagnostics for sparse separable reconciliation on linear equalities."""

    iterations: int
    max_normalized_gap: float
    max_abs_gap: float


# SEPARABLE_LINEAR_EQUALITY_RECONCILIATION_V2
# SEPARABLE_MONETARY_SCALE_EQUIVARIANCE_V3
def balance_separable_linear_equalities(
    prior: np.ndarray,
    constraint_matrix: object,
    targets: np.ndarray,
    *,
    signed_quadratic_mask: np.ndarray | None = None,
    signed_quadratic_scale: np.ndarray | None = None,
    equality_conditioning_scale: np.ndarray | None = None,
    equality_tolerance_scale: np.ndarray | None = None,
    max_iter: int = 50_000,
    rtol: float = 1e-12,
    atol: float = 1e-10,
) -> tuple[np.ndarray, KLLinearStats]:
    """Reconcile KL and signed-quadratic variables subject to ``A @ x = b``.

    Entries not selected by ``signed_quadratic_mask`` use the non-negative
    information-projection penalty

        x log(x/q) - x + q,

    so a zero KL prior remains a structural zero. Selected signed entries use the
    monetary-scale-compatible local quadratic form of KL,

        (x-q)^2 / (2 s),

    where ``s`` has the same monetary unit as ``x`` and ``q``. A zero signed scale
    fixes that variable exactly at its prior value. With ``s=abs(q)`` this is the
    second-order KL curvature around the signed prior, so the mixed objective is
    homogeneous of degree one under a change of monetary unit. This permits
    intrinsically signed accounts such as inventory/valuables final demand without
    decomposing them into artificial positive/negative epsilon supports.

    The dual-family state is closed form:

        x_KL = q_KL * exp(-A_KL.T @ lambda)
        x_S  = q_S - s_S * (A_S.T @ lambda).

    ``equality_conditioning_scale`` is an optional non-negative numerical row-scale
    floor. Dividing an equality and its target by that scale changes only dual
    coordinates, never ``A @ x = b`` or the primal objective.

    ``equality_tolerance_scale`` is a separate optional non-negative reference
    magnitude for the relative feasibility tolerance. It is needed when an
    economically scaled equality such as ``inputs = x`` becomes the algebraically
    equivalent homogeneous row ``inputs - x = 0`` after ``x`` is made endogenous.
    It never changes the equality, objective, structural support, or row
    conditioning. Source-control rows should leave it at zero.
    """
    q = np.asarray(prior, dtype="float64")
    original_shape = q.shape
    flat = q.reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Linear reconciliation requires a finite non-empty prior.")

    if signed_quadratic_mask is None:
        signed_mask = np.zeros(flat.size, dtype=bool)
    else:
        signed_mask = np.asarray(signed_quadratic_mask, dtype=bool).reshape(-1)
        if signed_mask.size != flat.size:
            raise ValueError("signed_quadratic_mask must match the flattened prior size.")

    if signed_quadratic_scale is None:
        signed_scale = np.zeros(flat.size, dtype="float64")
        if bool(signed_mask.any()):
            raise ValueError(
                "signed_quadratic_scale is required when signed quadratic variables are selected."
            )
    else:
        signed_scale = np.asarray(signed_quadratic_scale, dtype="float64").reshape(-1)
        if signed_scale.size != flat.size:
            raise ValueError("signed_quadratic_scale must match the flattened prior size.")
        if not np.isfinite(signed_scale).all() or bool((signed_scale < 0.0).any()):
            raise ValueError("signed_quadratic_scale must be finite and non-negative.")
        if bool((signed_scale[~signed_mask] != 0.0).any()):
            raise ValueError("signed_quadratic_scale must be zero outside signed variables.")

    kl_mask = ~signed_mask
    if bool((flat[kl_mask] < 0.0).any()):
        raise ValueError("KL variables require finite non-negative priors.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if (
        not np.isfinite(rtol)
        or not np.isfinite(atol)
        or rtol < 0.0
        or atol < 0.0
        or (rtol == 0.0 and atol == 0.0)
    ):
        raise ValueError(
            "Linear reconciliation tolerances must be finite, non-negative, and not both zero."
        )

    A = sparse.csr_matrix(constraint_matrix, dtype="float64")
    b = np.asarray(targets, dtype="float64").reshape(-1)
    raw_constraint_shape = A.shape
    if raw_constraint_shape is None:
        raise ValueError("Linear reconciliation constraint matrix has no shape.")
    constraint_shape = cast(tuple[int, int], raw_constraint_shape)
    rows, columns = constraint_shape
    if A.ndim != 2 or rows != b.size or columns != flat.size:
        raise ValueError(
            "Linear reconciliation dimensions disagree: "
            f"A={constraint_shape}, targets={b.shape}, prior_size={flat.size}."
        )
    if b.size == 0 or not np.isfinite(b).all() or not np.isfinite(A.data).all():
        raise ValueError("Linear reconciliation constraints and targets must be finite and non-empty.")

    if equality_conditioning_scale is None:
        conditioning_scale = np.zeros(b.size, dtype="float64")
    else:
        conditioning_scale = np.asarray(
            equality_conditioning_scale, dtype="float64"
        ).reshape(-1)
        if conditioning_scale.size != b.size:
            raise ValueError(
                "equality_conditioning_scale must match the number of equality rows."
            )
        if not np.isfinite(conditioning_scale).all() or bool(
            (conditioning_scale < 0.0).any()
        ):
            raise ValueError(
                "equality_conditioning_scale must be finite and non-negative."
            )

    # SEPARABLE_HOMOGENEOUS_TOLERANCE_REFERENCE_V4
    if equality_tolerance_scale is None:
        tolerance_scale = np.zeros(b.size, dtype="float64")
    else:
        tolerance_scale = np.asarray(
            equality_tolerance_scale, dtype="float64"
        ).reshape(-1)
        if tolerance_scale.size != b.size:
            raise ValueError(
                "equality_tolerance_scale must match the number of equality rows."
            )
        if not np.isfinite(tolerance_scale).all() or bool(
            (tolerance_scale < 0.0).any()
        ):
            raise ValueError(
                "equality_tolerance_scale must be finite and non-negative."
            )

    # Non-negative KL zero priors are structural zeros. Signed variables with zero
    # declared scale are fixed at their prior value. Remove both from the active
    # optimisation while retaining the exact contribution of fixed signed values.
    active_kl = kl_mask & (flat > 0.0)
    active_signed = signed_mask & (signed_scale > 0.0)
    active = active_kl | active_signed
    fixed = np.zeros_like(flat)
    fixed_signed = signed_mask & ~active_signed
    fixed[fixed_signed] = flat[fixed_signed]

    fixed_contribution = np.asarray(A @ fixed).reshape(-1)
    active_target = b - fixed_contribution
    active_A = A[:, active]
    active_q = flat[active]
    active_signed_mask = signed_mask[active]
    active_scale = signed_scale[active]

    row_nnz = np.diff(active_A.indptr)
    unsupported = row_nnz == 0
    unsupported_reference = np.maximum.reduce(
        [np.abs(b), np.abs(fixed_contribution), tolerance_scale, np.ones_like(b)]
    )
    unsupported_tolerance = atol + rtol * unsupported_reference
    unsupported_gap = np.abs(active_target) > unsupported_tolerance
    if bool(np.any(unsupported & unsupported_gap)):
        row = int(np.flatnonzero(unsupported & unsupported_gap)[0])
        raise ValueError(
            "Linear reconciliation is infeasible on structural support: "
            f"constraint_row={row}, target={b[row]:.12g}, "
            f"fixed_contribution={fixed_contribution[row]:.12g}."
        )

    def feasibility(
        result: np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        actual = np.asarray(A @ result).reshape(-1)
        if not np.isfinite(actual).all():
            raise ValueError("Linear reconciliation produced non-finite equality totals.")
        gap = np.abs(actual - b)
        tolerance = atol + rtol * np.maximum.reduce(
            [np.abs(actual), np.abs(b), tolerance_scale]
        )
        normalized = np.divide(
            gap,
            tolerance,
            out=np.full_like(gap, np.inf),
            where=tolerance > 0.0,
        )
        normalized[(tolerance == 0.0) & (gap == 0.0)] = 0.0
        return (
            float(np.max(normalized, initial=0.0)),
            float(np.max(gap, initial=0.0)),
            actual,
            tolerance,
            normalized,
        )

    if not bool(active.any()):
        max_norm, max_abs, _actual, _tolerance, _normalized = feasibility(fixed)
        if max_norm > 1.0:
            raise ValueError(
                "Linear reconciliation has no adjustable support but its fixed prior "
                f"violates the equalities: max_normalized_gap={max_norm:.6g}."
            )
        return fixed.reshape(original_shape), KLLinearStats(0, max_norm, max_abs)

    # Condition by the authoritative residual equality, coefficients, and any
    # caller-declared target-system monetary scale; never infer a row scale from
    # donor-prior mass. This is algebraically neutral and keeps the dual O(1),
    # including homogeneous Step-2 accounting rows whose literal target is zero.
    abs_A = active_A.copy()
    abs_A.data = np.abs(abs_A.data)
    row_coefficient_scale = np.asarray(abs_A.max(axis=1).toarray()).reshape(-1)
    row_scale = np.maximum.reduce(
        [
            np.ones_like(active_target),
            np.abs(active_target),
            row_coefficient_scale,
            conditioning_scale,
        ]
    )
    A_scaled = (sparse.diags(1.0 / row_scale) @ active_A).tocsr()
    b_scaled = active_target / row_scale

    # Homogeneous rows with an explicit tolerance reference recover the numerical
    # scale that was present before their nonzero RHS became endogenous. If that
    # requested acceptance is tighter than Krylov's historical conditioned fatol,
    # tighten only enough for those rows; hard source rows do not drive this floor.
    polish_fatol = 1e-13
    referenced_rows = tolerance_scale > 0.0
    if bool(np.any(referenced_rows)):
        referenced_floor = (
            atol
            + rtol
            * np.maximum(
                np.abs(b[referenced_rows]), tolerance_scale[referenced_rows]
            )
        ) / row_scale[referenced_rows]
        positive_floor = referenced_floor[referenced_floor > 0.0]
        if positive_floor.size:
            polish_fatol = min(
                polish_fatol, 0.5 * float(np.min(positive_floor))
            )

    active_kl_mask = ~active_signed_mask
    q_kl = active_q[active_kl_mask]
    q_signed = active_q[active_signed_mask]
    s_signed = active_scale[active_signed_mask]

    def state(multiplier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dual_load = np.asarray(A_scaled.T @ multiplier).reshape(-1)
        values = np.empty_like(active_q)
        exponent = np.empty(0, dtype="float64")
        if bool(active_kl_mask.any()):
            exponent = np.clip(-dual_load[active_kl_mask], -500.0, 500.0)
            values[active_kl_mask] = q_kl * np.exp(exponent)
        if bool(active_signed_mask.any()):
            signed_load = dual_load[active_signed_mask]
            values[active_signed_mask] = q_signed - s_signed * signed_load
        return values, exponent

    def objective(multiplier: np.ndarray) -> float:
        _values, exponent = state(multiplier)
        dual_load = np.asarray(A_scaled.T @ multiplier).reshape(-1)
        value = float(b_scaled @ multiplier)
        if bool(active_kl_mask.any()):
            value += float(np.sum(q_kl * np.expm1(exponent)))
        if bool(active_signed_mask.any()):
            signed_load = dual_load[active_signed_mask]
            value += float(
                -q_signed @ signed_load
                + 0.5 * np.sum(s_signed * signed_load**2)
            )
        return value

    def gradient(multiplier: np.ndarray) -> np.ndarray:
        values, _ = state(multiplier)
        return b_scaled - np.asarray(A_scaled @ values).reshape(-1)

    def evaluate_candidate(multiplier: np.ndarray) -> tuple[np.ndarray, float, float]:
        candidate = np.asarray(multiplier, dtype="float64").reshape(-1)
        if candidate.size != b.size or not np.isfinite(candidate).all():
            raise ValueError(
                "Linear reconciliation produced a non-finite or mis-sized dual candidate."
            )
        active_values, _ = state(candidate)
        if not np.isfinite(active_values).all():
            raise ValueError("Linear reconciliation produced non-finite primal values.")
        result = fixed.copy()
        result[active] = active_values
        max_norm, max_abs, _actual, _tolerance, _normalized = feasibility(result)
        return result, max_norm, max_abs

    start = np.zeros(b.size, dtype="float64")
    first = minimize(
        objective,
        start,
        jac=gradient,
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "maxfun": max(15_000, 20 * max_iter),
            "ftol": 0.0,
            "gtol": 1e-12,
            "maxls": 100,
            "maxcor": 30,
        },
    )
    result, max_norm, max_abs = evaluate_candidate(
        np.asarray(first.x, dtype="float64")
    )

    # Every state has the stationarity form of the separable convex objective.
    # Krylov is only a feasibility polish when L-BFGS has not yet reached Ax=b.
    polished = None
    if max_norm > 1.0:
        polished = root(
            gradient,
            np.asarray(first.x, dtype="float64"),
            method="krylov",
            options={"fatol": polish_fatol, "maxiter": min(max_iter, 5_000)},
        )
        result, max_norm, max_abs = evaluate_candidate(
            np.asarray(polished.x, dtype="float64")
        )

    if max_norm > 1.0:
        polish_success = None if polished is None else bool(polished.success)
        polish_message = "not_needed" if polished is None else str(polished.message)
        _norm, _abs, actual, tolerance, normalized = feasibility(result)
        worst_row = int(np.argmax(normalized))
        raise ValueError(
            "Separable linear reconciliation did not satisfy its equalities: "
            f"max_normalized_gap={max_norm:.6g}, max_abs_gap={max_abs:.12g}, "
            f"worst_row={worst_row}, actual={actual[worst_row]:.12g}, "
            f"target={b[worst_row]:.12g}, tolerance={tolerance[worst_row]:.12g}, "
            f"row_scale={row_scale[worst_row]:.12g}, "
            f"tolerance_scale={tolerance_scale[worst_row]:.12g}, "
            f"polish_fatol={polish_fatol:.12g}, "
            f"n_signed_quadratic={int(active_signed_mask.sum())}, "
            f"lbfgs_success={bool(first.success)}, lbfgs_message={first.message!s}, "
            f"lbfgs_nit={int(getattr(first, 'nit', 0))}, "
            f"lbfgs_nfev={int(getattr(first, 'nfev', 0))}, "
            f"polish_success={polish_success!r}, polish_message={polish_message}."
        )

    polish_iterations = 0 if polished is None else int(getattr(polished, "nit", 0))
    return result.reshape(original_shape), KLLinearStats(
        iterations=int(getattr(first, "nit", 0)) + polish_iterations,
        max_normalized_gap=max_norm,
        max_abs_gap=max_abs,
    )


def kl_balance_linear_equalities(
    prior: np.ndarray,
    constraint_matrix: object,
    targets: np.ndarray,
    *,
    max_iter: int = 50_000,
    rtol: float = 1e-12,
    atol: float = 1e-10,
) -> tuple[np.ndarray, KLLinearStats]:
    """Return the non-negative KL projection subject to ``A @ x = b``.

    This backward-compatible API is the all-KL specialization of
    ``balance_separable_linear_equalities``. Zero priors remain structural zeros.
    """
    return balance_separable_linear_equalities(
        prior,
        constraint_matrix,
        targets,
        max_iter=max_iter,
        rtol=rtol,
        atol=atol,
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

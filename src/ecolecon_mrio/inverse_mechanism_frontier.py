# GENERIC_INVERSE_MRIO_MECHANISM_API_V1
"""Low-memory inverse final-use / coefficient-flow optimisation for monetary MRIOs.

The public object in this module is deliberately agnostic to transition scenarios,
sector roles, environmental extensions, historical-rate benchmarks, and plotting.
It solves a generic inverse accounting problem around one square MRIO coefficient
state.  Callers supply arbitrary linear gross-output-change targets and an
independent elementwise coefficient-change box.

For baseline ``(A0, x0, f0)`` and counterfactual ``x1 = x0 + dx`` the change
identity is

    (I - A0) dx = df + u,

where ``u = delta_A @ x1``.  When ``x1 >= 0`` and the admissible coefficient change
is the independent box ``L <= delta_A <= U``, the exact rowwise image is

    L @ x1 <= u <= U @ x1.

The solver therefore needs no LP variable per coefficient edge.  It adds sparse
rows incrementally to SciPy's bundled HiGHS model so EXIO-scale callers do not
materialize a second assembled Python-side constraint matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

import numpy as np
from scipy.sparse import csr_matrix


InverseSolveStatus = Literal["optimal", "infeasible", "unbounded", "numerical_failure"]


@dataclass(frozen=True)
class InverseMechanismSolution:
    """One audited inverse-MRIO solution or explicit solver failure state."""

    status: InverseSolveStatus
    message: str
    delta_x: np.ndarray | None
    structural_flow: np.ndarray | None
    delta_final_use: np.ndarray | None
    j_final_use: float
    j_structure: float
    final_use_l1: float
    structure_l1: float
    accompaniment_l1: float
    min_output: float
    max_accounting_abs_residual: float
    max_target_abs_residual: float
    min_lower_capacity_slack: float
    min_upper_capacity_slack: float

    @property
    def optimal(self) -> bool:
        """Return whether this result contains a complete optimal witness."""

        return self.status == "optimal"


class InverseMrioMechanismProblem:
    """Persistent low-memory HiGHS model for generic inverse MRIO mechanism solves.

    Parameters
    ----------
    A0, x0, f0
        Baseline square monetary coefficient matrix, gross output, and complete
        final-use vector on one common basis. ``f0`` may contain signed accounting
        items; gross output and ``A0`` must be non-negative.
    lower_headroom, upper_headroom
        Sparse elementwise coefficient-change bounds ``L <= 0 <= U`` around
        ``A0``.  The box is assumed independent across coefficient cells.
    target_matrix
        Sparse matrix ``C`` defining caller-owned linear output-change targets
        ``C @ (x1-x0) = rhs``.  The API assigns no economic meaning to its rows.
    accompaniment_indices
        Optional output coordinates whose gross absolute ``delta_x`` is minimized
        only as a lexicographic witness selector after mechanism burdens are fixed.
    """

    def __init__(
        self,
        A0: Any,
        x0: np.ndarray,
        f0: np.ndarray,
        lower_headroom: Any,
        upper_headroom: Any,
        target_matrix: Any,
        *,
        accompaniment_indices: np.ndarray | None = None,
        feasibility_tolerance: float = 1.0e-9,
        objective_relative_tolerance: float = 1.0e-9,
        objective_absolute_tolerance: float = 1.0e-7,
        row_chunk_size: int = 128,
        verbose: bool = False,
    ) -> None:
        self.A0 = self._canonical_matrix(A0, "A0")
        self.L = self._canonical_matrix(lower_headroom, "lower_headroom")
        self.U = self._canonical_matrix(upper_headroom, "upper_headroom")
        self.C = self._canonical_matrix(target_matrix, "target_matrix", square=False)
        self.x0 = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        self.f0 = np.asarray(f0, dtype=np.float64).reshape(-1).copy()
        self.n = int(self.x0.size)
        c_shape = self.C.shape
        if c_shape is None:
            raise ValueError("target_matrix must expose a two-dimensional shape")
        self.m = int(c_shape[0])
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.objective_relative_tolerance = float(objective_relative_tolerance)
        self.objective_absolute_tolerance = float(objective_absolute_tolerance)
        self.row_chunk_size = int(row_chunk_size)
        self.verbose = bool(verbose)

        if self.n <= 0:
            raise ValueError("inverse MRIO problem requires a non-empty output vector")
        a_shape, lower_shape, upper_shape = self.A0.shape, self.L.shape, self.U.shape
        if a_shape is None or lower_shape is None or upper_shape is None:
            raise ValueError("MRIO coefficient matrices must expose shapes")
        if a_shape != (self.n, self.n) or lower_shape != (self.n, self.n) or upper_shape != (self.n, self.n):
            raise ValueError("A0 and coefficient-headroom matrices must match x0 on one square basis")
        if c_shape[1] != self.n or self.m <= 0:
            raise ValueError("target_matrix must contain at least one row on the MRIO basis")
        if self.f0.shape != (self.n,):
            raise ValueError("f0 must align one-to-one with x0")
        if not np.isfinite(self.x0).all() or not np.isfinite(self.f0).all():
            raise ValueError("x0/f0 contain non-finite values")
        if self.x0.size and float(self.x0.min()) < -self.feasibility_tolerance:
            raise ValueError("baseline gross output x0 must be non-negative")
        if self.A0.data.size and float(self.A0.data.min()) < -self.feasibility_tolerance:
            raise ValueError("baseline input coefficients A0 must be non-negative")
        if self.L.data.size and float(self.L.data.max()) > self.feasibility_tolerance:
            raise ValueError("lower_headroom must satisfy L <= 0")
        if self.U.data.size and float(self.U.data.min()) < -self.feasibility_tolerance:
            raise ValueError("upper_headroom must satisfy U >= 0")
        if self.feasibility_tolerance <= 0.0 or not np.isfinite(self.feasibility_tolerance):
            raise ValueError("feasibility_tolerance must be positive and finite")
        if self.objective_relative_tolerance < 0.0 or self.objective_absolute_tolerance < 0.0:
            raise ValueError("objective face tolerances must be non-negative")
        if self.row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive")

        if accompaniment_indices is None:
            self.accompaniment_indices = np.empty(0, dtype=np.int32)
        else:
            idx = np.asarray(accompaniment_indices, dtype=np.int64).reshape(-1)
            if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.n):
                raise ValueError("accompaniment_indices contain an out-of-range coordinate")
            if np.unique(idx).size != idx.size:
                raise ValueError("accompaniment_indices must be unique")
            self.accompaniment_indices = idx.astype(np.int32, copy=True)

        self.F0 = float(np.abs(self.f0).sum())
        self._Lx0 = np.asarray(self.L.dot(self.x0), dtype=np.float64).reshape(-1)
        self._Ux0 = np.asarray(self.U.dot(self.x0), dtype=np.float64).reshape(-1)
        intermediate = np.asarray(self.A0.dot(self.x0), dtype=np.float64).reshape(-1)
        self.M0 = float(np.abs(intermediate).sum())
        if not np.isfinite(self.F0) or self.F0 <= 0.0:
            raise ValueError("baseline complete-final-use L1 denominator must be positive")
        if not np.isfinite(self.M0) or self.M0 <= 0.0:
            raise ValueError("baseline intermediate-use L1 denominator must be positive")
        self.baseline_accounting_residual = self.x0 - intermediate - self.f0

        self._core = self._load_highs_core()
        self._inf = float(self._core.kHighsInf)
        self._highs = self._core._Highs()
        self._configure_highs()
        self._define_columns()
        self._define_rows()
        self._active_cost_indices = np.empty(0, dtype=np.int32)

    @staticmethod
    def _canonical_matrix(raw: Any, label: str, *, square: bool = True) -> csr_matrix:
        matrix = cast(csr_matrix, csr_matrix(raw, dtype=np.float64, copy=False))
        shape = matrix.shape
        if shape is None or len(shape) != 2:
            raise ValueError(f"{label} must be two-dimensional")
        if square and shape[0] != shape[1]:
            raise ValueError(f"{label} must be square")
        if not matrix.has_canonical_format or bool(matrix.data.size and (matrix.data == 0.0).any()):
            matrix = matrix.copy()
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError(f"{label} contains non-finite stored values")
        return matrix

    @staticmethod
    def _load_highs_core() -> Any:
        """Load SciPy's bundled HiGHS bindings behind one checked runtime boundary."""

        try:
            core = import_module("scipy.optimize._highspy._core")
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError(
                "inverse MRIO optimisation requires SciPy's bundled incremental HiGHS bindings"
            ) from exc
        if not hasattr(core, "_Highs") or not hasattr(core, "kHighsInf"):
            raise RuntimeError("installed SciPy lacks the incremental HiGHS API required by inverse MRIO optimisation")
        return core

    def _require_ok(self, status: Any, action: str) -> None:
        if int(status) < 0:
            raise RuntimeError(f"HiGHS failed while {action}: {status}")

    def _configure_highs(self) -> None:
        for name, value in (
            ("output_flag", bool(self.verbose)),
            ("log_to_console", bool(self.verbose)),
            ("primal_feasibility_tolerance", float(self.feasibility_tolerance)),
            ("dual_feasibility_tolerance", float(self.feasibility_tolerance)),
        ):
            self._require_ok(self._highs.setOptionValue(name, value), f"setting option {name}")

    def _define_columns(self) -> None:
        n = self.n
        p = int(self.accompaniment_indices.size)
        self._dx = slice(0, n)
        self._u = slice(n, 2 * n)
        self._df = slice(2 * n, 3 * n)
        self._abs_u = slice(3 * n, 4 * n)
        self._abs_f = slice(4 * n, 5 * n)
        self._acc = slice(5 * n, 5 * n + p)
        self.num_columns = 5 * n + p

        cost = np.zeros(self.num_columns, dtype=np.float64)
        lower = np.full(self.num_columns, -self._inf, dtype=np.float64)
        upper = np.full(self.num_columns, self._inf, dtype=np.float64)
        lower[self._dx] = -self.x0
        lower[self._abs_u] = 0.0
        lower[self._abs_f] = 0.0
        if p:
            lower[self._acc] = 0.0
        starts = np.zeros(self.num_columns + 1, dtype=np.int32)
        self._require_ok(
            self._highs.addCols(
                self.num_columns,
                cost,
                lower,
                upper,
                0,
                starts,
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            ),
            "adding inverse-MRIO columns",
        )

    def _add_rows(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        *,
        label: str,
    ) -> tuple[int, int]:
        row_start = int(self._highs.getNumRow())
        self._require_ok(
            self._highs.addRows(
                int(lower.size),
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
                int(data.size),
                np.asarray(indptr, dtype=np.int32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(data, dtype=np.float64),
            ),
            f"adding {label} rows",
        )
        return row_start, int(self._highs.getNumRow())

    def _add_accounting_rows(self) -> tuple[int, int]:
        """Stream ``(I-A0)dx-u-df=0`` without forming ``I-A0``."""

        first = int(self._highs.getNumRow())
        A = self.A0
        for block_start in range(0, self.n, self.row_chunk_size):
            block_stop = min(self.n, block_start + self.row_chunk_size)
            starts = [0]
            indices: list[int] = []
            values: list[float] = []
            for row in range(block_start, block_stop):
                lo, hi = int(A.indptr[row]), int(A.indptr[row + 1])
                diag_seen = False
                for pos in range(lo, hi):
                    col = int(A.indices[pos])
                    value = -float(A.data[pos])
                    if col == row:
                        value += 1.0
                        diag_seen = True
                    if value != 0.0:
                        indices.append(col)
                        values.append(value)
                if not diag_seen:
                    indices.append(row)
                    values.append(1.0)
                indices.extend((self._u.start + row, self._df.start + row))
                values.extend((-1.0, -1.0))
                starts.append(len(indices))
            rows = block_stop - block_start
            self._add_rows(
                np.zeros(rows),
                np.zeros(rows),
                np.asarray(starts),
                np.asarray(indices),
                np.asarray(values),
                label="accounting",
            )
        return first, int(self._highs.getNumRow())

    def _add_target_rows(self) -> tuple[int, int]:
        C = self.C
        return self._add_rows(
            np.zeros(self.m),
            np.zeros(self.m),
            C.indptr,
            C.indices,
            C.data,
            label="target",
        )

    def _add_capacity_rows(self, matrix: csr_matrix, *, lower_side: bool) -> tuple[int, int]:
        """Stream one side of ``L x1 <= u <= U x1`` in ``dx`` coordinates."""

        first = int(self._highs.getNumRow())
        base_product = self._Lx0 if lower_side else self._Ux0
        for block_start in range(0, self.n, self.row_chunk_size):
            block_stop = min(self.n, block_start + self.row_chunk_size)
            starts = [0]
            indices: list[int] = []
            values: list[float] = []
            for row in range(block_start, block_stop):
                lo, hi = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
                for pos in range(lo, hi):
                    indices.append(int(matrix.indices[pos]))
                    coefficient = float(matrix.data[pos])
                    values.append(coefficient if lower_side else -coefficient)
                indices.append(self._u.start + row)
                values.append(-1.0 if lower_side else 1.0)
                starts.append(len(indices))
            rows = block_stop - block_start
            upper = (
                -base_product[block_start:block_stop]
                if lower_side
                else base_product[block_start:block_stop]
            )
            self._add_rows(
                np.full(rows, -self._inf),
                upper,
                np.asarray(starts),
                np.asarray(indices),
                np.asarray(values),
                label="lower coefficient capacity" if lower_side else "upper coefficient capacity",
            )
        return first, int(self._highs.getNumRow())

    def _add_abs_rows(self, value_slice: slice, abs_slice: slice, label: str) -> tuple[int, int]:
        first = int(self._highs.getNumRow())
        for sign in (1.0, -1.0):
            rows = self.n
            indptr = np.arange(0, 2 * rows + 1, 2, dtype=np.int32)
            indices = np.empty(2 * rows, dtype=np.int32)
            data = np.empty(2 * rows, dtype=np.float64)
            base = np.arange(rows, dtype=np.int32)
            indices[0::2] = int(value_slice.start) + base
            indices[1::2] = int(abs_slice.start) + base
            data[0::2] = sign
            data[1::2] = -1.0
            self._add_rows(
                np.full(rows, -self._inf),
                np.zeros(rows),
                indptr,
                indices,
                data,
                label=f"{label} absolute-value",
            )
        return first, int(self._highs.getNumRow())

    def _add_sum_guard(self, variable_slice: slice, label: str) -> int:
        row = int(self._highs.getNumRow())
        size = int(variable_slice.stop) - int(variable_slice.start)
        indptr = np.asarray([0, size], dtype=np.int32)
        indices = np.arange(int(variable_slice.start), int(variable_slice.stop), dtype=np.int32)
        self._add_rows(
            np.asarray([-self._inf]),
            np.asarray([self._inf]),
            indptr,
            indices,
            np.ones(size, dtype=np.float64),
            label=label,
        )
        return row

    def _add_accompaniment_rows(self) -> tuple[int, int]:
        p = int(self.accompaniment_indices.size)
        first = int(self._highs.getNumRow())
        if p == 0:
            return first, first
        for sign in (1.0, -1.0):
            indptr = np.arange(0, 2 * p + 1, 2, dtype=np.int32)
            indices = np.empty(2 * p, dtype=np.int32)
            data = np.empty(2 * p, dtype=np.float64)
            indices[0::2] = self.accompaniment_indices
            indices[1::2] = np.arange(self._acc.start, self._acc.stop, dtype=np.int32)
            data[0::2] = sign
            data[1::2] = -1.0
            self._add_rows(
                np.full(p, -self._inf),
                np.zeros(p),
                indptr,
                indices,
                data,
                label="accompaniment absolute-value",
            )
        return first, int(self._highs.getNumRow())

    def _define_rows(self) -> None:
        self._rows_accounting = self._add_accounting_rows()
        self._rows_target = self._add_target_rows()
        self._rows_capacity_lower = self._add_capacity_rows(self.L, lower_side=True)
        self._rows_capacity_upper = self._add_capacity_rows(self.U, lower_side=False)
        self._rows_abs_u = self._add_abs_rows(self._u, self._abs_u, "structural flow")
        self._rows_abs_f = self._add_abs_rows(self._df, self._abs_f, "final-use change")
        self._row_final_cap = self._add_sum_guard(self._abs_f, "final-use cap")
        self._row_structure_face = self._add_sum_guard(self._abs_u, "structure objective face")
        self._row_final_face = self._add_sum_guard(self._abs_f, "final-use objective face")
        self._rows_accompaniment = self._add_accompaniment_rows()

    def _set_row_upper(self, row: int, upper: float) -> None:
        self._require_ok(
            self._highs.changeRowBounds(int(row), -self._inf, float(upper)),
            f"changing upper bound for row {row}",
        )

    def _set_target_rhs(self, rhs: np.ndarray) -> None:
        values = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if values.shape != (self.m,) or not np.isfinite(values).all():
            raise ValueError(f"target rhs must be a finite vector of length {self.m}")
        start, stop = self._rows_target
        for offset, value in enumerate(values):
            self._require_ok(
                self._highs.changeRowBounds(start + offset, float(value), float(value)),
                "updating target equality",
            )
        if stop - start != self.m:
            raise RuntimeError("internal target-row count drift")
        self._target_rhs = values.copy()

    def _set_objective(self, variable_slice: slice | None) -> None:
        if self._active_cost_indices.size:
            self._require_ok(
                self._highs.changeColsCost(
                    int(self._active_cost_indices.size),
                    self._active_cost_indices,
                    np.zeros(self._active_cost_indices.size, dtype=np.float64),
                ),
                "resetting lexicographic objective",
            )
        if variable_slice is None or int(variable_slice.stop) <= int(variable_slice.start):
            self._active_cost_indices = np.empty(0, dtype=np.int32)
            return
        indices = np.arange(int(variable_slice.start), int(variable_slice.stop), dtype=np.int32)
        self._require_ok(
            self._highs.changeColsCost(
                int(indices.size), indices, np.ones(indices.size, dtype=np.float64)
            ),
            "setting lexicographic objective",
        )
        self._active_cost_indices = indices

    def _reset_guards(self) -> None:
        self._set_row_upper(self._row_final_cap, self._inf)
        self._set_row_upper(self._row_structure_face, self._inf)
        self._set_row_upper(self._row_final_face, self._inf)

    def _face_tolerance(self, objective: float) -> float:
        return max(
            self.objective_absolute_tolerance,
            self.objective_relative_tolerance * max(1.0, abs(float(objective))),
        )

    def _status(self) -> tuple[InverseSolveStatus, str]:
        raw = self._highs.getModelStatus()
        code = int(raw)
        enum = self._core.HighsModelStatus
        if code == int(enum.kOptimal):
            return "optimal", str(raw)
        if code == int(enum.kInfeasible):
            return "infeasible", str(raw)
        if code == int(enum.kUnbounded):
            return "unbounded", str(raw)
        return "numerical_failure", str(raw)

    def _run_highs(self) -> tuple[InverseSolveStatus, str, np.ndarray | None, np.ndarray | None]:
        run_status = self._highs.run()
        if int(run_status) < 0:
            return "numerical_failure", f"HiGHS run failure: {run_status}", None, None
        status, message = self._status()
        if status != "optimal":
            return status, message, None, None
        solution = self._highs.getSolution()
        columns = np.asarray(solution.col_value, dtype=np.float64)
        rows = np.asarray(solution.row_value, dtype=np.float64)
        if columns.shape != (self.num_columns,) or rows.shape != (int(self._highs.getNumRow()),):
            return "numerical_failure", "HiGHS returned an unexpected solution shape", None, None
        if not np.isfinite(columns).all() or not np.isfinite(rows).all():
            return "numerical_failure", "HiGHS returned non-finite primal values", None, None
        return "optimal", message, columns, rows

    def _failure(self, status: InverseSolveStatus, message: str) -> InverseMechanismSolution:
        return InverseMechanismSolution(
            status=status,
            message=message,
            delta_x=None,
            structural_flow=None,
            delta_final_use=None,
            j_final_use=np.nan,
            j_structure=np.nan,
            final_use_l1=np.nan,
            structure_l1=np.nan,
            accompaniment_l1=np.nan,
            min_output=np.nan,
            max_accounting_abs_residual=np.nan,
            max_target_abs_residual=np.nan,
            min_lower_capacity_slack=np.nan,
            min_upper_capacity_slack=np.nan,
        )

    def _solution(self, message: str, columns: np.ndarray, rows: np.ndarray) -> InverseMechanismSolution:
        dx = columns[self._dx].copy()
        u = columns[self._u].copy()
        df = columns[self._df].copy()
        x1 = self.x0 + dx
        final_l1 = float(np.abs(df).sum())
        structure_l1 = float(np.abs(u).sum())
        acc_l1 = (
            float(np.abs(dx[self.accompaniment_indices]).sum())
            if self.accompaniment_indices.size
            else 0.0
        )
        a0, a1 = self._rows_accounting
        t0, t1 = self._rows_target
        l0, l1 = self._rows_capacity_lower
        u0, u1 = self._rows_capacity_upper
        accounting_residual = rows[a0:a1]
        target_residual = rows[t0:t1] - self._target_rhs
        # The lower-capacity row is L*dx-u <= -L*x0, so slack is
        # (-L*x0) - row_value = u - L*x1.  The upper side is analogous.
        lower_slack = -self._Lx0 - rows[l0:l1]
        upper_slack = self._Ux0 - rows[u0:u1]
        return InverseMechanismSolution(
            status="optimal",
            message=message,
            delta_x=dx,
            structural_flow=u,
            delta_final_use=df,
            j_final_use=final_l1 / self.F0,
            j_structure=structure_l1 / self.M0,
            final_use_l1=final_l1,
            structure_l1=structure_l1,
            accompaniment_l1=acc_l1,
            min_output=float(x1.min(initial=np.inf)),
            max_accounting_abs_residual=float(np.max(np.abs(accounting_residual), initial=0.0)),
            max_target_abs_residual=float(np.max(np.abs(target_residual), initial=0.0)),
            min_lower_capacity_slack=float(lower_slack.min(initial=np.inf)),
            min_upper_capacity_slack=float(upper_slack.min(initial=np.inf)),
        )

    def _solve_stage(self) -> tuple[InverseMechanismSolution, np.ndarray | None, np.ndarray | None]:
        status, message, columns, rows = self._run_highs()
        if status != "optimal" or columns is None or rows is None:
            return self._failure(status, message), None, None
        return self._solution(message, columns, rows), columns, rows

    def _minimize_with_lexicographic_followups(
        self,
        *,
        primary: Literal["structure", "final"],
        secondary: Literal["structure", "final"] | None,
    ) -> InverseMechanismSolution:
        primary_slice = self._abs_u if primary == "structure" else self._abs_f
        primary_guard = self._row_structure_face if primary == "structure" else self._row_final_face
        self._set_objective(primary_slice)
        first, first_columns, _ = self._solve_stage()
        if not first.optimal or first_columns is None:
            return first
        primary_value = (
            float(first_columns[self._abs_u].sum())
            if primary == "structure"
            else float(first_columns[self._abs_f].sum())
        )
        self._set_row_upper(primary_guard, primary_value + self._face_tolerance(primary_value))

        if secondary is not None:
            secondary_slice = self._abs_u if secondary == "structure" else self._abs_f
            secondary_guard = self._row_structure_face if secondary == "structure" else self._row_final_face
            self._set_objective(secondary_slice)
            second, second_columns, _ = self._solve_stage()
            if not second.optimal or second_columns is None:
                return second
            secondary_value = (
                float(second_columns[self._abs_u].sum())
                if secondary == "structure"
                else float(second_columns[self._abs_f].sum())
            )
            self._set_row_upper(
                secondary_guard,
                secondary_value + self._face_tolerance(secondary_value),
            )

        if self.accompaniment_indices.size:
            self._set_objective(self._acc)
            final, _, _ = self._solve_stage()
            return final
        # No accompaniment selector: return a solution after the last objective
        # face was fixed so the returned state is governed by the same guards.
        self._set_objective(None)
        final, _, _ = self._solve_stage()
        return final

    def solve_epsilon(self, target_rhs: np.ndarray, eta_final_use: float) -> InverseMechanismSolution:
        """Minimize structural burden under one normalized final-use L1 cap.

        Selection is lexicographic: minimum structural L1, then minimum final-use
        L1 on that optimal face, then minimum declared accompaniment.  This picks a
        nondominated burden pair without introducing a weighted mechanism objective.
        """

        eta = float(eta_final_use)
        if not np.isfinite(eta) or eta < 0.0:
            raise ValueError("eta_final_use must be finite and non-negative")
        self._set_target_rhs(target_rhs)
        self._reset_guards()
        self._set_row_upper(self._row_final_cap, eta * self.F0)
        return self._minimize_with_lexicographic_followups(primary="structure", secondary="final")

    def solve_min_final_use(self, target_rhs: np.ndarray) -> InverseMechanismSolution:
        """Return the leftmost burden point: min final use, then min structure."""

        self._set_target_rhs(target_rhs)
        self._reset_guards()
        return self._minimize_with_lexicographic_followups(primary="final", secondary="structure")

    def solve_pure_final_use(self, target_rhs: np.ndarray) -> InverseMechanismSolution:
        """Minimize final-use burden subject to exactly zero structural flow ``u``."""

        self._set_target_rhs(target_rhs)
        self._reset_guards()
        self._set_row_upper(self._row_structure_face, 0.0)
        return self._minimize_with_lexicographic_followups(primary="final", secondary=None)

    def solve_pure_structure(self, target_rhs: np.ndarray) -> InverseMechanismSolution:
        """Minimize structural burden subject to exactly zero final-use change."""

        self._set_target_rhs(target_rhs)
        self._reset_guards()
        self._set_row_upper(self._row_final_face, 0.0)
        return self._minimize_with_lexicographic_followups(primary="structure", secondary=None)

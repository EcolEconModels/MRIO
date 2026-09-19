"""Scaled inverse final-use / coefficient-flow optimisation for monetary MRIOs.

The public solver is deliberately agnostic to transition scenarios, sector roles,
environmental meanings, historical benchmarks, and plotting.  It solves the generic
change-accounting problem

    (I - A0) dx = df + u,   u = delta_A @ (x0 + dx),

with x0 + dx >= 0 and an independent coefficient box L <= delta_A <= U.  For
non-negative counterfactual output, coefficient realizability has the exact rowwise
image L @ (x0 + dx) <= u <= U @ (x0 + dx), so no coefficient-edge decision
variables are needed.

HiGHS receives a dimensionless/economically scaled coordinate representation.  Very
small coefficients in that scaled representation are a numerical backend
approximation only; the original A0/L/U/C/guard matrices remain authoritative and
every reported optimum is rechecked against them with scale-aware residual tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

import numpy as np
from scipy.sparse import csr_matrix

InverseSolveStatus = Literal["optimal", "infeasible", "unbounded", "numerical_failure"]
ColdStartSolver = Literal["choose", "simplex", "ipm"]

_HIGHS_MIN_SMALL_MATRIX_VALUE = 1.0e-12
_HIGHS_DEFAULT_SMALL_MATRIX_VALUE = 1.0e-9


@dataclass(frozen=True)
class InverseMechanismSolution:
    """One original-coordinate audited inverse-MRIO solution or explicit failure."""

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
    max_accounting_scaled_residual: float
    max_target_abs_residual: float
    max_target_scaled_residual: float
    min_lower_capacity_slack: float
    min_upper_capacity_slack: float
    max_capacity_scaled_violation: float
    max_guard_scaled_violation: float

    @property
    def optimal(self) -> bool:
        """Return whether this result contains a complete accepted witness."""

        return self.status == "optimal"


class InverseMrioMechanismProblem:
    """Persistent scaled HiGHS model for generic inverse MRIO mechanism solves.

    ``target_matrix`` defines equality targets C @ dx = rhs.  ``output_guard_matrix``
    optionally defines additional caller-owned linear output-change rows G @ dx;
    each solve may set lower/upper bounds on those rows.  The API assigns no
    environmental or transition meaning to either matrix.

    Solver coordinates are scaled by a positive monetary row scale derived from the
    supplied baseline accounts.  Stored matrices are never mutated.  Coefficients
    at or below ``solver_matrix_floor`` *after* this reversible scaling are omitted
    from the HiGHS representation and censused.  This is a numerical approximation,
    not a scientific sparsity rule: every candidate optimum is recomputed against
    the complete original matrices and rejected as ``numerical_failure`` when any
    scale-aware original-coordinate residual exceeds ``postcheck_tolerance``.
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
        output_guard_matrix: Any | None = None,
        accompaniment_indices: np.ndarray | None = None,
        feasibility_tolerance: float = 1.0e-9,
        postcheck_tolerance: float | None = None,
        objective_relative_tolerance: float = 1.0e-9,
        objective_absolute_tolerance: float = 1.0e-7,
        row_chunk_size: int = 128,
        solver_matrix_floor: float = _HIGHS_MIN_SMALL_MATRIX_VALUE,
        cold_start_solver: ColdStartSolver = "choose",
        verbose: bool = False,
    ) -> None:
        self.A0 = self._canonical_matrix(A0, "A0")
        self.L = self._canonical_matrix(lower_headroom, "lower_headroom")
        self.U = self._canonical_matrix(upper_headroom, "upper_headroom")
        self.C = self._canonical_matrix(target_matrix, "target_matrix", square=False)
        self.x0 = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        self.f0 = np.asarray(f0, dtype=np.float64).reshape(-1).copy()
        self.n = int(self.x0.size)
        if self.n <= 0:
            raise ValueError("inverse MRIO problem requires a non-empty output vector")
        a_shape = cast(tuple[int, int], self.A0.shape)
        l_shape = cast(tuple[int, int], self.L.shape)
        u_shape = cast(tuple[int, int], self.U.shape)
        c_shape = cast(tuple[int, int], self.C.shape)
        if a_shape != (self.n, self.n) or l_shape != a_shape or u_shape != a_shape:
            raise ValueError("A0 and coefficient-headroom matrices must match x0 on one square basis")
        if c_shape[1] != self.n or c_shape[0] <= 0:
            raise ValueError("target_matrix must contain at least one row on the MRIO basis")
        self.m = int(c_shape[0])
        if self.f0.shape != (self.n,):
            raise ValueError("f0 must align one-to-one with x0")
        if not np.isfinite(self.x0).all() or not np.isfinite(self.f0).all():
            raise ValueError("x0/f0 contain non-finite values")

        self.feasibility_tolerance = float(feasibility_tolerance)
        self.postcheck_tolerance = float(
            self.feasibility_tolerance if postcheck_tolerance is None else postcheck_tolerance
        )
        self.objective_relative_tolerance = float(objective_relative_tolerance)
        self.objective_absolute_tolerance = float(objective_absolute_tolerance)
        self.row_chunk_size = int(row_chunk_size)
        self.solver_matrix_floor = float(solver_matrix_floor)
        self.cold_start_solver = cast(ColdStartSolver, str(cold_start_solver))
        self.verbose = bool(verbose)
        if not np.isfinite(self.feasibility_tolerance) or self.feasibility_tolerance <= 0.0:
            raise ValueError("feasibility_tolerance must be positive and finite")
        if not np.isfinite(self.postcheck_tolerance) or self.postcheck_tolerance <= 0.0:
            raise ValueError("postcheck_tolerance must be positive and finite")
        if self.objective_relative_tolerance < 0.0 or self.objective_absolute_tolerance < 0.0:
            raise ValueError("objective face tolerances must be non-negative")
        if self.row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive")
        if not (_HIGHS_MIN_SMALL_MATRIX_VALUE <= self.solver_matrix_floor <= 1.0e-9):
            raise ValueError("solver_matrix_floor must lie in [1e-12, 1e-9]")
        if self.cold_start_solver not in {"choose", "simplex", "ipm"}:
            raise ValueError("cold_start_solver must be 'choose', 'simplex', or 'ipm'")
        if self.x0.size and float(self.x0.min()) < -self.feasibility_tolerance:
            raise ValueError("baseline gross output x0 must be non-negative")
        if self.A0.data.size and float(self.A0.data.min()) < -self.feasibility_tolerance:
            raise ValueError("baseline input coefficients A0 must be non-negative")
        if self.L.data.size and float(self.L.data.max()) > self.feasibility_tolerance:
            raise ValueError("lower_headroom must satisfy L <= 0")
        if self.U.data.size and float(self.U.data.min()) < -self.feasibility_tolerance:
            raise ValueError("upper_headroom must satisfy U >= 0")
        self._validate_nonnegative_lower_corner()

        if output_guard_matrix is None:
            self.G = csr_matrix((0, self.n), dtype=np.float64)
        else:
            self.G = self._canonical_matrix(output_guard_matrix, "output_guard_matrix", square=False)
            if cast(tuple[int, int], self.G.shape)[1] != self.n:
                raise ValueError("output_guard_matrix must align to x0")
        self.g = int(cast(tuple[int, int], self.G.shape)[0])

        if accompaniment_indices is None:
            self.accompaniment_indices = np.empty(0, dtype=np.int32)
        else:
            idx = np.asarray(accompaniment_indices, dtype=np.int64).reshape(-1)
            if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.n):
                raise ValueError("accompaniment_indices contain an out-of-range coordinate")
            if np.unique(idx).size != idx.size:
                raise ValueError("accompaniment_indices must be unique")
            self.accompaniment_indices = idx.astype(np.int32, copy=True)

        intermediate = np.asarray(self.A0.dot(self.x0), dtype=np.float64).reshape(-1)
        self.F0 = float(np.abs(self.f0).sum())
        self.M0 = float(np.abs(intermediate).sum())
        if not np.isfinite(self.F0) or self.F0 <= 0.0:
            raise ValueError("baseline complete-final-use L1 denominator must be positive")
        if not np.isfinite(self.M0) or self.M0 <= 0.0:
            raise ValueError("baseline intermediate-use L1 denominator must be positive")
        self.baseline_accounting_residual = self.x0 - intermediate - self.f0
        self._Lx0 = np.asarray(self.L.dot(self.x0), dtype=np.float64).reshape(-1)
        self._Ux0 = np.asarray(self.U.dot(self.x0), dtype=np.float64).reshape(-1)

        # One positive monetary scale per product is used for dx, df and u.  For
        # ordinary positive-output cells this is x0; the other terms make zero-output
        # or signed-final-use rows finite without inventing a scientific bound.
        self._scale = np.maximum.reduce(
            [np.ones(self.n), np.abs(self.x0), np.abs(self.f0), np.abs(intermediate)]
        ).astype(np.float64)
        if not np.isfinite(self._scale).all() or bool((self._scale <= 0.0).any()):
            raise RuntimeError("inverse-MRIO variable scaling is non-positive or non-finite")
        absC = self.C.copy(); absC.data = np.abs(absC.data)
        self._target_scale = np.maximum(1.0, np.asarray(absC.dot(self._scale)).reshape(-1))
        absG = self.G.copy(); absG.data = np.abs(absG.data)
        self._guard_scale = np.maximum(1.0, np.asarray(absG.dot(self._scale)).reshape(-1)) if self.g else np.empty(0)
        self._active_guard_lower = np.full(self.g, -np.inf, dtype=np.float64)
        self._active_guard_upper = np.full(self.g, np.inf, dtype=np.float64)
        self._target_rhs = np.zeros(self.m, dtype=np.float64)

        # Diagnostics refer to the scaled matrix actually submitted to HiGHS.
        self._solver_input_nonzeros = 0
        self._solver_pruned_nonzeros = 0
        self._solver_smallest_abs_input = np.inf
        self._solver_smallest_abs_retained = np.inf
        self._solver_largest_abs_retained = 0.0
        self._solver_max_abs_pruned = 0.0
        self._capacity_rows_added = False
        self._run_count = 0

        self._core = self._load_highs_core()
        self._inf = float(self._core.kHighsInf)
        self._highs = self._core._Highs()
        self._configure_highs()
        self._define_columns()
        self._define_rows()
        self._active_cost_indices = np.empty(0, dtype=np.int32)
        if self.verbose:
            print(f"inverse-MRIO numerical diagnostics (initial): {self.numerical_matrix_diagnostics}", flush=True)

    @staticmethod
    def _canonical_matrix(raw: Any, label: str, *, square: bool = True) -> csr_matrix:
        matrix = cast(csr_matrix, csr_matrix(raw, dtype=np.float64, copy=True))
        shape = matrix.shape
        if shape is None or len(shape) != 2:
            raise ValueError(f"{label} must be two-dimensional")
        if square and shape[0] != shape[1]:
            raise ValueError(f"{label} must be square")
        matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError(f"{label} contains non-finite stored values")
        return matrix

    def _validate_nonnegative_lower_corner(self) -> None:
        tol = float(self.feasibility_tolerance)
        for row in range(self.n):
            a_left, a_right = int(self.A0.indptr[row]), int(self.A0.indptr[row + 1])
            l_left, l_right = int(self.L.indptr[row]), int(self.L.indptr[row + 1])
            a_cols = np.asarray(self.A0.indices[a_left:a_right], dtype=np.intp)
            l_cols = np.asarray(self.L.indices[l_left:l_right], dtype=np.intp)
            cols = np.union1d(a_cols, l_cols)
            if cols.size == 0:
                continue
            baseline = np.zeros(cols.size); lower = np.zeros(cols.size)
            if a_cols.size:
                baseline[np.searchsorted(cols, a_cols)] = self.A0.data[a_left:a_right]
            if l_cols.size:
                lower[np.searchsorted(cols, l_cols)] = self.L.data[l_left:l_right]
            values = baseline + lower
            bad = np.flatnonzero(values < -tol)
            if bad.size:
                pos = int(bad[0])
                raise ValueError(
                    "coefficient lower corner would make A0 + L negative: "
                    f"row={row}, column={int(cols[pos])}, value={values[pos]:.6g}"
                )

    @staticmethod
    def _load_highs_core() -> Any:
        try:
            core = import_module("scipy.optimize._highspy._core")
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("inverse MRIO optimisation requires SciPy's bundled HiGHS bindings") from exc
        if not hasattr(core, "_Highs") or not hasattr(core, "kHighsInf"):
            raise RuntimeError("installed SciPy lacks the incremental HiGHS API")
        return core

    def _require_ok(self, status: Any, action: str) -> None:
        if int(status) < 0:
            raise RuntimeError(f"HiGHS failed while {action}: {status}")

    def _require_exact_ok(self, status: Any, action: str) -> None:
        if int(status) != 0:
            raise RuntimeError(f"HiGHS returned non-OK status while {action}: {status}")

    def _configure_highs(self) -> None:
        for name, value in (
            ("output_flag", bool(self.verbose)),
            ("log_to_console", bool(self.verbose)),
            ("small_matrix_value", float(self.solver_matrix_floor)),
            ("primal_feasibility_tolerance", float(self.feasibility_tolerance)),
            ("dual_feasibility_tolerance", float(self.feasibility_tolerance)),
        ):
            self._require_ok(self._highs.setOptionValue(name, value), f"setting option {name}")

    @property
    def coefficient_capacity_materialized(self) -> bool:
        return bool(self._capacity_rows_added)

    @property
    def numerical_matrix_diagnostics(self) -> dict[str, int | float | bool | str]:
        smallest_input = 0.0 if not np.isfinite(self._solver_smallest_abs_input) else float(self._solver_smallest_abs_input)
        smallest_retained = 0.0 if not np.isfinite(self._solver_smallest_abs_retained) else float(self._solver_smallest_abs_retained)
        dynamic = (
            float(self._solver_largest_abs_retained / smallest_retained)
            if smallest_retained > 0.0 else 0.0
        )
        return {
            "solver_matrix_floor": float(self.solver_matrix_floor),
            "backend_previous_default_small_matrix_value": float(_HIGHS_DEFAULT_SMALL_MATRIX_VALUE),
            "scaled_matrix_input_nonzeros": int(self._solver_input_nonzeros),
            "scaled_matrix_pruned_nonzeros": int(self._solver_pruned_nonzeros),
            "scaled_matrix_smallest_abs_input": smallest_input,
            "scaled_matrix_smallest_abs_retained": smallest_retained,
            "scaled_matrix_largest_abs_retained": float(self._solver_largest_abs_retained),
            "scaled_matrix_dynamic_range": dynamic,
            "scaled_matrix_max_abs_pruned": float(self._solver_max_abs_pruned),
            "postcheck_tolerance": float(self.postcheck_tolerance),
            "cold_start_solver": str(self.cold_start_solver),
            "capacity_rows_materialized": bool(self._capacity_rows_added),
            "solver_run_count": int(self._run_count),
        }

    def _prepare_rows_for_highs(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        *,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lo = np.asarray(lower, dtype=np.float64).reshape(-1)
        hi = np.asarray(upper, dtype=np.float64).reshape(-1)
        ptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
        idx = np.asarray(indices, dtype=np.int32).reshape(-1)
        values = np.asarray(data, dtype=np.float64).reshape(-1)
        if hi.shape != lo.shape or ptr.shape != (lo.size + 1,) or idx.size != values.size:
            raise RuntimeError(f"{label}: malformed sparse row arrays")
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label}: non-finite solver coefficient")
        absolute = np.abs(values)
        nz = absolute > 0.0
        self._solver_input_nonzeros += int(np.count_nonzero(nz))
        if bool(nz.any()):
            self._solver_smallest_abs_input = min(self._solver_smallest_abs_input, float(absolute[nz].min()))
        keep = absolute > float(self.solver_matrix_floor)
        dropped = nz & ~keep
        self._solver_pruned_nonzeros += int(np.count_nonzero(dropped))
        if bool(dropped.any()):
            self._solver_max_abs_pruned = max(self._solver_max_abs_pruned, float(absolute[dropped].max()))
        if bool(keep.any()):
            kept_abs = absolute[keep]
            self._solver_smallest_abs_retained = min(self._solver_smallest_abs_retained, float(kept_abs.min()))
            self._solver_largest_abs_retained = max(self._solver_largest_abs_retained, float(kept_abs.max()))

        counts = np.zeros(lo.size, dtype=np.int32)
        for row in range(lo.size):
            left, right = int(ptr[row]), int(ptr[row + 1])
            counts[row] = int(np.count_nonzero(keep[left:right]))
        out_ptr = np.empty(lo.size + 1, dtype=np.int32)
        out_ptr[0] = 0; np.cumsum(counts, out=out_ptr[1:])
        out_idx = idx[keep].astype(np.int32, copy=True)
        out_values = values[keep].astype(np.float64, copy=True)
        if out_values.size and float(np.min(np.abs(out_values))) <= self.solver_matrix_floor:
            raise RuntimeError(f"{label}: solver matrix preparation left a coefficient at the HiGHS floor")
        return lo.copy(), hi.copy(), out_ptr, out_idx, out_values

    def _define_columns(self) -> None:
        n = self.n; p = int(self.accompaniment_indices.size)
        self._dx = slice(0, n); self._u = slice(n, 2*n); self._df = slice(2*n, 3*n)
        self._abs_u = slice(3*n, 4*n); self._abs_f = slice(4*n, 5*n); self._acc = slice(5*n, 5*n+p)
        self.num_columns = 5*n+p
        cost = np.zeros(self.num_columns)
        lower = np.full(self.num_columns, -self._inf); upper = np.full(self.num_columns, self._inf)
        lower[self._dx] = -self.x0 / self._scale
        lower[self._abs_u] = 0.0; lower[self._abs_f] = 0.0
        if p: lower[self._acc] = 0.0
        starts = np.zeros(self.num_columns + 1, dtype=np.int32)
        self._require_ok(self._highs.addCols(
            self.num_columns, cost, lower, upper, 0, starts,
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        ), "adding inverse-MRIO columns")

    def _add_rows(self, lower: np.ndarray, upper: np.ndarray, indptr: np.ndarray,
                  indices: np.ndarray, data: np.ndarray, *, label: str) -> tuple[int, int]:
        first = int(self._highs.getNumRow())
        lo, hi, ptr, idx, values = self._prepare_rows_for_highs(lower, upper, indptr, indices, data, label=label)
        status = self._highs.addRows(int(lo.size), lo, hi, int(values.size), ptr, idx, values)
        self._require_exact_ok(status, f"adding {label} rows")
        return first, int(self._highs.getNumRow())

    def _add_accounting_rows(self) -> tuple[int, int]:
        first = int(self._highs.getNumRow()); A = self.A0
        for block_start in range(0, self.n, self.row_chunk_size):
            block_stop = min(self.n, block_start + self.row_chunk_size)
            starts=[0]; indices:list[int]=[]; values:list[float]=[]
            for row in range(block_start, block_stop):
                rscale=float(self._scale[row]); left,right=int(A.indptr[row]),int(A.indptr[row+1]); diag=False
                for pos in range(left,right):
                    col=int(A.indices[pos]); coefficient=-float(A.data[pos])*float(self._scale[col])/rscale
                    if col==row:
                        coefficient += float(self._scale[row])/rscale; diag=True
                    if coefficient != 0.0:
                        indices.append(col); values.append(coefficient)
                if not diag:
                    indices.append(row); values.append(float(self._scale[row])/rscale)
                indices.extend((self._u.start+row,self._df.start+row)); values.extend((-1.0,-1.0)); starts.append(len(indices))
            rows=block_stop-block_start
            self._add_rows(np.zeros(rows),np.zeros(rows),np.asarray(starts),np.asarray(indices),np.asarray(values),label="scaled accounting")
        return first,int(self._highs.getNumRow())

    def _scaled_sparse_rows(self, matrix: csr_matrix, row_scale: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        ptr=matrix.indptr.astype(np.int32,copy=True); idx=matrix.indices.astype(np.int32,copy=True); data=matrix.data.astype(float,copy=True)
        for row in range(cast(tuple[int,int],matrix.shape)[0]):
            left,right=int(ptr[row]),int(ptr[row+1])
            if left<right:
                data[left:right] *= self._scale[idx[left:right]]/float(row_scale[row])
        return ptr,idx,data

    def _add_target_rows(self) -> tuple[int,int]:
        ptr,idx,data=self._scaled_sparse_rows(self.C,self._target_scale)
        return self._add_rows(np.zeros(self.m),np.zeros(self.m),ptr,idx,data,label="scaled target")

    def _add_output_guard_rows(self) -> tuple[int,int]:
        if self.g==0:
            first=int(self._highs.getNumRow()); return first,first
        ptr,idx,data=self._scaled_sparse_rows(self.G,self._guard_scale)
        return self._add_rows(np.full(self.g,-self._inf),np.full(self.g,self._inf),ptr,idx,data,label="scaled output guard")

    def _add_capacity_rows(self, matrix: csr_matrix, *, lower_side: bool) -> tuple[int,int]:
        first=int(self._highs.getNumRow()); base=self._Lx0 if lower_side else self._Ux0
        for block_start in range(0,self.n,self.row_chunk_size):
            block_stop=min(self.n,block_start+self.row_chunk_size); starts=[0]; indices=[]; values=[]
            for row in range(block_start,block_stop):
                rscale=float(self._scale[row]); left,right=int(matrix.indptr[row]),int(matrix.indptr[row+1])
                for pos in range(left,right):
                    col=int(matrix.indices[pos]); coeff=float(matrix.data[pos])*float(self._scale[col])/rscale
                    indices.append(col); values.append(coeff if lower_side else -coeff)
                indices.append(self._u.start+row); values.append(-1.0 if lower_side else 1.0); starts.append(len(indices))
            rows=block_stop-block_start
            upper=(-base[block_start:block_stop] if lower_side else base[block_start:block_stop])/self._scale[block_start:block_stop]
            self._add_rows(np.full(rows,-self._inf),upper,np.asarray(starts),np.asarray(indices),np.asarray(values),label="scaled lower capacity" if lower_side else "scaled upper capacity")
        return first,int(self._highs.getNumRow())

    def _add_abs_rows(self,value_slice:slice,abs_slice:slice,label:str)->tuple[int,int]:
        first=int(self._highs.getNumRow()); rows=self.n
        for sign in (1.0,-1.0):
            ptr=np.arange(0,2*rows+1,2,dtype=np.int32); idx=np.empty(2*rows,dtype=np.int32); data=np.empty(2*rows)
            base=np.arange(rows,dtype=np.int32); idx[0::2]=int(value_slice.start)+base; idx[1::2]=int(abs_slice.start)+base
            data[0::2]=sign; data[1::2]=-1.0
            self._add_rows(np.full(rows,-self._inf),np.zeros(rows),ptr,idx,data,label=f"{label} absolute-value")
        return first,int(self._highs.getNumRow())

    def _add_sum_guard(self,variable_slice:slice,weights:np.ndarray,label:str)->int:
        row=int(self._highs.getNumRow()); size=int(variable_slice.stop)-int(variable_slice.start)
        w=np.asarray(weights,dtype=float).reshape(-1)
        if w.shape!=(size,) or not np.isfinite(w).all() or bool((w<0).any()): raise RuntimeError(f"{label}: invalid weights")
        self._add_rows(np.asarray([-self._inf]),np.asarray([self._inf]),np.asarray([0,size],dtype=np.int32),np.arange(int(variable_slice.start),int(variable_slice.stop),dtype=np.int32),w,label=label)
        return row

    def _add_accompaniment_rows(self)->tuple[int,int]:
        p=int(self.accompaniment_indices.size); first=int(self._highs.getNumRow())
        if p==0: return first,first
        for sign in (1.0,-1.0):
            ptr=np.arange(0,2*p+1,2,dtype=np.int32); idx=np.empty(2*p,dtype=np.int32); data=np.empty(2*p)
            idx[0::2]=self.accompaniment_indices; idx[1::2]=np.arange(self._acc.start,self._acc.stop,dtype=np.int32)
            data[0::2]=sign; data[1::2]=-1.0
            self._add_rows(np.full(p,-self._inf),np.zeros(p),ptr,idx,data,label="accompaniment absolute-value")
        return first,int(self._highs.getNumRow())

    def _define_rows(self)->None:
        self._rows_accounting=self._add_accounting_rows(); self._rows_target=self._add_target_rows(); self._rows_output_guard=self._add_output_guard_rows()
        self._rows_capacity_lower=(-1,-1); self._rows_capacity_upper=(-1,-1)
        self._rows_abs_u=self._add_abs_rows(self._u,self._abs_u,"structural flow"); self._rows_abs_f=self._add_abs_rows(self._df,self._abs_f,"final-use change")
        self._row_final_cap=self._add_sum_guard(self._abs_f,self._scale,"final-use cap")
        self._row_structure_face=self._add_sum_guard(self._abs_u,self._scale,"structure objective face")
        self._row_final_face=self._add_sum_guard(self._abs_f,self._scale,"final-use objective face")
        self._rows_accompaniment=self._add_accompaniment_rows()

    def _ensure_capacity_rows(self)->None:
        if self._capacity_rows_added: return
        self._rows_capacity_lower=self._add_capacity_rows(self.L,lower_side=True); self._rows_capacity_upper=self._add_capacity_rows(self.U,lower_side=False); self._capacity_rows_added=True
        if self.verbose: print(f"inverse-MRIO numerical diagnostics (with capacity): {self.numerical_matrix_diagnostics}",flush=True)

    def _set_row_bounds(self,row:int,lower:float,upper:float)->None:
        self._require_ok(self._highs.changeRowBounds(int(row),float(lower),float(upper)),f"changing bounds for row {row}")

    def _set_row_upper(self,row:int,upper:float)->None: self._set_row_bounds(row,-self._inf,upper)

    def _set_target_rhs(self,rhs:np.ndarray)->None:
        values=np.asarray(rhs,dtype=float).reshape(-1)
        if values.shape!=(self.m,) or not np.isfinite(values).all(): raise ValueError(f"target rhs must be finite length {self.m}")
        start,stop=self._rows_target
        for off,value in enumerate(values):
            scaled=float(value)/float(self._target_scale[off]); self._set_row_bounds(start+off,scaled,scaled)
        if stop-start!=self.m: raise RuntimeError("internal target-row count drift")
        self._target_rhs=values.copy()

    def _set_output_guard_bounds(self,lower:np.ndarray|None,upper:np.ndarray|None)->None:
        if self.g==0:
            if lower is not None or upper is not None: raise ValueError("guard bounds supplied without output_guard_matrix")
            return
        lo=np.full(self.g,-np.inf) if lower is None else np.asarray(lower,dtype=float).reshape(-1)
        hi=np.full(self.g,np.inf) if upper is None else np.asarray(upper,dtype=float).reshape(-1)
        if lo.shape!=(self.g,) or hi.shape!=(self.g,) or np.isnan(lo).any() or np.isnan(hi).any(): raise ValueError("guard bounds must align and contain no NaN")
        if bool((lo>hi).any()): raise ValueError("guard lower bound exceeds upper bound")
        start,stop=self._rows_output_guard
        if stop-start!=self.g: raise RuntimeError("internal output-guard row count drift")
        for i in range(self.g):
            slo=-self._inf if np.isneginf(lo[i]) else float(lo[i]/self._guard_scale[i]); shi=self._inf if np.isposinf(hi[i]) else float(hi[i]/self._guard_scale[i])
            self._set_row_bounds(start+i,slo,shi)
        self._active_guard_lower=lo.copy(); self._active_guard_upper=hi.copy()

    def _cost_weights(self,variable_slice:slice)->np.ndarray:
        if variable_slice==self._abs_u: return self._scale/self.M0
        if variable_slice==self._abs_f: return self._scale/self.F0
        if variable_slice==self._acc:
            if self.accompaniment_indices.size==0: return np.empty(0)
            w=self._scale[self.accompaniment_indices]; return w/max(1.0,float(w.max(initial=1.0)))
        return np.ones(int(variable_slice.stop)-int(variable_slice.start))

    def _set_objective(self,variable_slice:slice|None)->None:
        if self._active_cost_indices.size:
            self._require_ok(self._highs.changeColsCost(int(self._active_cost_indices.size),self._active_cost_indices,np.zeros(self._active_cost_indices.size)),"resetting lexicographic objective")
        if variable_slice is None or int(variable_slice.stop)<=int(variable_slice.start): self._active_cost_indices=np.empty(0,dtype=np.int32); return
        idx=np.arange(int(variable_slice.start),int(variable_slice.stop),dtype=np.int32); weights=self._cost_weights(variable_slice)
        self._require_ok(self._highs.changeColsCost(int(idx.size),idx,weights),"setting lexicographic objective"); self._active_cost_indices=idx

    def _reset_guards(self)->None:
        self._set_row_upper(self._row_final_cap,self._inf); self._set_row_upper(self._row_structure_face,self._inf); self._set_row_upper(self._row_final_face,self._inf)

    def _face_tolerance(self,objective:float)->float:
        return max(self.objective_absolute_tolerance,self.objective_relative_tolerance*max(1.0,abs(float(objective))))

    def _status(self)->tuple[InverseSolveStatus,str]:
        raw=self._highs.getModelStatus(); code=int(raw); enum=self._core.HighsModelStatus
        if code==int(enum.kOptimal): return "optimal",str(raw)
        if code==int(enum.kInfeasible): return "infeasible",str(raw)
        if code==int(enum.kUnbounded): return "unbounded",str(raw)
        return "numerical_failure",str(raw)

    def _configure_run_solver(self)->None:
        if self._run_count==0 and self.cold_start_solver!="choose":
            self._require_ok(self._highs.setOptionValue("solver",str(self.cold_start_solver)),"setting cold-start solver")
            if self.cold_start_solver=="ipm": self._require_ok(self._highs.setOptionValue("run_crossover","on"),"enabling IPM crossover")
        elif self._run_count>0:
            self._require_ok(self._highs.setOptionValue("solver","simplex"),"switching continuation solves to simplex")

    def _run_highs(self)->tuple[InverseSolveStatus,str,np.ndarray|None,np.ndarray|None]:
        self._configure_run_solver(); run_status=self._highs.run(); self._run_count += 1
        if int(run_status)<0: return "numerical_failure",f"HiGHS run failure: {run_status}",None,None
        status,message=self._status()
        if status!="optimal": return status,message,None,None
        solution=self._highs.getSolution(); columns=np.asarray(solution.col_value,dtype=float); rows=np.asarray(solution.row_value,dtype=float)
        if columns.shape!=(self.num_columns,) or rows.shape!=(int(self._highs.getNumRow()),): return "numerical_failure","HiGHS returned unexpected solution shape",None,None
        if not np.isfinite(columns).all() or not np.isfinite(rows).all(): return "numerical_failure","HiGHS returned non-finite primal values",None,None
        return "optimal",message,columns,rows

    def _failure(self,status:InverseSolveStatus,message:str)->InverseMechanismSolution:
        return InverseMechanismSolution(status,message,None,None,None,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)

    @staticmethod
    def _scaled_inf(residual:np.ndarray,scale:np.ndarray)->float:
        return float(np.max(np.abs(residual)/np.maximum(1.0,np.asarray(scale,dtype=float)),initial=0.0))

    def _solution(self,message:str,columns:np.ndarray)->InverseMechanismSolution:
        dx=columns[self._dx]*self._scale; u=columns[self._u]*self._scale; df=columns[self._df]*self._scale; x1=self.x0+dx
        final_l1=float(np.abs(df).sum()); structure_l1=float(np.abs(u).sum()); acc_l1=float(np.abs(dx[self.accompaniment_indices]).sum()) if self.accompaniment_indices.size else 0.0
        accounting=dx-np.asarray(self.A0.dot(dx)).reshape(-1)-df-u
        absA=self.A0.copy(); absA.data=np.abs(absA.data)
        accounting_scale=np.abs(dx)+np.abs(df)+np.abs(u)+np.asarray(absA.dot(np.abs(dx))).reshape(-1)
        target=np.asarray(self.C.dot(dx)).reshape(-1)-self._target_rhs; absC=self.C.copy(); absC.data=np.abs(absC.data)
        target_scale=np.abs(self._target_rhs)+np.asarray(absC.dot(np.abs(dx))).reshape(-1)
        low=np.asarray(self.L.dot(x1)).reshape(-1); high=np.asarray(self.U.dot(x1)).reshape(-1)
        lower_slack=u-low; upper_slack=high-u; cap_violation=np.maximum(low-u,0.0)+np.maximum(u-high,0.0)
        cap_scale=np.maximum.reduce([np.ones(self.n),np.abs(u),np.abs(low),np.abs(high)])
        guard_violation=0.0
        if self.g:
            activity=np.asarray(self.G.dot(dx)).reshape(-1); scale=np.maximum(1.0,np.abs(activity))
            finite_lo=np.isfinite(self._active_guard_lower); finite_hi=np.isfinite(self._active_guard_upper)
            if bool(finite_lo.any()): scale[finite_lo]=np.maximum(scale[finite_lo],np.abs(self._active_guard_lower[finite_lo]))
            if bool(finite_hi.any()): scale[finite_hi]=np.maximum(scale[finite_hi],np.abs(self._active_guard_upper[finite_hi]))
            violation=np.zeros(self.g)
            violation[finite_lo]=np.maximum(self._active_guard_lower[finite_lo]-activity[finite_lo],0.0)
            violation[finite_hi]=np.maximum(violation[finite_hi],np.maximum(activity[finite_hi]-self._active_guard_upper[finite_hi],0.0))
            guard_violation=float(np.max(violation/scale,initial=0.0))
        accounting_scaled=self._scaled_inf(accounting,accounting_scale); target_scaled=self._scaled_inf(target,target_scale); cap_scaled=float(np.max(cap_violation/cap_scale,initial=0.0)); output_scaled=float(np.max(np.maximum(-x1,0.0)/np.maximum(1.0,self._scale),initial=0.0))
        worst=max(accounting_scaled,target_scaled,cap_scaled,guard_violation,output_scaled)
        if not np.isfinite(worst) or worst>self.postcheck_tolerance:
            return self._failure("numerical_failure",f"original-coordinate postcheck failed: worst_scaled_residual={worst:.6g}, tolerance={self.postcheck_tolerance:.6g}")
        return InverseMechanismSolution("optimal",message,dx.copy(),u.copy(),df.copy(),final_l1/self.F0,structure_l1/self.M0,final_l1,structure_l1,acc_l1,float(x1.min(initial=np.inf)),float(np.max(np.abs(accounting),initial=0.0)),accounting_scaled,float(np.max(np.abs(target),initial=0.0)),target_scaled,float(lower_slack.min(initial=np.inf)),float(upper_slack.min(initial=np.inf)),cap_scaled,guard_violation)

    def _solve_stage(self)->tuple[InverseMechanismSolution,np.ndarray|None]:
        status,message,columns,_rows=self._run_highs()
        if status!="optimal" or columns is None: return self._failure(status,message),None
        solution=self._solution(message,columns)
        return solution,columns if solution.optimal else None

    def _minimize_with_lexicographic_followups(
        self,
        *,
        primary: Literal["structure", "final"],
        secondary: Literal["structure", "final"] | None,
    ) -> InverseMechanismSolution:
        """Solve the declared lexicographic objectives without an empty follow-up run.

        The primary objective is always solved.  When a secondary burden objective is
        requested, its optimum is found on the primary-optimal face.  A third solve is
        performed only when the caller explicitly supplied accompaniment indices; in
        that case the existing accompaniment witness rule is unchanged.  Otherwise the
        last optimized witness is already the lexicographic solution and is returned
        directly rather than asking HiGHS to solve the same face with a zero objective.
        """

        primary_slice = self._abs_u if primary == "structure" else self._abs_f
        primary_guard = self._row_structure_face if primary == "structure" else self._row_final_face
        self._set_objective(primary_slice)
        first, cols = self._solve_stage()
        if not first.optimal or cols is None:
            return first
        last = first
        primary_value = first.structure_l1 if primary == "structure" else first.final_use_l1
        self._set_row_upper(primary_guard, primary_value + self._face_tolerance(primary_value))

        if secondary is not None:
            secondary_slice = self._abs_u if secondary == "structure" else self._abs_f
            secondary_guard = self._row_structure_face if secondary == "structure" else self._row_final_face
            self._set_objective(secondary_slice)
            second, cols2 = self._solve_stage()
            if not second.optimal or cols2 is None:
                return second
            last = second
            secondary_value = second.structure_l1 if secondary == "structure" else second.final_use_l1
            self._set_row_upper(secondary_guard, secondary_value + self._face_tolerance(secondary_value))

        if self.accompaniment_indices.size:
            self._set_objective(self._acc)
            final, _ = self._solve_stage()
            return final
        return last

    def _prepare_solve(self,target_rhs:np.ndarray,guard_lower:np.ndarray|None,guard_upper:np.ndarray|None)->None:
        self._set_target_rhs(target_rhs); self._set_output_guard_bounds(guard_lower,guard_upper); self._reset_guards()

    def solve_epsilon(self,target_rhs:np.ndarray,eta_final_use:float,*,guard_lower:np.ndarray|None=None,guard_upper:np.ndarray|None=None)->InverseMechanismSolution:
        eta=float(eta_final_use)
        if not np.isfinite(eta) or eta<0.0: raise ValueError("eta_final_use must be finite and non-negative")
        self._ensure_capacity_rows(); self._prepare_solve(target_rhs,guard_lower,guard_upper); self._set_row_upper(self._row_final_cap,eta*self.F0)
        return self._minimize_with_lexicographic_followups(primary="structure",secondary="final")

    def solve_min_final_use(self,target_rhs:np.ndarray,*,guard_lower:np.ndarray|None=None,guard_upper:np.ndarray|None=None)->InverseMechanismSolution:
        self._ensure_capacity_rows(); self._prepare_solve(target_rhs,guard_lower,guard_upper)
        return self._minimize_with_lexicographic_followups(primary="final",secondary="structure")

    def solve_pure_final_use(self,target_rhs:np.ndarray,*,guard_lower:np.ndarray|None=None,guard_upper:np.ndarray|None=None)->InverseMechanismSolution:
        self._prepare_solve(target_rhs,guard_lower,guard_upper); self._set_row_upper(self._row_structure_face,0.0)
        return self._minimize_with_lexicographic_followups(primary="final",secondary=None)

    def solve_pure_structure(self,target_rhs:np.ndarray,*,guard_lower:np.ndarray|None=None,guard_upper:np.ndarray|None=None)->InverseMechanismSolution:
        self._ensure_capacity_rows(); self._prepare_solve(target_rhs,guard_lower,guard_upper); self._set_row_upper(self._row_final_face,0.0)
        return self._minimize_with_lexicographic_followups(primary="structure",secondary=None)

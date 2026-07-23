from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd

FlowRole = Literal["DOM", "IMP", "TOTAL"]
CheckStatus = Literal["pass", "fail", "not_applicable"]
CheckSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class InvariantCheck:
    check_id: str
    category: str
    status: CheckStatus
    severity: CheckSeverity
    flow_role: FlowRole
    block: str
    max_abs_gap: float | None
    tolerance: float | None
    affected_labels: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class InvariantReport:
    checks: tuple[InvariantCheck, ...]

    @property
    def ok(self) -> bool:
        return not any(check.status == "fail" and check.severity == "error" for check in self.checks)

    def require_ok(self) -> None:
        if self.ok:
            return
        failed = [check for check in self.checks if check.status == "fail" and check.severity == "error"]
        summary = "; ".join(f"{check.check_id}: {check.explanation}" for check in failed[:10])
        raise ValueError(f"IOT invariant validation failed: {summary}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check_id": check.check_id,
                    "category": check.category,
                    "status": check.status,
                    "severity": check.severity,
                    "flow_role": check.flow_role,
                    "block": check.block,
                    "max_abs_gap": check.max_abs_gap,
                    "tolerance": check.tolerance,
                    "affected_labels": "|".join(check.affected_labels),
                    "explanation": check.explanation,
                }
                for check in self.checks
            ]
        )


@dataclass(frozen=True)
class RowIdentityPolicy:
    total_final_use_column: str | None = None
    total_use_column: str | None = None
    output_row: str | None = None
    imports_row: str | None = None
    total_supply_basic_row: str | None = None


def _normalise_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = frame.copy().astype("float64")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    if out.index.has_duplicates:
        duplicates = out.index[out.index.duplicated()].astype(str).tolist()
        raise ValueError(f"{name} contains duplicate row labels: {duplicates[:20]!r}")
    if out.columns.has_duplicates:
        duplicates = out.columns[out.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{name} contains duplicate column labels: {duplicates[:20]!r}")
    return out


def _comparison_check(
    *,
    check_id: str,
    category: str,
    flow_role: FlowRole,
    block: str,
    observed: pd.Series,
    derived: pd.Series,
    rtol: float,
    atol: float,
    explanation: str,
) -> InvariantCheck:
    observed_aligned = observed.astype("float64")
    derived_aligned = derived.reindex(observed_aligned.index).astype("float64")
    values_observed = observed_aligned.to_numpy(dtype="float64")
    values_derived = derived_aligned.to_numpy(dtype="float64")
    finite = np.isfinite(values_observed) & np.isfinite(values_derived)
    tolerance = atol + rtol * np.maximum(np.abs(values_observed), np.abs(values_derived))
    gaps = np.abs(values_observed - values_derived)
    passed = finite & (gaps <= tolerance)
    failed_index = cast(pd.Index, observed_aligned.index)
    failed_labels = failed_index[~passed].astype(str).tolist()
    finite_gaps = gaps[np.isfinite(gaps)]
    finite_tolerances = tolerance[np.isfinite(tolerance)]
    max_abs_gap = float(finite_gaps.max()) if finite_gaps.size else float("inf")
    max_tolerance = float(finite_tolerances.max()) if finite_tolerances.size else float(atol)
    return InvariantCheck(
        check_id=check_id,
        category=category,
        status="pass" if not failed_labels else "fail",
        severity="error",
        flow_role=flow_role,
        block=block,
        max_abs_gap=max_abs_gap,
        tolerance=max_tolerance,
        affected_labels=tuple(failed_labels[:50]),
        explanation=explanation,
    )


def _missing_role_check(
    *, check_id: str, flow_role: FlowRole, block: str, explanation: str
) -> InvariantCheck:
    return InvariantCheck(
        check_id=check_id,
        category="row_identity",
        status="not_applicable",
        severity="info",
        flow_role=flow_role,
        block=block,
        max_abs_gap=None,
        tolerance=None,
        affected_labels=(),
        explanation=explanation,
    )


def validate_structural_and_row_invariants(
    *,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    flow_role: FlowRole,
    final_demand_columns: Sequence[str],
    V: pd.DataFrame | None = None,
    policy: RowIdentityPolicy = RowIdentityPolicy(),
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> InvariantReport:
    """Validate generated-table structure and row-side accounting identities.

    This function deliberately uses strict numerical tolerances. Independently
    rounded published Eurostat controls remain in the existing source-rounding
    validation path and are not accepted through this generated-table validator.
    """
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unsupported flow_role: {flow_role!r}")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")

    Z_ord = _normalise_frame(Z, name="Z")
    Y_ord = _normalise_frame(Y, name="Y")
    if not Z_ord.index.equals(Z_ord.columns):
        raise ValueError("Z must be square with identical ordered row and column labels")
    if not Y_ord.index.equals(Z_ord.index):
        raise ValueError("Y rows must exactly match the ordered Z product domain")

    selected = tuple(map(str, final_demand_columns))
    if not selected:
        raise ValueError("final_demand_columns must contain at least one column")
    selected_index = pd.Index(selected)
    if selected_index.has_duplicates:
        duplicates = selected_index[selected_index.duplicated()].astype(str).tolist()
        raise ValueError(f"final_demand_columns contains duplicates: {duplicates[:20]!r}")
    missing = [code for code in selected if code not in set(Y_ord.columns)]
    if missing:
        raise ValueError(f"Selected final-demand columns are absent from Y: {missing[:20]!r}")
    atomic_Y = Y_ord.loc[:, list(selected)]
    if not np.isfinite(Z_ord.to_numpy(dtype="float64")).all():
        raise ValueError("Z contains non-finite transaction values")
    if not np.isfinite(atomic_Y.to_numpy(dtype="float64")).all():
        raise ValueError("Selected final-demand block contains non-finite values")

    intermediate_use = Z_ord.sum(axis=1).astype("float64")
    final_use = atomic_Y.sum(axis=1).astype("float64")
    row_use_total = intermediate_use.add(final_use, fill_value=0.0).astype("float64")
    checks: list[InvariantCheck] = []

    if policy.total_final_use_column is None:
        checks.append(
            _missing_role_check(
                check_id="row.total_final_use",
                flow_role=flow_role,
                block="Y",
                explanation="No total-final-use control column was declared by policy.",
            )
        )
    else:
        column = str(policy.total_final_use_column)
        if column not in Y_ord.columns:
            raise ValueError(f"Declared total-final-use column is absent from Y: {column!r}")
        checks.append(
            _comparison_check(
                check_id="row.total_final_use",
                category="row_identity",
                flow_role=flow_role,
                block="Y",
                observed=cast(pd.Series, Y_ord[column]),
                derived=final_use,
                rtol=rtol,
                atol=atol,
                explanation="Total final use must equal the selected non-overlapping final-demand basis.",
            )
        )

    if policy.total_use_column is None:
        checks.append(
            _missing_role_check(
                check_id="row.total_use",
                flow_role=flow_role,
                block="Y",
                explanation="No total-use control column was declared by policy.",
            )
        )
    else:
        column = str(policy.total_use_column)
        if column not in Y_ord.columns:
            raise ValueError(f"Declared total-use column is absent from Y: {column!r}")
        checks.append(
            _comparison_check(
                check_id="row.total_use",
                category="row_identity",
                flow_role=flow_role,
                block="Y",
                observed=cast(pd.Series, Y_ord[column]),
                derived=row_use_total,
                rtol=rtol,
                atol=atol,
                explanation="Total use must equal intermediate use plus atomic final use.",
            )
        )

    V_ord = None if V is None else _normalise_frame(V, name="V")
    if V_ord is not None and not V_ord.columns.equals(Z_ord.columns):
        raise ValueError("V columns must exactly match the ordered Z product domain")

    def require_v_row(label: str, role_name: str) -> pd.Series:
        if V_ord is None:
            raise ValueError(f"{role_name} row was declared but V is absent")
        if label not in V_ord.index:
            raise ValueError(f"Declared {role_name} row is absent from V: {label!r}")
        row = V_ord.loc[label]
        if not isinstance(row, pd.Series):
            raise ValueError(f"Declared {role_name} row is not unique: {label!r}")
        if not np.isfinite(row.to_numpy(dtype="float64")).all():
            raise ValueError(f"Declared {role_name} row contains non-finite values: {label!r}")
        return row.astype("float64")

    if flow_role == "DOM":
        if policy.output_row is None:
            checks.append(
                _missing_role_check(
                    check_id="row.domestic_output",
                    flow_role=flow_role,
                    block="V",
                    explanation="No domestic-output row was declared by policy.",
                )
            )
        else:
            checks.append(
                _comparison_check(
                    check_id="row.domestic_output",
                    category="row_identity",
                    flow_role=flow_role,
                    block="V",
                    observed=require_v_row(str(policy.output_row), "output"),
                    derived=row_use_total,
                    rtol=rtol,
                    atol=atol,
                    explanation="Domestic output must equal domestic intermediate plus final uses.",
                )
            )
    elif flow_role == "IMP":
        if policy.imports_row is None:
            checks.append(
                _missing_role_check(
                    check_id="row.imported_supply",
                    flow_role=flow_role,
                    block="V",
                    explanation="No imports-supply row was declared by policy.",
                )
            )
        else:
            checks.append(
                _comparison_check(
                    check_id="row.imported_supply",
                    category="row_identity",
                    flow_role=flow_role,
                    block="V",
                    observed=require_v_row(str(policy.imports_row), "imports"),
                    derived=row_use_total,
                    rtol=rtol,
                    atol=atol,
                    explanation="Imported supply must equal all imported intermediate and final uses.",
                )
            )
    else:
        if policy.total_supply_basic_row is None:
            checks.append(
                _missing_role_check(
                    check_id="row.total_supply_equals_use",
                    flow_role=flow_role,
                    block="V",
                    explanation="No total-supply-at-basic-prices row was declared by policy.",
                )
            )
        else:
            total_supply = require_v_row(str(policy.total_supply_basic_row), "total supply")
            checks.append(
                _comparison_check(
                    check_id="row.total_supply_equals_use",
                    category="supply_use",
                    flow_role=flow_role,
                    block="V|Z|Y",
                    observed=total_supply,
                    derived=row_use_total,
                    rtol=rtol,
                    atol=atol,
                    explanation="Total supply at basic prices must equal total intermediate and final use.",
                )
            )
            if policy.output_row is not None and policy.imports_row is not None:
                output = require_v_row(str(policy.output_row), "output")
                imports = require_v_row(str(policy.imports_row), "imports")
                checks.append(
                    _comparison_check(
                        check_id="row.total_supply_components",
                        category="supply_identity",
                        flow_role=flow_role,
                        block="V",
                        observed=total_supply,
                        derived=output.add(imports, fill_value=0.0),
                        rtol=rtol,
                        atol=atol,
                        explanation="Total supply at basic prices must equal domestic output plus imports.",
                    )
                )
            else:
                checks.append(
                    _missing_role_check(
                        check_id="row.total_supply_components",
                        flow_role=flow_role,
                        block="V",
                        explanation="Output and imports rows were not both declared by policy.",
                    )
                )

    return InvariantReport(checks=tuple(checks))

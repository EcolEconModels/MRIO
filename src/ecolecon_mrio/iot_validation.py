from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd

FlowRole = Literal["DOM", "IMP", "TOTAL"]
CheckStatus = Literal["pass", "fail", "not_applicable", "not_tested"]
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
    published: float | None = None
    derived: float | None = None
    signed_gap: float | None = None
    relative_gap: float | None = None


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
                    "published": check.published,
                    "derived": check.derived,
                    "signed_gap": check.signed_gap,
                    "max_abs_gap": check.max_abs_gap,
                    "relative_gap": check.relative_gap,
                    "tolerance": check.tolerance,
                    "affected_labels": "|".join(check.affected_labels),
                    "explanation": check.explanation,
                }
                for check in self.checks
            ]
        )


@dataclass(frozen=True)
class BlockMetadata:
    """Declare comparison-critical unit and valuation metadata for one block."""

    unit: str
    valuation: str
    price_measure: str = "current"
    scale: float = 1.0


@dataclass(frozen=True)
class RowIdentityPolicy:
    total_final_use_column: str | None = None
    total_use_column: str | None = None
    output_row: str | None = None
    imports_row: str | None = None
    supply_equals_use_row: str | None = None


@dataclass(frozen=True)
class ColumnIdentityPolicy:
    """Map source rows to generated production-column accounting roles.

    ``valuation_adjustment_rows`` is explicit and may be empty. The validator
    never infers whether product taxes, subsidies, margins or other valuation
    rows belong in intermediate consumption from their labels.
    """

    intermediate_consumption_row: str | None = None
    gross_value_added_row: str | None = None
    output_row: str | None = None
    valuation_adjustment_rows: tuple[str, ...] = ()


@dataclass(frozen=True)
class CrossFlowPolicy:
    """Map TOTAL-table controls to DOM/IMP cross-flow accounting roles."""

    total_output_row: str | None = None
    total_imports_row: str | None = None
    imported_use_intermediate_row: str | None = None
    imported_use_final_row: str | None = None



@dataclass(frozen=True)
class AdditivePartition:
    """Declare one aggregate equal to a complete, non-overlapping component set."""

    check_id: str
    aggregate: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class FinalDemandPartitionPolicy:
    """Declare generated final-demand aggregate/detail identities."""

    partitions: tuple[AdditivePartition, ...] = ()
    allow_component_reuse: bool = False


@dataclass(frozen=True)
class ValueAddedPartitionPolicy:
    """Declare generated value-added aggregate/component identities."""

    partitions: tuple[AdditivePartition, ...] = ()
    allow_component_reuse: bool = False

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


def _gap_diagnostics(
    observed: np.ndarray,
    derived: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None]:
    finite = np.isfinite(observed) & np.isfinite(derived)
    if not finite.any():
        return None, None, None, None
    signed = observed - derived
    absolute = np.abs(signed)
    masked_absolute = np.where(finite, absolute, -np.inf)
    flat_position = int(np.argmax(masked_absolute))
    published = float(observed.flat[flat_position])
    calculated = float(derived.flat[flat_position])
    signed_gap = float(signed.flat[flat_position])
    denominator = max(abs(published), abs(calculated))
    relative_gap = 0.0 if denominator == 0.0 else signed_gap / denominator
    return published, calculated, signed_gap, float(relative_gap)


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
    derived_aligned = derived.astype("float64")
    if not observed_aligned.index.equals(derived_aligned.index):
        raise ValueError(f"{check_id}: compared series must have identical ordered labels")
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
    published, calculated, signed_gap, relative_gap = _gap_diagnostics(
        values_observed,
        values_derived,
    )
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
        published=published,
        derived=calculated,
        signed_gap=signed_gap,
        relative_gap=relative_gap,
    )


def _frame_comparison_check(
    *,
    check_id: str,
    category: str,
    block: str,
    flow_role: FlowRole = "TOTAL",
    observed: pd.DataFrame,
    derived: pd.DataFrame,
    rtol: float,
    atol: float,
    explanation: str,
) -> InvariantCheck:
    if not observed.index.equals(derived.index) or not observed.columns.equals(derived.columns):
        raise ValueError(f"{check_id}: compared frames must have identical ordered labels")
    observed_values = observed.to_numpy(dtype="float64")
    derived_values = derived.to_numpy(dtype="float64")
    finite = np.isfinite(observed_values) & np.isfinite(derived_values)
    tolerance = atol + rtol * np.maximum(np.abs(observed_values), np.abs(derived_values))
    gaps = np.abs(observed_values - derived_values)
    passed = finite & (gaps <= tolerance)
    failed_positions = np.argwhere(~passed)
    failed_labels = tuple(
        f"{observed.index[int(row)]}|{observed.columns[int(column)]}"
        for row, column in failed_positions[:50]
    )
    finite_gaps = gaps[np.isfinite(gaps)]
    finite_tolerances = tolerance[np.isfinite(tolerance)]
    published, calculated, signed_gap, relative_gap = _gap_diagnostics(
        observed_values,
        derived_values,
    )
    return InvariantCheck(
        check_id=check_id,
        category=category,
        status="pass" if not failed_labels else "fail",
        severity="error",
        flow_role=flow_role,
        block=block,
        max_abs_gap=float(finite_gaps.max()) if finite_gaps.size else float("inf"),
        tolerance=float(finite_tolerances.max()) if finite_tolerances.size else float(atol),
        affected_labels=failed_labels,
        explanation=explanation,
        published=published,
        derived=calculated,
        signed_gap=signed_gap,
        relative_gap=relative_gap,
    )


def _missing_role_check(
    *,
    check_id: str,
    flow_role: FlowRole,
    block: str,
    explanation: str,
    category: str = "row_identity",
) -> InvariantCheck:
    return InvariantCheck(
        check_id=check_id,
        category=category,
        status="not_applicable",
        severity="info",
        flow_role=flow_role,
        block=block,
        max_abs_gap=None,
        tolerance=None,
        affected_labels=(),
        explanation=explanation,
    )


def _not_tested_check(
    *,
    check_id: str,
    category: str,
    flow_role: FlowRole,
    block: str,
    explanation: str,
) -> InvariantCheck:
    return InvariantCheck(
        check_id=check_id,
        category=category,
        status="not_tested",
        severity="info",
        flow_role=flow_role,
        block=block,
        max_abs_gap=None,
        tolerance=None,
        affected_labels=(),
        explanation=explanation,
    )


def _require_finite_row(
    frame: pd.DataFrame | None,
    label: str,
    role_name: str,
    *,
    block_name: str,
) -> pd.Series:
    if frame is None:
        raise ValueError(f"{role_name} row was declared but {block_name} is absent")
    if label not in frame.index:
        raise ValueError(f"Declared {role_name} row is absent from {block_name}: {label!r}")
    row = frame.loc[label]
    if not isinstance(row, pd.Series):
        raise ValueError(f"Declared {role_name} row is not unique: {label!r}")
    if not np.isfinite(row.to_numpy(dtype="float64")).all():
        raise ValueError(f"Declared {role_name} row contains non-finite values: {label!r}")
    return row.astype("float64")


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
                    observed=_require_finite_row(V_ord, str(policy.output_row), "output", block_name="V"),
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
                    observed=_require_finite_row(V_ord, str(policy.imports_row), "imports", block_name="V"),
                    derived=row_use_total,
                    rtol=rtol,
                    atol=atol,
                    explanation="Imported supply must equal all imported intermediate and final uses.",
                )
            )
    else:
        if policy.supply_equals_use_row is None:
            checks.append(
                _missing_role_check(
                    check_id="row.total_supply_equals_use",
                    flow_role=flow_role,
                    block="V",
                    explanation="No same-valuation supply control row was declared by policy.",
                )
            )
        else:
            total_supply = _require_finite_row(
                V_ord, str(policy.supply_equals_use_row), "total supply", block_name="V"
            )
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
                    explanation="Supply and use must balance at the common valuation selected by policy.",
                )
            )
            if policy.output_row is not None and policy.imports_row is not None:
                output = _require_finite_row(V_ord, str(policy.output_row), "output", block_name="V")
                imports = _require_finite_row(V_ord, str(policy.imports_row), "imports", block_name="V")
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
                        explanation="The selected supply control must equal domestic output plus imports at the declared valuation.",
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


def validate_column_invariants(
    *,
    Z: pd.DataFrame,
    V: pd.DataFrame | None,
    flow_role: FlowRole,
    policy: ColumnIdentityPolicy = ColumnIdentityPolicy(),
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> InvariantReport:
    """Validate generated production-column intermediate-input and value-added accounts.

    The policy explicitly declares every valuation-adjustment row added to the
    column sum of ``Z``. This function does not infer tax, subsidy or margin
    treatment from row labels and does not apply published-source rounding.
    """
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unsupported flow_role: {flow_role!r}")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")

    Z_ord = _normalise_frame(Z, name="Z")
    if not Z_ord.index.equals(Z_ord.columns):
        raise ValueError("Z must be square with identical ordered row and column labels")
    if not np.isfinite(Z_ord.to_numpy(dtype="float64")).all():
        raise ValueError("Z contains non-finite transaction values")

    V_ord = None if V is None else _normalise_frame(V, name="V")
    if V_ord is not None and not V_ord.columns.equals(Z_ord.columns):
        raise ValueError("V columns must exactly match the ordered Z product domain")

    adjustment_labels = tuple(map(str, policy.valuation_adjustment_rows))
    adjustment_index = pd.Index(adjustment_labels)
    if adjustment_index.has_duplicates:
        duplicates = adjustment_index[adjustment_index.duplicated()].astype(str).tolist()
        raise ValueError(f"valuation_adjustment_rows contains duplicates: {duplicates[:20]!r}")

    intermediate = Z_ord.sum(axis=0).astype("float64")
    for label in adjustment_labels:
        intermediate = intermediate.add(
            _require_finite_row(V_ord, label, "valuation adjustment", block_name="V"),
            fill_value=0.0,
        ).astype("float64")

    checks: list[InvariantCheck] = []
    if policy.intermediate_consumption_row is None:
        checks.append(
            _missing_role_check(
                check_id="column.intermediate_consumption",
                category="column_identity",
                flow_role=flow_role,
                block="V|Z",
                explanation="No intermediate-consumption row was declared by policy.",
            )
        )
    else:
        observed_intermediate = _require_finite_row(
            V_ord,
            str(policy.intermediate_consumption_row),
            "intermediate consumption",
            block_name="V",
        )
        checks.append(
            _comparison_check(
                check_id="column.intermediate_consumption",
                category="column_identity",
                flow_role=flow_role,
                block="V|Z",
                observed=observed_intermediate,
                derived=intermediate,
                rtol=rtol,
                atol=atol,
                explanation=(
                    "Intermediate consumption must equal the Z column sum plus exactly "
                    "the valuation-adjustment rows declared by policy."
                ),
            )
        )

    if policy.output_row is None or policy.gross_value_added_row is None:
        checks.append(
            _missing_role_check(
                check_id="column.output_components",
                category="column_identity",
                flow_role=flow_role,
                block="V|Z",
                explanation="Output and gross-value-added rows were not both declared by policy.",
            )
        )
    else:
        output = _require_finite_row(
            V_ord, str(policy.output_row), "output", block_name="V"
        )
        value_added = _require_finite_row(
            V_ord,
            str(policy.gross_value_added_row),
            "gross value added",
            block_name="V",
        )
        checks.append(
            _comparison_check(
                check_id="column.output_components",
                category="column_identity",
                flow_role=flow_role,
                block="V|Z",
                observed=output,
                derived=intermediate.add(value_added, fill_value=0.0),
                rtol=rtol,
                atol=atol,
                explanation=(
                    "Output must equal policy-defined intermediate consumption plus "
                    "gross value added."
                ),
            )
        )

    return InvariantReport(checks=tuple(checks))


def validate_cross_flow_invariants(
    *,
    total_Z: pd.DataFrame,
    total_Y: pd.DataFrame,
    domestic_Z: pd.DataFrame,
    domestic_Y: pd.DataFrame,
    imports_Z: pd.DataFrame,
    imports_Y: pd.DataFrame,
    final_demand_columns: Sequence[str],
    total_V: pd.DataFrame | None = None,
    total_M: pd.DataFrame | None = None,
    policy: CrossFlowPolicy = CrossFlowPolicy(),
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> InvariantReport:
    """Validate generated TOTAL, DOM and IMP transaction and control bridges."""
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")

    total_Z_ord = _normalise_frame(total_Z, name="total_Z")
    domestic_Z_ord = _normalise_frame(domestic_Z, name="domestic_Z")
    imports_Z_ord = _normalise_frame(imports_Z, name="imports_Z")
    total_Y_ord = _normalise_frame(total_Y, name="total_Y")
    domestic_Y_ord = _normalise_frame(domestic_Y, name="domestic_Y")
    imports_Y_ord = _normalise_frame(imports_Y, name="imports_Y")

    for name, frame in (("total_Z", total_Z_ord), ("domestic_Z", domestic_Z_ord), ("imports_Z", imports_Z_ord)):
        if not frame.index.equals(frame.columns):
            raise ValueError(f"{name} must be square with identical ordered row and column labels")
        if not np.isfinite(frame.to_numpy(dtype="float64")).all():
            raise ValueError(f"{name} contains non-finite transaction values")
    if not domestic_Z_ord.index.equals(total_Z_ord.index) or not imports_Z_ord.index.equals(total_Z_ord.index):
        raise ValueError("TOTAL, DOM and IMP Z blocks must have identical ordered product domains")

    for name, frame in (("total_Y", total_Y_ord), ("domestic_Y", domestic_Y_ord), ("imports_Y", imports_Y_ord)):
        if not frame.index.equals(total_Z_ord.index):
            raise ValueError(f"{name} rows must exactly match the ordered Z product domain")

    selected = tuple(map(str, final_demand_columns))
    if not selected:
        raise ValueError("final_demand_columns must contain at least one column")
    selected_index = pd.Index(selected)
    if selected_index.has_duplicates:
        duplicates = selected_index[selected_index.duplicated()].astype(str).tolist()
        raise ValueError(f"final_demand_columns contains duplicates: {duplicates[:20]!r}")
    for name, frame in (("total_Y", total_Y_ord), ("domestic_Y", domestic_Y_ord), ("imports_Y", imports_Y_ord)):
        missing = [column for column in selected if column not in set(frame.columns)]
        if missing:
            raise ValueError(f"Selected final-demand columns are absent from {name}: {missing[:20]!r}")

    total_atomic_Y = total_Y_ord.loc[:, list(selected)]
    domestic_atomic_Y = domestic_Y_ord.loc[:, list(selected)]
    imports_atomic_Y = imports_Y_ord.loc[:, list(selected)]
    for name, frame in (("total_Y", total_atomic_Y), ("domestic_Y", domestic_atomic_Y), ("imports_Y", imports_atomic_Y)):
        if not np.isfinite(frame.to_numpy(dtype="float64")).all():
            raise ValueError(f"Selected final-demand block in {name} contains non-finite values")

    checks: list[InvariantCheck] = [
        _frame_comparison_check(
            check_id="cross_flow.intermediate_transactions",
            category="cross_flow",
            block="Z_TOTAL|Z_DOM|Z_IMP",
            observed=total_Z_ord,
            derived=domestic_Z_ord + imports_Z_ord,
            rtol=rtol,
            atol=atol,
            explanation="TOTAL intermediate transactions must equal DOM plus IMP cell by cell.",
        ),
        _frame_comparison_check(
            check_id="cross_flow.final_uses",
            category="cross_flow",
            block="Y_TOTAL|Y_DOM|Y_IMP",
            observed=total_atomic_Y,
            derived=domestic_atomic_Y + imports_atomic_Y,
            rtol=rtol,
            atol=atol,
            explanation="TOTAL atomic final uses must equal DOM plus IMP cell by cell.",
        ),
    ]

    total_V_ord = None if total_V is None else _normalise_frame(total_V, name="total_V")
    if total_V_ord is not None and not total_V_ord.columns.equals(total_Z_ord.columns):
        raise ValueError("total_V columns must exactly match the ordered Z product domain")
    total_M_ord = None if total_M is None else _normalise_frame(total_M, name="total_M")

    domestic_supply = domestic_Z_ord.sum(axis=1).add(domestic_atomic_Y.sum(axis=1), fill_value=0.0).astype("float64")
    imported_supply = imports_Z_ord.sum(axis=1).add(imports_atomic_Y.sum(axis=1), fill_value=0.0).astype("float64")

    bridge_specs = (
        ("cross_flow.domestic_output_bridge", policy.total_output_row, "V_TOTAL|Z_DOM|Y_DOM", "TOTAL-table output", domestic_supply, "TOTAL-table output must equal DOM intermediate plus final uses by product."),
        ("cross_flow.imported_supply_bridge", policy.total_imports_row, "V_TOTAL|Z_IMP|Y_IMP", "TOTAL-table imports", imported_supply, "TOTAL-table imports must equal all IMP intermediate and final uses by product."),
    )
    for check_id, label, block, role_name, derived, explanation in bridge_specs:
        if label is None:
            checks.append(_missing_role_check(check_id=check_id, category="cross_flow", flow_role="TOTAL", block=block, explanation=f"No {role_name} row was declared by policy."))
        else:
            checks.append(_comparison_check(check_id=check_id, category="cross_flow", flow_role="TOTAL", block=block, observed=_require_finite_row(total_V_ord, str(label), role_name, block_name="total_V"), derived=derived, rtol=rtol, atol=atol, explanation=explanation))

    if policy.imported_use_intermediate_row is None:
        checks.append(_missing_role_check(check_id="cross_flow.imported_intermediate_use", category="cross_flow", flow_role="TOTAL", block="V_TOTAL|Z_IMP", explanation="No imported-intermediate-use control row was declared by policy."))
    else:
        checks.append(_comparison_check(check_id="cross_flow.imported_intermediate_use", category="cross_flow", flow_role="TOTAL", block="V_TOTAL|Z_IMP", observed=_require_finite_row(total_V_ord, str(policy.imported_use_intermediate_row), "imported intermediate use", block_name="total_V"), derived=imports_Z_ord.sum(axis=0).astype("float64"), rtol=rtol, atol=atol, explanation="Imported-use controls under production columns must equal IMP Z column sums."))

    if policy.imported_use_final_row is None:
        checks.append(_missing_role_check(check_id="cross_flow.imported_final_use", category="cross_flow", flow_role="TOTAL", block="M_TOTAL|Y_IMP", explanation="No imported-final-use control row was declared by policy."))
    else:
        if total_M_ord is None:
            raise ValueError("imported final use row was declared but total_M is absent")
        missing_M = [column for column in selected if column not in set(total_M_ord.columns)]
        if missing_M:
            raise ValueError(f"Selected final-demand columns are absent from total_M: {missing_M[:20]!r}")
        observed_final = _require_finite_row(total_M_ord.loc[:, list(selected)], str(policy.imported_use_final_row), "imported final use", block_name="total_M")
        checks.append(_comparison_check(check_id="cross_flow.imported_final_use", category="cross_flow", flow_role="TOTAL", block="M_TOTAL|Y_IMP", observed=observed_final, derived=imports_atomic_Y.sum(axis=0).astype("float64"), rtol=rtol, atol=atol, explanation="Imported-use controls under final-use columns must equal IMP Y column sums."))

    return InvariantReport(checks=tuple(checks))


def _validated_partitions(
    partitions: Sequence[AdditivePartition],
    *,
    policy_name: str,
    allow_component_reuse: bool,
) -> tuple[AdditivePartition, ...]:
    normalized = tuple(
        AdditivePartition(
            check_id=str(partition.check_id),
            aggregate=str(partition.aggregate),
            components=tuple(map(str, partition.components)),
        )
        for partition in partitions
    )
    check_ids = pd.Index([partition.check_id for partition in normalized])
    if check_ids.has_duplicates:
        duplicates = check_ids[check_ids.duplicated()].astype(str).tolist()
        raise ValueError(f"{policy_name} contains duplicate check_id values: {duplicates[:20]!r}")
    aggregates = pd.Index([partition.aggregate for partition in normalized])
    if aggregates.has_duplicates:
        duplicates = aggregates[aggregates.duplicated()].astype(str).tolist()
        raise ValueError(f"{policy_name} contains duplicate aggregate roles: {duplicates[:20]!r}")
    for partition in normalized:
        if not partition.check_id:
            raise ValueError(f"{policy_name} contains an empty check_id")
        if not partition.aggregate:
            raise ValueError(f"{policy_name} contains an empty aggregate label")
        if not partition.components:
            raise ValueError(
                f"{policy_name} partition {partition.check_id!r} must contain at least one component"
            )
        component_index = pd.Index(partition.components)
        if component_index.has_duplicates:
            duplicates = component_index[component_index.duplicated()].astype(str).tolist()
            raise ValueError(
                f"{policy_name} partition {partition.check_id!r} contains duplicate components: "
                f"{duplicates[:20]!r}"
            )
        if partition.aggregate in set(partition.components):
            raise ValueError(
                f"{policy_name} partition {partition.check_id!r} includes its aggregate as a component"
            )
    if not allow_component_reuse:
        owners_by_component: dict[str, str] = {}
        reused: list[str] = []
        for partition in normalized:
            for component in partition.components:
                previous = owners_by_component.setdefault(component, partition.check_id)
                if previous != partition.check_id:
                    reused.append(component)
        if reused:
            raise ValueError(
                f"{policy_name} reuses components across partitions: "
                f"{sorted(set(reused))[:20]!r}"
            )
    return normalized


def validate_final_demand_partitions(
    *,
    Y: pd.DataFrame,
    flow_role: FlowRole,
    policy: FinalDemandPartitionPolicy = FinalDemandPartitionPolicy(),
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> InvariantReport:
    """Validate exact generated aggregate/detail identities across Y columns."""
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unsupported flow_role: {flow_role!r}")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")
    Y_ord = _normalise_frame(Y, name="Y")
    partitions = _validated_partitions(
        policy.partitions,
        policy_name="final-demand partition policy",
        allow_component_reuse=policy.allow_component_reuse,
    )
    if not partitions:
        return InvariantReport(
            checks=(
                _missing_role_check(
                    check_id="partition.final_demand",
                    category="partition_identity",
                    flow_role=flow_role,
                    block="Y",
                    explanation="No final-demand aggregate/detail partitions were declared by policy.",
                ),
            )
        )

    checks: list[InvariantCheck] = []
    for partition in partitions:
        required = (partition.aggregate, *partition.components)
        missing = [label for label in required if label not in set(Y_ord.columns)]
        if missing:
            raise ValueError(
                f"Declared final-demand partition {partition.check_id!r} is missing Y columns: "
                f"{missing[:20]!r}"
            )
        block = Y_ord.loc[:, list(required)]
        if not np.isfinite(block.to_numpy(dtype="float64")).all():
            raise ValueError(
                f"Declared final-demand partition {partition.check_id!r} contains non-finite values"
            )
        observed = cast(pd.Series, Y_ord[partition.aggregate]).astype("float64")
        derived = Y_ord.loc[:, list(partition.components)].sum(axis=1).astype("float64")
        checks.append(
            _comparison_check(
                check_id=partition.check_id,
                category="partition_identity",
                flow_role=flow_role,
                block="Y",
                observed=observed,
                derived=derived,
                rtol=rtol,
                atol=atol,
                explanation=(
                    f"Final-demand aggregate {partition.aggregate!r} must equal the complete "
                    f"declared component set {partition.components!r}."
                ),
            )
        )
    return InvariantReport(checks=tuple(checks))


def validate_value_added_partitions(
    *,
    V: pd.DataFrame | None,
    flow_role: FlowRole,
    policy: ValueAddedPartitionPolicy = ValueAddedPartitionPolicy(),
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> InvariantReport:
    """Validate exact generated value-added aggregate/component identities across V rows."""
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unsupported flow_role: {flow_role!r}")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")
    partitions = _validated_partitions(
        policy.partitions,
        policy_name="value-added partition policy",
        allow_component_reuse=policy.allow_component_reuse,
    )
    if not partitions:
        return InvariantReport(
            checks=(
                _missing_role_check(
                    check_id="partition.value_added",
                    category="partition_identity",
                    flow_role=flow_role,
                    block="V",
                    explanation="No value-added aggregate/component partitions were declared by policy.",
                ),
            )
        )
    if V is None:
        raise ValueError("Value-added partitions were declared but V is absent")
    V_ord = _normalise_frame(V, name="V")

    checks: list[InvariantCheck] = []
    for partition in partitions:
        required = (partition.aggregate, *partition.components)
        missing = [label for label in required if label not in set(V_ord.index)]
        if missing:
            raise ValueError(
                f"Declared value-added partition {partition.check_id!r} is missing V rows: "
                f"{missing[:20]!r}"
            )
        block = V_ord.loc[list(required), :]
        if not np.isfinite(block.to_numpy(dtype="float64")).all():
            raise ValueError(
                f"Declared value-added partition {partition.check_id!r} contains non-finite values"
            )
        observed = _require_finite_row(
            V_ord,
            partition.aggregate,
            "value-added aggregate",
            block_name="V",
        )
        derived = V_ord.loc[list(partition.components), :].sum(axis=0).astype("float64")
        checks.append(
            _comparison_check(
                check_id=partition.check_id,
                category="partition_identity",
                flow_role=flow_role,
                block="V",
                observed=observed,
                derived=derived,
                rtol=rtol,
                atol=atol,
                explanation=(
                    f"Value-added aggregate {partition.aggregate!r} must equal the complete "
                    f"declared component set {partition.components!r}."
                ),
            )
        )
    return InvariantReport(checks=tuple(checks))

def _normalise_axis_parent_map(
    mapping: Mapping[str, str] | None,
    *,
    child_labels: pd.Index,
    parent_labels: pd.Index,
    axis_name: str,
) -> dict[str, str]:
    child = child_labels.astype(str)
    parent = parent_labels.astype(str)
    if mapping is None:
        if not child.equals(parent):
            raise ValueError(
                f"{axis_name} aggregation mapping is absent, so child and parent labels "
                "must be identical and ordered"
            )
        return {label: label for label in child}

    normalized = {str(key): str(value) for key, value in mapping.items()}
    if len(normalized) != len(mapping):
        raise ValueError(
            f"{axis_name} aggregation mapping has keys that collide after string normalization"
        )

    child_set = set(child)
    mapping_set = set(normalized)
    missing_children = sorted(child_set - mapping_set)
    extra_children = sorted(mapping_set - child_set)
    if missing_children or extra_children:
        raise ValueError(
            f"{axis_name} aggregation mapping must cover exactly the child labels; "
            f"missing={missing_children[:20]!r}, extra={extra_children[:20]!r}"
        )

    parent_set = set(parent)
    unknown_parents = sorted(set(normalized.values()) - parent_set)
    if unknown_parents:
        raise ValueError(
            f"{axis_name} aggregation mapping references unknown parent labels: "
            f"{unknown_parents[:20]!r}"
        )
    uncovered_parents = sorted(parent_set - set(normalized.values()))
    if uncovered_parents:
        raise ValueError(
            f"{axis_name} aggregation mapping leaves parent labels uncovered: "
            f"{uncovered_parents[:20]!r}"
        )
    return normalized


def _reaggregate_frame(
    expanded: pd.DataFrame,
    *,
    parent_index: pd.Index,
    parent_columns: pd.Index,
    row_parent_by_child: Mapping[str, str] | None,
    column_parent_by_child: Mapping[str, str] | None,
) -> pd.DataFrame:
    row_map = _normalise_axis_parent_map(
        row_parent_by_child,
        child_labels=expanded.index,
        parent_labels=parent_index,
        axis_name="row",
    )
    column_map = _normalise_axis_parent_map(
        column_parent_by_child,
        child_labels=expanded.columns,
        parent_labels=parent_columns,
        axis_name="column",
    )

    parent_row_positions = {
        label: position for position, label in enumerate(parent_index.astype(str))
    }
    parent_column_positions = {
        label: position for position, label in enumerate(parent_columns.astype(str))
    }
    row_aggregation = np.zeros((len(parent_index), len(expanded.index)), dtype="float64")
    column_aggregation = np.zeros(
        (len(parent_columns), len(expanded.columns)), dtype="float64"
    )
    for child_position, child_label in enumerate(expanded.index.astype(str)):
        row_aggregation[parent_row_positions[row_map[child_label]], child_position] = 1.0
    for child_position, child_label in enumerate(expanded.columns.astype(str)):
        column_aggregation[
            parent_column_positions[column_map[child_label]], child_position
        ] = 1.0

    values = (
        row_aggregation
        @ expanded.to_numpy(dtype="float64")
        @ column_aggregation.T
    )
    return pd.DataFrame(
        values,
        index=parent_index,
        columns=parent_columns,
        dtype="float64",
    )


def validate_reaggregation_conservation(
    *,
    source: pd.DataFrame,
    expanded: pd.DataFrame,
    flow_role: FlowRole,
    check_id: str,
    block: str,
    row_parent_by_child: Mapping[str, str] | None = None,
    column_parent_by_child: Mapping[str, str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> InvariantReport:
    # Validate that an expanded labelled block reaggregates to its source block.
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unsupported flow_role: {flow_role!r}")
    if not str(check_id):
        raise ValueError("check_id must be non-empty")
    if not str(block):
        raise ValueError("block must be non-empty")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")

    source_ord = _normalise_frame(source, name="source")
    expanded_ord = _normalise_frame(expanded, name="expanded")
    if not np.isfinite(source_ord.to_numpy(dtype="float64")).all():
        raise ValueError("source contains non-finite values")
    if not np.isfinite(expanded_ord.to_numpy(dtype="float64")).all():
        raise ValueError("expanded contains non-finite values")

    reaggregated = _reaggregate_frame(
        expanded_ord,
        parent_index=source_ord.index,
        parent_columns=source_ord.columns,
        row_parent_by_child=row_parent_by_child,
        column_parent_by_child=column_parent_by_child,
    )
    return InvariantReport(
        checks=(
            _frame_comparison_check(
                check_id=str(check_id),
                category="reaggregation_conservation",
                flow_role=flow_role,
                block=str(block),
                observed=source_ord,
                derived=reaggregated,
                rtol=rtol,
                atol=atol,
                explanation=(
                    "The expanded block must recover the authoritative source block "
                    "exactly after applying the declared child-to-parent mappings."
                ),
            ),
        )
    )


def validate_applicability_contract(
    *,
    values: pd.DataFrame,
    applicability: pd.DataFrame,
    flow_role: FlowRole,
    block: str,
    check_id: str = "structural.applicability",
) -> InvariantReport:
    """Validate the distinction between applicable values and structural N/A cells."""
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unsupported flow_role: {flow_role!r}")
    values_ord = values.copy().astype("float64")
    values_ord.index = values_ord.index.astype(str)
    values_ord.columns = values_ord.columns.astype(str)
    if values_ord.index.has_duplicates or values_ord.columns.has_duplicates:
        raise ValueError("values contains duplicate labels")

    mask = applicability.copy()
    mask.index = mask.index.astype(str)
    mask.columns = mask.columns.astype(str)
    if mask.index.has_duplicates or mask.columns.has_duplicates:
        raise ValueError("applicability contains duplicate labels")
    if not values_ord.index.equals(mask.index) or not values_ord.columns.equals(mask.columns):
        raise ValueError("values and applicability must have identical ordered labels")
    if not all(dtype == bool for dtype in mask.dtypes):  # noqa: E721
        raise ValueError("applicability must contain boolean values only")

    value_array = values_ord.to_numpy(dtype="float64")
    mask_array = mask.to_numpy(dtype=bool)
    invalid_applicable = mask_array & ~np.isfinite(value_array)
    invalid_non_applicable = ~mask_array & np.isfinite(value_array)
    invalid = invalid_applicable | invalid_non_applicable
    positions = np.argwhere(invalid)
    labels = tuple(
        f"{values_ord.index[int(row)]}|{values_ord.columns[int(column)]}"
        for row, column in positions[:50]
    )
    return InvariantReport(
        checks=(
            InvariantCheck(
                check_id=str(check_id),
                category="structural_applicability",
                status="pass" if not labels else "fail",
                severity="error",
                flow_role=flow_role,
                block=str(block),
                max_abs_gap=None,
                tolerance=None,
                affected_labels=labels,
                explanation=(
                    "Applicable cells must contain finite values, while structurally "
                    "non-applicable cells must remain non-finite rather than being "
                    "converted to economic zeros."
                ),
            ),
        )
    )


def validate_metadata_compatibility(
    *,
    left: BlockMetadata,
    right: BlockMetadata,
    flow_role: FlowRole,
    block: str,
    check_id: str = "structural.metadata_compatibility",
) -> InvariantReport:
    """Validate unit, valuation, price-measure and scaling compatibility."""
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unsupported flow_role: {flow_role!r}")
    if not np.isfinite(float(left.scale)) or not np.isfinite(float(right.scale)):
        raise ValueError("metadata scales must be finite and positive")
    if left.scale <= 0.0 or right.scale <= 0.0:
        raise ValueError("metadata scales must be finite and positive")

    mismatches = tuple(
        field_name
        for field_name in ("unit", "valuation", "price_measure", "scale")
        if getattr(left, field_name) != getattr(right, field_name)
    )
    return InvariantReport(
        checks=(
            InvariantCheck(
                check_id=str(check_id),
                category="structural_metadata",
                status="pass" if not mismatches else "fail",
                severity="error",
                flow_role=flow_role,
                block=str(block),
                max_abs_gap=None,
                tolerance=None,
                affected_labels=mismatches,
                explanation=(
                    "Blocks may be compared or added only when unit, valuation, price "
                    "measure and monetary scaling are identical."
                ),
            ),
        )
    )

"""Generic accounting validation for Eurostat product-by-product IOT triplets.

The validator is deliberately read-only.  It turns the dataset-level policy in
``eurostat_iot_source_policy`` into explicit source or generated-table checks;
it does not construct totals, fill absent observations, or infer applicability.
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.eurostat_iot_source_policy import EurostatIOTAccountingPolicy
from ecolecon_mrio.iot_flow_semantics import (
    rounded_source_control_diagnostic,
    source_atomic_final_demand,
)
from ecolecon_mrio.iot_validation import (
    FlowRole,
    InvariantCheck,
    InvariantReport,
    _comparison_check,
    _frame_comparison_check,
)

ValidationMode = Literal["source", "generated"]
_FLOWS: tuple[FlowRole, ...] = ("DOM", "IMP", "TOTAL")
_RTOL = 1e-9
_ATOL = 1e-6


def _p5_partition_ids(atomic_columns: Sequence[str]) -> set[str]:
    """Return only the P5 identities applicable to the selected atomic basis."""
    selected = set(map(str, atomic_columns))
    if {"P51G", "P52", "P53"}.issubset(selected):
        return {"full.y2.p5m", "full.y3.p5_nested", "full.y4.p5_flat"}
    if {"P51G", "P5M"}.issubset(selected):
        return {"full.y3.p5_nested"}
    return set()


def _check(
    check_id: str,
    *,
    status: Literal["pass", "fail", "not_applicable", "not_tested"],
    severity: Literal["error", "warning", "info"],
    flow: FlowRole = "TOTAL",
    block: str,
    explanation: str,
    affected: Sequence[str] = (),
) -> InvariantCheck:
    return InvariantCheck(
        check_id=check_id,
        category="full_table",
        status=status,
        severity=severity,
        flow_role=flow,
        block=block,
        max_abs_gap=None,
        tolerance=None,
        affected_labels=tuple(map(str, affected))[:50],
        explanation=explanation,
    )


def _not_tested(check_id: str, *, flow: FlowRole, block: str, explanation: str) -> InvariantCheck:
    return _check(
        check_id,
        status="not_tested",
        severity="info",
        flow=flow,
        block=block,
        explanation=explanation,
    )


def _nonnegative_check(
    values: pd.Series | pd.DataFrame,
    *,
    check_id: str,
    flow: FlowRole,
    block: str,
    explanation: str,
) -> InvariantCheck:
    """Check an economically non-negative generated or published domain."""
    affected: list[str]
    if isinstance(values, pd.Series):
        numeric_series = values.astype("float64")
        positions = np.flatnonzero(numeric_series.to_numpy(dtype="float64") < -_ATOL)
        affected = [str(numeric_series.index[int(position)]) for position in positions[:50]]
    else:
        numeric_frame = values.astype("float64")
        positions = np.argwhere(numeric_frame.to_numpy(dtype="float64") < -_ATOL)
        affected = [
            f"{numeric_frame.index[int(row)]}|{numeric_frame.columns[int(column)]}"
            for row, column in positions[:50]
        ]
    return _check(
        check_id,
        status="fail" if affected else "pass",
        severity="error",
        flow=flow,
        block=block,
        affected=affected,
        explanation=explanation,
    )


def _source_series_check(
    *,
    check_id: str,
    flow: FlowRole,
    block: str,
    observed: pd.Series,
    derived: pd.Series,
    component_term_count: int,
    quantum: float | None,
    explanation: str,
) -> InvariantCheck:
    """Use the single established source-rounding diagnostic for a vector."""
    if not observed.index.equals(derived.index):
        return _check(
            check_id,
            status="fail",
            severity="error",
            flow=flow,
            block=block,
            explanation=f"{explanation} Compared vectors have different ordered labels.",
        )
    if quantum is None or not np.isfinite(quantum) or quantum <= 0.0:
        return _not_tested(
            check_id,
            flow=flow,
            block=block,
            explanation=f"{explanation} Source rounding quantum is unavailable.",
        )
    diagnostics: list[tuple[str, Mapping[str, object]]] = []
    for label in observed.index.astype(str):
        try:
            diagnostics.append(
                (
                    str(label),
                    rounded_source_control_diagnostic(
                        identity=check_id,
                        published=float(observed.loc[label]),
                        derived=float(derived.loc[label]),
                        component_term_count=component_term_count,
                        source_rounding_quantum=float(quantum),
                        raise_on_failure=False,
                    ),
                )
            )
        except ValueError as error:
            return _check(
                check_id,
                status="fail",
                severity="error",
                flow=flow,
                block=block,
                explanation=f"{explanation} Source values are structurally unusable: {error}",
            )
    def diagnostic_float(diagnostic: Mapping[str, object], key: str) -> float:
        value = diagnostic.get(key)
        if not isinstance(value, (float, int)):
            raise ValueError(f"{check_id}: source rounding diagnostic lacks numeric {key!r}")
        return float(value)

    failed = [(label, item) for label, item in diagnostics if not bool(item["ok"])]
    worst_label, worst = max(
        diagnostics,
        key=lambda item: diagnostic_float(item[1], "absolute_gap"),
    )
    published = diagnostic_float(worst, "published_value")
    derived_value = diagnostic_float(worst, "derived_value")
    denominator = max(abs(published), abs(derived_value))
    return InvariantCheck(
        check_id=check_id,
        category="full_table",
        status="fail" if failed else "pass",
        severity="warning" if failed else "info",
        flow_role=flow,
        block=block,
        published=published,
        derived=derived_value,
        signed_gap=diagnostic_float(worst, "signed_gap"),
        max_abs_gap=diagnostic_float(worst, "absolute_gap"),
        relative_gap=(published - derived_value) / denominator if denominator else 0.0,
        tolerance=diagnostic_float(worst, "rounding_bound"),
        affected_labels=tuple(label for label, _ in failed[:50]),
        explanation=f"{explanation} Worst observed label: {worst_label}.",
    )


def _source_frame_check(
    *,
    check_id: str,
    flow: FlowRole,
    block: str,
    observed: pd.DataFrame,
    derived: pd.DataFrame,
    component_term_count: int,
    quantum: float | None,
    explanation: str,
) -> InvariantCheck:
    if not observed.index.equals(derived.index) or not observed.columns.equals(derived.columns):
        return _check(
            check_id,
            status="fail",
            severity="error",
            flow=flow,
            block=block,
            explanation=f"{explanation} Compared frames have different ordered labels.",
        )
    observed_vector = observed.stack(future_stack=True)
    derived_vector = derived.stack(future_stack=True)
    if not isinstance(observed_vector, pd.Series) or not isinstance(derived_vector, pd.Series):
        return _check(
            check_id,
            status="fail",
            severity="error",
            flow=flow,
            block=block,
            explanation=f"{explanation} Stacked source frames are not one-dimensional labelled vectors.",
        )
    observed_vector.index = observed_vector.index.map(lambda item: f"{item[0]}|{item[1]}")
    derived_vector.index = derived_vector.index.map(lambda item: f"{item[0]}|{item[1]}")
    return _source_series_check(
        check_id=check_id,
        flow=flow,
        block=block,
        observed=observed_vector.astype("float64"),
        derived=derived_vector.astype("float64"),
        component_term_count=component_term_count,
        quantum=quantum,
        explanation=explanation,
    )


def _comparison(
    *,
    mode: ValidationMode,
    check_id: str,
    flow: FlowRole,
    block: str,
    observed: pd.Series,
    derived: pd.Series,
    component_term_count: int,
    quantum: float | None,
    explanation: str,
) -> InvariantCheck:
    if mode == "source":
        return _source_series_check(
            check_id=check_id,
            flow=flow,
            block=block,
            observed=observed,
            derived=derived,
            component_term_count=component_term_count,
            quantum=quantum,
            explanation=explanation,
        )
    return _comparison_check(
        check_id=check_id,
        category="full_table",
        flow_role=flow,
        block=block,
        observed=observed,
        derived=derived,
        rtol=_RTOL,
        atol=_ATOL,
        explanation=explanation,
    )


def _frame_comparison(
    *,
    mode: ValidationMode,
    check_id: str,
    flow: FlowRole,
    block: str,
    observed: pd.DataFrame,
    derived: pd.DataFrame,
    component_term_count: int,
    quantum: float | None,
    explanation: str,
) -> InvariantCheck:
    if mode == "source":
        return _source_frame_check(
            check_id=check_id,
            flow=flow,
            block=block,
            observed=observed,
            derived=derived,
            component_term_count=component_term_count,
            quantum=quantum,
            explanation=explanation,
        )
    return _frame_comparison_check(
        check_id=check_id,
        category="full_table",
        flow_role=flow,
        block=block,
        observed=observed,
        derived=derived,
        rtol=_RTOL,
        atol=_ATOL,
        explanation=explanation,
    )


def _quantum(parsed: ParsedEurostatIOT) -> float | None:
    value = parsed.meta.get("source_rounding_quantum")
    return None if value is None else float(cast(float, value))


def _safe_row(
    frame: pd.DataFrame | None,
    label: str,
    *,
    flow: FlowRole,
    block: str,
    checks: list[InvariantCheck],
    check_id: str,
) -> pd.Series | None:
    if frame is None:
        return None
    if label not in frame.index:
        return None
    row = frame.loc[label]
    if not isinstance(row, pd.Series):
        return None
    return row.astype("float64")


def _safe_column(
    frame: pd.DataFrame,
    label: str,
    *,
    flow: FlowRole,
    checks: list[InvariantCheck],
    check_id: str,
) -> pd.Series | None:
    if label not in frame.columns:
        return None
    column = frame.loc[:, label]
    if not isinstance(column, pd.Series):
        return None
    return column.astype("float64")


def _structural_checks(
    variants: Mapping[str, ParsedEurostatIOT],
    *,
    policy: EurostatIOTAccountingPolicy,
    mode: ValidationMode,
    checks: list[InvariantCheck],
) -> dict[FlowRole, pd.DataFrame]:
    atomic: dict[FlowRole, pd.DataFrame] = {}
    for flow in _FLOWS:
        parsed = variants.get(flow)
        if parsed is None:
            for identity in ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "atomic", "schema"):
                checks.append(
                    _check(
                        f"full.{identity}.{flow.lower()}",
                        status="fail",
                        severity="error",
                        flow=flow,
                        block="Z|Y|V|M",
                        explanation=f"Mandatory {flow} flow is absent.",
                    )
                )
            continue
        blocks = (("Z", parsed.Z), ("Y", parsed.Y), ("V", parsed.V), ("M", parsed.M))
        checks.append(
            _check(
                f"full.s1.{flow.lower()}",
                status="pass" if not parsed.Z.index.has_duplicates and not parsed.Z.columns.has_duplicates else "fail",
                severity="error",
                flow=flow,
                block="Z",
                affected=tuple(parsed.Z.index[parsed.Z.index.duplicated()].astype(str))
                + tuple(parsed.Z.columns[parsed.Z.columns.duplicated()].astype(str)),
                explanation="Z row and column labels must each be unique.",
            )
        )
        checks.append(
            _check(
                f"full.s2.{flow.lower()}",
                status="pass" if parsed.Z.index.equals(parsed.Z.columns) else "fail",
                severity="error",
                flow=flow,
                block="Z",
                explanation="Z must be square with one identical ordered product domain.",
            )
        )
        checks.append(
            _check(
                f"full.s3.{flow.lower()}",
                status="pass" if parsed.Y.index.equals(parsed.Z.index) else "fail",
                severity="error",
                flow=flow,
                block="Y|Z",
                explanation="Y product rows must exactly match the ordered Z product domain.",
            )
        )
        checks.append(
            _check(
                f"full.s4.{flow.lower()}",
                status="not_applicable" if parsed.V is None else "pass" if parsed.V.columns.equals(parsed.Z.columns) else "fail",
                severity="info" if parsed.V is None else "error",
                flow=flow,
                block="V|Z",
                explanation="V product columns must exactly match Z when V is supplied.",
            )
        )
        checks.append(
            _check(
                f"full.s5.{flow.lower()}",
                status="not_applicable" if parsed.M is None else "pass" if parsed.M.columns.equals(parsed.Y.columns) else "fail",
                severity="info" if parsed.M is None else "error",
                flow=flow,
                block="M|Y",
                explanation="M displayed-use columns must exactly match Y when M is supplied.",
            )
        )
        duplicates: list[str] = []
        non_finite = 0
        core_non_finite = 0
        for name, frame in blocks:
            if frame is None:
                continue
            duplicates.extend(f"{name}.row:{item}" for item in frame.index[frame.index.duplicated()].astype(str))
            duplicates.extend(f"{name}.column:{item}" for item in frame.columns[frame.columns.duplicated()].astype(str))
            values = frame.to_numpy(dtype="float64")
            if mode == "generated" and name in {"Z", "Y"}:
                core_non_finite += int((~np.isfinite(values)).sum())
            else:
                non_finite += int(
                    np.isinf(values).sum()
                    if mode == "generated"
                    else (~np.isfinite(values)).sum()
                )
        checks.append(
            _check(
                f"full.s6.{flow.lower()}",
                status="pass" if not duplicates else "fail",
                severity="error",
                flow=flow,
                block="Z|Y|V|M",
                affected=duplicates,
                explanation="No parsed block axis may contain duplicate source codes.",
            )
        )
        checks.append(
            _check(
                f"full.s7.{flow.lower()}",
                status="pass" if non_finite == 0 and core_non_finite == 0 else "fail",
                severity="error",
                flow=flow,
                block="Z|Y|V|M",
                affected=(str(non_finite + core_non_finite),) if non_finite or core_non_finite else (),
                explanation=(
                    "Generated Z and Y must be finite; V and M may retain NaN only for missing/non-applicable cells, but no generated block may contain infinity."
                    if mode == "generated"
                    else "All parsed applicable source cells must be finite under the current parser contract."
                ),
            )
        )
        try:
            selected_atomic = source_atomic_final_demand(parsed.Y)
        except ValueError as exc:
            selected_atomic = None
            atomic_error = str(exc)
            atomic_affected = tuple(
                code
                for code in ("P51G", "P52", "P53", "P5M")
                if code not in parsed.Y.columns
            )
        else:
            atomic_error = None
            atomic_affected = ()
        nested_p5 = (
            selected_atomic is not None
            and {"P51G", "P5M"}.issubset(selected_atomic.columns)
            and "P52" not in selected_atomic.columns
            and "P53" not in selected_atomic.columns
        )
        permitted_missing_p5 = {"P52", "P53"} if nested_p5 else set()
        missing_columns = [
            code
            for code in policy.required_final_demand_columns
            if code not in parsed.Y.columns and code not in permitted_missing_p5
        ]
        checks.append(
            _check(
                f"full.schema.y.{flow.lower()}",
                status="pass" if not missing_columns else "fail",
                severity="error",
                flow=flow,
                block="Y",
                affected=missing_columns,
                explanation="The selected naio_10_cp1700 policy requires its declared displayed final-demand domain.",
            )
        )
        missing_rows: list[str] = []
        for block_name, frame in (("V", parsed.V), ("M", parsed.M)):
            if frame is None:
                missing_rows.extend(f"{block_name}:{row}" for row in policy.required_rows_by_flow[flow])
            else:
                missing_rows.extend(
                    f"{block_name}:{row}"
                    for row in policy.required_rows_by_flow[flow]
                    if row not in frame.index
                )
        checks.append(
            _check(
                f"full.schema.rows.{flow.lower()}",
                status="pass" if not missing_rows else "fail",
                severity="error",
                flow=flow,
                block="V|M",
                affected=missing_rows,
                explanation="The selected naio_10_cp1700 policy requires its declared supplementary controls in V and M.",
            )
        )
        checks.append(
            _check(
                f"full.atomic.{flow.lower()}",
                status="pass" if selected_atomic is not None and not missing_columns else "fail",
                severity="error",
                flow=flow,
                block="Y",
                affected=(
                    atomic_affected
                    if atomic_error is not None
                    else tuple(
                        code
                        for code in policy.atomic_final_demand_columns
                        if code not in parsed.Y.columns and code not in permitted_missing_p5
                    )
                ),
                explanation="A complete recognised non-overlapping final-demand basis is mandatory for this policy.",
            )
        )
        if selected_atomic is not None:
            atomic[flow] = selected_atomic

        # Intermediate transactions and declared use/supply controls are gross
        # monetary flows. Signed stock changes, taxes and residual income rows
        # are deliberately absent from these policy lists.
        checks.append(
            _nonnegative_check(
                parsed.Z,
                check_id=f"full.domain.z.{flow.lower()}",
                flow=flow,
                block="Z",
                explanation="Intermediate transactions must be non-negative.",
            )
        )
        for column in policy.nonnegative_y_columns_by_flow.get(flow, ()):
            values = _safe_column(
                parsed.Y, column, flow=flow, checks=checks,
                check_id=f"full.domain.y.{flow.lower()}.{column.lower()}",
            )
            if values is not None:
                checks.append(
                    _nonnegative_check(
                        values,
                        check_id=f"full.domain.y.{flow.lower()}.{column.lower()}",
                        flow=flow,
                        block="Y",
                        explanation=f"{column} is a non-negative use control.",
                    )
                )
        for row in policy.nonnegative_v_rows_by_flow.get(flow, ()):
            values = _safe_row(
                parsed.V, row, flow=flow, block="V", checks=checks,
                check_id=f"full.domain.v.{flow.lower()}.{row.lower()}",
            )
            if values is not None:
                checks.append(
                    _nonnegative_check(
                        values,
                        check_id=f"full.domain.v.{flow.lower()}.{row.lower()}",
                        flow=flow,
                        block="V",
                        explanation=f"{row} is a non-negative production or supply control.",
                    )
                )

    checks.append(
        _not_tested(
            "full.s8.applicability",
            flow="TOTAL",
            block="Z|Y|V|M",
            explanation="Source observation applicability was destroyed by parser zero-filling; no mask is available.",
        )
    )
    checks.append(
        _not_tested(
            "full.s9.unit_valuation",
            flow="TOTAL",
            block="Z|Y|V|M",
            explanation="Parsed tables retain a unit filter but not enough unit, scale, price-basis and CIF/FOB metadata to prove every comparison compatible.",
        )
    )
    if all(flow in variants and flow in atomic for flow in _FLOWS):
        dom, imp, total = variants["DOM"], variants["IMP"], variants["TOTAL"]
        domains_ok = (
            dom.Z.index.equals(imp.Z.index)
            and dom.Z.index.equals(total.Z.index)
            and dom.Z.columns.equals(imp.Z.columns)
            and dom.Z.columns.equals(total.Z.columns)
            and atomic["DOM"].columns.equals(atomic["IMP"].columns)
            and atomic["DOM"].columns.equals(atomic["TOTAL"].columns)
        )
        checks.append(
            _check(
                "full.s10.triplet_domains",
                status="pass" if domains_ok else "fail",
                severity="error",
                block="Z|atomic_Y",
                explanation="DOM, IMP and TOTAL must share identical ordered product and atomic-final-demand domains.",
            )
        )
    else:
        checks.append(
            _not_tested(
                "full.s10.triplet_domains",
                flow="TOTAL",
                block="Z|atomic_Y",
                explanation="Triplet-domain comparison requires every mandatory flow and atomic final-demand basis.",
            )
        )
    return atomic


def _available(variants: Mapping[str, ParsedEurostatIOT], flow: FlowRole) -> ParsedEurostatIOT | None:
    return variants.get(flow)


def validate_eurostat_iot_full_table(
    variants: Mapping[str, ParsedEurostatIOT],
    *,
    policy: EurostatIOTAccountingPolicy,
    mode: ValidationMode = "source",
) -> InvariantReport:
    """Validate one complete DOM/IMP/TOTAL triplet under a declared policy.

    Source mode delegates every numerical published-control comparison to
    :func:`rounded_source_control_diagnostic`; generated mode uses the existing
    strict invariant comparison helpers.  Structural defects always remain
    error-level failures.
    """
    if mode not in {"source", "generated"}:
        raise ValueError(f"Unsupported validation mode: {mode!r}")
    checks: list[InvariantCheck] = []
    atomic = _structural_checks(variants, policy=policy, mode=mode, checks=checks)

    def compare(
        check_id: str,
        flow: FlowRole,
        block: str,
        observed: pd.Series | None,
        derived: pd.Series | None,
        terms: int,
        explanation: str,
    ) -> None:
        if observed is None or derived is None:
            checks.append(_not_tested(check_id, flow=flow, block=block, explanation=explanation))
            return
        parsed = _available(variants, flow)
        assert parsed is not None
        checks.append(
            _comparison(
                mode=mode,
                check_id=check_id,
                flow=flow,
                block=block,
                observed=observed.astype("float64"),
                derived=derived.astype("float64"),
                component_term_count=terms,
                quantum=_quantum(parsed),
                explanation=explanation,
            )
        )

    def frame_compare(
        check_id: str,
        flow: FlowRole,
        block: str,
        observed: pd.DataFrame | None,
        derived: pd.DataFrame | None,
        terms: int,
        explanation: str,
    ) -> None:
        if observed is None or derived is None:
            checks.append(_not_tested(check_id, flow=flow, block=block, explanation=explanation))
            return
        parsed = _available(variants, flow)
        assert parsed is not None
        checks.append(
            _frame_comparison(
                mode=mode,
                check_id=check_id,
                flow=flow,
                block=block,
                observed=observed.astype("float64"),
                derived=derived.astype("float64"),
                component_term_count=terms,
                quantum=_quantum(parsed),
                explanation=explanation,
            )
        )

    # Y1--Y13: policy partitions, role assertions, and row-use controls.
    for flow in _FLOWS:
        parsed = _available(variants, flow)
        if parsed is None:
            for partition in policy.final_demand_partitions:
                checks.append(
                    _not_tested(
                        f"{partition.check_id}.{flow.lower()}",
                        flow=flow,
                        block="Y",
                        explanation="Flow is absent.",
                    )
                )
            checks.extend(
                [
                    _not_tested(f"full.y9.tfu.{flow.lower()}", flow=flow, block="Y", explanation="Flow is absent."),
                    _not_tested(f"full.y13.tu.{flow.lower()}", flow=flow, block="Z|Y", explanation="Flow is absent."),
                ]
            )
            continue
        active_p5_partitions = _p5_partition_ids(
            () if flow not in atomic else tuple(map(str, atomic[flow].columns))
        )
        for partition in policy.final_demand_partitions:
            check_id = f"{partition.check_id}.{flow.lower()}"
            if partition.check_id.startswith("full.y2.p5") or partition.check_id.startswith("full.y3.p5") or partition.check_id.startswith("full.y4.p5"):
                if partition.check_id not in active_p5_partitions:
                    checks.append(
                        _not_tested(
                            check_id,
                            flow=flow,
                            block="Y",
                            explanation="This displayed P5 identity is outside the selected non-overlapping investment basis.",
                        )
                    )
                    continue
            aggregate = _safe_column(parsed.Y, partition.aggregate, flow=flow, checks=checks, check_id=check_id)
            components = [
                _safe_column(parsed.Y, item, flow=flow, checks=checks, check_id=check_id)
                for item in partition.components
            ]
            derived = None if any(item is None for item in components) else pd.concat(cast(list[pd.Series], components), axis=1).sum(axis=1)
            compare(
                check_id,
                flow,
                "Y",
                aggregate,
                derived,
                len(partition.components),
                f"Published {partition.aggregate} must equal its declared non-overlapping component set.",
            )
        tfu_id = f"full.y9.tfu.{flow.lower()}"
        tfu = _safe_column(parsed.Y, "TFU", flow=flow, checks=checks, check_id=tfu_id)
        atomic_y = atomic.get(flow)
        compare(
            tfu_id,
            flow,
            "Y",
            tfu,
            None if atomic_y is None else atomic_y.sum(axis=1),
            0 if atomic_y is None else len(atomic_y.columns),
            "TFU must equal the declared atomic non-overlapping final-demand basis.",
        )
        tu_id = f"full.y13.tu.{flow.lower()}"
        tu = _safe_column(parsed.Y, "TU", flow=flow, checks=checks, check_id=tu_id)
        compare(
            tu_id,
            flow,
            "Z|Y",
            tu,
            None if atomic_y is None else parsed.Z.sum(axis=1).add(atomic_y.sum(axis=1)),
            len(parsed.Z.columns) + (0 if atomic_y is None else len(atomic_y.columns)),
            "TU must equal intermediate use plus the declared atomic final-demand basis.",
        )
    atomic_set = set().union(*(set(frame.columns) for frame in atomic.values()))
    checks.append(
        _check(
            "full.y10.atomic_excludes_tfu",
            status="pass" if "TFU" not in atomic_set else "fail",
            severity="error",
            block="Y",
            explanation="TFU is a displayed total control and cannot be atomic final demand.",
        )
    )
    checks.append(
        _check(
            "full.y11.atomic_excludes_tu",
            status="pass" if "TU" not in atomic_set else "fail",
            severity="error",
            block="Y",
            explanation="TU is a displayed total-use control and cannot be atomic final demand.",
        )
    )
    checks.append(
        _not_tested(
            "full.y12.total_column_semantics",
            flow="TOTAL",
            block="Y",
            explanation="The source dictionary labels TOTAL only as Total; its exact accounting role is not authorized by the parsed metadata.",
        )
    )
    checks.append(
        _check(
            "full.y7.p6_d0_role",
            status="pass" if policy.row_roles.get("P6_D0") == "exports_to_non_eu" else "fail",
            severity="error",
            block="Y",
            explanation="P6_D0 is declared as the non-EU export control, not an atomic additive component.",
        )
    )
    checks.append(
        _check(
            "full.y8.p6_eu_role",
            status="pass" if policy.row_roles.get("P6_U2") and policy.row_roles.get("P6_U3") else "fail",
            severity="error",
            block="Y",
            explanation="P6_U2 and P6_U3 are declared EU export controls nested below P6_B0.",
        )
    )

    # C1--C4: displayed column totals, without summing overlapping Y columns.
    for flow in _FLOWS:
        parsed = _available(variants, flow)
        if parsed is None:
            continue
        cpa_v = _safe_row(parsed.V, "CPA_TOTAL", flow=flow, block="V", checks=checks, check_id=f"full.c1.cpa_v.{flow.lower()}")
        compare(
            f"full.c1.cpa_v.{flow.lower()}", flow, "V|Z", cpa_v, parsed.Z.sum(axis=0), len(parsed.Z.index), "CPA_TOTAL in V must equal the Z column sum."
        )
        cpa_m = _safe_row(parsed.M, "CPA_TOTAL", flow=flow, block="M", checks=checks, check_id=f"full.c2.cpa_m.{flow.lower()}")
        ordinary_y_totals = parsed.Y.drop(columns=[policy.total_control_column], errors="ignore").sum(axis=0)
        compare(
            f"full.c2.cpa_m.{flow.lower()}", flow, "M|Y", None if cpa_m is None else cpa_m.reindex(ordinary_y_totals.index), ordinary_y_totals, len(parsed.Y.index), "CPA_TOTAL in M must equal every ordinary displayed Y column sum; TOTAL is reconciled to V separately."
        )
        checks.append(
            _check(
                f"full.c3.aggregate_controls.{flow.lower()}",
                status=(
                    "pass"
                    if all(
                        next(
                            (
                                check.status == "pass"
                                for check in checks
                                if check.check_id == f"{partition}.{flow.lower()}"
                            ),
                            False,
                        )
                        for partition in ("full.y1.p3", *_p5_partition_ids(tuple(map(str, atomic.get(flow, pd.DataFrame()).columns))))
                    )
                    else "not_tested"
                ),
                severity="info",
                flow=flow,
                block="M|Y",
                explanation="Displayed aggregate M controls are paired with their own Y column totals; overlapping details are validated separately in Y partitions.",
            )
        )
    if all(_available(variants, flow) is not None for flow in _FLOWS):
        cross_total = _available(variants, "TOTAL")
        cross_dom = _available(variants, "DOM")
        cross_imp = _available(variants, "IMP")
        assert cross_total is not None and cross_dom is not None and cross_imp is not None
        for block, check_id in (("V", "full.c4.cpa_v_cross_flow"), ("M", "full.c4.cpa_m_cross_flow")):
            total = _safe_row(cross_total.V if block == "V" else cross_total.M, "CPA_TOTAL", flow="TOTAL", block=block, checks=checks, check_id=check_id)
            dom = _safe_row(cross_dom.V if block == "V" else cross_dom.M, "CPA_TOTAL", flow="DOM", block=block, checks=checks, check_id=check_id)
            imp = _safe_row(cross_imp.V if block == "V" else cross_imp.M, "CPA_TOTAL", flow="IMP", block=block, checks=checks, check_id=check_id)
            compare(check_id, "TOTAL", block, total, None if dom is None or imp is None else dom.add(imp), 2, "TOTAL CPA_TOTAL must explicitly agree with DOM plus IMP.")

    # P2 bridge and production accounts only use the declared TOTAL/DOM orientations.
    total = _available(variants, "TOTAL")
    dom = _available(variants, "DOM")
    imp = _available(variants, "IMP")
    if total is not None:
        p2_v = _safe_row(total.V, "P2_ADJ", flow="TOTAL", block="V", checks=checks, check_id="full.p2_1.total_v")
        d21_v = _safe_row(total.V, "D21X31", flow="TOTAL", block="V", checks=checks, check_id="full.p2_1.total_v")
        compare("full.p2_1.total_v", "TOTAL", "V|Z", p2_v, None if d21_v is None else total.Z.sum(axis=0).add(d21_v), len(total.Z.index) + 1, "TOTAL P2_ADJ must equal Z column sums plus D21X31.")
        p2_m = _safe_row(total.M, "P2_ADJ", flow="TOTAL", block="M", checks=checks, check_id="full.p2_2.total_m")
        d21_m = _safe_row(total.M, "D21X31", flow="TOTAL", block="M", checks=checks, check_id="full.p2_2.total_m")
        ordinary_y_totals = total.Y.drop(columns=[policy.total_control_column], errors="ignore").sum(axis=0)
        compare("full.p2_2.total_m", "TOTAL", "M|Y", None if p2_m is None else p2_m.reindex(ordinary_y_totals.index), None if d21_m is None else ordinary_y_totals.add(d21_m.reindex(ordinary_y_totals.index)), len(total.Y.index) + 1, "TOTAL M P2_ADJ must equal every ordinary displayed Y column sum plus D21X31; TOTAL is reconciled to V separately.")
    else:
        checks.append(_not_tested("full.p2_1.total_v", flow="TOTAL", block="V|Z", explanation="TOTAL flow is absent."))
        checks.append(_not_tested("full.p2_2.total_m", flow="TOTAL", block="M|Y", explanation="TOTAL flow is absent."))
    checks.append(_check("full.p2_dom_not_derived", status="not_applicable", severity="info", flow="DOM", block="V|Z", explanation="DOM P2_ADJ is not derived from DOM-origin Z because it is not a DOM-flow use identity."))
    checks.append(_check("full.p2_3.aggregate_controls", status="pass", severity="info", block="M|Y", explanation="P2_ADJ aggregate columns use their displayed Y controls; P3/P5/P6 overlap is never summed into an atomic total."))
    checks.append(_check("full.p2_4.d21_role", status="pass" if policy.row_roles.get("D21X31") == "taxes_less_subsidies_on_products" else "fail", severity="error", block="V|M", explanation="D21X31 is the declared product-tax valuation bridge."))
    checks.append(
        _check(
            "full.p2_5.d29_excluded",
            status=(
                "pass"
                if policy.row_roles.get("D29X39")
                == "other_taxes_less_other_subsidies_on_production"
                else "fail"
            ),
            severity="error",
            block="V|M",
            explanation="D29X39 is declared as an other-production-tax control and is excluded from the P2_ADJ bridge.",
        )
    )

    production_variants: tuple[tuple[FlowRole, ParsedEurostatIOT | None], ...] = (
        ("DOM", dom),
        ("TOTAL", total),
    )
    for flow, parsed in production_variants:
        if parsed is None:
            for check_id in ("v1.p1", "v2.b1g", "v3.b2a3g", "v4.d1"):
                checks.append(_not_tested(f"full.{check_id}.{flow.lower()}", flow=flow, block="V", explanation="Flow is absent."))
            continue
        p1 = _safe_row(parsed.V, "P1", flow=flow, block="V", checks=checks, check_id=f"full.v1.p1.{flow.lower()}")
        p2 = _safe_row(parsed.V, "P2_ADJ", flow=flow, block="V", checks=checks, check_id=f"full.v1.p1.{flow.lower()}")
        b1 = _safe_row(parsed.V, "B1G", flow=flow, block="V", checks=checks, check_id=f"full.v1.p1.{flow.lower()}")
        compare(f"full.v1.p1.{flow.lower()}", flow, "V", p1, None if p2 is None or b1 is None else p2.add(b1), 2, "P1 must equal P2_ADJ plus B1G in the declared production account.")
        components = [_safe_row(parsed.V, code, flow=flow, block="V", checks=checks, check_id=f"full.v2.b1g.{flow.lower()}") for code in ("D1", "D29X39", "P51C", "B2A3N")]
        compare(f"full.v2.b1g.{flow.lower()}", flow, "V", b1, None if any(item is None for item in components) else pd.concat(cast(list[pd.Series], components), axis=1).sum(axis=1), 4, "B1G must equal D1 plus D29X39 plus P51C plus B2A3N.")
        b2g = _safe_row(parsed.V, "B2A3G", flow=flow, block="V", checks=checks, check_id=f"full.v3.b2a3g.{flow.lower()}")
        b2n = _safe_row(parsed.V, "B2A3N", flow=flow, block="V", checks=checks, check_id=f"full.v3.b2a3g.{flow.lower()}")
        p51c = _safe_row(parsed.V, "P51C", flow=flow, block="V", checks=checks, check_id=f"full.v3.b2a3g.{flow.lower()}")
        compare(f"full.v3.b2a3g.{flow.lower()}", flow, "V", b2g, None if b2n is None or p51c is None else b2n.add(p51c), 2, "B2A3G must equal B2A3N plus P51C.")
        d1 = _safe_row(parsed.V, "D1", flow=flow, block="V", checks=checks, check_id=f"full.v4.d1.{flow.lower()}")
        d11 = _safe_row(parsed.V, "D11", flow=flow, block="V", checks=checks, check_id=f"full.v4.d1.{flow.lower()}")
        if d1 is None or d11 is None:
            checks.append(_not_tested(f"full.v4.d1.{flow.lower()}", flow=flow, block="V", explanation="D1 or D11 is unavailable."))
        else:
            checks.append(
                _nonnegative_check(
                    d1 - d11,
                    check_id=f"full.v4.d1.{flow.lower()}",
                    flow=flow,
                    block="V",
                    explanation="D1 must be at least D11; the unobserved difference is employers' social contributions.",
                )
            )
    if total is not None and dom is not None:
        for row in policy.production_account_rows:
            total_row = _safe_row(total.V, row, flow="TOTAL", block="V", checks=checks, check_id=f"full.v6.dom_total.{row.lower()}")
            dom_row = _safe_row(dom.V, row, flow="DOM", block="V", checks=checks, check_id=f"full.v6.dom_total.{row.lower()}")
            compare(f"full.v6.dom_total.{row.lower()}", "TOTAL", "V_TOTAL|V_DOM", total_row, dom_row, 1, "Shared DOM/TOTAL production-account controls must agree; they are not IMP-additive.")
        checks.append(_check("full.v5.p1_dom_total", status="pass", severity="info", block="V_TOTAL|V_DOM", explanation="P1 equality is included in the explicit DOM/TOTAL production-account comparisons."))
    else:
        checks.append(_not_tested("full.v5.p1_dom_total", flow="TOTAL", block="V_TOTAL|V_DOM", explanation="DOM and TOTAL flows are both required."))
    checks.append(_check("full.v7.no_blanket_v_additivity", status="not_applicable", severity="info", block="V", explanation="Production-account rows are intentionally not tested as TOTAL = DOM + IMP; IMP has no additive production account."))

    # R1--R5 and imports controls use the declared flow orientations.
    if total is not None and dom is not None and "DOM" in atomic:
        p1_total = _safe_row(total.V, "P1", flow="TOTAL", block="V", checks=checks, check_id="full.r1.domestic_output")
        compare("full.r1.domestic_output", "TOTAL", "V_TOTAL|Z_DOM|Y_DOM", p1_total, dom.Z.sum(axis=1).add(atomic["DOM"].sum(axis=1)), len(dom.Z.columns) + len(atomic["DOM"].columns), "TOTAL P1 must bridge domestic row uses.")
    else:
        checks.append(_not_tested("full.r1.domestic_output", flow="TOTAL", block="V_TOTAL|Z_DOM|Y_DOM", explanation="DOM/TOTAL flows and atomic Y are required."))
    checks.append(
        _not_tested(
            "full.r2.imports_supply",
            flow="TOTAL",
            block="V_TOTAL|Z_IMP|Y_IMP",
            explanation=(
                "TOTAL P7 is a supply-side imports-CIF control by product, while the "
                "IMP transaction row total is imported-product use. The current "
                "naio_10_cp1700 evidence does not authorize a strict productwise bridge."
            ),
        )
    )
    if total is not None and "TOTAL" in atomic:
        tu_total = _safe_column(total.Y, "TU", flow="TOTAL", checks=checks, check_id="full.r3.total_use")
        compare("full.r3.total_use", "TOTAL", "Y_TOTAL|Z_TOTAL", tu_total, total.Z.sum(axis=1).add(atomic["TOTAL"].sum(axis=1)), len(total.Z.columns) + len(atomic["TOTAL"].columns), "TOTAL TU must equal total row uses.")
    else:
        checks.append(_not_tested("full.r3.total_use", flow="TOTAL", block="Y_TOTAL|Z_TOTAL", explanation="TOTAL flow and atomic Y are required."))
    if total is not None and dom is not None and imp is not None:
        frame_compare("full.r4.total_equals_dom_plus_imp_z", "TOTAL", "Z_TOTAL|Z_DOM|Z_IMP", total.Z, dom.Z.add(imp.Z), 2, "TOTAL Z must equal DOM plus IMP cell by cell.")
    else:
        checks.append(_not_tested("full.r4.total_equals_dom_plus_imp_z", flow="TOTAL", block="Z_TOTAL|Z_DOM|Z_IMP", explanation="Complete flow triplet is required."))
    if all(flow in atomic for flow in _FLOWS):
        frame_compare("full.r5.total_equals_dom_plus_imp_atomic_y", "TOTAL", "Y_TOTAL|Y_DOM|Y_IMP", atomic["TOTAL"], atomic["DOM"].add(atomic["IMP"]), 2, "TOTAL atomic Y must equal DOM plus IMP cell by cell.")
    else:
        checks.append(_not_tested("full.r5.total_equals_dom_plus_imp_atomic_y", flow="TOTAL", block="Y_TOTAL|Y_DOM|Y_IMP", explanation="Complete atomic final-demand triplet is required."))
    if dom is not None and imp is not None:
        dom_imp_v = _safe_row(dom.V, "IMP", flow="DOM", block="V", checks=checks, check_id="full.i2.dom_v_imp")
        compare("full.i2.dom_v_imp", "DOM", "V_DOM|Z_IMP", dom_imp_v, imp.Z.sum(axis=0), len(imp.Z.index), "DOM V IMP must equal imported intermediate-use column sums.")
        dom_imp_m = _safe_row(dom.M, "IMP", flow="DOM", block="M", checks=checks, check_id="full.i3.dom_m_imp")
        ordinary_imp_y = imp.Y.drop(columns=[policy.total_control_column], errors="ignore").sum(axis=0)
        compare("full.i3.dom_m_imp", "DOM", "M_DOM|Y_IMP", None if dom_imp_m is None else dom_imp_m.reindex(ordinary_imp_y.index), ordinary_imp_y, len(imp.Y.index), "DOM M IMP must equal imported uses for every ordinary displayed final-use/control column; TOTAL is reconciled to V separately.")
    else:
        checks.append(_not_tested("full.i2.dom_v_imp", flow="DOM", block="V_DOM|Z_IMP", explanation="DOM and IMP flows are required."))
        checks.append(_not_tested("full.i3.dom_m_imp", flow="DOM", block="M_DOM|Y_IMP", explanation="DOM and IMP flows are required."))

    if total is not None:
        for partition in policy.import_partitions:
            aggregate = _safe_row(
                total.V,
                partition.aggregate,
                flow="TOTAL",
                block="V",
                checks=checks,
                check_id=f"{partition.check_id}.v",
            )
            components = [
                _safe_row(
                    total.V,
                    code,
                    flow="TOTAL",
                    block="V",
                    checks=checks,
                    check_id=f"{partition.check_id}.v",
                )
                for code in partition.components
            ]
            derived = (
                None
                if any(item is None for item in components)
                else pd.concat(cast(list[pd.Series], components), axis=1).sum(
                    axis=1, min_count=len(components)
                )
            )
            compare(
                f"{partition.check_id}.v",
                "TOTAL",
                "V",
                aggregate,
                derived,
                len(partition.components),
                f"{partition.aggregate} must equal its declared geographic imports partition on the supply/product axis.",
            )
            checks.append(
                _check(
                    f"{partition.check_id}.m",
                    status="not_applicable",
                    severity="info",
                    flow="TOTAL",
                    block="M",
                    explanation="P7/P7_* are supply-by-product rows and are not final-use controls in M.",
                )
            )
        checks.append(
            _not_tested(
                "full.i4.imports_scalar",
                flow="TOTAL",
                block="V_TOTAL|Z_IMP|Y_IMP",
                explanation=(
                    "The current source evidence does not authorize a strict scalar "
                    "bridge between imports-CIF supply P7 and transformed imported use."
                ),
            )
        )
    else:
        for check_id in (
            "full.i4.imports_scalar",
            "full.i5.p7_geography.v",
            "full.i5.p7_geography.m",
            "full.i6.p7_eu.v",
            "full.i6.p7_eu.m",
        ):
            checks.append(_not_tested(check_id, flow="TOTAL", block="V|M", explanation="TOTAL flow is absent."))
    competing_imports = [] if total is None or total.V is None else [code for code in ("P7_CIF", "P7_FOB") if code in total.V.index]
    checks.append(_check("full.i7.cif_fob_ambiguity", status="fail" if competing_imports else "pass", severity="error" if competing_imports else "info", block="V_TOTAL", affected=competing_imports, explanation="Competing CIF/FOB imports controls require an explicit valuation policy and cannot be selected silently."))
    checks.append(_check("full.i8.import_orientation", status="pass", severity="info", block="V|M", explanation="DOM IMP is an imported-use control while TOTAL P7 is imports supply; the validator compares only their declared orientations."))

    if total is not None:
        ts = _safe_row(total.V, "TS_BP", flow="TOTAL", block="V", checks=checks, check_id="full.sp1.ts_components")
        p1 = _safe_row(total.V, "P1", flow="TOTAL", block="V", checks=checks, check_id="full.sp1.ts_components")
        p7 = _safe_row(total.V, "P7", flow="TOTAL", block="V", checks=checks, check_id="full.sp1.ts_components")
        compare("full.sp1.ts_components", "TOTAL", "V", ts, None if p1 is None or p7 is None else p1.add(p7), 2, "TS_BP must equal P1 plus P7 by product.")
        checks.append(
            _not_tested(
                "full.sp2.ts_equals_tu",
                flow="TOTAL",
                block="V|Y",
                explanation=(
                    "TS_BP is a supply-side control and TU is transaction-row use; "
                    "their productwise valuation bridge is not authorized by the current source evidence."
                ),
            )
        )
        checks.append(
            _not_tested(
                "full.sp3.output_imports_use",
                flow="TOTAL",
                block="V|Z|Y",
                explanation=(
                    "P1+P7 is supply-side while TOTAL Z+Y is transaction use; "
                    "the current source evidence does not authorize a strict productwise bridge."
                ),
            )
        )
        checks.append(
            _not_tested(
                "full.sp4.supply_use_scalar",
                flow="TOTAL",
                block="V|Y",
                explanation=(
                    "A scalar supply/use comparison remains diagnostic until valuation and transformation "
                    "compatibility between P7/TS_BP and transaction use is established."
                ),
            )
        )
        p7 = _safe_row(total.V, "P7", flow="TOTAL", block="V", checks=checks, check_id="full.sp5.imports_geography_scalar")
        b0 = _safe_row(total.V, "P7_B0", flow="TOTAL", block="V", checks=checks, check_id="full.sp5.imports_geography_scalar")
        d0 = _safe_row(total.V, "P7_D0", flow="TOTAL", block="V", checks=checks, check_id="full.sp5.imports_geography_scalar")
        compare("full.sp5.imports_geography_scalar", "TOTAL", "V", None if p7 is None else pd.Series([float(p7.sum())], index=pd.Index(["national_total"])), None if b0 is None or d0 is None else pd.Series([float(b0.sum() + d0.sum())], index=pd.Index(["national_total"])), 2 * len(p7) if p7 is not None else 1, "National P7 must reconcile to its declared geographic components.")
    else:
        for check_id in ("full.sp1.ts_components", "full.sp2.ts_equals_tu", "full.sp3.output_imports_use", "full.sp4.supply_use_scalar", "full.sp5.imports_geography_scalar"):
            checks.append(_not_tested(check_id, flow="TOTAL", block="V|Y", explanation="TOTAL flow is absent."))

    # VM1--VM10: only named rows and the named TOTAL control, never blanket rows.
    for flow, rows in policy.vm_total_rows_by_flow.items():
        parsed = _available(variants, cast(FlowRole, flow))
        if parsed is None:
            continue
        for row in rows:
            check_id = f"full.vm.{flow.lower()}.{row.lower()}"
            v_row = _safe_row(parsed.V, row, flow=cast(FlowRole, flow), block="V", checks=checks, check_id=check_id)
            m_row = _safe_row(parsed.M, row, flow=cast(FlowRole, flow), block="M", checks=checks, check_id=check_id)
            m_total = None if m_row is None or policy.total_control_column not in m_row.index else pd.Series([float(m_row.loc[policy.total_control_column])], index=pd.Index(["total_control"]))
            v_total = None if v_row is None else pd.Series([float(v_row.sum())], index=pd.Index(["total_control"]))
            compare(check_id, cast(FlowRole, flow), "M[TOTAL]|V", m_total, v_total, len(v_row) if v_row is not None else 1, "The named M TOTAL control must equal the V product-column grand total for this declared row.")
    checks.append(_check("full.vm.no_blanket_m_additivity", status="not_applicable", severity="info", block="M", explanation="The validator does not equate complete DOM and TOTAL M rows or split M rows independently."))

    # GDP remains deliberately unpromoted: official formulas need controls not exposed here.
    for check_id, explanation in (
        ("full.gdp1.production", "GDP production requires a verified national taxes-less-subsidies-on-products control, which is not established by this parsed IOT alone."),
        ("full.gdp2.expenditure", "GDP expenditure requires verified purchasers-price national final-use controls and territorial-adjustment treatment, which are not exposed here."),
        ("full.gdp3.income", "GDP income requires verified taxes-on-production-and-imports coverage beyond the parsed row labels."),
        ("full.gdp4.consistency", "Three-way GDP consistency is not tested until all three GDP approaches are independently authorized."),
        ("full.gdp5.b2a3g_scalar", "The stronger B2A3G product-column identity is evaluated; its national GDP interpretation remains untested."),
    ):
        checks.append(_not_tested(check_id, flow="TOTAL", block="V|M|Y", explanation=explanation))
    return InvariantReport(checks=tuple(checks))

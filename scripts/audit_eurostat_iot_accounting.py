#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT, load_eurostat_iot_variants
from ecolecon_mrio.iot_identities import (
    is_imports_row,
    supplementary_accounting_identity_kind,
)
from ecolecon_mrio.iot_flow_semantics import (
    rounded_source_control_diagnostic,
    source_atomic_final_demand,
)
from ecolecon_mrio.iot_validation import FlowRole, InvariantCheck, InvariantReport

FLOW_ROLES = ("DOM", "IMP", "TOTAL")
PARTITIONS = {
    "P3": ("P3_S13", "P3_S14", "P3_S15"),
    "P5": ("P51G", "P52", "P53"),
}


def _parse_flows(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--stk-flows must be a non-empty JSON list")
    flows = [str(item).upper() for item in parsed]
    invalid = [flow for flow in flows if flow not in FLOW_ROLES]
    if invalid:
        raise ValueError(f"Unsupported stock-flow roles: {invalid!r}")
    if len(set(flows)) != len(flows):
        raise ValueError("--stk-flows contains duplicates")
    return flows


def _relative_gap(published: float, derived: float) -> float:
    denominator = max(abs(published), abs(derived))
    return 0.0 if denominator == 0.0 else (published - derived) / denominator


def _diagnostic_float(diagnostic: Mapping[str, object], key: str) -> float:
    return float(cast(float, diagnostic[key]))


def _source_check(
    *,
    check_id: str,
    flow_role: str,
    block: str,
    diagnostics: Sequence[tuple[str, Mapping[str, object]]],
    explanation: str,
) -> InvariantCheck:
    if not diagnostics:
        return InvariantCheck(
            check_id=check_id,
            category="source_control",
            status="not_tested",
            severity="info",
            flow_role=cast(FlowRole, flow_role),
            block=block,
            max_abs_gap=None,
            tolerance=None,
            affected_labels=(),
            explanation=explanation,
        )
    failed = [(label, item) for label, item in diagnostics if not bool(item["ok"])]
    worst_label, worst = max(
        diagnostics,
        key=lambda pair: _diagnostic_float(pair[1], "absolute_gap"),
    )
    published = _diagnostic_float(worst, "published_value")
    derived = _diagnostic_float(worst, "derived_value")
    return InvariantCheck(
        check_id=check_id,
        category="source_control",
        status="fail" if failed else "pass",
        severity="warning" if failed else "info",
        flow_role=cast(FlowRole, flow_role),
        block=block,
        published=published,
        derived=derived,
        signed_gap=_diagnostic_float(worst, "signed_gap"),
        max_abs_gap=_diagnostic_float(worst, "absolute_gap"),
        relative_gap=_relative_gap(published, derived),
        tolerance=_diagnostic_float(worst, "rounding_bound"),
        affected_labels=tuple(label for label, _ in failed[:50]),
        explanation=f"{explanation} Worst observed label: {worst_label}.",
    )


def _rounded_vector_check(
    *,
    check_id: str,
    flow_role: str,
    block: str,
    published: pd.Series,
    derived: pd.Series,
    component_term_count: int,
    quantum: float | None,
    explanation: str,
) -> InvariantCheck:
    if quantum is None or not np.isfinite(quantum) or quantum <= 0.0:
        return _source_check(
            check_id=check_id,
            flow_role=flow_role,
            block=block,
            diagnostics=(),
            explanation=f"{explanation} Source rounding quantum is unavailable.",
        )
    if not published.index.equals(derived.index):
        return InvariantCheck(
            check_id=check_id,
            category="structural",
            status="fail",
            severity="error",
            flow_role=cast(FlowRole, flow_role),
            block=block,
            max_abs_gap=None,
            tolerance=None,
            affected_labels=(),
            explanation=f"{explanation} Compared vectors have different ordered labels.",
        )
    diagnostics: list[tuple[str, Mapping[str, object]]] = []
    for label in published.index.astype(str):
        diagnostics.append(
            (
                str(label),
                rounded_source_control_diagnostic(
                    identity=check_id,
                    published=float(published.loc[label]),
                    derived=float(derived.loc[label]),
                    component_term_count=component_term_count,
                    source_rounding_quantum=float(quantum),
                    raise_on_failure=False,
                ),
            )
        )
    return _source_check(
        check_id=check_id,
        flow_role=flow_role,
        block=block,
        diagnostics=diagnostics,
        explanation=explanation,
    )


def _rounded_frame_check(
    *,
    check_id: str,
    block: str,
    published: pd.DataFrame,
    derived: pd.DataFrame,
    component_term_count: int,
    quantum: float | None,
    explanation: str,
) -> InvariantCheck:
    if not published.index.equals(derived.index) or not published.columns.equals(derived.columns):
        return InvariantCheck(
            check_id=check_id,
            category="structural",
            status="fail",
            severity="error",
            flow_role="TOTAL",
            block=block,
            max_abs_gap=None,
            tolerance=None,
            affected_labels=(),
            explanation=f"{explanation} Compared frames have different ordered labels.",
        )
    if quantum is None or not np.isfinite(quantum) or quantum <= 0.0:
        return _source_check(
            check_id=check_id,
            flow_role="TOTAL",
            block=block,
            diagnostics=(),
            explanation=f"{explanation} Source rounding quantum is unavailable.",
        )
    diagnostics: list[tuple[str, Mapping[str, object]]] = []
    for row in published.index.astype(str):
        for column in published.columns.astype(str):
            diagnostics.append(
                (
                    f"{row}|{column}",
                    rounded_source_control_diagnostic(
                        identity=check_id,
                        published=float(published.loc[row, column]),
                        derived=float(derived.loc[row, column]),
                        component_term_count=component_term_count,
                        source_rounding_quantum=float(quantum),
                        raise_on_failure=False,
                    ),
                )
            )
    return _source_check(
        check_id=check_id,
        flow_role="TOTAL",
        block=block,
        diagnostics=diagnostics,
        explanation=explanation,
    )


def _domain_rows(
    variants: Mapping[str, ParsedEurostatIOT],
    *,
    dataset: str,
    geo: str,
    year: int,
    unit: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for flow, parsed in variants.items():
        quantum = parsed.meta.get("source_rounding_quantum")
        for block_name, frame in (("Z", parsed.Z), ("Y", parsed.Y), ("V", parsed.V), ("M", parsed.M)):
            rows.append(
                {
                    "dataset": dataset,
                    "geo": geo,
                    "year": year,
                    "flow_role": flow,
                    "block": block_name,
                    "present": frame is not None,
                    "n_rows": 0 if frame is None else len(frame.index),
                    "n_columns": 0 if frame is None else len(frame.columns),
                    "row_duplicates": False if frame is None else bool(frame.index.has_duplicates),
                    "column_duplicates": False if frame is None else bool(frame.columns.has_duplicates),
                    "non_finite_cells": 0 if frame is None else int((~np.isfinite(frame.to_numpy(dtype="float64"))).sum()),
                    "unit_code": unit,
                    "valuation": "not_resolved_from_current_parser_metadata",
                    "source_rounding_quantum": quantum,
                    "parser_missingness_contract": "sparse SDMX observations are currently filled with zero by parse_eurostat_iot",
                }
            )
    return rows


def audit_variants(
    variants: Mapping[str, ParsedEurostatIOT],
    *,
    requested_flows: Sequence[str],
) -> InvariantReport:
    checks: list[InvariantCheck] = []
    for flow in requested_flows:
        parsed = variants.get(flow)
        if parsed is None:
            checks.append(
                InvariantCheck(
                    check_id=f"source.structure.{flow.lower()}",
                    category="structural",
                    status="fail",
                    severity="error",
                    flow_role=cast(FlowRole, flow),
                    block="Z|Y|V|M",
                    max_abs_gap=None,
                    tolerance=None,
                    affected_labels=(),
                    explanation=f"Requested flow role {flow!r} was not loaded.",
                )
            )
            continue
        structural_errors: list[str] = []
        if parsed.Z.index.has_duplicates or parsed.Z.columns.has_duplicates:
            structural_errors.append("duplicate Z labels")
        if not parsed.Z.index.equals(parsed.Z.columns):
            structural_errors.append("Z row and column domains differ")
        if not parsed.Y.index.equals(parsed.Z.index):
            structural_errors.append("Y rows differ from Z domain")
        if parsed.V is not None and not parsed.V.columns.equals(parsed.Z.columns):
            structural_errors.append("V columns differ from Z domain")
        if parsed.M is not None and not parsed.M.columns.equals(parsed.Y.columns):
            structural_errors.append("M columns differ from Y domain")
        if not np.isfinite(parsed.Z.to_numpy(dtype="float64")).all():
            structural_errors.append("Z contains non-finite values")
        if not np.isfinite(parsed.Y.to_numpy(dtype="float64")).all():
            structural_errors.append("Y contains non-finite values")
        try:
            source_atomic_final_demand(parsed.Y)
        except ValueError as error:
            structural_errors.append(str(error))
        checks.append(
            InvariantCheck(
                check_id=f"source.structure.{flow.lower()}",
                category="structural",
                status="fail" if structural_errors else "pass",
                severity="error",
                flow_role=cast(FlowRole, flow),
                block="Z|Y|V|M",
                max_abs_gap=None,
                tolerance=None,
                affected_labels=tuple(structural_errors[:50]),
                explanation="Published source domains and the non-overlapping final-demand basis must be structurally usable.",
            )
        )
        quantum_obj = parsed.meta.get("source_rounding_quantum")
        quantum = None if quantum_obj is None else float(quantum_obj)
        for aggregate, components in PARTITIONS.items():
            columns = set(parsed.Y.columns.astype(str))
            if aggregate not in columns or not set(components).issubset(columns):
                checks.append(
                    InvariantCheck(
                        check_id=f"source.partition.{aggregate.lower()}.{flow.lower()}",
                        category="source_control",
                        status="not_tested",
                        severity="info",
                        flow_role=cast(FlowRole, flow),
                        block="Y",
                        max_abs_gap=None,
                        tolerance=None,
                        affected_labels=(),
                        explanation=f"Published {aggregate} aggregate and its complete declared detail partition are not both present.",
                    )
                )
                continue
            published_obj = parsed.Y[aggregate]
            if not isinstance(published_obj, pd.Series):
                checks.append(
                    InvariantCheck(
                        check_id=f"source.partition.{aggregate.lower()}.{flow.lower()}.aggregate_unique",
                        category="structural",
                        status="fail",
                        severity="error",
                        flow_role=cast(FlowRole, flow),
                        block="Y",
                        max_abs_gap=None,
                        tolerance=None,
                        affected_labels=(aggregate,),
                        explanation=f"Published aggregate column {aggregate!r} must occur exactly once in Y.",
                    )
                )
                continue
            checks.append(
                _rounded_vector_check(
                    check_id=f"source.partition.{aggregate.lower()}.{flow.lower()}",
                    flow_role=flow,
                    block="Y",
                    published=published_obj.astype("float64"),
                    derived=parsed.Y.loc[:, list(components)].sum(axis=1).astype("float64"),
                    component_term_count=len(components),
                    quantum=quantum,
                    explanation=f"Published {aggregate} must agree with its independently rounded complete detail partition within the source rounding bound.",
                )
            )

    if all(flow in variants for flow in FLOW_ROLES):
        total = variants["TOTAL"]
        domestic = variants["DOM"]
        imported = variants["IMP"]
        quanta = {
            float(parsed.meta["source_rounding_quantum"])
            for parsed in (total, domestic, imported)
            if parsed.meta.get("source_rounding_quantum") is not None
        }
        quantum = next(iter(quanta)) if len(quanta) == 1 else None
        checks.append(
            _rounded_frame_check(
                check_id="source.cross_flow.Z",
                block="Z_TOTAL|Z_DOM|Z_IMP",
                published=total.Z.astype("float64"),
                derived=domestic.Z.astype("float64") + imported.Z.astype("float64"),
                component_term_count=2,
                quantum=quantum,
                explanation="Published TOTAL intermediate transactions must equal independently rounded DOM plus IMP transactions within the source rounding bound.",
            )
        )
        try:
            total_y = source_atomic_final_demand(total.Y)
            domestic_y = source_atomic_final_demand(domestic.Y)
            imported_y = source_atomic_final_demand(imported.Y)
            checks.append(
                _rounded_frame_check(
                    check_id="source.cross_flow.Y",
                    block="Y_TOTAL|Y_DOM|Y_IMP",
                    published=total_y,
                    derived=domestic_y + imported_y,
                    component_term_count=2,
                    quantum=quantum,
                    explanation="Published TOTAL atomic final uses must equal independently rounded DOM plus IMP final uses within the source rounding bound.",
                )
            )
        except ValueError as error:
            checks.append(
                InvariantCheck(
                    check_id="source.cross_flow.Y",
                    category="structural",
                    status="fail",
                    severity="error",
                    flow_role="TOTAL",
                    block="Y_TOTAL|Y_DOM|Y_IMP",
                    max_abs_gap=None,
                    tolerance=None,
                    affected_labels=(),
                    explanation=f"Atomic final-demand selection failed: {error}",
                )
            )
    return InvariantReport(checks=tuple(checks))



def _dimension_label(parsed: ParsedEurostatIOT, *, dimension: str, code: str) -> str:
    dimensions = parsed.meta.get("dimensions")
    if not isinstance(dimensions, Mapping):
        return str(code)
    labels = dimensions.get(str(dimension))
    if not isinstance(labels, Mapping):
        return str(code)
    value = labels.get(str(code))
    return str(code) if value is None else str(value)


def _current_role_hint(
    *,
    block: str,
    axis: str,
    code: str,
    atomic_final_demand: set[str],
) -> str:
    if block == "Z":
        return "transaction_product"
    if block == "Y" and axis == "row":
        return "supplied_product"
    if block == "Y" and axis == "column":
        if code in atomic_final_demand:
            return "atomic_final_demand"
        if code in PARTITIONS:
            return "published_aggregate_control"
        if code in {"TFU", "TU"}:
            return "published_total_control"
        return "unresolved_final_use_or_control"
    if block == "V" and axis == "column":
        return "using_product"
    if block == "M" and axis == "column":
        return "final_use_column"
    if axis == "row" and block in {"V", "M"}:
        identity = supplementary_accounting_identity_kind(code)
        if identity is not None:
            return f"compatibility_identity_hint:{identity}"
        if is_imports_row(code):
            return "compatibility_identity_hint:imports"
        return "unresolved_supplementary_row"
    return "unresolved"


def _schema_rows(
    variants: Mapping[str, ParsedEurostatIOT],
    *,
    dataset: str,
    geo: str,
    year: int,
    row_dim: str,
    col_dim: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for flow, parsed in variants.items():
        try:
            atomic = set(source_atomic_final_demand(parsed.Y).columns.astype(str))
        except ValueError:
            atomic = set()
        block_specs = (
            ("Z", parsed.Z, row_dim, col_dim),
            ("Y", parsed.Y, row_dim, col_dim),
            ("V", parsed.V, row_dim, col_dim),
            ("M", parsed.M, row_dim, col_dim),
        )
        for block, frame, index_dimension, column_dimension in block_specs:
            if frame is None:
                continue
            for axis, labels, dimension in (
                ("row", frame.index.astype(str), index_dimension),
                ("column", frame.columns.astype(str), column_dimension),
            ):
                for position, code in enumerate(labels):
                    rows.append(
                        {
                            "dataset": dataset,
                            "geo": geo,
                            "year": year,
                            "flow_role": flow,
                            "block": block,
                            "axis": axis,
                            "position": position,
                            "dimension": dimension,
                            "code": str(code),
                            "label": _dimension_label(
                                parsed,
                                dimension=dimension,
                                code=str(code),
                            ),
                            "current_role_hint": _current_role_hint(
                                block=block,
                                axis=axis,
                                code=str(code),
                                atomic_final_demand=atomic,
                            ),
                            "role_authority": "compatibility_hint_only_not_validated_economic_role",
                        }
                    )
    return rows


def _supplementary_profile_rows(
    variants: Mapping[str, ParsedEurostatIOT],
    *,
    dataset: str,
    geo: str,
    year: int,
    row_dim: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for flow, parsed in variants.items():
        for block, frame in (("V", parsed.V), ("M", parsed.M)):
            if frame is None:
                continue
            for position, row_code in enumerate(frame.index.astype(str)):
                values = frame.iloc[position, :].to_numpy(dtype="float64")
                finite = np.isfinite(values)
                finite_values = values[finite]
                identity = supplementary_accounting_identity_kind(row_code)
                role_hint = (
                    f"compatibility_identity_hint:{identity}"
                    if identity is not None
                    else "compatibility_identity_hint:imports"
                    if is_imports_row(row_code)
                    else "unresolved_supplementary_row"
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "geo": geo,
                        "year": year,
                        "flow_role": flow,
                        "block": block,
                        "row_position": position,
                        "row_code": str(row_code),
                        "row_label": _dimension_label(
                            parsed,
                            dimension=row_dim,
                            code=str(row_code),
                        ),
                        "current_role_hint": role_hint,
                        "role_authority": "compatibility_hint_only_not_validated_economic_role",
                        "n_cells": int(values.size),
                        "n_finite": int(finite.sum()),
                        "n_non_finite": int((~finite).sum()),
                        "n_zero": int((finite_values == 0.0).sum()),
                        "n_nonzero": int((finite_values != 0.0).sum()),
                        "finite_sum": float(finite_values.sum()) if finite_values.size else None,
                        "finite_min": float(finite_values.min()) if finite_values.size else None,
                        "finite_max": float(finite_values.max()) if finite_values.size else None,
                        "applicability_status": (
                            "not_resolvable_from_parsed_table_because_sparse_sdmx_cells_are_filled_with_zero"
                        ),
                    }
                )
    return rows


_SCHEMA_RESOLUTION_QUESTIONS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "p2_adj_valuation",
        "Determine the valuation meaning of P2_ADJ and whether it is an intermediate-consumption control or includes valuation adjustments.",
        ("P2_ADJ", "P2ADJ"),
        ("compatibility_identity_hint:intermediate_use",),
    ),
    (
        "supply_use_valuation",
        "Determine whether TS_BP and TU are expressed at a common valuation and which bridge components are required.",
        ("TS_BP", "TOTAL_SUPPLY_AT_BASIC_PRICES", "TU"),
        ("compatibility_identity_hint:total_supply", "published_total_control"),
    ),
    (
        "final_demand_exhaustiveness",
        "Confirm which P3, P5 and P6 detail families are exhaustive in the Austria source and which aggregates remain authoritative controls.",
        ("P3", "P3_S13", "P3_S14", "P3_S15", "P5", "P51G", "P52", "P53", "P6"),
        ("atomic_final_demand", "published_aggregate_control"),
    ),
    (
        "supplementary_block_orientation",
        "Confirm the economic orientation and role coverage of V versus M supplementary blocks.",
        (),
        ("unresolved_supplementary_row",),
    ),
    (
        "tax_margin_roles",
        "Locate product-tax, subsidy, trade-margin and transport-margin controls and determine their valuation role.",
        (),
        (),
    ),
    (
        "imports_valuation",
        "Confirm the authoritative imports row, CIF or FOB valuation, and its bridge to complete imported uses.",
        ("P7", "P7_CIF", "P7_FOB", "IMPORTS"),
        ("compatibility_identity_hint:imports",),
    ),
    (
        "source_missingness",
        "Resolve whether sparse absent SDMX observations mean zero, missing, confidential or structurally non-applicable before applicability policy is encoded.",
        (),
        (),
    ),
)


def _schema_resolution_question_rows(
    schema_rows: Sequence[Mapping[str, object]],
    supplementary_rows: Sequence[Mapping[str, object]],
    *,
    requested_flows: Sequence[str],
) -> list[dict[str, object]]:
    evidence = [dict(row) for row in schema_rows]
    supplementary = [dict(row) for row in supplementary_rows]
    all_rows = evidence + supplementary
    rows: list[dict[str, object]] = []
    for question_id, decision_required, codes, hints in _SCHEMA_RESOLUTION_QUESTIONS:
        candidates: list[dict[str, object]] = []
        if question_id == "supplementary_block_orientation":
            candidates = supplementary
        elif question_id == "tax_margin_roles":
            keywords = ("tax", "subsid", "margin")
            candidates = [
                row
                for row in all_rows
                if any(
                    keyword in f"{row.get('code', row.get('row_code', ''))} {row.get('label', row.get('row_label', ''))}".casefold()
                    for keyword in keywords
                )
            ]
        elif question_id == "source_missingness":
            candidates = supplementary
        else:
            code_set = set(codes)
            hint_set = set(hints)
            candidates = [
                row
                for row in all_rows
                if str(row.get("code", row.get("row_code", ""))) in code_set
                or str(row.get("current_role_hint", "")) in hint_set
            ]

        candidate_codes = sorted(
            {
                str(row.get("code", row.get("row_code", "")))
                for row in candidates
                if str(row.get("code", row.get("row_code", "")))
            }
        )
        candidate_labels = sorted(
            {
                str(row.get("label", row.get("row_label", "")))
                for row in candidates
                if str(row.get("label", row.get("row_label", "")))
            }
        )
        flows = sorted({str(row.get("flow_role", "")) for row in candidates if row.get("flow_role")})
        blocks = sorted({str(row.get("block", "")) for row in candidates if row.get("block")})
        rows.append(
            {
                "question_id": question_id,
                "status": "evidence_collected_not_resolved" if candidates else "no_candidate_found_not_resolved",
                "candidate_count": len(candidates),
                "candidate_codes": "|".join(candidate_codes),
                "candidate_labels": "|".join(candidate_labels),
                "flows_present": "|".join(flows),
                "all_requested_flows_represented": set(map(str, requested_flows)).issubset(set(flows)),
                "blocks_present": "|".join(blocks),
                "decision_required": decision_required,
                "evidence_authority": "descriptive_source_evidence_not_accounting_policy",
            }
        )
    return rows

def write_audit_outputs(
    *,
    variants: Mapping[str, ParsedEurostatIOT],
    requested_flows: Sequence[str],
    outdir: Path,
    dataset: str,
    geo: str,
    year: int,
    unit: str,
    row_dim: str,
    col_dim: str,
) -> InvariantReport:
    report = audit_variants(variants, requested_flows=requested_flows)
    outdir.mkdir(parents=True, exist_ok=True)
    checks = report.to_frame()
    checks.to_csv(outdir / "iot_invariant_checks.csv", index=False)
    checks.loc[checks["status"] == "fail"].to_csv(outdir / "iot_invariant_failures.csv", index=False)
    pd.DataFrame(
        _domain_rows(variants, dataset=dataset, geo=geo, year=year, unit=unit)
    ).to_csv(outdir / "iot_domain_summary.csv", index=False)
    schema_rows = _schema_rows(
        variants,
        dataset=dataset,
        geo=geo,
        year=year,
        row_dim=row_dim,
        col_dim=col_dim,
    )
    supplementary_rows = _supplementary_profile_rows(
        variants,
        dataset=dataset,
        geo=geo,
        year=year,
        row_dim=row_dim,
    )
    pd.DataFrame(schema_rows).to_csv(outdir / "iot_source_schema.csv", index=False)
    pd.DataFrame(supplementary_rows).to_csv(
        outdir / "iot_supplementary_row_profiles.csv", index=False
    )
    pd.DataFrame(
        _schema_resolution_question_rows(
            schema_rows,
            supplementary_rows,
            requested_flows=requested_flows,
        )
    ).to_csv(outdir / "iot_schema_resolution_questions.csv", index=False)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only accounting audit of cached or downloadable Eurostat TOTAL, DOM and IMP IOT sources."
    )
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--dataset", default="naio_10_cp1700")
    parser.add_argument("--row-dim", default="prd_ava")
    parser.add_argument("--col-dim", default="prd_use")
    parser.add_argument("--stk-flow-dim", default="stk_flow")
    parser.add_argument("--stk-flows", default='["DOM","IMP","TOTAL"]')
    parser.add_argument("--unit", default="MIO_EUR")
    parser.add_argument("--storage-folder", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--outdir", default="outputs/audits/eurostat_iot_accounting/AT/2021")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(None if argv is None else list(argv))
    flows = _parse_flows(args.stk_flows)
    variants = load_eurostat_iot_variants(
        dataset=args.dataset,
        base_filters={
            "geo": args.geo,
            "time": str(args.year),
            "freq": "A",
            "unit": args.unit,
        },
        variant_dim=args.stk_flow_dim,
        variant_values=flows,
        row_dim=args.row_dim,
        col_dim=args.col_dim,
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )
    report = write_audit_outputs(
        variants=variants,
        requested_flows=flows,
        outdir=Path(args.outdir),
        dataset=args.dataset,
        geo=args.geo,
        year=args.year,
        unit=args.unit,
        row_dim=args.row_dim,
        col_dim=args.col_dim,
    )
    print(f"Wrote {len(report.checks)} checks to {Path(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

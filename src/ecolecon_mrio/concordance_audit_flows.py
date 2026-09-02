from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import build_eurostat_atomic_final_demand, download_eurostat_dataset, parse_eurostat_iot, read_eurostat_raw
from ecolecon_mrio.eurostat_iot import select_eurostat_atomic_final_demand_columns
from ecolecon_mrio.exio_eurostat_concordance_audit import (
    ConcordanceAuditResult,
    DEFAULT_MIN_ABS_MEUR,
    DEFAULT_REL_THRESHOLD,
    EPS,
    METRIC_TOTAL_USE,
    METRIC_X_OUTPUT_PROXY_TOTAL_USE,
    METRIC_Y_FINAL_DEMAND_TOTAL,
    METRIC_Z_CELL,
    METRIC_Z_COL_INTERMEDIATE_PURCHASES,
    METRIC_Z_ROW_INTERMEDIATE_SALES,
    _display_code_label,
    _display_contributor_string,
    _markdown_table,
    _severity,
    _severity_rank,
    concordance_rel_gap,
    fmt_meur,
    fmt_pct,
    fmt_ratio,
    build_audit_metadata,
    build_concordance_audit,
    human_metric_label,
    load_default_label_maps,
    write_concordance_audit_outputs,
)
from ecolecon_mrio.region_aggregation import RegionScope, load_mrio_iot_scope_slice


REPO_ROOT = Path(__file__).resolve().parents[2]

FLOW_MODE_MAP = {
    "domestic": {"exio_flow": "domestic", "eurostat_flow": "DOM"},
    "imports": {"exio_flow": "imports", "eurostat_flow": "IMP"},
    "total": {"exio_flow": "total", "eurostat_flow": "TOTAL"},
}

EXPORT_EXCLUSION_CAVEAT = (
    "Important: Eurostat final demand uses atomic final-demand columns excluding exports. "
    "Therefore this audit is destination/use-side: it compares products used by the selected country. "
    "TOTAL means domestic + imported products used by the selected country, not a production-side total including exports."
)


@dataclass(frozen=True)
class FlowAuditRun:
    mode: str
    exio_flow: str
    eurostat_flow: str
    result: ConcordanceAuditResult


def normalize_exio_flow(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"dom", "domestic"}:
        return "domestic"
    if v in {"imp", "import", "imports"}:
        return "imports"
    if v in {"tot", "total"}:
        return "total"
    raise ValueError(f"Unknown EXIO flow: {value!r}")


def normalize_eurostat_flow(value: str) -> str:
    v = str(value).strip().upper()
    if v in {"DOM", "IMP", "TOTAL"}:
        return v
    raise ValueError(f"Unknown Eurostat stk_flow: {value!r}; expected DOM, IMP, or TOTAL")


def resolve_comparison_flows(
    *,
    comparison_mode: str | None,
    exio_flow: str | None,
    eurostat_flow: str | None,
    allow_flow_mismatch: bool = False,
) -> tuple[str, str, str]:
    if comparison_mode is not None:
        mode = str(comparison_mode).strip().lower()
        if mode not in FLOW_MODE_MAP:
            raise ValueError(f"Unknown comparison_mode={comparison_mode!r}; expected domestic, imports, or total.")
        pair = FLOW_MODE_MAP[mode]
        return mode, pair["exio_flow"], pair["eurostat_flow"]

    if exio_flow is None and eurostat_flow is None:
        return "domestic", "domestic", "DOM"

    exio_norm = normalize_exio_flow(exio_flow or "domestic")
    euro_norm = normalize_eurostat_flow(eurostat_flow or "DOM")
    expected = {"domestic": "DOM", "imports": "IMP", "total": "TOTAL"}[exio_norm]
    inferred_mode = {("domestic", "DOM"): "domestic", ("imports", "IMP"): "imports", ("total", "TOTAL"): "total"}.get((exio_norm, euro_norm), f"exio_{exio_norm}__eurostat_{euro_norm}")
    if not allow_flow_mismatch and euro_norm != expected:
        raise ValueError(
            f"Inconsistent flow pair: exio_flow={exio_norm!r}, eurostat_flow={euro_norm!r}. Expected eurostat_flow={expected!r} for exio_flow={exio_norm!r}. Use --allow-flow-mismatch only for debugging."
        )
    return inferred_mode, exio_norm, euro_norm


def build_flow_total_use_proxy(Z: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    proxy = (Z.sum(axis=1) + Y.sum(axis=1)).astype("float64")
    proxy.name = "EXIO_FLOW_TOTAL_USE_PROXY"
    return proxy


def run_single_flow_audit(
    *,
    country: str,
    year: int,
    exio_year: int | None,
    comparison_mode: str,
    exio_flow: str,
    eurostat_flow: str,
    mapping_csv: Path,
    output_dir: Path,
    rel_threshold: float,
    min_abs_meur: float,
    identity_mode: str,
    eurostat_dataset: str,
    eurostat_unit: str,
    exio_system: str,
    verbose: bool,
    full_subcomponent_z_scan: bool,
    run_command: str | None = None,
    force_download: bool = False,
) -> ConcordanceAuditResult:
    data_root = default_data_root()
    resolved_exio_year = exio_year if exio_year is not None else year

    euro_filters = {
        "geo": country,
        "time": str(year),
        "freq": "A",
        "unit": eurostat_unit,
        "stk_flow": eurostat_flow,
    }
    euro_dataset_obj = download_eurostat_dataset(
        eurostat_dataset,
        filters=euro_filters,
        storage_folder=data_root / "eurostat",
        force_download=force_download,
    )
    parsed = parse_eurostat_iot(read_eurostat_raw(euro_dataset_obj.raw_path), row_dim="prd_ava", col_dim="prd_use")
    eurostat_Y_atomic = build_eurostat_atomic_final_demand(parsed, include_exports=False)
    selected_final_demand_columns = select_eurostat_atomic_final_demand_columns(parsed, include_exports=False)

    scope_flow = {"domestic": "internal", "imports": "external_imports", "total": "total"}[exio_flow]
    exio = load_mrio_iot_scope_slice(
        "exiobase",
        resolved_exio_year,
        scope=RegionScope(code=str(country), members=(str(country),)),
        exio_system=exio_system,
        flow=scope_flow,
        data_root=data_root,
        force_download=force_download,
    )
    exio_Z_flow = exio.Z
    exio_Y_flow = exio.Y
    exio_x_flow_proxy = build_flow_total_use_proxy(exio_Z_flow, exio_Y_flow)

    exio_labels, euro_labels = load_default_label_maps(REPO_ROOT)
    metadata = build_audit_metadata(
        country=country,
        year=year,
        exio_year=resolved_exio_year,
        eurostat_dataset=eurostat_dataset,
        eurostat_request_url=euro_dataset_obj.request_url,
        mapping_file=mapping_csv,
        exio_flow=exio_flow,
        eurostat_unit=eurostat_unit,
    )
    metadata.update(
        {
            "comparison_mode": comparison_mode,
            "exio_flow": exio_flow,
            "eurostat_flow": eurostat_flow,
            "rel_threshold": rel_threshold,
            "min_abs_meur": min_abs_meur,
            "eurostat_final_demand_basis": "atomic_final_demand_excluding_exports",
            "eurostat_exports_included": False,
            "eurostat_final_demand_columns_available": [str(col) for col in parsed.Y.columns],
            "eurostat_final_demand_columns_selected": [str(col) for col in selected_final_demand_columns],
            "eurostat_final_demand_columns_excluded": [str(col) for col in parsed.Y.columns if str(col) not in set(selected_final_demand_columns)],
            "eurostat_row_codes_not_in_intermediate_columns": parsed.meta.get("row_codes_not_in_intermediate_columns", []),
            "eurostat_intermediate_columns_not_in_rows": parsed.meta.get("intermediate_columns_not_in_rows", []),
            "exio_x_is_proxy": True,
            "exio_x_proxy_basis": "Z row total + Y row total for selected EXIO flow slice",
            "eurostat_x_is_proxy": True,
            "eurostat_x_proxy_basis": "Z row total + Y row total for selected Eurostat stk_flow",
            "flow_interpretation": {
                "domestic": (
                    "EXIOBASE domestic means origin == selected country and destination/use == selected country; Eurostat DOM is the domestic-flow table."
                ),
                "imports": (
                    "EXIOBASE imports mean origin != selected country and destination/use == selected country, summed across all foreign origins; Eurostat IMP is the import-flow table."
                ),
                "total": (
                    "EXIOBASE total means all origins used by the selected country and is mapped independently using TOTAL-specific concordance weights; Eurostat TOTAL is the total-flow table. This independent TOTAL is a mapping sensitivity and reconciliation diagnostic, while canonical disaggregated TOTAL outputs are derived as DOM + IMP."
                ),
            }.get(comparison_mode, ""),
        }
    )
    if run_command:
        metadata["run_command"] = run_command

    result = build_concordance_audit(
        country=country,
        year=year,
        exio_x=exio_x_flow_proxy,
        exio_Z=exio_Z_flow,
        exio_Y=exio_Y_flow,
        eurostat_Z=parsed.Z,
        eurostat_Y=eurostat_Y_atomic,
        mapping_df=pd.read_csv(mapping_csv, dtype=str).fillna(""),
        exio_labels=exio_labels,
        eurostat_labels=euro_labels,
        metadata=metadata,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
        identity_mode=identity_mode,
        verbose=verbose,
        full_subcomponent_z_scan=full_subcomponent_z_scan,
    )

    write_concordance_audit_outputs(result, output_dir)
    return result


def _flow_value_series(result: ConcordanceAuditResult, *, database: str, metric: str) -> pd.Series:
    if database == "EXIOBASE-mapped":
        if metric == METRIC_Z_CELL:
            return result.mapped_z.stack(dropna=False) if result.mapped_z is not None else pd.Series(dtype="float64")
        if metric == METRIC_Z_ROW_INTERMEDIATE_SALES:
            return result.mapped_z.sum(axis=1) if result.mapped_z is not None else pd.Series(dtype="float64")
        if metric == METRIC_Z_COL_INTERMEDIATE_PURCHASES:
            return result.mapped_z.sum(axis=0) if result.mapped_z is not None else pd.Series(dtype="float64")
        if metric == METRIC_Y_FINAL_DEMAND_TOTAL:
            return result.mapped_y if result.mapped_y is not None else pd.Series(dtype="float64")
        if metric == METRIC_TOTAL_USE:
            return result.mapped_z.sum(axis=1) + (result.mapped_y if result.mapped_y is not None else 0.0)
        if metric == METRIC_X_OUTPUT_PROXY_TOTAL_USE:
            return result.mapped_x if result.mapped_x is not None else pd.Series(dtype="float64")
    else:
        if metric == METRIC_Z_CELL:
            return result.eurostat_z.stack(dropna=False) if result.eurostat_z is not None else pd.Series(dtype="float64")
        if metric == METRIC_Z_ROW_INTERMEDIATE_SALES:
            return result.eurostat_z.sum(axis=1) if result.eurostat_z is not None else pd.Series(dtype="float64")
        if metric == METRIC_Z_COL_INTERMEDIATE_PURCHASES:
            return result.eurostat_z.sum(axis=0) if result.eurostat_z is not None else pd.Series(dtype="float64")
        if metric == METRIC_Y_FINAL_DEMAND_TOTAL:
            return result.eurostat_y_total if result.eurostat_y_total is not None else pd.Series(dtype="float64")
        if metric == METRIC_TOTAL_USE:
            return result.eurostat_z.sum(axis=1) + (result.eurostat_y_total if result.eurostat_y_total is not None else 0.0)
        if metric == METRIC_X_OUTPUT_PROXY_TOTAL_USE:
            return result.eurostat_x if result.eurostat_x is not None else pd.Series(dtype="float64")
    return pd.Series(dtype="float64")


def _matrix_key_frame(frame: pd.DataFrame, *, metric: str, database: str, labels: Mapping[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metric == METRIC_Z_CELL:
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "database": database,
                    "metric": metric,
                    "supplier_code": str(row.supplier_code),
                    "supplier_label": str(row.supplier_label),
                    "buyer_code": str(row.buyer_code),
                    "buyer_label": str(row.buyer_label),
                    "target_code": "",
                    "target_label": "",
                    "domestic_value": 0.0,
                    "imports_value": 0.0,
                    "domestic_plus_imports": 0.0,
                    "total_value": 0.0,
                    "signed_gap": 0.0,
                    "abs_gap": 0.0,
                    "rel_gap": 0.0,
                    "flag": False,
                    "severity": "OK",
                    "interpretation": "",
                }
            )
    return pd.DataFrame(rows)


def build_flow_reconciliation(
    *,
    domestic: FlowAuditRun,
    imports: FlowAuditRun,
    total: FlowAuditRun,
    rel_threshold: float,
    min_abs_meur: float,
    eurostat_labels: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    databases = {
        "Eurostat": (domestic.result.eurostat_z, imports.result.eurostat_z, total.result.eurostat_z, domestic.result.eurostat_y_total, imports.result.eurostat_y_total, total.result.eurostat_y_total, domestic.result.eurostat_x, imports.result.eurostat_x, total.result.eurostat_x),
        "EXIOBASE-mapped": (domestic.result.mapped_z, imports.result.mapped_z, total.result.mapped_z, domestic.result.mapped_y, imports.result.mapped_y, total.result.mapped_y, domestic.result.mapped_x, imports.result.mapped_x, total.result.mapped_x),
    }
    def _recon_row(*, database: str, metric: str, supplier_code: str = "", supplier_label: str = "", buyer_code: str = "", buyer_label: str = "", target_code: str = "", target_label: str = "", domestic_value: float = 0.0, imports_value: float = 0.0, total_value: float = 0.0) -> dict[str, Any]:
        domestic_plus_imports = float(domestic_value) + float(imports_value)
        signed_gap = domestic_plus_imports - float(total_value)
        abs_gap = abs(signed_gap)
        rel_gap = abs_gap / max(abs(domestic_plus_imports), abs(float(total_value)), EPS)
        flag = abs_gap >= min_abs_meur and rel_gap >= rel_threshold
        return {
            "database": database,
            "metric": metric,
            "supplier_code": supplier_code,
            "supplier_label": supplier_label,
            "buyer_code": buyer_code,
            "buyer_label": buyer_label,
            "target_code": target_code,
            "target_label": target_label,
            "domestic_value": float(domestic_value),
            "imports_value": float(imports_value),
            "domestic_plus_imports": domestic_plus_imports,
            "total_value": float(total_value),
            "signed_gap": signed_gap,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "flag": flag,
            "severity": _severity(rel_gap, rel_threshold),
            "interpretation": (
                "DOM + IMP does not equal TOTAL for this metric above the selected threshold."
                if flag
                else "DOM + IMP approximately equals TOTAL for this metric."
            ),
        }

    def _align_square(frame: pd.DataFrame | None, codes: list[str]) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(0.0, index=codes, columns=codes, dtype="float64")
        return frame.reindex(index=codes, columns=codes).fillna(0.0).astype("float64")

    def _align_series(frame: pd.Series | None, codes: list[str]) -> pd.Series:
        if frame is None or frame.empty:
            return pd.Series(0.0, index=codes, dtype="float64")
        return frame.reindex(index=codes).fillna(0.0).astype("float64")

    def _lookup_label(frames: tuple[pd.DataFrame, ...], code: str, *, code_column: str, label_column: str) -> str:
        code_text = str(code)
        for frame in frames:
            if frame.empty or code_column not in frame.columns or label_column not in frame.columns:
                continue
            matches = frame.loc[frame[code_column].astype(str).eq(code_text), label_column]
            if not matches.empty:
                return str(matches.iloc[0])
        return code_text

    for database, pieces in databases.items():
        dom_z, imp_z, tot_z, dom_y, imp_y, tot_y, dom_x, imp_x, tot_x = pieces
        dom_z = dom_z if dom_z is not None else pd.DataFrame(dtype="float64")
        imp_z = imp_z if imp_z is not None else pd.DataFrame(dtype="float64")
        tot_z = tot_z if tot_z is not None else pd.DataFrame(dtype="float64")
        dom_y = dom_y if dom_y is not None else pd.Series(dtype="float64")
        imp_y = imp_y if imp_y is not None else pd.Series(dtype="float64")
        tot_y = tot_y if tot_y is not None else pd.Series(dtype="float64")
        dom_x = dom_x if dom_x is not None else pd.Series(dtype="float64")
        imp_x = imp_x if imp_x is not None else pd.Series(dtype="float64")
        tot_x = tot_x if tot_x is not None else pd.Series(dtype="float64")

        z_codes = sorted(set(dom_z.index.astype(str)).union(set(dom_z.columns.astype(str)), set(imp_z.index.astype(str)), set(imp_z.columns.astype(str)), set(tot_z.index.astype(str)), set(tot_z.columns.astype(str))))
        dom_z_aligned = _align_square(dom_z, z_codes)
        imp_z_aligned = _align_square(imp_z, z_codes)
        tot_z_aligned = _align_square(tot_z, z_codes)
        dom_z_row_total = dom_z_aligned.sum(axis=1)
        imp_z_row_total = imp_z_aligned.sum(axis=1)
        tot_z_row_total = tot_z_aligned.sum(axis=1)
        dom_z_col_total = dom_z_aligned.sum(axis=0)
        imp_z_col_total = imp_z_aligned.sum(axis=0)
        tot_z_col_total = tot_z_aligned.sum(axis=0)

        row_codes = sorted(set(dom_z_aligned.index.astype(str)).union(set(imp_z_aligned.index.astype(str)), set(tot_z_aligned.index.astype(str)), set(dom_y.index.astype(str)), set(imp_y.index.astype(str)), set(tot_y.index.astype(str)), set(dom_x.index.astype(str)), set(imp_x.index.astype(str)), set(tot_x.index.astype(str))))
        dom_y_aligned = _align_series(dom_y, row_codes)
        imp_y_aligned = _align_series(imp_y, row_codes)
        tot_y_aligned = _align_series(tot_y, row_codes)
        dom_x_aligned = _align_series(dom_x, row_codes)
        imp_x_aligned = _align_series(imp_x, row_codes)
        tot_x_aligned = _align_series(tot_x, row_codes)

        for supplier_code in z_codes:
            for buyer_code in z_codes:
                rows.append(
                    _recon_row(
                        database=database,
                        metric=METRIC_Z_CELL,
                        supplier_code=supplier_code,
                        supplier_label=eurostat_labels.get(supplier_code, supplier_code),
                        buyer_code=buyer_code,
                        buyer_label=eurostat_labels.get(buyer_code, buyer_code),
                        domestic_value=float(dom_z_aligned.loc[supplier_code, buyer_code]),
                        imports_value=float(imp_z_aligned.loc[supplier_code, buyer_code]),
                        total_value=float(tot_z_aligned.loc[supplier_code, buyer_code]),
                    )
                )
        for code in row_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_Z_ROW_INTERMEDIATE_SALES,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_z_row_total.get(code, 0.0)),
                    imports_value=float(imp_z_row_total.get(code, 0.0)),
                    total_value=float(tot_z_row_total.get(code, 0.0)),
                )
            )

        col_codes = sorted(set(dom_z_aligned.columns.astype(str)).union(set(imp_z_aligned.columns.astype(str)), set(tot_z_aligned.columns.astype(str))))
        for code in col_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_Z_COL_INTERMEDIATE_PURCHASES,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_z_col_total.get(code, 0.0)),
                    imports_value=float(imp_z_col_total.get(code, 0.0)),
                    total_value=float(tot_z_col_total.get(code, 0.0)),
                )
            )

        y_codes = sorted(set(dom_y_aligned.index.astype(str)).union(set(imp_y_aligned.index.astype(str)), set(tot_y_aligned.index.astype(str))))
        for code in y_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_Y_FINAL_DEMAND_TOTAL,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_y_aligned.get(code, 0.0)),
                    imports_value=float(imp_y_aligned.get(code, 0.0)),
                    total_value=float(tot_y_aligned.get(code, 0.0)),
                )
            )

        x_codes = sorted(set(dom_x_aligned.index.astype(str)).union(set(imp_x_aligned.index.astype(str)), set(tot_x_aligned.index.astype(str))))
        for code in x_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_TOTAL_USE,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_x_aligned.get(code, 0.0)),
                    imports_value=float(imp_x_aligned.get(code, 0.0)),
                    total_value=float(tot_x_aligned.get(code, 0.0)),
                )
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["severity_rank"] = frame["severity"].map(_severity_rank).fillna(-1).astype(int)
        frame = frame.sort_values(["database", "severity_rank", "abs_gap", "rel_gap"], ascending=[True, False, False, False], kind="stable").drop(columns=["severity_rank"]).reset_index(drop=True)
    return frame


def _reconciliation_markdown(frame: pd.DataFrame, country: str, year: int) -> str:
    lines = [
        f"# Flow reconciliation audit: {country} {year}",
        "",
        "## What this checks",
        "This report checks whether domestic + imports equals total for both Eurostat and mapped EXIOBASE.",
        "- Eurostat: DOM + IMP should approximately equal TOTAL.",
        "- EXIOBASE-mapped: domestic + imports should approximately equal total after applying the EXIOBASE→Eurostat concordance.",
        "",
        "This is a consistency check on the flow slicing and Eurostat `stk_flow` handling. It is not the main concordance comparison.",
        "",
        "## Summary",
    ]
    summary = frame.groupby("database", dropna=False)["flag"].sum().reset_index()
    summary.columns = ["Database", "Number of flagged reconciliation rows"]
    lines.append(_markdown_table(summary, columns=["Database", "Number of flagged reconciliation rows"], max_rows=20))
    lines.extend([
        "",
        "## Top reconciliation flags",
    ])
    flagged = frame[frame["flag"]].copy()
    if not flagged.empty:
        flagged["severity_rank"] = flagged["severity"].map(_severity_rank).fillna(-1).astype(int)
        flagged = flagged.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"]).head(20)
        display_rows: list[dict[str, Any]] = []
        for row in flagged.itertuples(index=False):
            if row.metric == METRIC_Z_CELL:
                sector_display = f"{_optional_display_code_label(row.supplier_code, row.supplier_label)} → {_optional_display_code_label(row.buyer_code, row.buyer_label)}"
            else:
                sector_display = _optional_display_code_label(row.target_code, row.target_label)
            display_rows.append(
                {
                    "Database": row.database,
                    "Readable metric": human_metric_label(str(row.metric)),
                    "Sector or supplier→buyer pair": sector_display,
                    "Domestic value, MEUR": fmt_meur(row.domestic_value),
                    "Imports value, MEUR": fmt_meur(row.imports_value),
                    "Domestic + imports, MEUR": fmt_meur(row.domestic_plus_imports),
                    "Total value, MEUR": fmt_meur(row.total_value),
                    "Difference, MEUR": fmt_meur(row.signed_gap),
                    "Relative gap, %": fmt_pct(row.rel_gap),
                    "Interpretation": row.interpretation,
                }
            )
        lines.append(_markdown_table(pd.DataFrame(display_rows), columns=["Database", "Readable metric", "Sector or supplier→buyer pair", "Domestic value, MEUR", "Imports value, MEUR", "Domestic + imports, MEUR", "Total value, MEUR", "Difference, MEUR", "Relative gap, %", "Interpretation"], max_rows=20))
    else:
        lines.append("No reconciliation rows exceeded the selected thresholds.")
    lines.extend([
        "",
        "## Notes",
        EXPORT_EXCLUSION_CAVEAT,
        f"- For imports, EXIOBASE imports are foreign-origin products used by {country}, summed across all non-{country} origins. Partner-origin detail is not retained in this audit.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _optional_display_code_label(code: object, label: object) -> str:
    code_text = str(code).strip()
    label_text = str(label).strip() if label is not None else ""
    if not code_text or code_text.lower() == "nan":
        return ""
    if not label_text or label_text.lower() == "nan":
        return code_text
    return f"{code_text} — {label_text}"


def _reconciliation_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display_rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        if row.metric == METRIC_Z_CELL:
            sector_display = f"{_optional_display_code_label(row.supplier_code, row.supplier_label)} → {_optional_display_code_label(row.buyer_code, row.buyer_label)}"
        else:
            sector_display = _optional_display_code_label(row.target_code, row.target_label)
        display_rows.append(
            {
                "Database": row.database,
                "Readable metric": human_metric_label(str(row.metric)),
                "Sector or supplier→buyer pair": sector_display,
                "Domestic value, MEUR": fmt_meur(row.domestic_value),
                "Imports value, MEUR": fmt_meur(row.imports_value),
                "Domestic + imports, MEUR": fmt_meur(row.domestic_plus_imports),
                "Total value, MEUR": fmt_meur(row.total_value),
                "Difference, MEUR": fmt_meur(row.signed_gap),
                "Relative gap, %": fmt_pct(row.rel_gap),
                "Interpretation": row.interpretation,
            }
        )
    return pd.DataFrame(display_rows)


def _top_flag_rows(frame: pd.DataFrame, *, sort_cols: list[str], max_rows: int = 10) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["severity_rank"] = out["severity"].map(_severity_rank).fillna(-1).astype(int)
    out = out.sort_values(["severity_rank", *sort_cols], ascending=[False] + [False] * len(sort_cols), kind="stable").drop(columns=["severity_rank"])
    return out.head(max_rows)


def _flow_flags_table(result: ConcordanceAuditResult, *, flow: str, eurostat_labels: Mapping[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = result.aggregate_totals[result.aggregate_totals["flag"]].copy()
    agg["flow"] = flow
    agg_rows = []
    for row in agg.itertuples(index=False):
        agg_rows.append(
            {
                "flow": flow,
                "metric": row.metric,
                "human_metric": human_metric_label(str(row.metric)),
                "target_code": row.target_code,
                "target_label": row.target_label,
                "target_display": _display_code_label(row.target_code, row.target_label),
                "exio_mapped_value": float(row.exio_mapped_value),
                "eurostat_value": float(row.eurostat_value),
                "signed_gap": float(row.signed_gap),
                "abs_gap": float(row.abs_gap),
                "rel_gap": float(row.rel_gap),
                "ratio_exio_to_eurostat": float(row.ratio_exio_to_eurostat),
                "severity": row.severity,
                "top_exio_contributors": _display_contributor_string(row.top_exio_contributors),
            }
        )
    agg_frame = pd.DataFrame(agg_rows)

    cells = result.z_cell_mismatches[result.z_cell_mismatches["flag"]].copy()
    cell_rows = []
    for row in cells.itertuples(index=False):
        cell_rows.append(
            {
                "flow": flow,
                "supplier_code": row.supplier_code,
                "supplier_label": row.supplier_label,
                "supplier_display": _display_code_label(row.supplier_code, row.supplier_label),
                "buyer_code": row.buyer_code,
                "buyer_label": row.buyer_label,
                "buyer_display": _display_code_label(row.buyer_code, row.buyer_label),
                "comparison_display": f"{_display_code_label(row.supplier_code, row.supplier_label)} → {_display_code_label(row.buyer_code, row.buyer_label)}",
                "exio_mapped_z": float(row.exio_mapped_z),
                "eurostat_z": float(row.eurostat_z),
                "signed_gap": float(row.signed_gap),
                "abs_gap": float(row.abs_gap),
                "rel_gap": float(row.rel_gap),
                "ratio_exio_to_eurostat": float(row.ratio_exio_to_eurostat),
                "severity": row.severity,
                "top_source_subsectors": _display_contributor_string(row.top_source_subsectors),
                "top_buyer_subsectors": _display_contributor_string(row.top_buyer_subsectors),
            }
        )
    cell_frame = pd.DataFrame(cell_rows)
    return agg_frame, cell_frame


def _flow_classification_frame(domestic: pd.DataFrame, imports: pd.DataFrame, total: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    def _flag_lookup(frame: pd.DataFrame) -> dict[tuple[str, ...], float]:
        lookup: dict[tuple[str, ...], float] = {}
        for row in frame.to_dict(orient="records"):
            if kind == "aggregate":
                lookup[(str(row.get("metric", "")), str(row.get("target_code", "")))] = float(row.get("abs_gap", 0.0))
            else:
                lookup[(str(row.get("metric", "")), str(row.get("supplier_code", "")), str(row.get("buyer_code", "")))] = float(row.get("abs_gap", 0.0))
        return lookup

    def _lookup_label(frames: tuple[pd.DataFrame, ...], code: str, *, code_column: str, label_column: str) -> str:
        code_text = str(code)
        for frame in frames:
            if frame.empty or code_column not in frame.columns or label_column not in frame.columns:
                continue
            matches = frame.loc[frame[code_column].astype(str).eq(code_text), label_column]
            if not matches.empty:
                return str(matches.iloc[0])
        return code_text

    rows: list[dict[str, Any]] = []
    if kind == "aggregate":
        dom_lookup = _flag_lookup(domestic)
        imp_lookup = _flag_lookup(imports)
        tot_lookup = _flag_lookup(total)
        aggregate_keys = sorted(set(dom_lookup) | set(imp_lookup) | set(tot_lookup))
        for metric, target_code in aggregate_keys:
            domestic_flag = (metric, target_code) in dom_lookup
            imports_flag = (metric, target_code) in imp_lookup
            total_flag = (metric, target_code) in tot_lookup
            if not (domestic_flag or imports_flag or total_flag):
                continue
            if domestic_flag and imports_flag:
                classification = "BOTH_DOMESTIC_AND_IMPORT"
            elif domestic_flag:
                classification = "DOMESTIC_DRIVEN"
            elif imports_flag:
                classification = "IMPORT_DRIVEN"
            elif total_flag:
                classification = "TOTAL_ONLY"
            else:
                classification = "NO_FLAG"
            rows.append(
                {
                    "kind": "aggregate",
                    "metric": metric,
                    "target_code": target_code,
                    "target_label": _lookup_label((domestic, imports, total), target_code, code_column="target_code", label_column="target_label"),
                    "supplier_code": "",
                    "supplier_label": "",
                    "buyer_code": "",
                    "buyer_label": "",
                    "domestic_flag": domestic_flag,
                    "imports_flag": imports_flag,
                    "total_flag": total_flag,
                    "domestic_abs_gap": dom_lookup.get((metric, target_code), 0.0),
                    "imports_abs_gap": imp_lookup.get((metric, target_code), 0.0),
                    "total_abs_gap": tot_lookup.get((metric, target_code), 0.0),
                    "flow_classification": classification,
                }
            )
    else:
        dom_lookup = _flag_lookup(domestic)
        imp_lookup = _flag_lookup(imports)
        tot_lookup = _flag_lookup(total)
        cell_keys = sorted(set(dom_lookup) | set(imp_lookup) | set(tot_lookup))
        for metric, supplier_code, buyer_code in cell_keys:
            domestic_flag = (metric, supplier_code, buyer_code) in dom_lookup
            imports_flag = (metric, supplier_code, buyer_code) in imp_lookup
            total_flag = (metric, supplier_code, buyer_code) in tot_lookup
            if not (domestic_flag or imports_flag or total_flag):
                continue
            if domestic_flag and imports_flag:
                classification = "BOTH_DOMESTIC_AND_IMPORT"
            elif domestic_flag:
                classification = "DOMESTIC_DRIVEN"
            elif imports_flag:
                classification = "IMPORT_DRIVEN"
            elif total_flag:
                classification = "TOTAL_ONLY"
            else:
                classification = "NO_FLAG"
            rows.append(
                {
                    "kind": "z_cell",
                    "metric": metric,
                    "target_code": "",
                    "target_label": "",
                    "supplier_code": supplier_code,
                    "supplier_label": _lookup_label((domestic, imports, total), supplier_code, code_column="supplier_code", label_column="supplier_label"),
                    "buyer_code": buyer_code,
                    "buyer_label": _lookup_label((domestic, imports, total), buyer_code, code_column="buyer_code", label_column="buyer_label"),
                    "domestic_flag": domestic_flag,
                    "imports_flag": imports_flag,
                    "total_flag": total_flag,
                    "domestic_abs_gap": dom_lookup.get((metric, supplier_code, buyer_code), 0.0),
                    "imports_abs_gap": imp_lookup.get((metric, supplier_code, buyer_code), 0.0),
                    "total_abs_gap": tot_lookup.get((metric, supplier_code, buyer_code), 0.0),
                    "flow_classification": classification,
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["kind", "metric", "flow_classification"], kind="stable").reset_index(drop=True)
    return frame


def _quality_metric_label(metric: str) -> str:
    metric_text = str(metric)
    if metric_text in {METRIC_X_OUTPUT_PROXY_TOTAL_USE, METRIC_TOTAL_USE}:
        return "TOTAL_USE_PROXY"
    return metric_text


def _quality_bucket_label(rel_gap: float, eurostat_value: float, exio_mapped_value: float) -> str:
    euro_abs = abs(float(eurostat_value))
    exio_abs = abs(float(exio_mapped_value))
    if euro_abs <= EPS and exio_abs > EPS:
        return "Eurostat zero / EXIO positive"
    if exio_abs <= EPS and euro_abs > EPS:
        return "EXIO zero / Eurostat positive"
    pct = abs(float(rel_gap)) * 100.0
    if pct <= 5.0:
        return "0–5%"
    if pct <= 10.0:
        return "5–10%"
    if pct <= 20.0:
        return "10–20%"
    if pct <= 50.0:
        return "20–50%"
    if pct <= 100.0:
        return "50–100%"
    return ">100%"


def _align_series_to_index(series: pd.Series | None, index: list[str]) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(0.0, index=index, dtype="float64")
    return series.reindex(index=index).fillna(0.0).astype("float64")


def _safe_ratio(numerator: object, denominator: object) -> float:
    try:
        numerator_value = float(numerator)
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return 0.0
    if abs(denominator_value) <= EPS:
        return 0.0
    return numerator_value / denominator_value


def _quality_sector_frame(flow: str, result: ConcordanceAuditResult, *, eurostat_labels: Mapping[str, str]) -> pd.DataFrame:
    rel_threshold = float(result.metadata.get("rel_threshold", DEFAULT_REL_THRESHOLD))
    rows: list[dict[str, Any]] = []
    aggregate_frame = result.aggregate_totals.copy()
    available_metrics = set(aggregate_frame["metric"].astype(str)) if not aggregate_frame.empty else set()
    total_use_metric = METRIC_TOTAL_USE if METRIC_TOTAL_USE in available_metrics else METRIC_X_OUTPUT_PROXY_TOTAL_USE if METRIC_X_OUTPUT_PROXY_TOTAL_USE in available_metrics else None
    aggregate_metrics = {METRIC_Y_FINAL_DEMAND_TOTAL}
    if total_use_metric is not None:
        aggregate_metrics.add(total_use_metric)
    aggregate_frame = aggregate_frame[aggregate_frame["metric"].astype(str).isin(aggregate_metrics)].copy()
    if not aggregate_frame.empty:
        for metric, metric_frame in aggregate_frame.groupby(aggregate_frame["metric"].astype(str), sort=False):
            metric_block = _quality_metric_label(metric)
            metric_frame = metric_frame.sort_values(["abs_gap", "rel_gap"], ascending=[False, False], kind="stable").reset_index(drop=True)
            eurostat_mass = float(metric_frame["eurostat_value"].abs().sum())
            abs_gap_mass = float(metric_frame["abs_gap"].sum())
            cumulative = metric_frame["abs_gap"].cumsum()
            for position, row in metric_frame.iterrows():
                abs_gap = float(row["abs_gap"])
                euro_value = float(row["eurostat_value"])
                exio_value = float(row["exio_mapped_value"])
                rows.append(
                    {
                        "flow": flow,
                        "block": metric_block,
                        "sector_code": str(row["target_code"]),
                        "sector_label": str(row["target_label"]),
                        "metric": metric_block,
                        "eurostat_value": euro_value,
                        "exio_mapped_value": exio_value,
                        "signed_gap": float(row["signed_gap"]),
                        "abs_gap": abs_gap,
                        "rel_gap": float(row["rel_gap"]),
                        "ratio": float(row["ratio_exio_to_eurostat"]),
                        "eurostat_mass_share": _safe_ratio(abs(euro_value), eurostat_mass),
                        "abs_gap_share": _safe_ratio(abs_gap, abs_gap_mass),
                        "cumulative_abs_gap_share": _safe_ratio(float(cumulative.iloc[position]), abs_gap_mass),
                        "flag": bool(row["flag"]),
                        "severity": str(row["severity"]),
                    }
                )

    z_row_col_frame = result.z_row_col_mismatches.copy() if result.z_row_col_mismatches is not None else pd.DataFrame()
    if not z_row_col_frame.empty:
        z_row_col_frame = z_row_col_frame[z_row_col_frame["metric"].astype(str).isin({METRIC_Z_ROW_INTERMEDIATE_SALES, METRIC_Z_COL_INTERMEDIATE_PURCHASES})].copy()
        for metric, metric_frame in z_row_col_frame.groupby(z_row_col_frame["metric"].astype(str), sort=False):
            metric_block = _quality_metric_label(metric)
            metric_frame = metric_frame.sort_values(["abs_gap", "rel_gap"], ascending=[False, False], kind="stable").reset_index(drop=True)
            eurostat_mass = float(metric_frame["eurostat_value"].abs().sum())
            abs_gap_mass = float(metric_frame["abs_gap"].sum())
            cumulative = metric_frame["abs_gap"].cumsum()
            for position, row in metric_frame.iterrows():
                abs_gap = float(row["abs_gap"])
                euro_value = float(row["eurostat_value"])
                exio_value = float(row["exio_mapped_value"])
                rows.append(
                    {
                        "flow": flow,
                        "block": metric_block,
                        "sector_code": str(row.get("target_code", row.get("supplier_code", row.get("buyer_code", "")))),
                        "sector_label": str(row.get("target_label", row.get("supplier_label", row.get("buyer_label", "")))),
                        "metric": metric_block,
                        "eurostat_value": euro_value,
                        "exio_mapped_value": exio_value,
                        "signed_gap": float(row["signed_gap"]),
                        "abs_gap": abs_gap,
                        "rel_gap": float(row["rel_gap"]),
                        "ratio": float(row.get("ratio_exio_to_eurostat", _safe_ratio(exio_value, euro_value))),
                        "eurostat_mass_share": _safe_ratio(abs(euro_value), eurostat_mass),
                        "abs_gap_share": _safe_ratio(abs_gap, abs_gap_mass),
                        "cumulative_abs_gap_share": _safe_ratio(float(cumulative.iloc[position]), abs_gap_mass),
                        "flag": bool(row["flag"]),
                        "severity": str(row["severity"]),
                    }
                )
    return pd.DataFrame(rows)


def _quality_z_cell_frame(flow: str, result: ConcordanceAuditResult, *, eurostat_labels: Mapping[str, str]) -> pd.DataFrame:
    rel_threshold = float(result.metadata.get("rel_threshold", DEFAULT_REL_THRESHOLD))
    min_abs_meur = float(result.metadata.get("min_abs_meur", DEFAULT_MIN_ABS_MEUR))
    exio_z = result.mapped_z if result.mapped_z is not None else pd.DataFrame(dtype="float64")
    euro_z = result.eurostat_z if result.eurostat_z is not None else pd.DataFrame(dtype="float64")
    if exio_z.empty and euro_z.empty:
        return pd.DataFrame()

    codes = sorted(set(exio_z.index.astype(str)).union(set(exio_z.columns.astype(str)), set(euro_z.index.astype(str)), set(euro_z.columns.astype(str))))
    exio_aligned = exio_z.reindex(index=codes, columns=codes).fillna(0.0).astype("float64")
    euro_aligned = euro_z.reindex(index=codes, columns=codes).fillna(0.0).astype("float64")

    cell_rows: list[dict[str, Any]] = []
    for supplier_code in codes:
        for buyer_code in codes:
            exio_value = float(exio_aligned.loc[supplier_code, buyer_code])
            euro_value = float(euro_aligned.loc[supplier_code, buyer_code])
            signed_gap = exio_value - euro_value
            abs_gap = abs(signed_gap)
            rel_gap = concordance_rel_gap(exio_value, euro_value)
            flagged = abs_gap >= min_abs_meur and rel_gap >= rel_threshold
            cell_rows.append(
                {
                    "flow": flow,
                    "block": "Z_CELL",
                    "metric": "Z_CELL",
                    "supplier_code": supplier_code,
                    "supplier_label": eurostat_labels.get(supplier_code, supplier_code),
                    "buyer_code": buyer_code,
                    "buyer_label": eurostat_labels.get(buyer_code, buyer_code),
                    "eurostat_value": euro_value,
                    "exio_mapped_value": exio_value,
                    "signed_gap": signed_gap,
                    "abs_gap": abs_gap,
                    "rel_gap": rel_gap,
                    "ratio": _safe_ratio(exio_value, euro_value),
                    "flag": flagged,
                    "severity": _severity(rel_gap, rel_threshold),
                }
            )

    frame = pd.DataFrame(cell_rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["abs_gap", "rel_gap"], ascending=[False, False], kind="stable").reset_index(drop=True)
    eurostat_mass = float(frame["eurostat_value"].abs().sum())
    abs_gap_mass = float(frame["abs_gap"].sum())
    frame["eurostat_mass_share"] = frame["eurostat_value"].abs().map(lambda value: _safe_ratio(value, eurostat_mass))
    frame["abs_gap_share"] = frame["abs_gap"].map(lambda value: _safe_ratio(value, abs_gap_mass))
    frame["cumulative_abs_gap_share"] = frame["abs_gap"].cumsum().map(lambda value: _safe_ratio(value, abs_gap_mass))
    return frame


def _quality_summary_frame(sector_quality: pd.DataFrame, z_cell_quality: pd.DataFrame) -> pd.DataFrame:
    summary_frames: list[pd.DataFrame] = []
    for source_frame, block_names in (
        (sector_quality, ["Z_ROW_INTERMEDIATE_SALES", "Z_COL_INTERMEDIATE_PURCHASES", "Y_FINAL_DEMAND_TOTAL", "TOTAL_USE_PROXY"]),
        (z_cell_quality, ["Z_CELL"]),
    ):
        for block in block_names:
            block_frame = source_frame[source_frame["block"].astype(str).eq(block)].copy()
            if block_frame.empty:
                continue
            eurostat_mass = float(block_frame["eurostat_value"].abs().sum()) if not block_frame.empty else 0.0
            exio_mapped_mass = float(block_frame["exio_mapped_value"].abs().sum())
            signed_gap_total = float(block_frame["signed_gap"].sum())
            absolute_discrepancy_mass = float(block_frame["abs_gap"].sum())
            flagged = block_frame[block_frame["flag"]].copy()
            flagged_cell_count = int(len(flagged))
            total_cell_count = int(len(block_frame))
            flagged_eurostat_mass = float(flagged["eurostat_value"].abs().sum())
            flagged_abs_gap = float(flagged["abs_gap"].sum())
            block_frame = block_frame.sort_values(["abs_gap", "rel_gap"], ascending=[False, False], kind="stable").reset_index(drop=True)
            within_20 = float(block_frame.loc[block_frame["rel_gap"].fillna(0.0).le(0.20), "eurostat_value"].abs().sum())
            top10_abs_gap = float(block_frame.head(10)["abs_gap"].sum())
            summary_frames.append(
                pd.DataFrame(
                    [
                        {
                            "flow": str(block_frame.iloc[0]["flow"]),
                            "block": block,
                            "eurostat_mass": eurostat_mass,
                            "exio_mapped_mass": exio_mapped_mass,
                            "signed_gap_total": signed_gap_total,
                            "absolute_discrepancy_mass": absolute_discrepancy_mass,
                            "wape_vs_eurostat": _safe_ratio(absolute_discrepancy_mass, eurostat_mass),
                            "matched_mass_share": _safe_ratio(float(np.minimum(block_frame["eurostat_value"].abs(), block_frame["exio_mapped_value"].abs()).sum()), float(np.maximum(block_frame["eurostat_value"].abs(), block_frame["exio_mapped_value"].abs()).sum())),
                            "share_of_eurostat_mass_within_20pct_gap": _safe_ratio(within_20, eurostat_mass),
                            "flagged_cell_count": flagged_cell_count,
                            "total_cell_count": total_cell_count,
                            "flagged_cell_share": _safe_ratio(flagged_cell_count, total_cell_count),
                            "flagged_eurostat_mass": flagged_eurostat_mass,
                            "flagged_eurostat_mass_share": _safe_ratio(flagged_eurostat_mass, eurostat_mass),
                            "flagged_abs_gap": flagged_abs_gap,
                            "flagged_abs_gap_share": _safe_ratio(flagged_abs_gap, absolute_discrepancy_mass),
                            "top10_abs_gap_share": _safe_ratio(top10_abs_gap, absolute_discrepancy_mass),
                        }
                    ]
                )
            )
    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)


def _gap_bucket_frame(sector_quality: pd.DataFrame, z_cell_quality: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([sector_quality, z_cell_quality], ignore_index=True, sort=False)
    if combined.empty:
        return combined

    bucket_order = {
        "0–5%": 0,
        "5–10%": 1,
        "10–20%": 2,
        "20–50%": 3,
        "50–100%": 4,
        ">100%": 5,
        "Eurostat zero / EXIO positive": 6,
        "EXIO zero / Eurostat positive": 7,
    }
    combined = combined.copy()
    combined["bucket"] = combined.apply(lambda row: _quality_bucket_label(float(row["rel_gap"]), float(row["eurostat_value"]), float(row["exio_mapped_value"])), axis=1)
    rows: list[dict[str, Any]] = []
    for (flow, block), block_frame in combined.groupby(["flow", "block"], sort=False):
        total_count = int(len(block_frame))
        eurostat_mass = float(block_frame["eurostat_value"].abs().sum())
        abs_gap_mass = float(block_frame["abs_gap"].sum())
        for bucket, bucket_frame in block_frame.groupby("bucket", sort=False):
            rows.append(
                {
                    "flow": flow,
                    "block": block,
                    "bucket": bucket,
                    "bucket_order": bucket_order.get(bucket, 99),
                    "cell_count": int(len(bucket_frame)),
                    "cell_count_share": _safe_ratio(len(bucket_frame), total_count),
                    "eurostat_mass": float(bucket_frame["eurostat_value"].abs().sum()),
                    "eurostat_mass_share": _safe_ratio(float(bucket_frame["eurostat_value"].abs().sum()), eurostat_mass),
                    "abs_gap": float(bucket_frame["abs_gap"].sum()),
                    "abs_gap_share": _safe_ratio(float(bucket_frame["abs_gap"].sum()), abs_gap_mass),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["flow", "block", "bucket_order"], kind="stable").drop(columns=["bucket_order"]).reset_index(drop=True)
    return frame


def _top_discrepancy_contributors_frame(sector_quality: pd.DataFrame, z_cell_quality: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([sector_quality, z_cell_quality], ignore_index=True, sort=False)
    if combined.empty:
        return combined

    rows: list[dict[str, Any]] = []
    for (flow, block), block_frame in combined.groupby(["flow", "block"], sort=False):
        block_frame = block_frame.sort_values(["abs_gap", "rel_gap"], ascending=[False, False], kind="stable").reset_index(drop=True)
        abs_gap_mass = float(block_frame["abs_gap"].sum())
        for rank, (_, row) in enumerate(block_frame.head(20).iterrows(), start=1):
            rows.append(
                {
                    "flow": flow,
                    "block": block,
                    "rank": rank,
                    "kind": "cell" if str(block) == "Z_CELL" else "sector",
                    "metric": row.get("metric", block),
                    "sector_code": row.get("sector_code", ""),
                    "sector_label": row.get("sector_label", ""),
                    "supplier_code": row.get("supplier_code", ""),
                    "supplier_label": row.get("supplier_label", ""),
                    "buyer_code": row.get("buyer_code", ""),
                    "buyer_label": row.get("buyer_label", ""),
                    "eurostat_value": float(row["eurostat_value"]),
                    "exio_mapped_value": float(row["exio_mapped_value"]),
                    "signed_gap": float(row["signed_gap"]),
                    "abs_gap": float(row["abs_gap"]),
                    "rel_gap": float(row["rel_gap"]),
                    "ratio": float(row["ratio"]),
                    "eurostat_mass_share": float(row.get("eurostat_mass_share", 0.0)),
                    "abs_gap_share": _safe_ratio(float(row["abs_gap"]), abs_gap_mass),
                    "cumulative_abs_gap_share": _safe_ratio(float(block_frame.head(rank)["abs_gap"].sum()), abs_gap_mass),
                    "flag": bool(row.get("flag", False)),
                    "severity": str(row.get("severity", "")),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["flow", "block", "rank"], kind="stable").reset_index(drop=True)
    return frame


def _format_quality_summary_md(quality_summary: pd.DataFrame, bucket_summary: pd.DataFrame) -> str:
    lines = ["## How much of the concordance agrees?", "", EXPORT_EXCLUSION_CAVEAT, "", "These are overlapping diagnostic views; do not sum across blocks as independent error shares.", ""]
    if quality_summary.empty:
        lines.append("No quality summary rows were produced.")
        return "\n".join(lines).rstrip() + "\n"

    display = quality_summary.copy()
    display["flow"] = display["flow"].astype(str)
    display["block"] = display["block"].astype(str)
    summary_columns = [
        "flow",
        "block",
        "eurostat_mass",
        "exio_mapped_mass",
        "absolute_discrepancy_mass",
        "wape_vs_eurostat",
        "matched_mass_share",
        "share_of_eurostat_mass_within_20pct_gap",
        "flagged_eurostat_mass_share",
        "top10_abs_gap_share",
    ]
    summary_view = display.loc[:, summary_columns].copy()
    summary_view["eurostat_mass"] = summary_view["eurostat_mass"].map(fmt_meur)
    summary_view["exio_mapped_mass"] = summary_view["exio_mapped_mass"].map(fmt_meur)
    summary_view["absolute_discrepancy_mass"] = summary_view["absolute_discrepancy_mass"].map(fmt_meur)
    summary_view["wape_vs_eurostat"] = summary_view["wape_vs_eurostat"].map(fmt_pct)
    summary_view["matched_mass_share"] = summary_view["matched_mass_share"].map(fmt_pct)
    summary_view["share_of_eurostat_mass_within_20pct_gap"] = summary_view["share_of_eurostat_mass_within_20pct_gap"].map(fmt_pct)
    summary_view["flagged_eurostat_mass_share"] = summary_view["flagged_eurostat_mass_share"].map(fmt_pct)
    summary_view["top10_abs_gap_share"] = summary_view["top10_abs_gap_share"].map(fmt_pct)
    lines.append(_markdown_table(summary_view, columns=summary_columns, max_rows=20))
    lines.extend(["", "### Gap buckets", ""])
    if bucket_summary.empty:
        lines.append("No gap bucket rows were produced.")
    else:
        bucket_view = bucket_summary.copy()
        bucket_view["eurostat_mass"] = bucket_view["eurostat_mass"].map(fmt_meur)
        bucket_view["eurostat_mass_share"] = bucket_view["eurostat_mass_share"].map(fmt_pct)
        bucket_view["abs_gap"] = bucket_view["abs_gap"].map(fmt_meur)
        bucket_view["abs_gap_share"] = bucket_view["abs_gap_share"].map(fmt_pct)
        bucket_view["cell_count_share"] = bucket_view["cell_count_share"].map(fmt_pct)
        lines.append(_markdown_table(bucket_view, columns=["flow", "block", "bucket", "cell_count", "cell_count_share", "eurostat_mass", "eurostat_mass_share", "abs_gap", "abs_gap_share"], max_rows=200))

    if not quality_summary.empty:
        lines.extend(["", "### Agreement by flow and block", ""])
        for row in quality_summary.itertuples(index=False):
            within_20_share = float(row.share_of_eurostat_mass_within_20pct_gap)
            flagged_share = float(row.flagged_eurostat_mass_share)
            top10_share = float(row.top10_abs_gap_share)
            if within_20_share >= 0.8 and flagged_share <= 0.2:
                agreement_text = "high agreement"
            elif within_20_share >= 0.5:
                agreement_text = "moderate agreement"
            else:
                agreement_text = "low agreement"
            lines.append(
                f"- {row.flow} / {row.block}: agreement is {agreement_text} because {within_20_share:.1%} of Eurostat mass sits within 20% gap, flagged Eurostat mass is {flagged_share:.1%}, and the top-10 discrepancies capture about {top10_share:.1%} of total absolute gap."
            )
    return "\n".join(lines).rstrip() + "\n"


def _paired_cell_display(eurostat_value: object, exio_value: object, signed_gap: object, rel_gap: object) -> str:
    return "\n".join([
        f"EU: {fmt_meur(eurostat_value)}",
        f"EX: {fmt_meur(exio_value)}",
        f"Δ: {fmt_meur(signed_gap)}",
        f"gap: {fmt_pct(rel_gap)}",
    ])


def _iot_pair_display_frame(
    *,
    euro_z: pd.DataFrame,
    exio_z: pd.DataFrame,
    euro_y: pd.Series,
    exio_y: pd.Series,
    labels: Mapping[str, str],
) -> pd.DataFrame:
    codes = sorted(set(euro_z.index.astype(str)).union(set(euro_z.columns.astype(str)), set(exio_z.index.astype(str)), set(exio_z.columns.astype(str)), set(euro_y.index.astype(str)), set(exio_y.index.astype(str))))
    z_columns = [code for code in codes]
    display_columns = [f"{code} — {labels.get(code, code)}" for code in z_columns] + ["Y_TOTAL", "TOTAL_USE"]
    display_index = [f"{code} — {labels.get(code, code)}" for code in codes]
    euro_z_aligned = euro_z.reindex(index=codes, columns=codes).fillna(0.0) if not euro_z.empty else pd.DataFrame(0.0, index=codes, columns=codes)
    exio_z_aligned = exio_z.reindex(index=codes, columns=codes).fillna(0.0) if not exio_z.empty else pd.DataFrame(0.0, index=codes, columns=codes)
    euro_y_aligned = _align_series_to_index(euro_y, codes)
    exio_y_aligned = _align_series_to_index(exio_y, codes)

    frame = pd.DataFrame(index=display_index, columns=display_columns, dtype=object)
    for row_code in codes:
        row_label = f"{row_code} — {labels.get(row_code, row_code)}"
        euro_row_total = float(euro_z_aligned.loc[row_code].sum() + euro_y_aligned.get(row_code, 0.0))
        exio_row_total = float(exio_z_aligned.loc[row_code].sum() + exio_y_aligned.get(row_code, 0.0))
        for col_code in codes:
            col_label = f"{col_code} — {labels.get(col_code, col_code)}"
            frame.loc[row_label, col_label] = _paired_cell_display(
                euro_z_aligned.loc[row_code, col_code],
                exio_z_aligned.loc[row_code, col_code],
                float(exio_z_aligned.loc[row_code, col_code]) - float(euro_z_aligned.loc[row_code, col_code]),
                concordance_rel_gap(exio_z_aligned.loc[row_code, col_code], euro_z_aligned.loc[row_code, col_code]),
            )
        frame.loc[row_label, "Y_TOTAL"] = _paired_cell_display(
            euro_y_aligned.get(row_code, 0.0),
            exio_y_aligned.get(row_code, 0.0),
            float(exio_y_aligned.get(row_code, 0.0)) - float(euro_y_aligned.get(row_code, 0.0)),
            concordance_rel_gap(exio_y_aligned.get(row_code, 0.0), euro_y_aligned.get(row_code, 0.0)),
        )
        frame.loc[row_label, "TOTAL_USE"] = _paired_cell_display(
            euro_row_total,
            exio_row_total,
            exio_row_total - euro_row_total,
            concordance_rel_gap(exio_row_total, euro_row_total),
        )
    return frame


def _write_paired_iot_workbook(
    *,
    workbook_path: Path,
    country: str,
    year: int,
    quality_summary: pd.DataFrame,
    sector_quality: pd.DataFrame,
    top_discrepancies: pd.DataFrame,
    flows: Mapping[str, FlowAuditRun],
    eurostat_labels: Mapping[str, str],
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.dataframe import dataframe_to_rows

    workbook = Workbook()
    workbook.remove(workbook.active)

    def _append_sheet(frame: pd.DataFrame, sheet_name: str, *, include_index: bool = False) -> None:
        worksheet = workbook.create_sheet(title=sheet_name)
        for row in dataframe_to_rows(frame, index=include_index, header=True):
            worksheet.append(list(row))
        worksheet.freeze_panes = "A2"
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for column_cells in worksheet.columns:
            width = 12
            for cell in column_cells:
                if cell.value is None:
                    continue
                width = min(60, max(width, len(str(cell.value)) + 2))
            worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width
        header_fill = PatternFill("solid", fgColor="D9EAF7")
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill

    def _add_numeric_sheet(sheet_name: str, frame: pd.DataFrame) -> None:
        _append_sheet(frame, sheet_name, include_index=True)

    def _add_rel_sheet(sheet_name: str, frame: pd.DataFrame) -> None:
        _append_sheet(frame, sheet_name, include_index=True)
        worksheet = workbook[sheet_name]
        max_row = worksheet.max_row
        max_col = worksheet.max_column
        if max_row >= 2 and max_col >= 2:
            value_range = f"B2:{get_column_letter(max_col)}{max_row}"
            worksheet.conditional_formatting.add(value_range, CellIsRule(operator="greaterThan", formula=["0.2"], fill=PatternFill("solid", fgColor="FFF2CC")))
            worksheet.conditional_formatting.add(value_range, CellIsRule(operator="greaterThan", formula=["0.5"], fill=PatternFill("solid", fgColor="F9CB9C")))
            worksheet.conditional_formatting.add(value_range, CellIsRule(operator="greaterThan", formula=["1.0"], fill=PatternFill("solid", fgColor="F4CCCC")))

    def _pair_display_matrix(flow_key: str, euro_frame: pd.DataFrame, exio_frame: pd.DataFrame) -> pd.DataFrame:
        if euro_frame is None or euro_frame.empty:
            return pd.DataFrame()
        codes = sorted(set(euro_frame.index.astype(str)).union(set(euro_frame.columns.astype(str)), set(exio_frame.index.astype(str) if exio_frame is not None else []), set(exio_frame.columns.astype(str) if exio_frame is not None else [])))
        euro_aligned = euro_frame.reindex(index=codes, columns=codes).fillna(0.0) if not euro_frame.empty else pd.DataFrame(0.0, index=codes, columns=codes)
        exio_aligned = exio_frame.reindex(index=codes, columns=codes).fillna(0.0) if exio_frame is not None and not exio_frame.empty else pd.DataFrame(0.0, index=codes, columns=codes)
        display = pd.DataFrame(index=[f"{code} — {eurostat_labels.get(code, code)}" for code in codes], columns=[f"{code} — {eurostat_labels.get(code, code)}" for code in codes], dtype=object)
        for supplier_code in codes:
            for buyer_code in codes:
                display.loc[f"{supplier_code} — {eurostat_labels.get(supplier_code, supplier_code)}", f"{buyer_code} — {eurostat_labels.get(buyer_code, buyer_code)}"] = _paired_cell_display(
                    euro_aligned.loc[supplier_code, buyer_code],
                    exio_aligned.loc[supplier_code, buyer_code],
                    exio_aligned.loc[supplier_code, buyer_code] - euro_aligned.loc[supplier_code, buyer_code],
                    concordance_rel_gap(exio_aligned.loc[supplier_code, buyer_code], euro_aligned.loc[supplier_code, buyer_code]),
                )
        return display

    def _pair_display_series(flow_key: str, euro_series: pd.Series, exio_series: pd.Series) -> pd.DataFrame:
        codes = sorted(set(euro_series.index.astype(str)).union(set(exio_series.index.astype(str))))
        rows = []
        for code in codes:
            euro_value = float(euro_series.get(code, 0.0))
            exio_value = float(exio_series.get(code, 0.0))
            rows.append(
                {
                    "sector_code": code,
                    "sector_label": eurostat_labels.get(code, code),
                    "pair_display": _paired_cell_display(euro_value, exio_value, exio_value - euro_value, concordance_rel_gap(exio_value, euro_value)),
                }
            )
        return pd.DataFrame(rows)

    def _sector_totals_pair_frame(flow_key: str, euro_z: pd.DataFrame, exio_z: pd.DataFrame, euro_y: pd.Series, exio_y: pd.Series) -> pd.DataFrame:
        codes = sorted(set(euro_z.index.astype(str)).union(set(euro_z.columns.astype(str)), set(exio_z.index.astype(str)), set(exio_z.columns.astype(str)), set(euro_y.index.astype(str)), set(exio_y.index.astype(str))))
        euro_z_aligned = euro_z.reindex(index=codes, columns=codes).fillna(0.0) if not euro_z.empty else pd.DataFrame(0.0, index=codes, columns=codes)
        exio_z_aligned = exio_z.reindex(index=codes, columns=codes).fillna(0.0) if not exio_z.empty else pd.DataFrame(0.0, index=codes, columns=codes)
        euro_y_aligned = euro_y.reindex(index=codes).fillna(0.0) if not euro_y.empty else pd.Series(0.0, index=codes, dtype="float64")
        exio_y_aligned = exio_y.reindex(index=codes).fillna(0.0) if not exio_y.empty else pd.Series(0.0, index=codes, dtype="float64")
        rows: list[dict[str, Any]] = []
        for code in codes:
            euro_row = float(euro_z_aligned.loc[code].sum())
            exio_row = float(exio_z_aligned.loc[code].sum())
            euro_col = float(euro_z_aligned[code].sum())
            exio_col = float(exio_z_aligned[code].sum())
            euro_y_value = float(euro_y_aligned.get(code, 0.0))
            exio_y_value = float(exio_y_aligned.get(code, 0.0))
            euro_total = euro_row + euro_y_value
            exio_total = exio_row + exio_y_value
            rows.append(
                {
                    "sector_code": code,
                    "sector_label": eurostat_labels.get(code, code),
                    "Z row pair_display": _paired_cell_display(euro_row, exio_row, exio_row - euro_row, concordance_rel_gap(exio_row, euro_row)),
                    "Z col pair_display": _paired_cell_display(euro_col, exio_col, exio_col - euro_col, concordance_rel_gap(exio_col, euro_col)),
                    "Y pair_display": _paired_cell_display(euro_y_value, exio_y_value, exio_y_value - euro_y_value, concordance_rel_gap(exio_y_value, euro_y_value)),
                    "TOTAL_USE pair_display": _paired_cell_display(euro_total, exio_total, exio_total - euro_total, concordance_rel_gap(exio_total, euro_total)),
                }
            )
        return pd.DataFrame(rows)

    def _single_sheet_sheet_name(prefix: str, suffix: str) -> str:
        return f"{prefix}_{suffix}"

    readme = pd.DataFrame(
        [
            {"item": "country", "value": country},
            {"item": "year", "value": year},
            {"item": "purpose", "value": "Paired Eurostat vs EXIOBASE concordance workbook."},
            {"item": "destination_use_caveat", "value": EXPORT_EXCLUSION_CAVEAT},
            {"item": "pair_display_format", "value": "EU / EX / Δ / gap"},
            {"item": "relative_gap_threshold_highlight", "value": ">20%, >50%, >100%"},
            {"item": "paired sheets", "value": "Cells contain Eurostat and EXIOBASE values, signed gap, and relative gap."},
        ]
    )
    _append_sheet(readme, "README")
    _append_sheet(quality_summary, "quality_summary")
    _append_sheet(sector_quality, "sector_quality")
    _append_sheet(top_discrepancies, "top_discrepancies")

    for flow_key, flow_run in flows.items():
        flow_prefix = {"domestic": "DOM", "imports": "IMP", "total": "TOTAL"}[flow_key]
        z_matrix = flow_run.result.eurostat_z if flow_run.result.eurostat_z is not None else pd.DataFrame()
        mapped_z = flow_run.result.mapped_z if flow_run.result.mapped_z is not None else pd.DataFrame()
        y_euro = flow_run.result.eurostat_y_total if flow_run.result.eurostat_y_total is not None else pd.Series(dtype="float64")
        y_exio = flow_run.result.mapped_y if flow_run.result.mapped_y is not None else pd.Series(dtype="float64")
        z_codes = sorted(set(z_matrix.index.astype(str)).union(set(z_matrix.columns.astype(str)), set(mapped_z.index.astype(str)), set(mapped_z.columns.astype(str))))
        z_euro_aligned = z_matrix.reindex(index=z_codes, columns=z_codes).fillna(0.0) if not z_matrix.empty else pd.DataFrame(0.0, index=z_codes, columns=z_codes)
        z_exio_aligned = mapped_z.reindex(index=z_codes, columns=z_codes).fillna(0.0) if not mapped_z.empty else pd.DataFrame(0.0, index=z_codes, columns=z_codes)

        pair_sheet = _pair_display_matrix(flow_key, z_matrix, mapped_z)
        _append_sheet(pair_sheet, f"{flow_prefix}_Z_pair_display", include_index=True)
        _add_numeric_sheet(f"{flow_prefix}_Z_eurostat", z_euro_aligned)
        _add_numeric_sheet(f"{flow_prefix}_Z_exio_mapped", z_exio_aligned)
        _add_numeric_sheet(f"{flow_prefix}_Z_signed_gap", (z_exio_aligned - z_euro_aligned).fillna(0.0))
        rel_gap_matrix = pd.DataFrame(index=z_codes, columns=z_codes, dtype="float64") if z_codes else pd.DataFrame()
        if z_codes:
            for supplier_code in z_codes:
                for buyer_code in z_codes:
                    rel_gap_matrix.loc[supplier_code, buyer_code] = concordance_rel_gap(z_exio_aligned.loc[supplier_code, buyer_code], z_euro_aligned.loc[supplier_code, buyer_code])
        _add_rel_sheet(f"{flow_prefix}_Z_rel_gap", rel_gap_matrix)

        y_pair = _pair_display_series(flow_key, y_euro, y_exio)
        _append_sheet(y_pair, f"{flow_prefix}_Y_pair_display")

        sector_totals = _sector_totals_pair_frame(flow_key, z_matrix, mapped_z, y_euro, y_exio)
        _append_sheet(sector_totals, f"{flow_prefix}_sector_totals_pair", include_index=False)

        iot_pair = _iot_pair_display_frame(euro_z=z_matrix, exio_z=mapped_z, euro_y=y_euro, exio_y=y_exio, labels=eurostat_labels)
        _append_sheet(iot_pair, f"{flow_prefix}_IOT_pair_display", include_index=True)

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(workbook_path)


def _frame_to_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame.copy()


def _build_consolidated_summary_md(
    *,
    country: str,
    year: int,
    domestic: FlowAuditRun,
    imports: FlowAuditRun,
    total: FlowAuditRun,
    quality_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    reconciliation: pd.DataFrame,
    aggregate_flags: pd.DataFrame,
    z_flags: pd.DataFrame,
    classification: pd.DataFrame,
    mapping_risk_warning: bool,
) -> str:
    def _classification_view(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        view = frame.copy()
        view["metric_display"] = view["metric"].map(human_metric_label)
        if "target_code" in view.columns:
            view["target_display"] = view.apply(lambda row: _optional_display_code_label(row.get("target_code", ""), row.get("target_label", "")), axis=1)
        else:
            view["target_display"] = ""
        if "supplier_code" in view.columns:
            view["supplier_display"] = view.apply(lambda row: _optional_display_code_label(row.get("supplier_code", ""), row.get("supplier_label", "")), axis=1)
        else:
            view["supplier_display"] = ""
        if "buyer_code" in view.columns:
            view["buyer_display"] = view.apply(lambda row: _optional_display_code_label(row.get("buyer_code", ""), row.get("buyer_label", "")), axis=1)
        else:
            view["buyer_display"] = ""
        view["comparison_display"] = view.apply(
            lambda row: row["target_display"] if str(row.get("kind", "")) == "aggregate" else f"{row['supplier_display']} → {row['buyer_display']}",
            axis=1,
        )
        return view

    def _reconciliation_view(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        display_rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            if row.metric == METRIC_Z_CELL:
                sector_display = f"{_optional_display_code_label(row.supplier_code, row.supplier_label)} → {_optional_display_code_label(row.buyer_code, row.buyer_label)}"
            else:
                sector_display = _optional_display_code_label(row.target_code, row.target_label)
            display_rows.append(
                {
                    "Database": row.database,
                    "Readable metric": human_metric_label(str(row.metric)),
                    "Sector or supplier→buyer pair": sector_display,
                    "Domestic value, MEUR": fmt_meur(row.domestic_value),
                    "Imports value, MEUR": fmt_meur(row.imports_value),
                    "Domestic + imports, MEUR": fmt_meur(row.domestic_plus_imports),
                    "Total value, MEUR": fmt_meur(row.total_value),
                    "Difference, MEUR": fmt_meur(row.signed_gap),
                    "Relative gap, %": fmt_pct(row.rel_gap),
                    "Interpretation": row.interpretation,
                }
            )
        return pd.DataFrame(display_rows)

    def _top(frame: pd.DataFrame, *, flow: str, kind: str, n: int = 10) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame[frame["flow"].astype(str).eq(flow)].copy()
        if out.empty:
            return out
        out["severity_rank"] = out["severity"].map(_severity_rank).fillna(-1).astype(int)
        out = out.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        return out.head(n)

    def _format_table(frame: pd.DataFrame, *, rename_map: Mapping[str, str], formatters: Mapping[str, Any], columns: list[str]) -> str:
        if frame.empty:
            return "No flagged discrepancies."
        view = frame.rename(columns=rename_map).copy()
        for column, formatter in formatters.items():
            if column in view.columns:
                view[column] = view[column].map(formatter)
        return _markdown_table(view, columns=columns, max_rows=10)

    lines = [
        f"# Consolidated EXIOBASE–Eurostat Concordance Audit: {country} {year}",
        "",
        "## What was compared",
        "| Flow | EXIOBASE slice | Eurostat table | Output folder |",
        "| --- | --- | --- | --- |",
        f"| Domestic | {country}-origin products used by {country} | stk_flow=DOM | domestic |",
        f"| Imports | Non-{country}-origin products used by {country}, foreign origins summed | stk_flow=IMP | imports |",
        f"| Total | All-origin products used by {country} | stk_flow=TOTAL | total |",
        "",
        "## Important caveat on x-like rows",
        "All x-like rows in these audits are flow-specific total-use proxy comparisons: Z row total + Y total.",
        "They are not independently observed gross-output comparisons.",
        "",
        EXPORT_EXCLUSION_CAVEAT,
        "",
        "Warning: multi-target EXIO→Eurostat mappings without explicit weights are normalized equally. Discrepancies involving such targets may reflect concordance allocation assumptions rather than database disagreement." if mapping_risk_warning else "",
        "",
        "## Executive summary",
        f"- Number of aggregate flags in domestic audit: {int(aggregate_flags[aggregate_flags['flow'].astype(str).eq('domestic')].shape[0])}",
        f"- Number of aggregate flags in imports audit: {int(aggregate_flags[aggregate_flags['flow'].astype(str).eq('imports')].shape[0])}",
        f"- Number of aggregate flags in total audit: {int(aggregate_flags[aggregate_flags['flow'].astype(str).eq('total')].shape[0])}",
        f"- Number of Z-cell flags in domestic audit: {int(domestic.result.z_cell_mismatches['flag'].sum())}",
        f"- Number of Z-cell flags in imports audit: {int(imports.result.z_cell_mismatches['flag'].sum())}",
        f"- Number of Z-cell flags in total audit: {int(total.result.z_cell_mismatches['flag'].sum())}",
        f"- Number of reconciliation flags where DOM + IMP != TOTAL: {int(reconciliation['flag'].sum())}",
        "",
        "## Biggest domestic discrepancies",
    ]
    dom_agg = _top(aggregate_flags, flow="domestic", kind="aggregate")
    dom_cell = _top(z_flags, flow="domestic", kind="z_cell")
    imp_agg = _top(aggregate_flags, flow="imports", kind="aggregate")
    imp_cell = _top(z_flags, flow="imports", kind="z_cell")
    tot_agg = _top(aggregate_flags, flow="total", kind="aggregate")
    tot_cell = _top(z_flags, flow="total", kind="z_cell")

    def _agg_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No flagged aggregate discrepancies."
        return _format_table(
            frame,
            rename_map={"human_metric": "Readable metric", "target_display": "Target sector", "exio_mapped_value": "EXIOBASE mapped value, MEUR", "eurostat_value": "Eurostat value, MEUR", "signed_gap": "Difference, MEUR", "rel_gap": "Relative gap", "ratio_exio_to_eurostat": "Ratio", "severity": "Severity", "top_exio_contributors": "Largest EXIOBASE contributors"},
            formatters={"EXIOBASE mapped value, MEUR": fmt_meur, "Eurostat value, MEUR": fmt_meur, "Difference, MEUR": fmt_meur, "Relative gap": fmt_pct, "Ratio": fmt_ratio},
            columns=["Readable metric", "Target sector", "EXIOBASE mapped value, MEUR", "Eurostat value, MEUR", "Difference, MEUR", "Relative gap", "Ratio", "Severity", "Largest EXIOBASE contributors"],
        )

    def _cell_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No flagged Z-cell discrepancies."
        return _format_table(
            frame,
            rename_map={"comparison_display": "Supplier → buyer", "exio_mapped_z": "EXIOBASE mapped flow, MEUR", "eurostat_z": "Eurostat flow, MEUR", "signed_gap": "Difference, MEUR", "rel_gap": "Relative gap", "ratio_exio_to_eurostat": "Ratio", "severity": "Severity", "top_source_subsectors": "Top source subsectors", "top_buyer_subsectors": "Top buyer subsectors"},
            formatters={"EXIOBASE mapped flow, MEUR": fmt_meur, "Eurostat flow, MEUR": fmt_meur, "Difference, MEUR": fmt_meur, "Relative gap": fmt_pct, "Ratio": fmt_ratio},
            columns=["Supplier → buyer", "EXIOBASE mapped flow, MEUR", "Eurostat flow, MEUR", "Difference, MEUR", "Relative gap", "Ratio", "Severity", "Top source subsectors", "Top buyer subsectors"],
        )

    lines.append(_agg_table(dom_agg))
    lines.extend(["", _cell_table(dom_cell), "", "## Biggest import discrepancies"])
    lines.append(_agg_table(imp_agg))
    lines.extend(["", _cell_table(imp_cell), "", "## Biggest total discrepancies"])
    lines.append(_agg_table(tot_agg))
    lines.extend(["", _cell_table(tot_cell), "", "## Discrepancies flagged in both domestic and imports"])
    both = classification[classification["flow_classification"].astype(str).eq("BOTH_DOMESTIC_AND_IMPORT")].copy()
    if not both.empty:
        lines.append(_markdown_table(_classification_view(both), columns=["kind", "metric_display", "comparison_display", "flow_classification"], max_rows=20))
    else:
        lines.append("No rows were flagged in both domestic and imports.")
    lines.extend(["", "## Discrepancies mostly driven by domestic flows"])
    dom_driven = classification[classification["flow_classification"].astype(str).eq("DOMESTIC_DRIVEN")].copy()
    if not dom_driven.empty:
        lines.append(_markdown_table(_classification_view(dom_driven), columns=["kind", "metric_display", "comparison_display", "flow_classification"], max_rows=20))
    else:
        lines.append("No domestic-driven rows were flagged.")
    lines.extend(["", "## Discrepancies mostly driven by imports"])
    imp_driven = classification[classification["flow_classification"].astype(str).eq("IMPORT_DRIVEN")].copy()
    if not imp_driven.empty:
        lines.append(_markdown_table(_classification_view(imp_driven), columns=["kind", "metric_display", "comparison_display", "flow_classification"], max_rows=20))
    else:
        lines.append("No import-driven rows were flagged.")
    lines.extend(["", "## Total-only discrepancies"])
    total_only = classification[classification["flow_classification"].astype(str).eq("TOTAL_ONLY")].copy()
    if not total_only.empty:
        lines.append(_markdown_table(_classification_view(total_only), columns=["kind", "metric_display", "comparison_display", "flow_classification"], max_rows=20))
    else:
        lines.append("No total-only rows were flagged.")
    lines.extend(["", _format_quality_summary_md(quality_summary, bucket_summary)])
    lines.extend(["", "## Flow reconciliation problems"])
    recon_flagged = reconciliation[reconciliation["flag"]].copy()
    if not recon_flagged.empty:
        recon_flagged["severity_rank"] = recon_flagged["severity"].map(_severity_rank).fillna(-1).astype(int)
        recon_flagged = recon_flagged.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        lines.append("TOTAL_USE is a derived check equal to Z row total + final demand. If it fails, inspect Z_ROW_INTERMEDIATE_SALES and Y_FINAL_DEMAND_TOTAL for the same sector.")
        lines.append(_markdown_table(_reconciliation_view(recon_flagged.head(20)), columns=["Database", "Readable metric", "Sector or supplier→buyer pair", "Domestic value, MEUR", "Imports value, MEUR", "Domestic + imports, MEUR", "Total value, MEUR", "Difference, MEUR", "Relative gap, %", "Interpretation"], max_rows=20))
    else:
        lines.append("No reconciliation mismatches exceeded the selected thresholds.")
    return "\n".join(lines).rstrip() + "\n"


def write_flow_consolidated_outputs(
    *,
    domestic: FlowAuditRun,
    imports: FlowAuditRun,
    total: FlowAuditRun,
    output_dir: Path,
    eurostat_labels: Mapping[str, str],
    rel_threshold: float,
    min_abs_meur: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sector_quality_frames: list[pd.DataFrame] = []
    z_quality_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    z_flag_frames: list[pd.DataFrame] = []
    for flow_name, flow_run in (("domestic", domestic), ("imports", imports), ("total", total)):
        agg_frame, cell_frame = _flow_flags_table(flow_run.result, flow=flow_name, eurostat_labels=eurostat_labels)
        aggregate_frames.append(agg_frame)
        z_flag_frames.append(cell_frame)
        sector_quality_frames.append(_quality_sector_frame(flow_name, flow_run.result, eurostat_labels=eurostat_labels))
        z_quality_frames.append(_quality_z_cell_frame(flow_name, flow_run.result, eurostat_labels=eurostat_labels))

    aggregate_flags = pd.concat(aggregate_frames, ignore_index=True) if aggregate_frames else pd.DataFrame()
    z_flags = pd.concat(z_flag_frames, ignore_index=True) if z_flag_frames else pd.DataFrame()
    sector_quality = pd.concat([frame for frame in sector_quality_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in sector_quality_frames) else pd.DataFrame()
    z_cell_quality = pd.concat([frame for frame in z_quality_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in z_quality_frames) else pd.DataFrame()
    quality_summary = _quality_summary_frame(sector_quality, z_cell_quality)
    gap_bucket_summary = _gap_bucket_frame(sector_quality, z_cell_quality)
    top_discrepancies = _top_discrepancy_contributors_frame(sector_quality, z_cell_quality)
    mapping_risk_warning = bool(any((not flow_run.result.mapping_coverage.empty) and (flow_run.result.mapping_coverage["mapped_targets"].astype(int) > 1).any() for flow_run in (domestic, imports, total)))
    reconciliation = build_flow_reconciliation(
        domestic=domestic,
        imports=imports,
        total=total,
        rel_threshold=rel_threshold,
        min_abs_meur=min_abs_meur,
        eurostat_labels=eurostat_labels,
    )
    classification = _flow_classification_frame(
        domestic=domestic.result.aggregate_totals[domestic.result.aggregate_totals["flag"]],
        imports=imports.result.aggregate_totals[imports.result.aggregate_totals["flag"]],
        total=total.result.aggregate_totals[total.result.aggregate_totals["flag"]],
        kind="aggregate",
    )
    # The classification helper above only sees aggregate rows; append z-cell rows via a second call.
    z_classification = _flow_classification_frame(
        domestic=domestic.result.z_cell_mismatches[domestic.result.z_cell_mismatches["flag"]],
        imports=imports.result.z_cell_mismatches[imports.result.z_cell_mismatches["flag"]],
        total=total.result.z_cell_mismatches[total.result.z_cell_mismatches["flag"]],
        kind="z_cell",
    )
    classification = pd.concat([classification, z_classification], ignore_index=True) if not z_classification.empty else classification

    # Coverage tables used for the quality workbook and summary.
    sector_quality = sector_quality.sort_values(["flow", "block", "abs_gap"], ascending=[True, True, False], kind="stable").reset_index(drop=True)
    z_cell_quality = z_cell_quality.sort_values(["flow", "abs_gap"], ascending=[True, False], kind="stable").reset_index(drop=True)

    paths = {
        "audit_flow_reconciliation.csv": output_dir / "audit_flow_reconciliation.csv",
        "audit_flow_reconciliation.md": output_dir / "audit_flow_reconciliation.md",
        "audit_consolidated_summary.md": output_dir / "audit_consolidated_summary.md",
        "audit_quality_summary.csv": output_dir / "audit_quality_summary.csv",
        "audit_quality_summary.md": output_dir / "audit_quality_summary.md",
        "consolidated_aggregate_flags.csv": output_dir / "consolidated_aggregate_flags.csv",
        "sector_quality_by_flow.csv": output_dir / "sector_quality_by_flow.csv",
        "z_cell_quality_by_flow.csv": output_dir / "z_cell_quality_by_flow.csv",
        "gap_bucket_summary.csv": output_dir / "gap_bucket_summary.csv",
        "top_discrepancy_contributors.csv": output_dir / "top_discrepancy_contributors.csv",
        "consolidated_z_cell_flags.csv": output_dir / "consolidated_z_cell_flags.csv",
        "consolidated_sector_flow_classification.csv": output_dir / "consolidated_sector_flow_classification.csv",
        "consolidated_reconciliation_flags.csv": output_dir / "consolidated_reconciliation_flags.csv",
        "paired_iot_exio_vs_eurostat.xlsx": output_dir / f"paired_iot_exio_vs_eurostat_{domestic.result.metadata.get('country', '')}_{domestic.result.metadata.get('year', 0)}.xlsx",
    }

    reconciliation.to_csv(paths["audit_flow_reconciliation.csv"], index=False)
    quality_summary.to_csv(paths["audit_quality_summary.csv"], index=False)
    sector_quality.to_csv(paths["sector_quality_by_flow.csv"], index=False)
    z_cell_quality.to_csv(paths["z_cell_quality_by_flow.csv"], index=False)
    gap_bucket_summary.to_csv(paths["gap_bucket_summary.csv"], index=False)
    top_discrepancies.to_csv(paths["top_discrepancy_contributors.csv"], index=False)
    Path(paths["audit_flow_reconciliation.md"]).write_text(_reconciliation_markdown(reconciliation, str(domestic.result.metadata.get("country", "")), int(domestic.result.metadata.get("year", 0))), encoding="utf-8")
    Path(paths["audit_consolidated_summary.md"]).write_text(
        _build_consolidated_summary_md(
            country=str(domestic.result.metadata.get("country", "")),
            year=int(domestic.result.metadata.get("year", 0)),
            domestic=domestic,
            imports=imports,
            total=total,
            quality_summary=quality_summary,
            bucket_summary=gap_bucket_summary,
            reconciliation=reconciliation,
            aggregate_flags=aggregate_flags,
            z_flags=z_flags,
            classification=classification,
            mapping_risk_warning=mapping_risk_warning,
        ),
        encoding="utf-8",
    )
    Path(paths["audit_quality_summary.md"]).write_text(_format_quality_summary_md(quality_summary, gap_bucket_summary), encoding="utf-8")
    aggregate_flags.to_csv(paths["consolidated_aggregate_flags.csv"], index=False)
    z_flags.to_csv(paths["consolidated_z_cell_flags.csv"], index=False)
    classification.to_csv(paths["consolidated_sector_flow_classification.csv"], index=False)
    reconciliation[reconciliation["flag"]].to_csv(paths["consolidated_reconciliation_flags.csv"], index=False)

    _write_paired_iot_workbook(
        workbook_path=paths["paired_iot_exio_vs_eurostat.xlsx"],
        country=str(domestic.result.metadata.get("country", "")),
        year=int(domestic.result.metadata.get("year", 0)),
        quality_summary=quality_summary,
        sector_quality=sector_quality,
        top_discrepancies=top_discrepancies,
        flows={"domestic": domestic, "imports": imports, "total": total},
        eurostat_labels=eurostat_labels,
    )
    return paths

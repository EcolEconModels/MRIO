from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import download_eurostat_dataset, parse_eurostat_iot, read_eurostat_raw
from ecolecon_mrio.exio_eurostat_concordance_audit import (
    ConcordanceAuditResult,
    DEFAULT_MIN_ABS_MEUR,
    DEFAULT_REL_THRESHOLD,
    EPS,
    METRIC_HUMAN_LABELS,
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
    fmt_meur,
    fmt_pct,
    fmt_ratio,
    build_audit_metadata,
    build_concordance_audit,
    human_metric_label,
    load_default_label_maps,
    write_concordance_audit_outputs,
)
from ecolecon_mrio.exiobase3 import load_exiobase3_region_slice


REPO_ROOT = Path(__file__).resolve().parents[2]

FLOW_MODE_MAP = {
    "domestic": {"exio_flow": "domestic", "eurostat_flow": "DOM"},
    "imports": {"exio_flow": "imports", "eurostat_flow": "IMP"},
    "total": {"exio_flow": "total", "eurostat_flow": "TOTAL"},
}


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
    force_rebuild_slice: bool = False,
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
    eurostat = parse_eurostat_iot(read_eurostat_raw(euro_dataset_obj.raw_path), row_dim="prd_ava", col_dim="prd_use")

    exio = load_exiobase3_region_slice(
        year=resolved_exio_year,
        region=country,
        system=exio_system,
        flow=exio_flow,
        cache_root=data_root,
        force_download=force_download,
        force_rebuild_slice=force_rebuild_slice,
    )
    exio_Z_flow = exio.Z_domestic
    exio_Y_flow = exio.Y_region
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
                    "EXIOBASE total means all origins used by the selected country; Eurostat TOTAL is the total-flow table. Total is mainly a reconciliation view and should not replace separate domestic/import diagnostics."
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
        eurostat_Z=eurostat.Z,
        eurostat_Y=eurostat.Y,
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
    metrics = [METRIC_Z_CELL, METRIC_Z_ROW_INTERMEDIATE_SALES, METRIC_Z_COL_INTERMEDIATE_PURCHASES, METRIC_Y_FINAL_DEMAND_TOTAL, METRIC_X_OUTPUT_PROXY_TOTAL_USE]

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
                        domestic_value=float(dom_z.reindex(index=z_codes, columns=z_codes).fillna(0.0).loc[supplier_code, buyer_code]) if not dom_z.empty else 0.0,
                        imports_value=float(imp_z.reindex(index=z_codes, columns=z_codes).fillna(0.0).loc[supplier_code, buyer_code]) if not imp_z.empty else 0.0,
                        total_value=float(tot_z.reindex(index=z_codes, columns=z_codes).fillna(0.0).loc[supplier_code, buyer_code]) if not tot_z.empty else 0.0,
                    )
                )

        row_codes = sorted(set(dom_z.index.astype(str)).union(set(imp_z.index.astype(str)), set(tot_z.index.astype(str)), set(dom_y.index.astype(str)), set(imp_y.index.astype(str)), set(tot_y.index.astype(str)), set(dom_x.index.astype(str)), set(imp_x.index.astype(str)), set(tot_x.index.astype(str))))
        for code in row_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_Z_ROW_INTERMEDIATE_SALES,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_z.sum(axis=1).reindex(row_codes, fill_value=0.0).get(code, 0.0)) if not dom_z.empty else 0.0,
                    imports_value=float(imp_z.sum(axis=1).reindex(row_codes, fill_value=0.0).get(code, 0.0)) if not imp_z.empty else 0.0,
                    total_value=float(tot_z.sum(axis=1).reindex(row_codes, fill_value=0.0).get(code, 0.0)) if not tot_z.empty else 0.0,
                )
            )

        col_codes = sorted(set(dom_z.columns.astype(str)).union(set(imp_z.columns.astype(str)), set(tot_z.columns.astype(str))))
        for code in col_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_Z_COL_INTERMEDIATE_PURCHASES,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_z.sum(axis=0).get(code, 0.0)) if not dom_z.empty else 0.0,
                    imports_value=float(imp_z.sum(axis=0).get(code, 0.0)) if not imp_z.empty else 0.0,
                    total_value=float(tot_z.sum(axis=0).get(code, 0.0)) if not tot_z.empty else 0.0,
                )
            )

        y_codes = sorted(set(dom_y.index.astype(str)).union(set(imp_y.index.astype(str)), set(tot_y.index.astype(str))))
        for code in y_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_Y_FINAL_DEMAND_TOTAL,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_y.get(code, 0.0)),
                    imports_value=float(imp_y.get(code, 0.0)),
                    total_value=float(tot_y.get(code, 0.0)),
                )
            )

        x_codes = sorted(set(dom_x.index.astype(str)).union(set(imp_x.index.astype(str)), set(tot_x.index.astype(str))))
        for code in x_codes:
            rows.append(
                _recon_row(
                    database=database,
                    metric=METRIC_X_OUTPUT_PROXY_TOTAL_USE,
                    target_code=code,
                    target_label=eurostat_labels.get(code, code),
                    domestic_value=float(dom_x.get(code, 0.0)),
                    imports_value=float(imp_x.get(code, 0.0)),
                    total_value=float(tot_x.get(code, 0.0)),
                )
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["severity_rank"] = frame["severity"].map(_severity_rank).fillna(-1).astype(int)
        frame = frame.sort_values(["database", "severity_rank", "abs_gap", "rel_gap"], ascending=[True, False, False, False], kind="stable").drop(columns=["severity_rank"]).reset_index(drop=True)
    return frame


def _reconciliation_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Flow reconciliation audit: AT 2020",
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
                sector_display = f"{_display_code_label(row.supplier_code, row.supplier_label)} → {_display_code_label(row.buyer_code, row.buyer_label)}"
            else:
                sector_display = _display_code_label(row.target_code, row.target_label)
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
        "- For imports, EXIOBASE imports are foreign-origin products used by AT, summed across all non-AT origins. Partner-origin detail is not retained in this audit.",
    ])
    return "\n".join(lines).rstrip() + "\n"


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
                    "target_label": domestic.loc[domestic["target_code"].astype(str).eq(target_code), "target_label"].iloc[0] if not domestic.empty and "target_code" in domestic.columns and (domestic["target_code"].astype(str).eq(target_code)).any() else target_code,
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
                    "supplier_label": domestic.loc[(domestic["supplier_code"].astype(str).eq(supplier_code)) & (domestic["buyer_code"].astype(str).eq(buyer_code)), "supplier_label"].iloc[0] if not domestic.empty and "supplier_code" in domestic.columns and "buyer_code" in domestic.columns and ((domestic["supplier_code"].astype(str).eq(supplier_code)) & (domestic["buyer_code"].astype(str).eq(buyer_code))).any() else supplier_code,
                    "buyer_code": buyer_code,
                    "buyer_label": domestic.loc[(domestic["supplier_code"].astype(str).eq(supplier_code)) & (domestic["buyer_code"].astype(str).eq(buyer_code)), "buyer_label"].iloc[0] if not domestic.empty and "supplier_code" in domestic.columns and "buyer_code" in domestic.columns and ((domestic["supplier_code"].astype(str).eq(supplier_code)) & (domestic["buyer_code"].astype(str).eq(buyer_code))).any() else buyer_code,
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


def _build_consolidated_summary_md(
    *,
    country: str,
    year: int,
    domestic: FlowAuditRun,
    imports: FlowAuditRun,
    total: FlowAuditRun,
    reconciliation: pd.DataFrame,
    aggregate_flags: pd.DataFrame,
    z_flags: pd.DataFrame,
    classification: pd.DataFrame,
) -> str:
    def _top(frame: pd.DataFrame, *, flow: str, kind: str, n: int = 10) -> pd.DataFrame:
        if frame.empty:
            return frame
        out = frame[frame["flow"].astype(str).eq(flow)].copy()
        if out.empty:
            return out
        out["severity_rank"] = out["severity"].map(_severity_rank).fillna(-1).astype(int)
        out = out.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        return out.head(n)

    lines = [
        f"# Consolidated EXIOBASE–Eurostat Concordance Audit: {country} {year}",
        "",
        "## What was compared",
        "| Flow | EXIOBASE slice | Eurostat table | Output folder |",
        "| --- | --- | --- | --- |",
        "| Domestic | AT-origin products used by AT | stk_flow=DOM | domestic |",
        "| Imports | Non-AT-origin products used by AT, foreign origins summed | stk_flow=IMP | imports |",
        "| Total | All-origin products used by AT | stk_flow=TOTAL | total |",
        "",
        "## Important caveat on x-like rows",
        "All x-like rows in these audits are flow-specific total-use proxy comparisons: Z row total + Y total.",
        "They are not independently observed gross-output comparisons.",
        "",
        "## Executive summary",
        f"- Number of aggregate flags in domestic audit: {int(domestic.result.aggregate_totals['flag'].sum())}",
        f"- Number of aggregate flags in imports audit: {int(imports.result.aggregate_totals['flag'].sum())}",
        f"- Number of aggregate flags in total audit: {int(total.result.aggregate_totals['flag'].sum())}",
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
        return _markdown_table(frame.rename(columns={"human_metric": "Readable metric", "target_display": "Target sector", "exio_mapped_value": "EXIOBASE mapped value, MEUR", "eurostat_value": "Eurostat value, MEUR", "signed_gap": "Difference, MEUR", "rel_gap": "Relative gap", "ratio_exio_to_eurostat": "Ratio", "severity": "Severity", "top_exio_contributors": "Largest EXIOBASE contributors"}), columns=["Readable metric", "Target sector", "EXIOBASE mapped value, MEUR", "Eurostat value, MEUR", "Difference, MEUR", "Relative gap", "Ratio", "Severity", "Largest EXIOBASE contributors"], max_rows=10)

    def _cell_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No flagged Z-cell discrepancies."
        return _markdown_table(frame.rename(columns={"comparison_display": "Supplier → buyer", "exio_mapped_z": "EXIOBASE mapped flow, MEUR", "eurostat_z": "Eurostat flow, MEUR", "signed_gap": "Difference, MEUR", "rel_gap": "Relative gap", "ratio_exio_to_eurostat": "Ratio", "severity": "Severity", "top_source_subsectors": "Top source subsectors", "top_buyer_subsectors": "Top buyer subsectors"}), columns=["Supplier → buyer", "EXIOBASE mapped flow, MEUR", "Eurostat flow, MEUR", "Difference, MEUR", "Relative gap", "Ratio", "Severity", "Top source subsectors", "Top buyer subsectors"], max_rows=10)

    lines.append(_agg_table(dom_agg))
    lines.extend(["", _cell_table(dom_cell), "", "## Biggest import discrepancies"])
    lines.append(_agg_table(imp_agg))
    lines.extend(["", _cell_table(imp_cell), "", "## Biggest total discrepancies"])
    lines.append(_agg_table(tot_agg))
    lines.extend(["", _cell_table(tot_cell), "", "## Discrepancies flagged in both domestic and imports"])
    both = classification[classification["flow_classification"].astype(str).eq("BOTH_DOMESTIC_AND_IMPORT")].copy()
    if not both.empty:
        lines.append(_markdown_table(both, columns=["kind", "metric", "target_code", "supplier_code", "buyer_code", "flow_classification"], max_rows=20))
    else:
        lines.append("No rows were flagged in both domestic and imports.")
    lines.extend(["", "## Discrepancies mostly driven by domestic flows"])
    dom_driven = classification[classification["flow_classification"].astype(str).eq("DOMESTIC_DRIVEN")].copy()
    if not dom_driven.empty:
        lines.append(_markdown_table(dom_driven, columns=["kind", "metric", "target_code", "supplier_code", "buyer_code", "flow_classification"], max_rows=20))
    else:
        lines.append("No domestic-driven rows were flagged.")
    lines.extend(["", "## Discrepancies mostly driven by imports"])
    imp_driven = classification[classification["flow_classification"].astype(str).eq("IMPORT_DRIVEN")].copy()
    if not imp_driven.empty:
        lines.append(_markdown_table(imp_driven, columns=["kind", "metric", "target_code", "supplier_code", "buyer_code", "flow_classification"], max_rows=20))
    else:
        lines.append("No import-driven rows were flagged.")
    lines.extend(["", "## Total-only discrepancies"])
    total_only = classification[classification["flow_classification"].astype(str).eq("TOTAL_ONLY")].copy()
    if not total_only.empty:
        lines.append(_markdown_table(total_only, columns=["kind", "metric", "target_code", "supplier_code", "buyer_code", "flow_classification"], max_rows=20))
    else:
        lines.append("No total-only rows were flagged.")
    lines.extend(["", "## Flow reconciliation problems"])
    recon_flagged = reconciliation[reconciliation["flag"]].copy()
    if not recon_flagged.empty:
        recon_flagged["severity_rank"] = recon_flagged["severity"].map(_severity_rank).fillna(-1).astype(int)
        recon_flagged = recon_flagged.sort_values(["severity_rank", "abs_gap", "rel_gap"], ascending=[False, False, False], kind="stable").drop(columns=["severity_rank"])
        lines.append(_markdown_table(recon_flagged.head(20), columns=["database", "metric", "target_code", "supplier_code", "buyer_code", "domestic_value", "imports_value", "domestic_plus_imports", "total_value", "signed_gap", "rel_gap", "interpretation"], max_rows=20))
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
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_frames: list[pd.DataFrame] = []
    z_frames: list[pd.DataFrame] = []
    for flow_name, flow_run in (("domestic", domestic), ("imports", imports), ("total", total)):
        agg_frame, cell_frame = _flow_flags_table(flow_run.result, flow=flow_name, eurostat_labels=eurostat_labels)
        aggregate_frames.append(agg_frame)
        z_frames.append(cell_frame)
    aggregate_flags = pd.concat(aggregate_frames, ignore_index=True) if aggregate_frames else pd.DataFrame()
    z_flags = pd.concat(z_frames, ignore_index=True) if z_frames else pd.DataFrame()
    reconciliation = build_flow_reconciliation(
        domestic=domestic,
        imports=imports,
        total=total,
        rel_threshold=DEFAULT_REL_THRESHOLD,
        min_abs_meur=DEFAULT_MIN_ABS_MEUR,
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

    paths = {
        "audit_flow_reconciliation.csv": output_dir / "audit_flow_reconciliation.csv",
        "audit_flow_reconciliation.md": output_dir / "audit_flow_reconciliation.md",
        "audit_consolidated_summary.md": output_dir / "audit_consolidated_summary.md",
        "consolidated_aggregate_flags.csv": output_dir / "consolidated_aggregate_flags.csv",
        "consolidated_z_cell_flags.csv": output_dir / "consolidated_z_cell_flags.csv",
        "consolidated_sector_flow_classification.csv": output_dir / "consolidated_sector_flow_classification.csv",
        "consolidated_reconciliation_flags.csv": output_dir / "consolidated_reconciliation_flags.csv",
    }

    reconciliation.to_csv(paths["audit_flow_reconciliation.csv"], index=False)
    Path(paths["audit_flow_reconciliation.md"]).write_text(_reconciliation_markdown(reconciliation), encoding="utf-8")
    Path(paths["audit_consolidated_summary.md"]).write_text(
        _build_consolidated_summary_md(
            country=str(domestic.result.metadata.get("country", "")),
            year=int(domestic.result.metadata.get("year", 0)),
            domestic=domestic,
            imports=imports,
            total=total,
            reconciliation=reconciliation,
            aggregate_flags=aggregate_flags,
            z_flags=z_flags,
            classification=classification,
        ),
        encoding="utf-8",
    )
    aggregate_flags.to_csv(paths["consolidated_aggregate_flags.csv"], index=False)
    z_flags.to_csv(paths["consolidated_z_cell_flags.csv"], index=False)
    classification.to_csv(paths["consolidated_sector_flow_classification.csv"], index=False)
    reconciliation[reconciliation["flag"]].to_csv(paths["consolidated_reconciliation_flags.csv"], index=False)
    return paths

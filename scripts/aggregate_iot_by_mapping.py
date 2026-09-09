#!/usr/bin/env python3
# pyright: reportReturnType=false, reportArgumentType=false, reportAbstractUsage=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportPossiblyUnboundVariable=false, reportOptionalMemberAccess=false

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_aggregation import (
    aggregate_environmental_extensions,
    aggregate_iot,
)
from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT
from ecolecon_mrio.eurostat_iot_source_policy import NAIO_10_CP1700_ACCOUNTING_POLICY
from ecolecon_mrio.iot_flow_semantics import (
    _normalize_nonnegative_accounting_control,
    atomic_final_demand,
    flow_use_total,
)
from ecolecon_mrio.mapping_csv import read_aggregation_mapping


def _vprint(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


AGGREGATION_MANAGED_OUTPUT_NAMES: tuple[str, ...] = (
    "Z_agg.csv",
    "Y_agg.csv",
    "x_agg.csv",
    "A_agg.csv",
    "V_agg.csv",
    "M_agg.csv",
    "use_total_agg.csv",
    "aggregation_info.json",
    "sector_contract.csv",
    "source_sector_contract.csv",
    "aggregation_edge_mapping_audit.csv",
    "aggregated_sector_value_per_physical_input.csv",
    "environmental_extensions.csv",
    "exio_environmental_extensions.csv",
    # Removed CO2-only sidecars remain managed only so reruns clear stale files.
    "aggregated_sector_co2_intensity.csv",
    "exio_final_demand_co2_intensity.csv",
    "aggregation_report.xlsx",
    # Removed reports retained here only so reruns clear stale artifacts.
    "aggregation_block_conservation.csv",
    "supplementary_row_contract.csv",
    "template_parent_total_audit.csv",
)


def _clear_managed_aggregation_outputs(outdir: Path) -> None:
    for name in AGGREGATION_MANAGED_OUTPUT_NAMES:
        (outdir / name).unlink(missing_ok=True)


class _Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, index_col=0)
    raise ValueError(f"Unsupported table format: {path}")


def _write_df_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_rows_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    """Write a row-oriented table without serializing a synthetic RangeIndex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_series_csv(series: pd.Series, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, header=True)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_text(path: Path, content: str, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _flatten_selected_sector_values(value: object) -> set[str]:
    children: set[str] = set()
    if value is None:
        return children
    if isinstance(value, dict):
        for nested in value.values():
            children.update(_flatten_selected_sector_values(nested))
        return children
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            children.update(_flatten_selected_sector_values(nested))
        return children
    text = str(value).strip()
    if text:
        children.add(text)
    return children


def _load_split_selected_children(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise SystemExit(f"Split selected sectors JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _flatten_selected_sector_values(data)


def _augment_mapping_from_split_child_edges(
    *,
    mapping: dict[str, str],
    mapping_df: pd.DataFrame,
    edge_metadata: pd.DataFrame | None,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Add materialized edge children to an aggregation mapping.

    Aggregation CSVs are often written against Eurostat target sectors or older
    raw child labels. Phase-4 split outputs can contain emitted edge labels such
    as ``p24.d@CPA_C20``. Those labels must map through their target side. This
    helper uses ``split_child_edges.csv`` metadata to map each current edge child
    to the mapped parent/concordance target when available, otherwise to the
    concordance target itself.
    """
    if edge_metadata is None or edge_metadata.empty:
        return dict(mapping), mapping_df.copy(), pd.DataFrame()

    out_mapping = {str(k): str(v) for k, v in mapping.items()}
    out_df = mapping_df.copy()
    audit_rows: list[dict[str, object]] = []
    synthetic_rows: list[dict[str, object]] = []

    def _cell(row: pd.Series, name: str) -> str:
        if name not in row.index:
            return ""
        value = row.get(name)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip()

    for _, row in edge_metadata.iterrows():
        child = _cell(row, "current_child_code")
        if not child or "@" not in child:
            continue
        source = _cell(row, "source_code")
        parent = _cell(row, "parent_code")
        concordance_target = _cell(row, "concordance_target_code") or parent

        if child in out_mapping:
            audit_rows.append(
                {
                    "current_child_code": child,
                    "source_code": source,
                    "parent_code": parent,
                    "concordance_target_code": concordance_target,
                    "mapped_to_sector": out_mapping[child],
                    "mapping_source": "existing_mapping",
                }
            )
            continue

        target_hint = concordance_target or parent
        mapped_to = ""
        mapping_source = ""

        # Prefer a legacy raw-source mapping only when its target is compatible
        # with this materialized edge target.  This recovers old rows such as
        # p24.d -> CPA_C20_others for p24.d@CPA_C20, without silently using that
        # row for an incompatible p24.d@CPA_C21 edge.
        if source and source in out_mapping:
            raw_mapped_to = out_mapping[source]
            raw_target = str(raw_mapped_to).strip()
            hint = str(target_hint).strip()
            raw_upper = raw_target.upper()
            hint_upper = hint.upper()
            if not hint or raw_upper == hint_upper or raw_upper.startswith(f"{hint_upper}_"):
                mapped_to = raw_mapped_to
                mapping_source = f"existing_mapping_for_source_code_{source}"

        candidate_keys = [k for k in (parent, concordance_target) if k]
        if not mapped_to:
            for candidate in candidate_keys:
                if candidate in out_mapping:
                    mapped_to = out_mapping[candidate]
                    mapping_source = f"existing_mapping_for_{candidate}"
                    break
        if not mapped_to:
            mapped_to = concordance_target or parent
            mapping_source = "identity_to_concordance_target"
        if not mapped_to:
            continue

        out_mapping[child] = mapped_to
        audit_rows.append(
            {
                "current_child_code": child,
                "source_code": source,
                "parent_code": parent,
                "concordance_target_code": concordance_target,
                "mapped_to_sector": mapped_to,
                "mapping_source": mapping_source,
            }
        )

        new_row = {column: "" for column in out_df.columns}
        for source_col in ("source_code", "from_sector"):
            if source_col in new_row:
                new_row[source_col] = child
        for target_col in ("target_code", "to_sector"):
            if target_col in new_row:
                new_row[target_col] = mapped_to
        if "kind" in new_row:
            new_row["kind"] = "aggregate"
        if "enabled" in new_row:
            new_row["enabled"] = "1"
        if "notes" in new_row:
            new_row["notes"] = "auto-added from split_child_edges.csv for Phase-4 edge child reaggregation"
        synthetic_rows.append(new_row)

    if synthetic_rows:
        out_df = pd.concat([out_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
        if "from_sector" in out_df.columns:
            out_df["from_sector"] = out_df["from_sector"].astype(str)
        if "to_sector" in out_df.columns:
            out_df["to_sector"] = out_df["to_sector"].astype(str)

    audit = pd.DataFrame(audit_rows)
    return out_mapping, out_df, audit


def _infer_variant_value(
    *,
    out_base: Path,
    source_dir: Path,
    variant_dim: str,
    eurostat_filters: dict[str, object] | None,
) -> str | None:
    prefix = f"{variant_dim}="
    for candidate in (out_base.name, source_dir.name):
        if candidate.startswith(prefix):
            return candidate.split("=", 1)[1] or None

    if eurostat_filters is None:
        return None
    value = eurostat_filters.get(variant_dim)
    if value is None:
        return None
    value_text = str(value).strip()
    return value_text or None


def write_aggregation_report_xlsx(
    out_path: str | Path,
    *,
    Z_before: pd.DataFrame,
    Y_before: pd.DataFrame,
    result,
    mapping_df: pd.DataFrame,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    before_sectors = pd.DataFrame({"sector": list(map(str, Z_before.index))})
    after_sectors = pd.DataFrame({"sector": list(map(str, result.Z.index))})

    counts = mapping_df.copy()
    if "enabled" not in counts.columns:
        counts["enabled"] = "1"

    with pd.ExcelWriter(out_path) as xw:
        mapping_df.to_excel(xw, sheet_name="mapping", index=False)
        before_sectors.to_excel(xw, sheet_name="sectors_before", index=False)
        after_sectors.to_excel(xw, sheet_name="sectors_after", index=False)

        summary = pd.DataFrame(
            {
                "n_sectors_before": [len(Z_before.index)],
                "n_sectors_after": [len(result.Z.index)],
                "Z_total_before": [float(Z_before.to_numpy().sum())],
                "Z_total_after": [float(result.Z.to_numpy().sum())],
                "atomic_final_demand_before": [float(atomic_final_demand(Y_before).to_numpy().sum())],
                "atomic_final_demand_after": [float(atomic_final_demand(result.Y).to_numpy().sum())],
                "total_use_before": [float(flow_use_total(Z_before, Y_before).sum())],
                "total_use_after": [float(flow_use_total(result.Z, result.Y).sum())],
            }
        )
        summary.to_excel(xw, sheet_name="summary", index=False)

    return out_path


def _require_columns(df: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)!r}")


def _compute_production_based_intensity(
    disaggregated_sector_emission_intensity: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, str]:
    _require_columns(
        disaggregated_sector_emission_intensity,
        {"disaggregated_sector", "emission", "monetary_output", "unit"},
        label="disaggregated_sector_emission_intensity",
    )
    emis_df = disaggregated_sector_emission_intensity.copy()
    sums = emis_df.groupby("disaggregated_sector")[["emission", "monetary_output"]].sum(
        min_count=1
    )
    out_by_sector = sums["monetary_output"]
    intensity_by_sector = sums["emission"] / sums["monetary_output"]
    unit_vals = emis_df["unit"].dropna().unique().tolist()
    unit_label = str(unit_vals[0]).strip() if len(unit_vals) == 1 else "mixed units"
    return out_by_sector, intensity_by_sector, unit_label


def _compute_use_phase_intensity_scalar(
    exio_final_demand_emission_intensity: pd.DataFrame,
    Y: pd.DataFrame,
) -> tuple[float, str]:
    _require_columns(
        exio_final_demand_emission_intensity,
        {"category", "emission", "monetary_final_demand", "unit"},
        label="exio_final_demand_emission_intensity",
    )
    if Y is None or Y.empty:
        raise ValueError("disaggregated_Y is required to compute use-phase intensity.")

    fy = exio_final_demand_emission_intensity.copy()
    fy["emission"] = pd.to_numeric(fy["emission"], errors="coerce")
    fy["monetary_final_demand"] = pd.to_numeric(fy["monetary_final_demand"], errors="coerce")
    grouped = fy.groupby("category")[["emission", "monetary_final_demand"]].sum()

    def _ratio_by_keys(keys: list[str]) -> float | None:
        sel = [c for c in grouped.index if any(k in str(c).lower() for k in keys)]
        if not sel:
            return None
        num = grouped.loc[sel, "emission"].sum()
        den = grouped.loc[sel, "monetary_final_demand"].sum()
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return None
        return float(num) / float(den)

    p3_ratio = _ratio_by_keys([
        "households",
        "non-profit organisations serving households",
        "npish",
        "government",
    ])
    p5_ratio = _ratio_by_keys(["gross fixed capital formation"])
    fy_intensity = {
        "P3": p3_ratio,
        "P3_S14": _ratio_by_keys(["households"]),
        "P3_S15": _ratio_by_keys(["non-profit organisations serving households", "npish"]),
        "P3_S13": _ratio_by_keys(["government"]),
        "P5": p5_ratio,
        "P51G": p5_ratio,
        "P52": _ratio_by_keys(["inventories"]),
        "P53": _ratio_by_keys(["valuables"]),
        "P6": _ratio_by_keys(["exports"]),
        "P6_B0": _ratio_by_keys(["exports"]),
        "P6_D0": _ratio_by_keys(["exports"]),
        "P6_U2": _ratio_by_keys(["exports"]),
        "P6_U3": _ratio_by_keys(["exports"]),
    }

    y_cols = list(map(str, Y.columns))
    dom_cols: list[str] = []
    if "P3" in y_cols:
        dom_cols.append("P3")
    else:
        dom_cols += [c for c in ["P3_S13", "P3_S14", "P3_S15"] if c in y_cols]
    if "P5" in y_cols:
        dom_cols.append("P5")
    elif "P51G" in y_cols:
        dom_cols.append("P51G")

    intensity_by_col = {c: fy_intensity.get(c) for c in dom_cols if fy_intensity.get(c) is not None}
    if not intensity_by_col:
        raise ValueError("Could not derive use-phase intensities for domestic final-demand columns.")

    intensities = pd.Series(intensity_by_col).dropna()
    y_dom = Y[intensities.index].astype("float64")
    total_emis = float((y_dom.mul(intensities, axis=1)).sum().sum())
    total_fd = float(y_dom.sum().sum())
    if not np.isfinite(total_emis) or not np.isfinite(total_fd) or total_fd == 0:
        raise ValueError("Invalid totals for use-phase intensity (emissions or final demand).")

    unit_vals = fy["unit"].dropna().unique().tolist()
    unit_label = str(unit_vals[0]).strip() if len(unit_vals) == 1 else "mixed units"
    return total_emis / total_fd, unit_label


def _parse_json_arg(s: str) -> dict:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON: {e}: {s!r}") from e
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected JSON object (dict) but got {type(obj).__name__}: {s!r}")
    return obj


def _pretty_sector_label(code_or_label: str) -> str:
    s = str(code_or_label)
    if s.startswith("ENERGY_"):
        return s.replace("_", " ").title()
    if "::" in s:
        left, right = s.split("::", 1)
        return f"{left}: {right}"
    return s


def _base_sector_from_code(code: str, base_product_codes: Sequence[str]) -> str | None:
    code_s = str(code)
    base_codes = [str(base) for base in base_product_codes]
    if code_s in base_codes:
        return code_s
    for base in base_codes:
        if code_s.startswith(f"{base}::") or code_s.startswith(f"{base}_"):
            return base
    return None


def _build_source_parent_by_input(
    mapping_df: pd.DataFrame,
    *,
    mapping_from_col: str,
    mapping_to_col: str,
    base_product_codes: Sequence[str],
) -> dict[str, str]:
    base_codes = [str(code) for code in base_product_codes]
    source_parent_by_input: dict[str, str] = {}

    if mapping_from_col not in mapping_df.columns or mapping_to_col not in mapping_df.columns:
        return source_parent_by_input

    for _, row in mapping_df.iterrows():
        if "enabled" in mapping_df.columns:
            enabled_value = row.get("enabled", True)
            if isinstance(enabled_value, str):
                enabled_text = enabled_value.strip().lower()
                if enabled_text in {"0", "false", "f", "no", "n", "off"}:
                    continue
            elif not bool(enabled_value):
                continue

        source_code = str(row[mapping_from_col]).strip()
        target_code = str(row[mapping_to_col]).strip()
        if not source_code or not target_code:
            continue

        explicit_value = row.get("source_parent_code")
        explicit_parent = (
            ""
            if explicit_value is None or pd.isna(explicit_value)
            else str(explicit_value).strip()
        )
        if explicit_parent and explicit_parent not in base_codes:
            raise ValueError(
                "Unknown explicit source parent in aggregation mapping: "
                f"source_code={source_code!r}, source_parent_code={explicit_parent!r}."
            )
        target_parent = _base_sector_from_code(target_code, base_codes)
        chosen_parent = explicit_parent or target_parent or _base_sector_from_code(
            source_code, base_codes
        )
        if chosen_parent is None:
            continue

        existing_parent = source_parent_by_input.get(source_code)
        if existing_parent is None:
            source_parent_by_input[source_code] = chosen_parent
            continue
        if existing_parent != chosen_parent:
            raise ValueError(
                "Conflicting parent lineage for source sector "
                f"{source_code!r}: {existing_parent!r} vs {chosen_parent!r}"
            )

    return source_parent_by_input


def _missing_sfc_co2_reason(variant_value: str | None, *, use_phase: bool) -> str:
    role = str(variant_value or "").strip().upper()
    if role == "IMP":
        if use_phase:
            return (
                "for IMP flow because import-origin weighted final-demand CO2 intensities "
                "are not implemented"
            )
        return (
            "for IMP flow because origin-weighted import production CO2 intensities are "
            "not implemented; using Austrian destination-region EXIO extension proxies "
            "would be misleading"
        )
    if role == "TOTAL":
        if use_phase:
            return (
                "for TOTAL flow because a combined domestic and import-origin final-demand "
                "CO2 intensity is not implemented"
            )
        return (
            "for TOTAL flow because a combined domestic-production and origin-weighted "
            "import CO2 intensity is not implemented; using domestic-only intensities "
            "would mislabel total supply"
        )
    return (
        f"for {role or 'this'} flow because the required CO2 intensity inputs are not "
        "available"
    )


def _write_excel_mapping_sheet(
    wb,
    *,
    sheet_name: str,
    mapping_df: pd.DataFrame,
    disaggregated_Y: pd.DataFrame | None = None,
    aggregated_Y: pd.DataFrame | None = None,
    aggregated_x: pd.Series | None = None,
    aggregated_sector_order: list[str] | None = None,
    aggregated_sector_labels: dict[str, str] | None = None,
    eurostat_label_map: dict[str, str] | None = None,
    eurostat_sector_co2_mapping: pd.DataFrame | None = None,
    disaggregated_sector_value_per_physical_input: pd.DataFrame | None = None,
    aggregated_sector_value_per_physical_input: pd.DataFrame | None = None,
    disaggregated_sector_emission_intensity: pd.DataFrame | None = None,
    aggregated_sector_emission_intensity: pd.DataFrame | None = None,
    exio_final_demand_emission_intensity: pd.DataFrame | None = None,
    mapping_from_col: str = "from_sector",
    mapping_to_col: str = "to_sector",
    variant_value: str | None = None,
    allow_missing_sfc_co2: bool = False,
    verbose: bool = False,
) -> None:
    import openpyxl
    from openpyxl.utils import get_column_letter

    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        wb.remove(ws)
    ws = wb.create_sheet(sheet_name)

    def _auto_width(ws) -> None:
        for col_cells in ws.columns:
            values = [cell.value for cell in col_cells if cell.value is not None]
            if not values:
                continue
            max_len = max(len(str(v)) for v in values)
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(max_len + 2, 60)

    def _split_final_demand(Y: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        cols = list(map(str, Y.columns))

        # Consumption
        if "P3" in cols:
            cons_cols = ["P3"]
        else:
            cons_cols = [c for c in cols if c in {"P3_S13", "P3_S14", "P3_S15"}]

        # Capital formation (avoid overlap with stocks)
        if "P5" in cols:
            cap_cols = ["P5"]
        else:
            cap_cols = [c for c in cols if c == "P51G"]

        # Exports
        if "P6" in cols:
            exp_cols = ["P6"]
        else:
            exp_cols = [c for c in cols if c in {"P6_B0", "P6_D0", "P6_U2", "P6_U3"}]

        # Stocks & valuables
        stock_cols = [c for c in cols if c in {"P52", "P53"}]

        Y_use = Y.copy()
        for c in Y_use.columns:
            Y_use[c] = pd.to_numeric(Y_use[c], errors="coerce")

        domestic = Y_use[cons_cols + cap_cols].sum(axis=1) if (cons_cols or cap_cols) else Y_use.sum(axis=1) * 0
        exports = Y_use[exp_cols].sum(axis=1) if exp_cols else Y_use.sum(axis=1) * 0
        stocks = Y_use[stock_cols].sum(axis=1) if stock_cols else Y_use.sum(axis=1) * 0
        return domestic.astype("float64"), exports.astype("float64"), stocks.astype("float64")

    # Prepare data with value-per-physical-input columns
    display_df = mapping_df.copy()

    # Add columns for disaggregated sectors when stressor/unit are unambiguous
    if (
        disaggregated_sector_value_per_physical_input is not None
        and not disaggregated_sector_value_per_physical_input.empty
    ):
        value_df = disaggregated_sector_value_per_physical_input.copy()
        if {"extension", "stressor", "unit"}.issubset(value_df.columns):
            unique_keys = value_df[["extension", "stressor", "unit"]].drop_duplicates()
            if len(unique_keys) == 1:
                sums = value_df.groupby("disaggregated_sector")[["quantity", "monetary_output"]].sum()
                unit_val = str(unique_keys["unit"].iloc[0]).strip()
                unit_label = unit_val if unit_val and unit_val.lower() != "nan" else "mixed units"
                header_energy = f"{mapping_from_col} Total energy ({unit_label})"
                header_value = f"{mapping_from_col} Total value (EUR)"
                display_df[header_energy] = display_df[mapping_from_col].map(sums["quantity"]).fillna("")
                display_df[header_value] = display_df[mapping_from_col].map(sums["monetary_output"]).fillna("")

    # Add CO2 monetary output + production-based intensity columns for disaggregated sectors.
    if disaggregated_sector_emission_intensity is None or disaggregated_sector_emission_intensity.empty:
        if not allow_missing_sfc_co2:
            raise ValueError("disaggregated_sector_emission_intensity is required for production-based CO2 intensity.")
        _vprint(
            verbose,
            "WARNING: SFC workbook mapping sheet is omitting production-based CO2 intensity columns "
            + _missing_sfc_co2_reason(variant_value, use_phase=False)
            + ".",
        )
    else:
        emis_df = disaggregated_sector_emission_intensity.copy()
        _require_columns(emis_df, {"extension", "stressor", "unit"}, label="disaggregated_sector_emission_intensity")
        unique_keys = emis_df[["extension", "stressor", "unit"]].drop_duplicates()
        if len(unique_keys) != 1:
            raise ValueError(
                "Production-based CO2 intensity is ambiguous: multiple extension/stressor/unit combinations found."
            )
        disagg_out_by_sector, disagg_intensity_by_sector, unit_label = _compute_production_based_intensity(emis_df)
        header_out = f"{mapping_from_col} Monetary output (M€)"
        header_int = f"{mapping_from_col} Mean CO2 intensity ({unit_label}/M€)"
        display_df[header_out] = display_df[mapping_from_col].map(disagg_out_by_sector).fillna("")
        display_df[header_int] = display_df[mapping_from_col].map(disagg_intensity_by_sector).fillna("")

    # Add domestic/exports/stocks final demand + CO2 intensity for domestic demand (per disaggregated sector)
    domestic_fd_by_sector = None
    exports_by_sector = None
    stocks_by_sector = None
    if disaggregated_Y is not None and not disaggregated_Y.empty:
        domestic_fd_by_sector, exports_by_sector, stocks_by_sector = _split_final_demand(disaggregated_Y)
        display_df[f"{mapping_from_col} Domestic final demand (M€)"] = display_df[
            mapping_from_col
        ].map(domestic_fd_by_sector).fillna("")
        display_df[f"{mapping_from_col} Exports (M€)"] = display_df[mapping_from_col].map(
            exports_by_sector
        ).fillna("")
        display_df[f"{mapping_from_col} Stocks & valuables (M€)"] = display_df[
            mapping_from_col
        ].map(stocks_by_sector).fillna("")

    # Add notes on energy-use stressors
    if "stressor" in display_df.columns and "notes" not in display_df.columns:
        stressor_notes = {
            "energy use - emission relevant": (
                "Energy that is combustion-relevant (used for emissions accounting)."
            ),
            "energy use - final": "Final energy delivered to end users.",
            "energy use - gross": "Total primary energy input before transformation losses.",
            "energy use - net": "Energy available after transformation losses.",
        }

        def _note_for_stressor(val: object) -> str:
            s = str(val).strip().lower()
            for key, note in stressor_notes.items():
                if key in s:
                    return note
            return ""

        display_df["notes"] = display_df["stressor"].map(_note_for_stressor)

    # Use-phase (F_Y) domestic-demand CO2 intensity (scalar applied to all sectors)
    fy_dom_intensity_scalar: float | None = None
    fy_unit_label = "mixed units"
    if exio_final_demand_emission_intensity is None or exio_final_demand_emission_intensity.empty:
        if not allow_missing_sfc_co2:
            raise ValueError("exio_final_demand_emission_intensity is required for use-phase CO2 intensity.")
        _vprint(
            verbose,
            "WARNING: SFC workbook mapping sheet is omitting use-phase CO2 intensity columns "
            + _missing_sfc_co2_reason(variant_value, use_phase=True)
            + ".",
        )
    else:
        if disaggregated_Y is None or disaggregated_Y.empty:
            raise ValueError("disaggregated_Y is required for use-phase CO2 intensity.")
        fy_dom_intensity_scalar, fy_unit_label = _compute_use_phase_intensity_scalar(
            exio_final_demand_emission_intensity,
            disaggregated_Y,
        )
        header_fd_int = f"{mapping_from_col} Mean CO2 intensity for domestic demand ({fy_unit_label}/M€)"
        display_df[header_fd_int] = fy_dom_intensity_scalar

    # Write the main mapping table
    headers = list(map(str, display_df.columns))
    ws.append(headers)

    def _cell_value(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, (int, float)):
            return float(v)
        return str(v)

    for row in display_df.itertuples(index=False):
        ws.append([_cell_value(v) for v in row])

    # Apply number formats for known numeric columns in the main mapping table
    header_to_col = {h: i + 1 for i, h in enumerate(headers)}
    for h, col_idx in header_to_col.items():
        if "Monetary output" in h:
            fmt = "0.00"
        elif "Mean CO2 intensity" in h:
            fmt = "0.0000"
        elif "Domestic final demand" in h or "Exports" in h or "Stocks" in h:
            fmt = "0.00"
        elif "Total energy" in h or "Total value" in h:
            fmt = "0.00"
        else:
            continue
        for r in range(2, len(display_df) + 2):
            ws.cell(row=r, column=col_idx).number_format = fmt

    _auto_width(ws)

    # Add aggregated sector totals (energy + value) by stressor in a separate tab
    if (
        aggregated_sector_value_per_physical_input is not None
        and not aggregated_sector_value_per_physical_input.empty
    ):
        sheet_title = "Re-aggregated Totals"
        if sheet_title in wb.sheetnames:
            wb.remove(wb[sheet_title])
        ws_v = wb.create_sheet(sheet_title)
        agg = None
        n_comp = None
        use_unit_column = True
        unit_label = None

        if disaggregated_sector_value_per_physical_input is not None:
            disagg_vals = disaggregated_sector_value_per_physical_input.copy()
            required = {
                "disaggregated_sector",
                "extension",
                "stressor",
                "unit",
                "quantity",
                "monetary_output",
            }
            if required.issubset(disagg_vals.columns):
                tmp = disagg_vals.merge(
                    mapping_df[[mapping_from_col, mapping_to_col]],
                    left_on="disaggregated_sector",
                    right_on=mapping_from_col,
                    how="left",
                )
                tmp = tmp[tmp[mapping_to_col].notna()].copy()
                if not tmp.empty:
                    agg = (
                        tmp.groupby([mapping_to_col, "extension", "stressor", "unit"], dropna=False)
                        [["quantity", "monetary_output"]]
                        .sum()
                    )
                    n_comp = tmp.groupby([mapping_to_col, "extension", "stressor", "unit"], dropna=False)[
                        mapping_from_col
                    ].nunique()

                    units = agg.index.get_level_values("unit").dropna().unique().tolist()
                    if len(units) == 1:
                        unit_label = str(units[0]).strip()
                        use_unit_column = False

                    stressor_notes = {
                        "energy use - emission relevant": (
                            "Energy that is combustion-relevant (used for emissions accounting)."
                        ),
                        "energy use - final": "Final energy delivered to end users.",
                        "energy use - gross": "Total primary energy input before transformation losses.",
                        "energy use - net": "Energy available after transformation losses.",
                    }

                    def _note_for_stressor(val: object) -> str:
                        s = str(val).strip().lower()
                        for key, note in stressor_notes.items():
                            if key in s:
                                return note
                        return ""

                    pass

        total_energy_header = "Total energy"
        if unit_label:
            total_energy_header = f"Total energy ({unit_label})"

        if use_unit_column:
            ws_v.append(
                [
                    mapping_to_col.replace("_", " ").title(),
                    "Extension",
                    "Stressor",
                    "Unit",
                    total_energy_header,
                    "Total value (M€)",
                    "N Components",
                    "Notes",
                ]
            )
        else:
            ws_v.append(
                [
                    mapping_to_col.replace("_", " ").title(),
                    "Extension",
                    "Stressor",
                    total_energy_header,
                    "Total value (M€)",
                    "N Components",
                    "Notes",
                ]
            )

        if agg is not None:
            for (sector, ext, stressor, unit), row in agg.iterrows():
                if use_unit_column:
                    ws_v.append(
                        [
                            str(sector),
                            str(ext),
                            str(stressor),
                            str(unit),
                            row["quantity"],
                            row["monetary_output"],
                            int(n_comp.get((sector, ext, stressor, unit), 0)) if n_comp is not None else 0,
                            _note_for_stressor(stressor),
                        ]
                    )
                else:
                    ws_v.append(
                        [
                            str(sector),
                            str(ext),
                            str(stressor),
                            row["quantity"],
                            row["monetary_output"],
                            int(n_comp.get((sector, ext, stressor, unit), 0)) if n_comp is not None else 0,
                            _note_for_stressor(stressor),
                        ]
                    )

        energy_col_idx = 5 if use_unit_column else 4
        value_col_idx = 6 if use_unit_column else 5
        for row in ws_v.iter_rows(min_row=2, min_col=energy_col_idx, max_col=value_col_idx):
            for cell in row:
                if cell.value is not None and not pd.isna(cell.value):
                    cell.number_format = "0.00"
        _auto_width(ws_v)

    # Add EXIO -> Eurostat CO2 mapping details
    if eurostat_sector_co2_mapping is not None and not eurostat_sector_co2_mapping.empty:
        sheet_title = "EXIO-Eurostat CO2 Mapping"
        if sheet_title in wb.sheetnames:
            wb.remove(wb[sheet_title])
        ws_m = wb.create_sheet(sheet_title)

        cols = [
            "aggregated_sector",
            "eurostat_sector_name",
            "exio_sector",
            "extension",
            "stressor",
            "unit",
            "emission_intensity",
            "method",
            "n_exio_sectors",
            "exio_weights",
            "mapping_notes",
        ]
        ws_m.append([c.replace("_", " ").title() for c in cols])

        df_m = eurostat_sector_co2_mapping.copy()
        label_map = eurostat_label_map or aggregated_sector_labels
        if label_map is not None and "aggregated_sector" in df_m.columns:
            df_m["eurostat_sector_name"] = df_m["aggregated_sector"].astype(str).map(
                lambda s: label_map.get(s, s)
            )
        if "aggregated_sector" in df_m.columns:
            df_m = df_m[df_m["aggregated_sector"].astype(str).str.startswith("CPA_")]
        for c in cols:
            if c not in df_m.columns:
                df_m[c] = ""

        def _split_exio_list(v) -> list[str]:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return []
            s = str(v).strip()
            if not s:
                return []
            return [x.strip() for x in s.split("|") if x.strip()]

        def _val(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            return v

        expanded_rows = []
        for _, r in df_m.iterrows():
            exio_list = _split_exio_list(r.get("exio_sectors", ""))
            if not exio_list:
                exio_list = [""]
            for exio in exio_list:
                row_dict = {c: r.get(c, "") for c in cols}
                row_dict["exio_sector"] = exio
                expanded_rows.append([_val(row_dict.get(c)) for c in cols])
        for row in expanded_rows:
            ws_m.append(row)

        for row in ws_m.iter_rows(min_row=2, min_col=7, max_col=7):
            for cell in row:
                if cell.value is not None and not pd.isna(cell.value):
                    cell.number_format = "0.0000"
        _auto_width(ws_m)

    # Add aggregated sector summary with domestic/exports/stocks + optional CO2 intensity columns.
    if aggregated_Y is not None and not aggregated_Y.empty:
        summary_has_prod_co2 = aggregated_sector_emission_intensity is not None and not aggregated_sector_emission_intensity.empty
        summary_has_use_co2 = fy_dom_intensity_scalar is not None
        if not summary_has_prod_co2:
            if not allow_missing_sfc_co2:
                raise ValueError("aggregated_sector_emission_intensity is required for the summary sheet.")
            _vprint(
                verbose,
                "WARNING: SFC workbook re-aggregated summary is omitting production-based CO2 columns "
                + _missing_sfc_co2_reason(variant_value, use_phase=False)
                + ".",
            )
        if not summary_has_use_co2:
            if not allow_missing_sfc_co2:
                raise ValueError("use-phase CO2 intensity is required for the summary sheet.")
            _vprint(
                verbose,
                "WARNING: SFC workbook re-aggregated summary is omitting use-phase CO2 columns "
                + _missing_sfc_co2_reason(variant_value, use_phase=True)
                + ".",
            )
        agg_domestic_fd, agg_exports, agg_stocks = _split_final_demand(aggregated_Y)

        sheet_title = "Re-aggregated Summary"
        if sheet_title in wb.sheetnames:
            wb.remove(wb[sheet_title])
        ws_s = wb.create_sheet(sheet_title)

        prod_unit_label = "mixed units"
        agg_emis_df = pd.DataFrame()
        intensity_by_sector = None
        output_by_sector = None
        if summary_has_prod_co2:
            assert aggregated_sector_emission_intensity is not None
            agg_emis_df = aggregated_sector_emission_intensity.copy()
            if "unit" in agg_emis_df.columns:
                uvals = agg_emis_df["unit"].dropna().unique().tolist()
                if len(uvals) == 1:
                    prod_unit_label = str(uvals[0]).strip()
            if "aggregated_sector" in agg_emis_df.columns:
                agg_emis_df["aggregated_sector"] = agg_emis_df["aggregated_sector"].astype(str)
            if "aggregated_sector" in agg_emis_df.columns and "emission_intensity" in agg_emis_df.columns:
                intensity_by_sector = agg_emis_df.set_index("aggregated_sector")["emission_intensity"]

        if aggregated_x is not None and not aggregated_x.empty:
            output_by_sector = aggregated_x.copy()
            output_by_sector.index = output_by_sector.index.astype(str)
        elif summary_has_prod_co2 and "aggregated_sector" in agg_emis_df.columns and "monetary_output" in agg_emis_df.columns:
            output_by_sector = agg_emis_df.set_index("aggregated_sector")["monetary_output"]

        summary_headers = [
            mapping_to_col.replace("_", " ").title(),
            "Monetary output (M€)",
        ]
        if summary_has_prod_co2:
            summary_headers.append(f"Mean CO2 intensity ({prod_unit_label}/M€)")
        summary_headers.extend(
            [
                "Domestic final demand (M€)",
                "Exports (M€)",
                "Stocks & valuables (M€)",
            ]
        )
        if summary_has_use_co2:
            summary_headers.append(f"Mean CO2 intensity for domestic demand ({fy_unit_label}/M€)")
        ws_s.append(summary_headers)

        if aggregated_sector_order:
            sector_index = pd.Index(list(map(str, aggregated_sector_order)))
        elif output_by_sector is not None:
            sector_index = output_by_sector.index
        else:
            sector_index = aggregated_Y.index.astype(str)

        for agg_sector in sector_index:
            agg_sector_s = str(agg_sector)
            row_values = [
                aggregated_sector_labels.get(agg_sector_s)
                if aggregated_sector_labels is not None
                else agg_sector_s,
                output_by_sector.get(agg_sector_s) if output_by_sector is not None else None,
            ]
            if summary_has_prod_co2:
                row_values.append(intensity_by_sector.get(agg_sector_s) if intensity_by_sector is not None else None)
            row_values.extend(
                [
                    agg_domestic_fd.get(agg_sector_s) if agg_domestic_fd is not None else None,
                    agg_exports.get(agg_sector_s) if agg_exports is not None else None,
                    agg_stocks.get(agg_sector_s) if agg_stocks is not None else None,
                ]
            )
            if summary_has_use_co2:
                row_values.append(fy_dom_intensity_scalar)
            ws_s.append(row_values)
        for row in ws_s.iter_rows(min_row=2, min_col=2, max_col=len(summary_headers)):
            for cell in row:
                if cell.value is None or pd.isna(cell.value):
                    continue
                header = str(ws_s.cell(row=1, column=cell.column).value)
                cell.number_format = "0.0000" if "intensity" in header.lower() else "0.00"
        _auto_width(ws_s)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{openpyxl.utils.get_column_letter(len(headers))}{len(display_df) + 1}"


def _clear_worksheet(ws) -> None:
    for row in ws.iter_rows():
        for cell in row:
            cell.value = None


def _template_table_layout(ws) -> dict:
    """
    Infer the 'Sheet 1' Eurostat export layout used by the SFC replication.

    We look for the header marker in A11 ('PRD_USE (Labels)') and for the first
    'Value added, gross' row to find where the product block ends.
    """
    header_row = None
    for r in range(1, ws.max_row + 1):
        if ws.cell(row=r, column=1).value == "PRD_USE (Labels)":
            header_row = r
            break
    if header_row is None:
        raise ValueError("Template workbook does not contain 'PRD_USE (Labels)' in column A.")

    # header width: last non-empty cell in header_row
    header_last_col = 0
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=header_row, column=c).value
        if v is not None and str(v).strip() != "":
            header_last_col = c
    if header_last_col < 3:
        raise ValueError("Template workbook header row seems empty/incomplete.")

    va_row = None
    for r in range(header_row + 1, ws.max_row + 1):
        if ws.cell(row=r, column=1).value == "Value added, gross":
            va_row = r
            break
    if va_row is None:
        raise ValueError("Template workbook does not contain 'Value added, gross' row.")

    total_row = header_row + 1
    first_product_row = header_row + 2
    last_product_row = va_row - 1
    n_products = last_product_row - first_product_row + 1
    if n_products <= 0:
        raise ValueError("Template workbook has no product rows between header and 'Value added, gross'.")

    total_col = 2
    first_product_col = 3
    last_product_col = first_product_col + n_products - 1
    first_fd_col = last_product_col + 1
    last_fd_col = header_last_col
    n_fd = last_fd_col - first_fd_col + 1

    return {
        "header_row": header_row,
        "header_last_col": header_last_col,
        "total_row": total_row,
        "first_product_row": first_product_row,
        "last_product_row": last_product_row,
        "n_products": n_products,
        "total_col": total_col,
        "first_product_col": first_product_col,
        "last_product_col": last_product_col,
        "first_fd_col": first_fd_col,
        "last_fd_col": last_fd_col,
        "n_fd": n_fd,
        "va_row": va_row,
    }


def _reshape_template_table(
    ws, *, target_n_products: int, target_n_final_demand: int, target_n_supplementary: int
) -> dict:
    """Resize style geometry to current source-owned product, FD and V/M axes."""
    layout = _template_table_layout(ws)
    current_n_products = int(layout["n_products"])
    delta = int(target_n_products) - current_n_products
    if delta > 0:
        first_fd_col = int(layout["first_fd_col"])
        first_lower_row = int(layout["va_row"])
        source_product_col = first_fd_col - 1
        source_product_row = first_lower_row - 1
        merged_ranges = [copy.copy(cell_range) for cell_range in ws.merged_cells.ranges]
        for cell_range in list(ws.merged_cells.ranges):
            ws.unmerge_cells(str(cell_range))
        ws.insert_cols(first_fd_col, amount=delta)
        ws.insert_rows(first_lower_row, amount=delta)

        source_letter = ws.cell(row=1, column=source_product_col).column_letter
        source_width = ws.column_dimensions[source_letter].width
        for column in range(first_fd_col, first_fd_col + delta):
            target_letter = ws.cell(row=1, column=column).column_letter
            ws.column_dimensions[target_letter].width = source_width
            for row in range(1, ws.max_row + 1):
                source = ws.cell(row=row, column=source_product_col)
                target = ws.cell(row=row, column=column)
                target._style = copy.copy(source._style)
        source_height = ws.row_dimensions[source_product_row].height
        for row in range(first_lower_row, first_lower_row + delta):
            ws.row_dimensions[row].height = source_height
            for column in range(1, ws.max_column + 1):
                source = ws.cell(row=source_product_row, column=column)
                target = ws.cell(row=row, column=column)
                target._style = copy.copy(source._style)

        # openpyxl deliberately does not translate merged ranges on insertion.
        for cell_range in merged_ranges:
            if cell_range.min_col >= first_fd_col:
                cell_range.shift(col_shift=delta)
            elif cell_range.max_col >= first_fd_col:
                cell_range.max_col += delta
            if cell_range.min_row >= first_lower_row:
                cell_range.shift(row_shift=delta)
            elif cell_range.max_row >= first_lower_row:
                cell_range.max_row += delta
            ws.merge_cells(str(cell_range))
    elif delta < 0:
        deleted = -delta
        first_deleted_col = int(layout["first_product_col"]) + target_n_products
        first_deleted_row = int(layout["first_product_row"]) + target_n_products
        merged_ranges = [copy.copy(cell_range) for cell_range in ws.merged_cells.ranges]
        for cell_range in list(ws.merged_cells.ranges):
            ws.unmerge_cells(str(cell_range))
        ws.delete_cols(first_deleted_col, amount=deleted)
        ws.delete_rows(first_deleted_row, amount=deleted)
        for cell_range in merged_ranges:
            removed_with_product_block = (
                first_deleted_col <= cell_range.min_col
                and cell_range.max_col < first_deleted_col + deleted
            ) or (
                first_deleted_row <= cell_range.min_row
                and cell_range.max_row < first_deleted_row + deleted
            )
            if removed_with_product_block:
                continue
            if cell_range.min_col >= first_deleted_col + deleted:
                cell_range.shift(col_shift=-deleted)
            elif cell_range.min_col < first_deleted_col <= cell_range.max_col:
                cell_range.max_col -= deleted
            if cell_range.min_row >= first_deleted_row + deleted:
                cell_range.shift(row_shift=-deleted)
            elif cell_range.min_row < first_deleted_row <= cell_range.max_row:
                cell_range.max_row -= deleted
            if cell_range.min_col <= cell_range.max_col and cell_range.min_row <= cell_range.max_row:
                ws.merge_cells(str(cell_range))
    layout = _template_table_layout(ws)
    fd_delta = int(target_n_final_demand) - int(layout["n_fd"])
    fd_boundary = int(layout["first_fd_col"]) + min(int(layout["n_fd"]), target_n_final_demand)
    if fd_delta > 0:
        ws.insert_cols(fd_boundary, amount=fd_delta)
        for column in range(fd_boundary, fd_boundary + fd_delta):
            for row in range(1, ws.max_row + 1):
                ws.cell(row, column)._style = copy.copy(ws.cell(row, fd_boundary - 1)._style)
    elif fd_delta < 0:
        ws.delete_cols(fd_boundary, amount=-fd_delta)
    layout = _template_table_layout(ws)
    current_supplementary = 0
    for row in range(int(layout["va_row"]), ws.max_row + 1):
        if ws.cell(row, 1).value is None or not str(ws.cell(row, 1).value).strip():
            break
        current_supplementary += 1
    supplementary_delta = int(target_n_supplementary) - current_supplementary
    supplementary_boundary = int(layout["va_row"]) + min(current_supplementary, target_n_supplementary)
    if supplementary_delta > 0:
        ws.insert_rows(supplementary_boundary, amount=supplementary_delta)
    elif supplementary_delta < 0:
        ws.delete_rows(supplementary_boundary, amount=-supplementary_delta)
    return _template_table_layout(ws)


def _current_eurostat_product_codes(parsed: ParsedEurostatIOT) -> list[str]:
    """Return the current parsed Eurostat product axis and verify metadata agrees."""
    matrix_rows = list(map(str, parsed.Z.index))
    matrix_columns = list(map(str, parsed.Z.columns))
    metadata_products = list(
        map(str, parsed.meta.get("identified_industry_columns", []))
    )
    if not metadata_products:
        raise ValueError("Current Eurostat parse does not expose an identified product domain.")
    if matrix_rows != matrix_columns or metadata_products != matrix_rows:
        raise ValueError(
            "Current Eurostat product domain differs between parsed Z and SDMX metadata."
        )
    return matrix_rows


def _current_eurostat_display_schema(
    parsed: ParsedEurostatIOT,
) -> tuple[list[str], list[str], list[str]]:
    """Return current source-owned workbook axes, excluding only Y's TOTAL control."""
    products = _current_eurostat_product_codes(parsed)
    final_demand = [
        str(code)
        for code in parsed.meta.get("identified_final_demand_columns", [])
        if str(code) != "TOTAL"
    ]
    supplementary = list(map(str, parsed.meta.get("identified_supplementary_rows", [])))
    if not final_demand or not supplementary:
        raise ValueError("Current Eurostat parse does not expose final-demand and supplementary axes.")
    return products, final_demand, supplementary




def _validate_parent_final_demand_reconstruction(
    *,
    product_codes_order: Sequence[str],
    base_product_codes: Sequence[str],
    parents_by_output: dict[str, tuple[str, ...]],
    fd_codes_order_no_total: Sequence[str],
    Y_before: pd.DataFrame,
    Y_ord: pd.DataFrame,
    source_parent_by_input: dict[str, str],
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> None:
    """Require every displayed final-use control to reaggregate to its parent."""
    product_codes = [str(code) for code in product_codes_order]
    base_codes = [str(code) for code in base_product_codes]
    fd_codes = [str(code) for code in fd_codes_order_no_total]
    children_by_parent: dict[str, list[str]] = {code: [] for code in base_codes}
    for output_code in product_codes:
        parents = parents_by_output.get(output_code, ())
        if len(parents) != 1 or parents[0] not in children_by_parent:
            raise ValueError(f"Invalid parent lineage for output {output_code!r}: {parents!r}")
        children_by_parent[parents[0]].append(output_code)

    before_parent_index = [
        source_parent_by_input.get(str(code)) or _base_sector_from_code(str(code), base_codes)
        for code in Y_before.index
    ]
    if any(parent is None for parent in before_parent_index):
        missing = [str(code) for code, parent in zip(Y_before.index, before_parent_index, strict=True) if parent is None]
        raise ValueError(f"Could not resolve source parents for final-demand rows: {missing!r}")
    source = Y_before.copy()
    source.index = pd.Index(before_parent_index)
    source = source.groupby(level=0, sort=False).sum().reindex(base_codes).fillna(0.0)

    def tfu(frame: pd.DataFrame) -> pd.Series:
        if {"P3", "P5", "P6"}.issubset(frame.columns):
            return frame["P3"] + frame["P5"] + frame["P6"]
        return frame["TFU"]

    for parent, children in children_by_parent.items():
        for code in fd_codes:
            if code == "TU" or (code != "TFU" and (code not in source or code not in Y_ord)):
                continue
            expected = float(tfu(source).loc[parent]) if code == "TFU" else float(source.loc[parent, code])
            actual = float(tfu(Y_ord).reindex(children).sum()) if code == "TFU" else float(Y_ord.loc[children, code].sum())
            if not np.isclose(actual, expected, rtol=rtol, atol=atol):
                raise ValueError(
                    "Parent final-demand reconstruction failed: "
                    f"parent={parent!r}, column={code!r}, actual={actual}, expected={expected}."
                )


def _resolve_product_template_alignment(
    *,
    template_xlsx: Path | None,
    base_product_codes: list[str],
    output_product_codes: list[str],
    mapping_df: pd.DataFrame | None = None,
    mapping_from_col: str = "from_sector",
    mapping_to_col: str = "to_sector",
    verbose: bool,
) -> tuple[Path | None, list[str], dict[str, str], dict[str, tuple[str, ...]]]:
    """Order output products by current Eurostat parent and mapping-row child order."""
    input_product_order = list(map(str, output_product_codes))
    output_set = set(input_product_order)
    base_codes = [str(code) for code in base_product_codes]
    parents_by_output: dict[str, tuple[str, ...]] = {}
    source_parent_by_input = _build_source_parent_by_input(
        mapping_df if mapping_df is not None else pd.DataFrame(),
        mapping_from_col=mapping_from_col,
        mapping_to_col=mapping_to_col,
        base_product_codes=base_codes,
    )

    mapped_parents: dict[str, list[str]] = {}
    if mapping_df is not None and mapping_from_col in mapping_df.columns and mapping_to_col in mapping_df.columns:
        for _, row in mapping_df.iterrows():
            output_code = str(row[mapping_to_col])
            parent_code = source_parent_by_input.get(str(row[mapping_from_col]))
            if parent_code is None:
                continue
            mapped_parents.setdefault(output_code, [])
            if parent_code not in mapped_parents[output_code]:
                mapped_parents[output_code].append(parent_code)

    for output_code in input_product_order:
        parent_candidates = mapped_parents.get(output_code)
        if parent_candidates:
            parents_by_output[output_code] = tuple(parent_candidates)
            continue
        inferred = _base_sector_from_code(output_code, base_codes)
        if inferred is not None:
            parents_by_output[output_code] = (inferred,)

    missing = [code for code in input_product_order if code not in parents_by_output]
    if missing:
        raise ValueError(f"Could not resolve current Eurostat parents for output products: {missing!r}")
    ambiguous = {code: parents for code, parents in parents_by_output.items() if len(parents) != 1}
    if ambiguous:
        raise ValueError(f"Output products span multiple current Eurostat parents: {ambiguous!r}")
    primary_parent_by_output = {code: parents[0] for code, parents in parents_by_output.items()}

    # Mapping rows are the authoritative child sequence inside each current Eurostat
    # parent. Unsplit products follow in their original source order.
    mapped_child_order: dict[str, list[str]] = {parent: [] for parent in base_codes}
    if mapping_df is not None and mapping_from_col in mapping_df.columns and mapping_to_col in mapping_df.columns:
        for _, row in mapping_df.iterrows():
            output_code = str(row[mapping_to_col])
            if output_code not in output_set:
                continue
            parent = source_parent_by_input.get(str(row[mapping_from_col])) or primary_parent_by_output.get(output_code)
            if parent in mapped_child_order and output_code not in mapped_child_order[parent]:
                mapped_child_order[parent].append(output_code)
    product_order: list[str] = []
    for parent in base_codes:
        for output_code in mapped_child_order[parent]:
            if output_code not in product_order:
                product_order.append(output_code)
        for output_code in input_product_order:
            if primary_parent_by_output[output_code] == parent and output_code not in product_order:
                product_order.append(output_code)
    if set(product_order) != output_set or len(product_order) != len(input_product_order):
        raise ValueError("Current-source/mapping product ordering lost or duplicated output products")
    return template_xlsx, product_order, primary_parent_by_output, parents_by_output


_FINAL_DEMAND_ENVIRONMENTAL_BOUNDARY = "scope_final_demand_direct"
_FINAL_DEMAND_ENVIRONMENTAL_DENOMINATOR_ROLE = "scope_final_demand"


def _partition_environmental_extensions_for_sector_reaggregation(
    environmental_extensions: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Separate product-axis rows from direct-final-demand rows.

    ``scope_final_demand_direct`` is already on its scientifically correct final-
    demand-category axis. It must bypass every product mapping while the other
    target boundaries keep the existing sector reaggregation path.
    """
    if environmental_extensions is None or environmental_extensions.empty:
        return None, None
    required = {"boundary", "denominator_role"}
    missing = required - set(environmental_extensions.columns)
    if missing:
        raise ValueError(
            "Target environmental sidecar is missing boundary metadata: "
            f"{sorted(missing)!r}."
        )
    frame = environmental_extensions.copy()
    final_mask = frame["boundary"].astype(str).eq(_FINAL_DEMAND_ENVIRONMENTAL_BOUNDARY)
    final_demand = frame.loc[final_mask].copy()
    sector = frame.loc[~final_mask].copy()
    if not final_demand.empty:
        roles = set(final_demand["denominator_role"].astype(str))
        if roles != {_FINAL_DEMAND_ENVIRONMENTAL_DENOMINATOR_ROLE}:
            raise ValueError(
                "scope_final_demand_direct rows must use denominator_role="
                f"{_FINAL_DEMAND_ENVIRONMENTAL_DENOMINATOR_ROLE!r}."
            )
    return (
        sector if not sector.empty else None,
        final_demand if not final_demand.empty else None,
    )


def _final_demand_environmental_rows_for_grid(
    final_demand_environmental_extensions: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Add compatibility axis positions while copying F_Y rows unchanged."""
    if (
        final_demand_environmental_extensions is None
        or final_demand_environmental_extensions.empty
    ):
        return None
    required_columns = [
        "sector",
        "boundary",
        "denominator_role",
        "extension",
        "stressor",
        "unit",
        "value",
        "monetary_output",
        "intensity",
    ]
    missing = set(required_columns) - set(final_demand_environmental_extensions.columns)
    if missing:
        raise ValueError(
            "Final-demand environmental rows are missing columns: "
            f"{sorted(missing)!r}."
        )
    out = final_demand_environmental_extensions.loc[:, required_columns].copy()
    group_cols = ["boundary", "denominator_role", "extension", "stressor", "unit"]
    out.insert(
        0,
        "sector_position",
        out.groupby(group_cols, sort=False, dropna=False).cumcount().add(1).astype("int64"),
    )
    return out.loc[
        :,
        [
            "sector_position",
            "sector",
            "boundary",
            "denominator_role",
            "extension",
            "stressor",
            "unit",
            "value",
            "monetary_output",
            "intensity",
        ],
    ].reset_index(drop=True)



def _complete_environmental_extension_sector_grid(
    environmental_extensions: pd.DataFrame | None,
    *,
    output: pd.Series,
    production_groups: pd.DataFrame,
) -> pd.DataFrame | None:
    """Complete explicit boundary-aware satellite groups onto the final IOT axis."""
    group_cols = ["boundary", "denominator_role", "extension", "stressor", "unit"]
    missing_group_columns = set(group_cols) - set(production_groups.columns)
    if missing_group_columns:
        raise ValueError(
            "Environmental production-group domain is missing explicit boundary metadata: "
            f"{sorted(missing_group_columns)!r}."
        )
    groups = production_groups.loc[:, group_cols].copy().drop_duplicates(ignore_index=True)
    if groups.empty:
        return None
    for column in group_cols:
        groups[column] = groups[column].astype(str).str.strip()
    if bool(groups.eq("").to_numpy().any()):
        raise ValueError("Environmental group metadata must not be blank.")
    required_columns = [
        "sector", "boundary", "denominator_role", "extension", "stressor", "unit",
        "value", "monetary_output", "intensity",
    ]
    if environmental_extensions is None:
        actual = pd.DataFrame(columns=required_columns)
    else:
        actual = environmental_extensions.copy()
    missing = set(required_columns) - set(actual.columns)
    if missing:
        raise ValueError(
            f"Aggregated environmental extensions are missing columns: {sorted(missing)!r}."
        )
    denominator = output.copy().astype("float64")
    denominator.index = denominator.index.astype(str)
    roles = set(groups["denominator_role"].astype(str))
    if len(roles) != 1:
        raise ValueError("Final environmental grid requires one denominator role per artifact.")
    allow_signed = roles == {"target_import_use"}
    if not np.isfinite(denominator.to_numpy(dtype="float64")).all():
        raise ValueError("Final environmental denominator must be finite.")
    if not allow_signed:
        denominator = _normalize_nonnegative_accounting_control(
            denominator,
            context="Final environmental denominator",
        )
    key_cols = ["sector"] + group_cols
    if bool(actual.duplicated(key_cols).any()):
        raise ValueError(
            "Aggregated environmental extensions contain duplicate sector/group rows "
            "(including boundary identity)."
        )
    actual_group_keys = actual.loc[:, group_cols].drop_duplicates(ignore_index=True)
    unknown_groups = actual_group_keys.merge(
        groups, on=group_cols, how="left", indicator=True, validate="one_to_one"
    )
    unknown_groups = unknown_groups.loc[unknown_groups["_merge"].eq("left_only")]
    if not unknown_groups.empty:
        raise ValueError(
            "Scientific environmental estimates contain groups outside raw production F "
            "identity domain: "
            f"{unknown_groups[group_cols].head(10).to_dict(orient='records')!r}."
        )
    unsupported = sorted(set(actual["sector"].astype(str)) - set(denominator.index))
    if unsupported:
        raise ValueError(
            f"Environmental estimates contain sectors outside the final IOT axis: {unsupported[:20]!r}."
        )
    if not actual.empty:
        observed = pd.to_numeric(actual["monetary_output"], errors="raise").to_numpy(
            dtype="float64"
        )
        expected = actual["sector"].astype(str).map(denominator).to_numpy(dtype="float64")
        if not np.allclose(observed, expected, rtol=1e-9, atol=1e-6, equal_nan=False):
            raise ValueError(
                "Supported environmental denominator disagrees with final IOT flow use/supply."
            )
    groups["__group_position"] = np.arange(len(groups), dtype="int64")
    sectors = pd.DataFrame(
        {
            "sector_position": np.arange(1, len(denominator) + 1, dtype="int64"),
            "sector": denominator.index.to_numpy(dtype=str),
            "monetary_output": denominator.to_numpy(dtype="float64"),
        }
    )
    grid = groups.merge(sectors, how="cross", sort=False)
    supported = actual.loc[:, key_cols + ["value", "intensity"]].copy()
    completed = grid.merge(
        supported, on=key_cols, how="left", sort=False, validate="one_to_one"
    )
    completed = completed.sort_values(
        ["__group_position", "sector_position"], kind="stable"
    ).drop(columns="__group_position")
    return completed.loc[
        :,
        [
            "sector_position", "sector", "boundary", "denominator_role", "extension",
            "stressor", "unit", "value", "monetary_output", "intensity",
        ],
    ].reset_index(drop=True)




def _production_environmental_group_domain(
    exio_environmental_extensions: pd.DataFrame,
    *,
    target_environmental_extensions: pd.DataFrame,
) -> pd.DataFrame:
    """Return selected production-F identities on the target sidecar's flow boundary.

    Raw EXIO F/F_Y rows are provenance and stressor-selection evidence. They do not
    define whether the target monetary object is DOM, IMP, or TOTAL. The paired
    target environmental sidecar owns that boundary and denominator role.
    """
    raw_required = {"extension", "kind", "stressor", "unit"}
    missing_raw = raw_required - set(exio_environmental_extensions.columns)
    if missing_raw:
        raise ValueError(
            "Raw EXIO environmental sidecar is missing production-group columns: "
            f"{sorted(missing_raw)!r}."
        )
    target_required = {"boundary", "denominator_role"}
    missing_target = target_required - set(target_environmental_extensions.columns)
    if missing_target:
        raise ValueError(
            "Target environmental sidecar is missing explicit boundary metadata: "
            f"{sorted(missing_target)!r}."
        )
    if target_environmental_extensions.empty:
        raise ValueError(
            "Raw EXIO production provenance requires a paired non-empty target environmental sidecar."
        )
    boundary_pairs = target_environmental_extensions.loc[
        :, ["boundary", "denominator_role"]
    ].copy()
    for column in ("boundary", "denominator_role"):
        boundary_pairs[column] = boundary_pairs[column].astype(str).str.strip()
    boundary_pairs = boundary_pairs.drop_duplicates(ignore_index=True)
    if len(boundary_pairs) != 1 or bool(boundary_pairs.eq("").to_numpy().any()):
        raise ValueError(
            "One target environmental artifact must identify exactly one non-blank "
            "boundary/denominator_role pair."
        )
    boundary = str(boundary_pairs.loc[0, "boundary"])
    denominator_role = str(boundary_pairs.loc[0, "denominator_role"])

    raw = exio_environmental_extensions.loc[
        exio_environmental_extensions["kind"].astype(str).eq("F"),
        ["extension", "stressor", "unit"],
    ].copy()
    raw.insert(0, "boundary", boundary)
    raw.insert(1, "denominator_role", denominator_role)
    return raw.drop_duplicates(ignore_index=True)


def _write_sector_contract(
    *,
    out_path: Path,
    product_order: Sequence[str],
    parent_by_sector: Mapping[str, str],
    verbose: bool,
) -> None:
    """Write the downstream sector-interface contract for one emitted IO system.

    The contract carries only structural identifiers: the canonical output
    position, the emitted sector, and its current published Eurostat parent. Monetary P1
    remains in the exported workbook so a consumer cannot accidentally pair a
    stale value vector with a newer workbook.
    """
    rows: list[dict[str, object]] = []
    for position, sector_id in enumerate(product_order, start=1):
        parent_id = parent_by_sector.get(str(sector_id))
        if parent_id is None:
            raise ValueError(f"Sector contract is missing a current Eurostat parent for {sector_id!r}.")
        # CPA product codes are structurally aligned to their NACE letter.
        parts = str(parent_id).split("_", maxsplit=1)
        if len(parts) != 2 or not parts[1] or not parts[1][0].isalpha():
            raise ValueError(f"Sector contract cannot derive NACE section from {parent_id!r}.")
        rows.append(
            {
                "position": position,
                "sector_id": str(sector_id),
                "parent_sector_id": str(parent_id),
                "nace_section": parts[1][0],
            }
        )
    _write_rows_csv(pd.DataFrame(rows), out_path, verbose=verbose)


def _write_source_sector_contract(
    *, out_path: Path, parent_order: Sequence[str], verbose: bool
) -> None:
    """Write source-row CPA65 identities for MPG4ALL auxiliary inputs.

    Capital stock and A64 employment use Eurostat's standard parent order.
    The legacy energy workbook uses its published source order, retained here
    as explicit contract data rather than an unlabelled MPG4ALL assumption.
    """
    energy_parent_order = (
        "CPA_A01","CPA_A02","CPA_A03","CPA_B","CPA_C10-12","CPA_C13-15","CPA_C16","CPA_C31_32",
        "CPA_C17","CPA_C18","CPA_C19","CPA_C20","CPA_C21","CPA_C22","CPA_C23","CPA_C24","CPA_C25",
        "CPA_C26","CPA_C27","CPA_C28","CPA_C33","CPA_C29","CPA_C30","CPA_D","CPA_E36","CPA_E37-39",
        "CPA_F","CPA_H49","CPA_H50","CPA_H51","CPA_G45","CPA_G46","CPA_G47","CPA_H52","CPA_H53",
        "CPA_I","CPA_J58","CPA_J59_60","CPA_J61","CPA_J62_63","CPA_K64","CPA_K65","CPA_K66","CPA_L68A",
        "CPA_L68B","CPA_M69_70","CPA_M71","CPA_M72","CPA_M73","CPA_M74_75","CPA_N77","CPA_N78","CPA_N79",
        "CPA_N80-82","CPA_O","CPA_P","CPA_Q86","CPA_Q87_88","CPA_R90-92","CPA_R93","CPA_S94","CPA_S95",
        "CPA_S96","CPA_T","CPA_U"
    )
    if len(parent_order) != 65 or set(parent_order) != set(energy_parent_order):
        raise ValueError("MPG4ALL source contract requires exactly the CPA65 parent domain.")
    rows: list[dict[str, object]] = []
    for source_system, order in (
        ("capital_stock", parent_order),
        ("employment_a64", parent_order),
        ("energy_balance", energy_parent_order),
    ):
        for position, parent_sector_id in enumerate(order, start=1):
            rows.append(
                {
                    "source_system": source_system,
                    "source_position": position,
                    "parent_sector_id": parent_sector_id,
                }
            )
    _write_rows_csv(pd.DataFrame(rows), out_path, verbose=verbose)



def _legacy_co2_production_view(
    environmental_extensions: pd.DataFrame | None,
    *,
    sector_col: str,
) -> pd.DataFrame | None:
    """Adapt one boundary-aware CO2 sector satellite to the legacy workbook view."""
    if environmental_extensions is None or environmental_extensions.empty:
        return None
    required = {
        "sector", "boundary", "extension", "stressor", "unit", "value", "monetary_output", "intensity"
    }
    if not required.issubset(environmental_extensions.columns):
        return None
    selection = (
        environmental_extensions["extension"].astype(str).eq("air_emissions")
        & environmental_extensions["stressor"].astype(str).str.casefold().eq("CO2 - combustion - air".casefold())
    )
    selected = cast(pd.DataFrame, environmental_extensions.loc[selection].copy())
    if selected.empty:
        return None
    selected = selected.rename(columns={"sector": sector_col, "value": "emission", "intensity": "emission_intensity"})
    return selected.loc[:, [
        sector_col, "extension", "stressor", "unit", "emission", "monetary_output", "emission_intensity"
    ]]


def _legacy_co2_final_demand_view(
    exio_environmental_extensions: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Adapt generic EXIO ``F_Y`` rows to the existing SFC CO2 presentation."""
    if exio_environmental_extensions is None or exio_environmental_extensions.empty:
        return None
    required = {
        "source_label",
        "extension",
        "kind",
        "stressor",
        "unit",
        "value",
        "monetary_denominator",
        "intensity",
    }
    if not required.issubset(exio_environmental_extensions.columns):
        return None
    selection = (
        (exio_environmental_extensions["kind"].astype(str) == "F_Y")
        & (exio_environmental_extensions["extension"].astype(str) == "air_emissions")
        & (
            exio_environmental_extensions["stressor"].astype(str).str.casefold()
            == "CO2 - combustion - air".casefold()
        )
    )
    selected = cast(pd.DataFrame, exio_environmental_extensions.loc[selection].copy())
    if selected.empty:
        return None
    selected = selected.rename(
        columns={
            "source_label": "category",
            "value": "emission",
            "monetary_denominator": "monetary_final_demand",
            "intensity": "final_demand_emission_intensity",
        }
    )
    return selected[
        [
            "extension",
            "stressor",
            "unit",
            "category",
            "emission",
            "monetary_final_demand",
            "final_demand_emission_intensity",
        ]
    ]

def _write_sfc_workbook(
    *,
    parsed: ParsedEurostatIOT,
    out_path: Path,
    template_xlsx: Path | None,
    eurostat_dataset: str,
    eurostat_filters: dict,
    eurostat_row_dim: str,
    eurostat_col_dim: str,
    eurostat_storage_folder: Path,
    variant_dim: str,
    variant_value: str | None,
    Z_before: pd.DataFrame,
    Y_before: pd.DataFrame,
    Z_after: pd.DataFrame,
    Y_after: pd.DataFrame,
    V_after: pd.DataFrame | None = None,
    M_after: pd.DataFrame | None = None,
    mapping_df: pd.DataFrame,
    mapping_from_col: str,
    mapping_to_col: str,
    verbose: bool,
    disaggregated_sector_value_per_physical_input: pd.DataFrame | None = None,
    aggregated_sector_value_per_physical_input: pd.DataFrame | None = None,
    disaggregated_sector_emission_intensity: pd.DataFrame | None = None,
    aggregated_sector_emission_intensity: pd.DataFrame | None = None,
    exio_final_demand_emission_intensity: pd.DataFrame | None = None,
    eurostat_sector_co2_mapping: pd.DataFrame | None = None,
    output_after: pd.Series | None = None,
) -> None:
    """Populate and round-trip validate one official Eurostat workbook layout."""
    from io import BytesIO

    import openpyxl

    number_format = "0.00"

    def _write_num(ws, *, row: int, col: int, value: float) -> None:
        cell = ws.cell(row=row, column=col)
        cell.value = float(value)
        cell.number_format = number_format

    filt = dict(eurostat_filters)
    if variant_value is not None:
        filt[variant_dim] = str(variant_value)
    dims = parsed.meta.get("dimensions", {})
    row_labels = dims.get(eurostat_row_dim, {})
    col_labels = dims.get(eurostat_col_dim, {})
    if template_xlsx is None:
        raise ValueError("Official Eurostat template is required for SFC workbook export")
    if not template_xlsx.is_file():
        raise FileNotFoundError(f"Official Eurostat template not found: {template_xlsx}")
    template_bytes = template_xlsx.read_bytes()
    base_product_codes, fd_codes_order_no_total, source_rows = _current_eurostat_display_schema(parsed)

    target_label_by_output: dict[str, str] = {}
    if "target_label" in mapping_df.columns and mapping_to_col in mapping_df.columns:
        for _, row in mapping_df.iterrows():
            output_code = str(row[mapping_to_col])
            label = str(row.get("target_label", "")).strip()
            if output_code and label and output_code not in target_label_by_output:
                target_label_by_output[output_code] = label

    template_xlsx, product_codes_order, _, parents_by_output = _resolve_product_template_alignment(
        template_xlsx=template_xlsx,
        base_product_codes=base_product_codes,
        output_product_codes=Z_after.index.astype(str).tolist(),
        mapping_df=mapping_df,
        mapping_from_col=mapping_from_col,
        mapping_to_col=mapping_to_col,
        verbose=verbose,
    )
    source_parent_by_input = _build_source_parent_by_input(
        mapping_df,
        mapping_from_col=mapping_from_col,
        mapping_to_col=mapping_to_col,
        base_product_codes=base_product_codes,
    )

    missing_fd = [
        code
        for code in fd_codes_order_no_total
        if code not in Y_after.columns and code not in {"TU", "TFU"}
    ]
    if missing_fd:
        raise ValueError(f"Transformed Y is missing official template columns: {missing_fd!r}")

    Z_ord = Z_after.loc[product_codes_order, product_codes_order].astype("float64")
    Y_ord = Y_after.loc[product_codes_order, :].astype("float64")
    V_ord = None
    if V_after is not None:
        V_ord = V_after.copy().astype("float64")
        V_ord.index = V_ord.index.astype(str)
        V_ord.columns = V_ord.columns.astype(str)
    M_ord = None
    if M_after is not None:
        M_ord = M_after.copy().astype("float64")
        M_ord.index = M_ord.index.astype(str)
        M_ord.columns = M_ord.columns.astype(str)
    if output_after is None:
        raise ValueError("Workbook export requires the authoritative output vector.")
    x_after = output_after.copy()
    x_after.index = x_after.index.astype(str)
    expected_output_index = pd.Index(Z_after.index.astype(str))
    if not x_after.index.is_unique or set(x_after.index) != set(expected_output_index):
        raise ValueError(
            "Workbook authoritative output vector must cover the transformed product axis exactly."
        )
    x_after = x_after.reindex(expected_output_index).astype("float64")
    if not np.isfinite(x_after.to_numpy(dtype="float64")).all():
        raise ValueError("Workbook authoritative output vector must be finite.")

    # The official workbook is always the layout authority. Reshaping inserts or
    # removes only product rows/columns, leaving final-demand and lower-row order intact.
    wb = openpyxl.load_workbook(BytesIO(template_bytes), data_only=False)
    if "Sheet 1" not in wb.sheetnames:
        raise ValueError(f"Template workbook missing required sheet 'Sheet 1': {template_xlsx}")
    ws = wb["Sheet 1"]
    layout = _template_table_layout(ws)
    layout = _reshape_template_table(
        ws,
        target_n_products=len(product_codes_order),
        target_n_final_demand=len(fd_codes_order_no_total),
        target_n_supplementary=len(source_rows),
    )

    hr = int(layout["header_row"])
    tr = int(layout["total_row"])
    r0 = int(layout["first_product_row"])
    c_total = int(layout["total_col"])
    c_prod0 = int(layout["first_product_col"])
    c_fd0 = int(layout["first_fd_col"])
    lower_start = int(layout["va_row"])

    if len(source_rows) != len(set(source_rows)):
        raise ValueError(f"Current Eurostat supplementary rows are not unique: {source_rows!r}")

    # Hard taint boundary: preserve formatting/merges/dimensions, but erase every
    # template cell value before generated economic data are written.
    from openpyxl.cell.cell import MergedCell
    for worksheet_row in ws.iter_rows():
        for cell in worksheet_row:
            if not isinstance(cell, MergedCell):
                cell.value = None

    ws.cell(row=hr, column=1).value = f"{str(eurostat_col_dim).upper()} (Labels)"
    ws.cell(row=hr, column=c_total).value = "Total"
    for j, code in enumerate(product_codes_order):
        label = target_label_by_output.get(code) or _pretty_sector_label(row_labels.get(code, str(code)))
        ws.cell(row=hr, column=c_prod0 + j).value = label
    for j, code in enumerate(fd_codes_order_no_total):
        ws.cell(row=hr, column=c_fd0 + j).value = _pretty_sector_label(col_labels.get(code, code))

    sector_label_map = {
        str(code): target_label_by_output.get(str(code), _pretty_sector_label(row_labels.get(str(code), str(code))))
        for code in product_codes_order
    }

    # Total row label
    ws.cell(row=tr, column=1).value = "Total"

    # Summary columns (these exist in the Eurostat XLSX export but should be derived from Z/Y)
    row_sum_Z = Z_ord.sum(axis=1)
    total_intermediate = float(row_sum_Z.sum())

    # Total final use (TFU) should be derived for consistency after splitting/balancing:
    # TFU = P3 + P5 + P6. (P5M and P51G are overlapping components of P5.)
    if {"P3", "P5", "P6"}.issubset(set(Y_ord.columns)):
        tfu = (Y_ord["P3"] + Y_ord["P5"] + Y_ord["P6"]).astype("float64")
        tfu_total = float(tfu.sum())
    elif "TFU" in Y_ord.columns:
        tfu = Y_ord["TFU"].astype("float64")
        tfu_total = float(tfu.sum())
    else:
        tfu = pd.Series(0.0, index=Y_ord.index)
        tfu_total = 0.0

    tu = (row_sum_Z + tfu).astype("float64")
    tu_total = float(total_intermediate + tfu_total)

    # Total row, column B: total intermediate consumption
    _write_num(ws, row=tr, col=c_total, value=total_intermediate)

    col_sums_Z = Z_ord.sum(axis=0)
    for j, code in enumerate(product_codes_order):
        _write_num(ws, row=tr, col=c_prod0 + j, value=float(col_sums_Z.loc[code]))

    col_sums_Y = Y_ord.sum(axis=0)
    for j, code in enumerate(fd_codes_order_no_total):
        if code == "TU":
            _write_num(ws, row=tr, col=c_fd0 + j, value=tu_total)
        elif code == "TFU":
            _write_num(ws, row=tr, col=c_fd0 + j, value=tfu_total)
        else:
            _write_num(ws, row=tr, col=c_fd0 + j, value=float(col_sums_Y.get(code, 0.0)))

    # Fill product rows (labels + numbers)
    for i, row_code in enumerate(product_codes_order):
        excel_r = r0 + i
        ws.cell(row=excel_r, column=1).value = sector_label_map.get(str(row_code), _pretty_sector_label(row_code))

        _write_num(ws, row=excel_r, col=c_total, value=float(row_sum_Z.loc[row_code]))

        # intermediate block
        row_vec = Z_ord.loc[row_code, product_codes_order]
        for j, col_code in enumerate(product_codes_order):
            _write_num(ws, row=excel_r, col=c_prod0 + j, value=float(row_vec.loc[col_code]))

        # final demand block (excluding TOTAL)
        for j, col_code in enumerate(fd_codes_order_no_total):
            if col_code == "TU":
                _write_num(ws, row=excel_r, col=c_fd0 + j, value=float(tu.loc[row_code]))
            elif col_code == "TFU":
                _write_num(ws, row=excel_r, col=c_fd0 + j, value=float(tfu.loc[row_code]))
            else:
                _write_num(ws, row=excel_r, col=c_fd0 + j, value=float(Y_ord.loc[row_code, col_code]))

    lower_row_vectors: dict[str, pd.Series] = {}
    lower_fd_vectors: dict[str, pd.Series] = {}

    def _match_row(frame: pd.DataFrame | None, row_code: str) -> str | None:
        if frame is None:
            return None
        index = [str(code) for code in frame.index]
        exact = [code for code in index if code == row_code]
        if len(exact) > 1:
            raise ValueError(f"Ambiguous supplementary row code {row_code!r}")
        return exact[0] if exact else None

    flow_role = str(variant_value or "TOTAL").upper()
    if flow_role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unknown workbook flow role {variant_value!r}")

    # Remove every inherited auxiliary sheet. The template is presentation-only.
    for sheet_name in list(wb.sheetnames):
        if sheet_name != "Sheet 1":
            wb.remove(wb[sheet_name])

    ws_sum = wb.create_sheet("Summary")
    ws_sum.append(["Field", "Value"])
    ws_sum.append(["Table", "Generated disaggregated/reaggregated Eurostat-shaped IOT"])
    ws_sum.append(["Dataset", str(eurostat_dataset)])
    ws_sum.append(["Geo", str(filt.get("geo", ""))])
    ws_sum.append(["Time", str(filt.get("time", ""))])
    ws_sum.append(["Frequency", str(filt.get("freq", ""))])
    ws_sum.append(["Unit", str(filt.get("unit", ""))])
    ws_sum.append(["Stock or flow", str(variant_value or "")])
    ws_sum.append(["Output semantics", "Domestic product output; shared activity denominator"])
    ws_sum.append(["Final-demand basis", "Atomic/non-overlapping P3, P5 and P6 categories"])
    ws_sum.append(["Template contract", "layout/styles/static labels only; no template numerical evidence"])
    ws_struct = wb.create_sheet("Structure")
    ws_struct.append(["Dimension", "Count"])
    ws_struct.append(["Products", len(product_codes_order)])
    ws_struct.append(["Final-demand display columns", len(fd_codes_order_no_total)])
    ws_struct.append(["Supplementary rows", len(source_rows)])

    lower_end = max(ws.max_row, lower_start + len(source_rows) - 1)

    for row in range(lower_start, lower_end + 1):
        for col in range(1, int(layout["header_last_col"]) + 1):
            ws.cell(row=row, column=col).value = None

    row_positions: dict[str, int] = {}
    row_labels_written: dict[str, str] = {}
    parsed_v = parsed.V.astype("float64") if parsed.V is not None else None
    parsed_m = parsed.M.astype("float64") if parsed.M is not None else None
    required_rows = (
        set(NAIO_10_CP1700_ACCOUNTING_POLICY.required_rows_by_flow.get(str(variant_value), ()))
        if eurostat_dataset == NAIO_10_CP1700_ACCOUNTING_POLICY.dataset
        else None
    )
    if V_ord is not None:
        missing_v_columns = sorted(set(product_codes_order) - set(V_ord.columns.astype(str)))
        if missing_v_columns:
            raise ValueError(f"Transformed V is missing applicable product columns: {missing_v_columns!r}")
    if M_ord is not None:
        missing_m_columns = sorted(set(fd_codes_order_no_total) - set(M_ord.columns.astype(str)))
        if missing_m_columns:
            raise ValueError(f"Transformed M is missing applicable final-demand columns: {missing_m_columns!r}")

    for offset, row_code in enumerate(source_rows):
        target_r = lower_start + offset
        row_label = _pretty_sector_label(row_labels.get(row_code, row_code))
        source_v_row = _match_row(parsed_v, row_code)
        source_m_row = _match_row(parsed_m, row_code)
        output_v_row = _match_row(V_ord, row_code)
        output_m_row = _match_row(M_ord, row_code)
        # Current SDMX/source policy plus transformed V/M exclusively own
        # applicability. Template filled/blank patterns are not source evidence.
        source_v_applicable = source_v_row is not None
        source_m_applicable = False
        if source_m_row is not None and parsed_m is not None:
            source_m_values = parsed_m.loc[source_m_row].reindex(
                fd_codes_order_no_total
            )
            source_m_applicable = bool(source_m_values.notna().any())
        if source_v_applicable and output_v_row is None:
            raise ValueError(
                f"Transformed V is missing source supplementary row {row_code!r}."
            )
        if source_m_applicable and output_m_row is None:
            raise ValueError(
                f"Transformed M is missing applicable source supplementary row {row_code!r}."
            )
        if (
            not source_m_applicable
            and (required_rows is None or row_code not in required_rows)
            and output_m_row is not None
            and M_ord is not None
            and bool(
                M_ord.loc[output_m_row]
                .reindex(fd_codes_order_no_total)
                .notna()
                .any()
            )
        ):
            raise ValueError(
                "Transformed M contains values for source-blank supplementary row "
                f"{row_code!r}."
            )

        # Supplementary blocks are already authoritative transformed data.  This
        # writer must serialize them, never reconstruct them from Z, x or labels.
        product_vector = (
            V_ord.loc[output_v_row].reindex(product_codes_order).astype("float64")
            if output_v_row is not None and V_ord is not None
            else pd.Series(np.nan, index=product_codes_order, dtype="float64")
        )
        fd_vector = (
            M_ord.loc[output_m_row].reindex(fd_codes_order_no_total).astype("float64")
            if output_m_row is not None and M_ord is not None
            else pd.Series(np.nan, index=fd_codes_order_no_total, dtype="float64")
        )
        ws.cell(row=target_r, column=1).value = row_label
        # The Eurostat-shaped column-B control is the signed total of the
        # authoritative transformed industry vector.  Leave it blank only when
        # the entire V row is economically non-applicable (not when its sum is
        # zero or negative).
        product_total = product_vector.sum(min_count=1)
        if not pd.isna(product_total):
            _write_num(ws, row=target_r, col=c_total, value=float(product_total))
        for j, output_code in enumerate(product_codes_order):
            value = product_vector.loc[output_code]
            if not pd.isna(value):
                _write_num(ws, row=target_r, col=c_prod0 + j, value=float(value))
        for j, fd_code in enumerate(fd_codes_order_no_total):
            value = fd_vector.loc[fd_code]
            if not pd.isna(value):
                _write_num(ws, row=target_r, col=c_fd0 + j, value=float(value))

        lower_row_vectors[row_code] = product_vector
        lower_fd_vectors[row_code] = fd_vector
        row_positions[row_code] = target_r
        row_labels_written[row_code] = row_label

    _validate_parent_final_demand_reconstruction(
        product_codes_order=product_codes_order,
        base_product_codes=base_product_codes,
        parents_by_output=parents_by_output,
        fd_codes_order_no_total=fd_codes_order_no_total,
        Y_before=Y_before,
        Y_ord=Y_ord,
        source_parent_by_input=source_parent_by_input,
    )
    _write_excel_mapping_sheet(
        wb,
        sheet_name="IO Aggregation Mapping",
        mapping_df=mapping_df,
        disaggregated_Y=Y_before,
        aggregated_Y=Y_after,
        aggregated_x=x_after,
        aggregated_sector_order=list(map(str, product_codes_order)),
        aggregated_sector_labels=sector_label_map,
        eurostat_label_map=row_labels,
        eurostat_sector_co2_mapping=eurostat_sector_co2_mapping,
        disaggregated_sector_value_per_physical_input=disaggregated_sector_value_per_physical_input,
        aggregated_sector_value_per_physical_input=aggregated_sector_value_per_physical_input,
        disaggregated_sector_emission_intensity=disaggregated_sector_emission_intensity,
        aggregated_sector_emission_intensity=aggregated_sector_emission_intensity,
        exio_final_demand_emission_intensity=exio_final_demand_emission_intensity,
        mapping_from_col=mapping_from_col,
        mapping_to_col=mapping_to_col,
        variant_value=variant_value,
        allow_missing_sfc_co2=(
            disaggregated_sector_emission_intensity is None
            or exio_final_demand_emission_intensity is None
            or (
                str(variant_dim).strip() == "stk_flow"
                and str(variant_value or "").strip().upper()
                in {"IMP", "IMPORT", "IMPORTS", "TOTAL"}
            )
        ),
        verbose=verbose,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    # Round-trip the saved workbook so stale-template, shape and serialization
    # failures cannot hide behind an in-memory validation pass.
    reopened = openpyxl.load_workbook(out_path, data_only=False)
    if "Sheet 1" not in reopened.sheetnames or "Summary" not in reopened.sheetnames or "Structure" not in reopened.sheetnames:
        raise ValueError(f"Saved workbook is missing required generated sheets: {out_path}")
    ws_reopened = reopened["Sheet 1"]

    # Codes are validated before serialization; labels prove that the saved
    # workbook retained the same canonical order on every visible axis.
    for j, product_code in enumerate(product_codes_order):
        expected_label = sector_label_map[product_code]
        if ws_reopened.cell(row=hr, column=c_prod0 + j).value != expected_label:
            raise ValueError(f"Workbook round-trip lost product-column order at {product_code!r}")
        if ws_reopened.cell(row=r0 + j, column=1).value != expected_label:
            raise ValueError(f"Workbook round-trip lost product-row order at {product_code!r}")
    for j, fd_code in enumerate(fd_codes_order_no_total):
        expected_label = _pretty_sector_label(col_labels.get(fd_code, fd_code))
        if ws_reopened.cell(row=hr, column=c_fd0 + j).value != expected_label:
            raise ValueError(
                f"Workbook round-trip lost current final-demand order at {fd_code!r}"
            )

    def _assert_roundtrip_value(label: str, actual: object, expected: float) -> None:
        if not isinstance(actual, (int, float)) or not np.isclose(float(actual), expected, rtol=1e-9, atol=1e-6):
            raise ValueError(f"Workbook round-trip validation failed for {label}: actual={actual!r}, expected={expected!r}")

    for i, row_code in enumerate(product_codes_order):
        excel_r = r0 + i
        for j, col_code in enumerate(product_codes_order):
            _assert_roundtrip_value(
                f"Z[{row_code!r}, {col_code!r}]",
                ws_reopened.cell(row=excel_r, column=c_prod0 + j).value,
                float(Z_ord.loc[row_code, col_code]),
            )
        for j, fd_code in enumerate(fd_codes_order_no_total):
            expected = float(tu.loc[row_code]) if fd_code == "TU" else float(tfu.loc[row_code]) if fd_code == "TFU" else float(Y_ord.loc[row_code, fd_code])
            _assert_roundtrip_value(
                f"Y[{row_code!r}, {fd_code!r}]",
                ws_reopened.cell(row=excel_r, column=c_fd0 + j).value,
                expected,
            )

    for row_code in source_rows:
        row = row_positions[row_code]
        if ws_reopened.cell(row=row, column=1).value != row_labels_written[row_code]:
            raise ValueError(f"Workbook round-trip validation lost supplementary row {row_code!r}.")
        expected_total = lower_row_vectors[row_code].sum(min_count=1)
        actual_total = ws_reopened.cell(row=row, column=c_total).value
        if pd.isna(expected_total):
            if isinstance(actual_total, (int, float)):
                raise ValueError(
                    "Workbook round-trip validation found numeric value for blank "
                    f"supplementary total row {row_code!r}: {actual_total!r}"
                )
        else:
            _assert_roundtrip_value(
                f"supplementary total row {row_code!r}", actual_total, float(expected_total)
            )
        for j, product_code in enumerate(product_codes_order):
            expected = lower_row_vectors[row_code].loc[product_code]
            actual = ws_reopened.cell(row=row, column=c_prod0 + j).value
            if pd.isna(expected):
                if isinstance(actual, (int, float)):
                    raise ValueError(
                        f"Workbook round-trip validation found numeric value for blank "
                        f"supplementary V[{row_code!r}, {product_code!r}]: {actual!r}"
                    )
            else:
                _assert_roundtrip_value(
                    f"supplementary V[{row_code!r}, {product_code!r}]", actual, float(expected)
                )
        for j, fd_code in enumerate(fd_codes_order_no_total):
            expected = lower_fd_vectors[row_code].loc[fd_code]
            actual = ws_reopened.cell(row=row, column=c_fd0 + j).value
            if pd.isna(expected):
                if isinstance(actual, (int, float)):
                    raise ValueError(
                        f"Workbook round-trip validation found inherited numeric value for "
                        f"supplementary M[{row_code!r}, {fd_code!r}]: {actual!r}"
                    )
            else:
                _assert_roundtrip_value(
                    f"supplementary M[{row_code!r}, {fd_code!r}]", actual, float(expected)
                )
    for row in range(lower_start + len(source_rows), lower_end + 1):
        for col in range(c_prod0, int(layout["header_last_col"]) + 1):
            if isinstance(ws_reopened.cell(row=row, column=col).value, (int, float)):
                raise ValueError(f"Workbook round-trip validation found stale template value at row={row}, col={col}.")
    reopened.close()
    _vprint(verbose, f"Wrote and round-trip validated: {out_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Aggregate (re-group) an IO table (Z,Y) using the canonical annotated mapping CSV format.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--classification",
        choices=["product", "industry"],
        default="product",
        help="Choose the source/target dictionary pair for the canonical mapping CSV.",
    )

    p.add_argument("--mapping-csv", required=True, help="Canonical annotated aggregation mapping CSV.")

    p.add_argument(
        "--require-from-contains",
        default=None,
        help="Safety check (CLI-only): require every from_sector in the mapping CSV to contain this substring "
        "(e.g. '::' to ensure mapping only targets disaggregated subsectors).",
    )

    inp = p.add_argument_group("Input")
    inp.add_argument("--Z", help="Path to Z table.")
    inp.add_argument("--Y", help="Path to Y table.")
    inp.add_argument(
        "--indir",
        help="Input directory. If it contains subfolders like 'stk_flow=DOM', all variants are processed.",
    )
    inp.add_argument("--variant-dim", default="stk_flow")
    inp.add_argument("--z-filename", default="Z_split.csv")
    inp.add_argument("--y-filename", default="Y_split.csv")
    inp.add_argument(
        "--split-selected-sectors-json",
        default=None,
        help="Optional exio_selected_sectors.json path for strict split-child coverage checks.",
    )
    inp.add_argument("--fail-on-unmapped", action="store_true", help="Fail if any input sector is unmapped.")
    inp.add_argument(
        "--fail-on-unmapped-split-children",
        action="store_true",
        help="Fail if any selected split child is unmapped.",
    )
    inp.add_argument(
        "--expect-n-sectors",
        type=int,
        default=None,
        help="Fail if the aggregated output does not have exactly this many sectors.",
    )

    out = p.add_argument_group("Output")
    out.add_argument("--outdir", required=True, help="Output directory.")
    out.add_argument("--no-xlsx", action="store_true", help="Skip XLSX report.")
    out.add_argument("--create-io-system", action="store_true", help="Create pymrio.IOSystem and calc_system().")
    out.add_argument(
        "--write-sfc-workbook",
        action="store_true",
        help="Write an additional Eurostat-export-shaped workbook for SFC/R scripts (one per variant).",
    )
    out.add_argument(
        "--sfc-template-xlsx",
        default=None,
        help=(
            "Eurostat-shaped XLSX layout template. Only styles, static labels and "
            "display order are reused; template numerical cells and metadata are ignored."
        ),
    )
    out.add_argument(
        "--sfc-outname",
        default="IOT_reaggregated_{variant}.xlsx",
        help="Output workbook filename template. Supports {variant}. Written inside each outdir/variant folder.",
    )
    out.add_argument("--eurostat-dataset", default=None, help="Eurostat dataset id (e.g. naio_10_cp1700).")
    out.add_argument(
        "--eurostat-filters",
        default=None,
        help="Base Eurostat filters as JSON (without stk_flow when processing variants).",
    )
    out.add_argument("--eurostat-row-dim", default=None, help="Eurostat row dimension id (e.g. prd_ava).")
    out.add_argument("--eurostat-col-dim", default=None, help="Eurostat column dimension id (e.g. prd_use).")
    out.add_argument("--eurostat-storage-folder", default="data/eurostat", help="Eurostat cache folder.")

    args = p.parse_args(argv)

    mapping, mapping_df = read_aggregation_mapping(args.mapping_csv)

    if args.require_from_contains is not None:
        bad = [k for k in mapping.keys() if args.require_from_contains not in str(k)]
        if bad:
            raise SystemExit(
                f"--require-from-contains {args.require_from_contains!r} failed for mapping rows: {bad[:10]}"
                + (f" (and {len(bad)-10} more)" if len(bad) > 10 else "")
            )

    keep_unmapped = not bool(args.fail_on_unmapped)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _process_one(Z_path: Path, Y_path: Path, out_base: Path) -> None:
        _vprint(args.verbose, f"Reading: {Z_path.resolve()}")
        Z = _read_table(Z_path).astype("float64")
        _vprint(args.verbose, f"Reading: {Y_path.resolve()}")
        Y = _read_table(Y_path).astype("float64")

        Z.index = Z.index.astype(str)
        Z.columns = Z.columns.astype(str)
        Y.index = Y.index.astype(str)
        Y.columns = Y.columns.astype(str)
        z_dir = Z_path.parent
        v_path = z_dir / "V_split.csv"
        V = _read_table(v_path).astype("float64") if v_path.exists() else None
        if V is not None:
            V.index = V.index.astype(str)
            V.columns = V.columns.astype(str)
        m_path = z_dir / "M_split.csv"
        M = _read_table(m_path).astype("float64") if m_path.exists() else None
        if M is not None:
            M.index = M.index.astype(str)
            M.columns = M.columns.astype(str)

        mapping_for_run = dict(mapping)
        mapping_df_for_run = mapping_df.copy()
        edge_mapping_audit = pd.DataFrame()
        edge_metadata_path = z_dir / "split_child_edges.csv"
        if edge_metadata_path.exists():
            _vprint(args.verbose, f"Reading: {edge_metadata_path.resolve()}")
            edge_metadata = pd.read_csv(edge_metadata_path)
            mapping_for_run, mapping_df_for_run, edge_mapping_audit = _augment_mapping_from_split_child_edges(
                mapping=mapping_for_run,
                mapping_df=mapping_df_for_run,
                edge_metadata=edge_metadata,
            )
            if not edge_mapping_audit.empty:
                _vprint(
                    args.verbose,
                    f"Added/verified {len(edge_mapping_audit)} edge-child aggregation mapping row(s).",
                )
        if args.fail_on_unmapped_split_children:
            selected_json = (
                Path(args.split_selected_sectors_json)
                if args.split_selected_sectors_json
                else z_dir / "exio_selected_sectors.json"
            )
            selected_children = _load_split_selected_children(selected_json)
            counts = mapping_df_for_run["from_sector"].astype(str).value_counts()
            missing_selected = sorted(child for child in selected_children if int(counts.get(child, 0)) == 0)
            duplicate_selected = sorted(child for child in selected_children if int(counts.get(child, 0)) != 1)
            if missing_selected or duplicate_selected:
                raise SystemExit(
                    "Strict aggregation mapping must contain every emitted split child exactly once: "
                    f"missing={missing_selected[:20]!r}, duplicates={duplicate_selected[:20]!r}"
                )
        input_sectors = set(Z.index.astype(str))
        mapped_sources = set(map(str, mapping_for_run.keys()))
        unmapped = sorted(input_sectors - mapped_sources)
        if args.fail_on_unmapped and unmapped:
            raise SystemExit(
                "Aggregation mapping leaves unmapped input sectors: "
                f"{unmapped[:20]}"
            )


        # Try to load disaggregated sector value-per-physical-input if they exist
        disaggregated_values = None
        values_path = z_dir / "disaggregated_sector_value_per_physical_input.csv"
        if values_path.exists():
            _vprint(args.verbose, f"Reading: {values_path.resolve()}")
            disaggregated_values = pd.read_csv(values_path)
            required_cols = {
                "disaggregated_sector",
                "extension",
                "stressor",
                "unit",
                "quantity",
                "monetary_output",
            }
            missing_cols = required_cols - set(disaggregated_values.columns)
            if missing_cols:
                _vprint(
                    args.verbose,
                    "Skipping sector value-per-physical-input aggregation; missing columns in "
                    f"{values_path.name}: {sorted(missing_cols)!r}",
                )
                disaggregated_values = None

        environmental_extensions = None
        environmental_path = z_dir / "environmental_extensions.csv"
        if environmental_path.exists():
            _vprint(args.verbose, f"Reading: {environmental_path.resolve()}")
            environmental_extensions = pd.read_csv(environmental_path)
            required_cols = {
                "sector",
                "boundary",
                "denominator_role",
                "extension",
                "stressor",
                "unit",
                "value",
                "monetary_output",
                "intensity",
            }
            missing_cols = required_cols - set(environmental_extensions.columns)
            if missing_cols:
                raise ValueError(
                    f"{environmental_path.name} missing required columns: "
                    f"{sorted(missing_cols)!r}"
                )

        exio_environmental_extensions = None
        exio_environmental_path = z_dir / "exio_environmental_extensions.csv"
        if exio_environmental_path.exists():
            _vprint(args.verbose, f"Reading: {exio_environmental_path.resolve()}")
            exio_environmental_extensions = pd.read_csv(exio_environmental_path)
            required_cols = {
                "source_label",
                "extension",
                "kind",
                "stressor",
                "unit",
                "value",
                "monetary_denominator",
                "intensity",
            }
            missing_cols = required_cols - set(exio_environmental_extensions.columns)
            if missing_cols:
                raise ValueError(
                    f"{exio_environmental_path.name} missing required columns: "
                    f"{sorted(missing_cols)!r}"
                )

        _vprint(args.verbose, "Calling: aggregate_iot(...)")
        t = _Timer()
        x_input_path = z_dir / "x_split.csv"
        if not x_input_path.is_file():
            raise SystemExit(f"Missing authoritative output vector beside {Z_path}: {x_input_path.name}")
        x_input_frame = pd.read_csv(x_input_path, index_col=0)
        if x_input_frame.shape[1] != 1:
            raise SystemExit(f"Expected one output column in {x_input_path}, found {x_input_frame.shape[1]}")
        x_input = x_input_frame.iloc[:, 0].astype("float64")
        x_input.index = x_input.index.astype(str)
        flow_role = (
            str(out_base.name).split("=", 1)[-1].upper()
            if str(out_base.name).startswith("stk_flow=")
            else "TOTAL"
        )
        res = aggregate_iot(
            Z,
            Y,
            mapping_for_run,
            V=V,
            M=M,
            output=x_input,
            flow_role=flow_role,
            keep_unmapped=keep_unmapped,
            create_io_system=args.create_io_system,
            compute_sector_value_per_physical_input=disaggregated_values is not None,
            disaggregated_sector_value_per_physical_input=disaggregated_values,
        )
        export_context: tuple[Path, dict, str | None, ParsedEurostatIOT] | None = None
        sector_parent_by_output: dict[str, str] | None = None
        source_parent_order: list[str] | None = None
        source_schema_args = (
            args.eurostat_dataset,
            args.eurostat_filters,
            args.eurostat_row_dim,
            args.eurostat_col_dim,
        )
        if args.write_sfc_workbook and (
            not args.sfc_template_xlsx or not all(source_schema_args)
        ):
            raise SystemExit(
                "--write-sfc-workbook requires --sfc-template-xlsx, "
                "--eurostat-dataset, --eurostat-filters, --eurostat-row-dim, "
                "and --eurostat-col-dim."
            )

        # Current Eurostat source metadata owns sector lineage independently of the
        # optional presentation path. Pipeline callers already provide this context
        # even when XLSX export is disabled, so workbook on/off cannot change the
        # canonical sector-contract artifact.
        if all(source_schema_args):
            from ecolecon_mrio.eurostat_iot import load_eurostat_iot

            eurostat_filters = _parse_json_arg(str(args.eurostat_filters))
            variant = _infer_variant_value(
                out_base=out_base,
                source_dir=z_dir,
                variant_dim=str(args.variant_dim),
                eurostat_filters=eurostat_filters,
            )
            source_filters = dict(eurostat_filters)
            if variant is not None:
                source_filters[str(args.variant_dim)] = variant
            parsed_schema = load_eurostat_iot(
                dataset=str(args.eurostat_dataset),
                filters=source_filters,
                row_dim=str(args.eurostat_row_dim),
                col_dim=str(args.eurostat_col_dim),
                storage_folder=Path(args.eurostat_storage_folder),
                force_download=False,
                force_reparse=False,
            )
            base_products = _current_eurostat_product_codes(parsed_schema)
            _, _, sector_parent_by_output, _ = _resolve_product_template_alignment(
                template_xlsx=None,
                base_product_codes=base_products,
                output_product_codes=res.Z.index.astype(str).tolist(),
                mapping_df=mapping_df_for_run,
                mapping_from_col="from_sector",
                mapping_to_col="to_sector",
                verbose=args.verbose,
            )
            source_parent_order = base_products
            if args.write_sfc_workbook:
                export_context = (
                    Path(args.sfc_template_xlsx), eurostat_filters, variant, parsed_schema
                )
        _clear_managed_aggregation_outputs(out_base)
        _vprint(args.verbose, f"Returned: aggregate_iot ({t.elapsed_s():.2f}s)")

        if args.expect_n_sectors is not None and len(res.Z.index) != int(args.expect_n_sectors):
            raise SystemExit(
                f"Aggregated output has {len(res.Z.index)} sectors, expected {int(args.expect_n_sectors)}."
            )

        sector_environmental_extensions, final_demand_environmental_extensions = (
            _partition_environmental_extensions_for_sector_reaggregation(
                environmental_extensions
            )
        )
        aggregated_environmental_extensions_actual = aggregate_environmental_extensions(
            sector_environmental_extensions,
            mapping_for_run,
            output=flow_use_total(Z, Y),
            keep_unmapped=keep_unmapped,
        )
        if exio_environmental_extensions is not None:
            if sector_environmental_extensions is None:
                raise ValueError(
                    "Raw EXIO production provenance requires paired product-axis target environmental rows."
                )
            production_groups = _production_environmental_group_domain(
                exio_environmental_extensions,
                target_environmental_extensions=sector_environmental_extensions,
            )
        else:
            production_groups = (
                None
                if sector_environmental_extensions is None
                else sector_environmental_extensions.loc[
                    :, ["boundary", "denominator_role", "extension", "stressor", "unit"]
                ].drop_duplicates(ignore_index=True)
            )
        aggregated_sector_environmental_extensions = (
            None
            if production_groups is None
            else _complete_environmental_extension_sector_grid(
                aggregated_environmental_extensions_actual,
                output=flow_use_total(res.Z, res.Y),
                production_groups=production_groups,
            )
        )
        final_demand_environmental_grid = _final_demand_environmental_rows_for_grid(
            final_demand_environmental_extensions
        )
        environmental_parts = [
            part
            for part in (
                aggregated_sector_environmental_extensions,
                final_demand_environmental_grid,
            )
            if part is not None and not part.empty
        ]
        aggregated_environmental_extensions = (
            pd.concat(environmental_parts, ignore_index=True)
            if environmental_parts
            else None
        )

        _write_df_csv(res.Z, out_base / "Z_agg.csv", verbose=args.verbose)
        _write_df_csv(res.Y, out_base / "Y_agg.csv", verbose=args.verbose)
        _write_series_csv(res.x, out_base / "x_agg.csv", verbose=args.verbose)
        if sector_parent_by_output is not None:
            _write_sector_contract(
                out_path=out_base / "sector_contract.csv",
                product_order=res.Z.index.astype(str).tolist(),
                parent_by_sector=sector_parent_by_output,
                verbose=args.verbose,
            )
            if source_parent_order is None:
                raise RuntimeError("Sector lineage was resolved without a current source-parent order.")
            from ecolecon_mrio.eurostat_target_product_schema import CPA65_CODES

            if tuple(source_parent_order) == CPA65_CODES:
                _write_source_sector_contract(
                    out_path=out_base / "source_sector_contract.csv",
                    parent_order=source_parent_order,
                    verbose=args.verbose,
                )
            else:
                _vprint(
                    args.verbose,
                    "Skipping source_sector_contract.csv: current Eurostat product domain is not CPA65.",
                )
        _write_series_csv(flow_use_total(res.Z, res.Y), out_base / "use_total_agg.csv", verbose=args.verbose)
        _write_df_csv(res.A, out_base / "A_agg.csv", verbose=args.verbose)
        if res.V is not None:
            _write_df_csv(res.V, out_base / "V_agg.csv", verbose=args.verbose)
        if res.M is not None:
            _write_df_csv(res.M, out_base / "M_agg.csv", verbose=args.verbose)
        _write_text(
            out_base / "aggregation_info.json",
            json.dumps(
                {
                    "mapping_csv": str(Path(args.mapping_csv).resolve()),
                    "n_map_entries": len(mapping_for_run),
                    "n_sectors_before": int(len(Z.index)),
                    "n_sectors_after": int(len(res.Z.index)),
                },
                indent=2,
            ),
            verbose=args.verbose,
        )
        if not edge_mapping_audit.empty:
            _write_df_csv(edge_mapping_audit, out_base / "aggregation_edge_mapping_audit.csv", verbose=args.verbose)

        # Output aggregated sector value-per-physical-input if available
        if res.aggregated_sector_value_per_physical_input is not None:
            _write_df_csv(
                res.aggregated_sector_value_per_physical_input,
                out_base / "aggregated_sector_value_per_physical_input.csv",
                verbose=args.verbose,
            )

        if aggregated_environmental_extensions is not None:
            _write_rows_csv(
                aggregated_environmental_extensions,
                out_base / "environmental_extensions.csv",
                verbose=args.verbose,
            )
        if exio_environmental_extensions is not None:
            _write_rows_csv(
                exio_environmental_extensions,
                out_base / "exio_environmental_extensions.csv",
                verbose=args.verbose,
            )

        disaggregated_co2 = _legacy_co2_production_view(
            sector_environmental_extensions,
            sector_col="disaggregated_sector",
        )
        aggregated_co2 = _legacy_co2_production_view(
            aggregated_environmental_extensions_actual,
            sector_col="aggregated_sector",
        )
        final_demand_co2 = _legacy_co2_final_demand_view(
            exio_environmental_extensions
        )

        if not args.no_xlsx:
            out_xlsx = out_base / "aggregation_report.xlsx"
            write_aggregation_report_xlsx(out_xlsx, Z_before=Z, Y_before=Y, result=res, mapping_df=mapping_df_for_run)
            _vprint(args.verbose, f"Wrote: {out_xlsx.resolve()}")

        if args.write_sfc_workbook:
            assert export_context is not None
            template, eurostat_filters, variant, parsed_schema = export_context
            out_name = str(args.sfc_outname).format(variant=(variant or "single"))
            out_path = out_base / out_name
            out_path.unlink(missing_ok=True)

            _vprint(args.verbose, "Calling: write SFC workbook ...")
            t2 = _Timer()
            _write_sfc_workbook(
                parsed=parsed_schema,
                out_path=out_path,
                template_xlsx=template,
                eurostat_dataset=str(args.eurostat_dataset),
                eurostat_filters=eurostat_filters,
                eurostat_row_dim=str(args.eurostat_row_dim),
                eurostat_col_dim=str(args.eurostat_col_dim),
                eurostat_storage_folder=Path(args.eurostat_storage_folder),
                variant_dim=str(args.variant_dim),
                variant_value=variant,
                Z_before=Z,
                Y_before=Y,
                Z_after=res.Z,
                Y_after=res.Y,
                V_after=res.V,
                M_after=res.M,
                mapping_df=mapping_df_for_run,
                mapping_from_col="from_sector",
                mapping_to_col="to_sector",
                verbose=args.verbose,
                disaggregated_sector_value_per_physical_input=disaggregated_values,
                aggregated_sector_value_per_physical_input=res.aggregated_sector_value_per_physical_input,
                disaggregated_sector_emission_intensity=disaggregated_co2,
                aggregated_sector_emission_intensity=aggregated_co2,
                exio_final_demand_emission_intensity=final_demand_co2,
                eurostat_sector_co2_mapping=None,
                output_after=res.x,
            )
            _vprint(args.verbose, f"Returned: write SFC workbook ({t2.elapsed_s():.2f}s)")

    if args.indir:
        indir = Path(args.indir)
        if not indir.exists():
            raise SystemExit(f"--indir not found: {indir}")

        variant_prefix = f"{args.variant_dim}="
        variant_dirs = sorted([p for p in indir.iterdir() if p.is_dir() and p.name.startswith(variant_prefix)])
        if variant_dirs:
            for vd in variant_dirs:
                Z_path = vd / args.z_filename
                Y_path = vd / args.y_filename
                if not (Z_path.exists() and Y_path.exists()):
                    raise SystemExit(f"Missing {args.z_filename} or {args.y_filename} under {vd}")
                out_base = outdir / vd.name
                out_base.mkdir(parents=True, exist_ok=True)
                _process_one(Z_path, Y_path, out_base)
            return 0

        Z_path = indir / args.z_filename
        Y_path = indir / args.y_filename
        if not (Z_path.exists() and Y_path.exists()):
            raise SystemExit(
                f"--indir provided but could not find {args.z_filename} and {args.y_filename} in {indir} "
                f"and no '{variant_prefix}*' subfolders were present."
            )
        _process_one(Z_path, Y_path, outdir)
        return 0

    if not (args.Z and args.Y):
        raise SystemExit("Provide either --indir OR both --Z and --Y.")
    _process_one(Path(args.Z), Path(args.Y), outdir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

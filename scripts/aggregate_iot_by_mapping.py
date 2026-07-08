#!/usr/bin/env python3
# pyright: reportReturnType=false, reportArgumentType=false, reportAbstractUsage=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportPossiblyUnboundVariable=false, reportOptionalMemberAccess=false

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_aggregation import (
    aggregate_iot,
)
from ecolecon_mrio.iot_identities import find_imports_vector, supplementary_accounting_identity_kind
from ecolecon_mrio.mapping_csv import read_aggregation_mapping


def _vprint(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


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
                "Y_total_before": [float(Y_before.to_numpy().sum())],
                "Y_total_after": [float(result.Y.to_numpy().sum())],
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
    sums = emis_df.groupby("disaggregated_sector")[["emission", "monetary_output"]].sum()
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

        target_parent = _base_sector_from_code(target_code, base_codes)
        chosen_parent = target_parent or _base_sector_from_code(source_code, base_codes)
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
            "for import flow because disaggregated_sector_co2_intensity.csv is not available. "
            "Import-origin weighted CO2 intensities are not implemented; using destination-region "
            "EXIO extension proxies would be misleading.",
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
            "for import flow because exio_final_demand_co2_intensity.csv is not available. "
            "Import-origin weighted final-demand CO2 intensities are not implemented.",
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
                "WARNING: SFC workbook re-aggregated summary is omitting production-based CO2 "
                "columns for import flow because aggregated_sector_co2_intensity.csv is not available.",
            )
        if not summary_has_use_co2:
            if not allow_missing_sfc_co2:
                raise ValueError("use-phase CO2 intensity is required for the summary sheet.")
            _vprint(
                verbose,
                "WARNING: SFC workbook re-aggregated summary is omitting use-phase CO2 "
                "columns for import flow because exio_final_demand_co2_intensity.csv is not available.",
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


def _reshape_template_table(ws, *, target_n_products: int) -> dict:
    layout = _template_table_layout(ws)
    current_n_products = int(layout["n_products"])
    delta = int(target_n_products) - current_n_products
    if delta > 0:
        ws.insert_cols(int(layout["first_fd_col"]), amount=delta)
        ws.insert_rows(int(layout["va_row"]), amount=delta)
    elif delta < 0:
        ws.delete_cols(int(layout["first_product_col"]) + target_n_products, amount=-delta)
        ws.delete_rows(int(layout["first_product_row"]) + target_n_products, amount=-delta)
    return _template_table_layout(ws)


def _supplementary_fd_row_from_current_shares(
    *,
    template_fd_values: pd.Series,
    current_fd_totals: pd.Series,
    product_total: float,
) -> pd.Series:
    out = pd.Series(index=template_fd_values.index.astype(str), dtype="float64")
    current = pd.to_numeric(current_fd_totals.reindex(out.index), errors="coerce")
    template = pd.to_numeric(template_fd_values.reindex(out.index), errors="coerce")

    split_groups = {
        "P3": ["P3_S13", "P3_S14", "P3_S15"],
        "P5": ["P51G", "P52", "P53"],
        "P6": ["P6_B0", "P6_D0", "P6_U2", "P6_U3"],
    }

    for total_code, detail_codes in split_groups.items():
        if total_code in out.index:
            if pd.notna(template.get(total_code)):
                out.loc[total_code] = float(template.loc[total_code])
            elif pd.notna(current.get(total_code)):
                out.loc[total_code] = float(current.loc[total_code])
        available_details = [code for code in detail_codes if code in out.index]
        if not available_details:
            continue
        detail_sum = float(current.reindex(available_details).fillna(0.0).sum())
        total_value = out.get(total_code)
        if pd.isna(total_value):
            total_value = template.get(total_code)
        if pd.isna(total_value):
            total_value = current.get(total_code)
        if pd.isna(total_value):
            continue
        if detail_sum > 0:
            shares = current.reindex(available_details).fillna(0.0) / detail_sum
            out.loc[available_details] = float(total_value) * shares
        else:
            out.loc[available_details] = 0.0

    for code in out.index:
        if code in {"P3", "P5", "P6", "TU", "TFU"}:
            continue
        if code == "P5M":
            if pd.notna(current.get(code)):
                out.loc[code] = float(current.loc[code])
            elif pd.notna(template.get(code)):
                out.loc[code] = float(template.loc[code])
            continue
        if pd.notna(template.get(code)) and pd.isna(out.get(code)):
            out.loc[code] = float(template.loc[code])

    tfu_value = 0.0
    for code in ("P3", "P5", "P6"):
        if code in out.index and pd.notna(out.get(code)):
            tfu_value += float(out.loc[code])
    if "TFU" in out.index:
        out.loc["TFU"] = tfu_value
    if "TU" in out.index:
        out.loc["TU"] = float(product_total) + tfu_value
    return out


def _validate_template_reaggregation(
    *,
    product_codes_order: Sequence[str],
    base_product_codes: Sequence[str],
    parents_by_output: dict[str, tuple[str, ...]],
    fd_codes_order_no_total: Sequence[str],
    Y_before: pd.DataFrame,
    tpl_ws,
    tpl_layout: dict[str, int],
    workbook_ws=None,
    workbook_layout: dict[str, int] | None = None,
    Z_ord: pd.DataFrame,
    Y_ord: pd.DataFrame,
    lower_row_vectors: dict[str, pd.Series],
    lower_fd_vectors: dict[str, pd.Series],
    source_parent_by_input: dict[str, str] | None = None,
    validate_template_parent_totals: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> None:
    product_codes = [str(code) for code in product_codes_order]
    base_codes = [str(code) for code in base_product_codes]
    fd_codes = [str(code) for code in fd_codes_order_no_total]
    tpl_product_cols = {
        code: int(tpl_layout["first_product_col"]) + idx for idx, code in enumerate(base_codes)
    }
    tpl_fd_cols = {
        code: int(tpl_layout["first_fd_col"]) + idx for idx, code in enumerate(fd_codes)
    }
    workbook_ws = tpl_ws if workbook_ws is None else workbook_ws
    workbook_layout = tpl_layout if workbook_layout is None else workbook_layout
    workbook_fd_cols = {
        code: int(workbook_layout["first_fd_col"]) + idx for idx, code in enumerate(fd_codes)
    }
    source_parent_by_input = {} if source_parent_by_input is None else dict(source_parent_by_input)

    children_by_parent: dict[str, list[str]] = {code: [] for code in base_codes}
    has_multi_parent_outputs = any(len(tuple(parents)) > 1 for parents in parents_by_output.values())
    for output_code in product_codes:
        for parent_code in parents_by_output.get(output_code, ()):
            if parent_code in children_by_parent:
                children_by_parent[parent_code].append(output_code)

    before_parent_index = [
        source_parent_by_input.get(str(code)) or _base_sector_from_code(str(code), base_codes)
        for code in Y_before.index
    ]
    valid_before_rows = [parent is not None for parent in before_parent_index]
    Y_before_valid = Y_before.loc[valid_before_rows, :].copy()
    Y_before_valid.index = [parent for parent in before_parent_index if parent is not None]
    Y_before_base = Y_before_valid.groupby(level=0).sum().reindex(index=base_codes).fillna(0.0).astype("float64")

    row_sum_Z = Z_ord.sum(axis=1).astype("float64")
    col_sum_Z = Z_ord.sum(axis=0).astype("float64")
    current_x = (row_sum_Z + Y_ord.sum(axis=1)).astype("float64")
    current_tfu = pd.Series(0.0, index=product_codes, dtype="float64")
    if {"P3", "P5", "P6"}.issubset(set(Y_ord.columns.astype(str))):
        current_tfu = (Y_ord["P3"] + Y_ord["P5"] + Y_ord["P6"]).astype("float64")
    elif "TFU" in Y_ord.columns:
        current_tfu = Y_ord["TFU"].astype("float64")
    current_tu = (row_sum_Z + current_tfu).astype("float64")
    current_fd_totals = pd.Series(
        {
            code: (
                float(current_tu.sum())
                if code == "TU"
                else float(current_tfu.sum())
                if code == "TFU"
                else float(Y_ord.get(code, pd.Series(0.0, index=product_codes)).sum())
            )
            for code in fd_codes
        },
        dtype="float64",
    )

    def _numeric_cell(row: int, col: int) -> float | None:
        raw = tpl_ws.cell(row=row, column=col).value
        if isinstance(raw, (int, float)):
            return float(raw)
        return None

    def _assert_close(label: str, actual: float, expected: float) -> None:
        if not np.isclose(float(actual), float(expected), rtol=rtol, atol=atol):
            raise ValueError(
                f"Template reaggregation validation failed for {label}: "
                f"actual={float(actual):.10f}, expected={float(expected):.10f}"
            )

    for output_idx, output_code in enumerate(product_codes, start=int(workbook_layout["first_product_row"])):
        expected_intermediate = float(row_sum_Z.loc[output_code])
        raw_intermediate = workbook_ws.cell(row=output_idx, column=int(workbook_layout["total_col"])).value
        actual_intermediate = float(raw_intermediate) if isinstance(raw_intermediate, (int, float)) else None
        if actual_intermediate is not None:
            _assert_close(
                f"product-row intermediate-total output={output_code}",
                actual_intermediate,
                expected_intermediate,
            )

        for fd_code, fd_col in workbook_fd_cols.items():
            raw_fd = workbook_ws.cell(row=output_idx, column=fd_col).value
            actual_fd = float(raw_fd) if isinstance(raw_fd, (int, float)) else None
            if actual_fd is None:
                continue
            if fd_code == "TU":
                expected_fd = float(current_tu.loc[output_code])
            elif fd_code == "TFU":
                expected_fd = float(current_tfu.loc[output_code])
            else:
                if fd_code not in Y_ord.columns:
                    continue
                expected_fd = float(Y_ord.loc[output_code, fd_code])
            _assert_close(
                f"product-row workbook fd output={output_code} fd={fd_code}",
                actual_fd,
                expected_fd,
            )

    total_row = int(workbook_layout["total_row"])
    raw_total_col = workbook_ws.cell(row=total_row, column=int(workbook_layout["total_col"])).value
    total_col_value = float(raw_total_col) if isinstance(raw_total_col, (int, float)) else None
    if total_col_value is not None:
        _assert_close("total-row intermediate-total", total_col_value, float(col_sum_Z.sum()))
    for fd_code, fd_col in workbook_fd_cols.items():
        raw_total = workbook_ws.cell(row=total_row, column=fd_col).value
        actual_total = float(raw_total) if isinstance(raw_total, (int, float)) else None
        if actual_total is None:
            continue
        expected_total = float(current_fd_totals.get(fd_code, 0.0))
        _assert_close(f"total-row fd={fd_code}", actual_total, expected_total)

    original_tfu = pd.Series(0.0, index=base_codes, dtype="float64")
    if {"P3", "P5", "P6"}.issubset(set(Y_before_base.columns.astype(str))):
        original_tfu = (Y_before_base["P3"] + Y_before_base["P5"] + Y_before_base["P6"]).astype("float64")
    elif "TFU" in Y_before_base.columns:
        original_tfu = Y_before_base["TFU"].astype("float64")

    if validate_template_parent_totals and not has_multi_parent_outputs:
        for parent_code in base_codes:
            child_codes = children_by_parent.get(parent_code, [])
            if not child_codes:
                continue
            for fd_code in fd_codes:
                if fd_code == "TU":
                    continue
                if fd_code == "TFU":
                    actual = float(current_tfu.reindex(child_codes).fillna(0.0).sum())
                    expected = float(original_tfu.loc[parent_code])
                else:
                    if fd_code not in Y_ord.columns or fd_code not in Y_before_base.columns:
                        continue
                    actual = float(Y_ord.reindex(index=child_codes, columns=[fd_code]).fillna(0.0)[fd_code].sum())
                    expected = float(Y_before_base.reindex(columns=[fd_code]).fillna(0.0).loc[parent_code, fd_code])
                _assert_close(f"product-row final-demand parent={parent_code} fd={fd_code}", actual, expected)

    for r in range(int(tpl_layout["va_row"]), tpl_ws.max_row + 1):
        label = tpl_ws.cell(row=r, column=1).value
        if label is None or str(label).strip() == "":
            continue
        label_text = str(label).strip()
        identity_kind = supplementary_accounting_identity_kind(label_text)
        row_values = lower_row_vectors.get(label_text)
        if row_values is not None:
            if identity_kind == "value_added":
                expected_by_output = (current_x - col_sum_Z).astype("float64")
            elif identity_kind == "output":
                expected_by_output = current_x.astype("float64")
            elif identity_kind == "intermediate_use":
                expected_by_output = col_sum_Z.astype("float64")
            elif identity_kind == "total_supply":
                imports_values = find_imports_vector(pd.DataFrame(lower_row_vectors).T, product_codes)
                if imports_values is None:
                    imports_values = pd.Series(0.0, index=product_codes, dtype="float64")
                expected_by_output = (current_x + imports_values.reindex(product_codes).fillna(0.0)).astype("float64")
            else:
                expected_by_output = None
            if expected_by_output is not None:
                for output_code in product_codes:
                    _assert_close(
                        f"lower-row product output={output_code} row={label_text}",
                        float(row_values.reindex(product_codes).fillna(0.0).loc[output_code]),
                        float(expected_by_output.loc[output_code]),
                    )
            else:
                if has_multi_parent_outputs or not validate_template_parent_totals:
                    continue
                for parent_code in base_codes:
                    expected = _numeric_cell(r, tpl_product_cols[parent_code])
                    if expected is None:
                        continue
                    actual = float(row_values.reindex(children_by_parent.get(parent_code, [])).fillna(0.0).sum())
                    _assert_close(f"lower-row product parent={parent_code} row={label_text}", actual, expected)

        fd_values = lower_fd_vectors.get(label_text)
        if fd_values is not None:
            if identity_kind == "intermediate_use":
                expected_fd_values = current_fd_totals
            elif identity_kind in {"value_added", "output", "total_supply"}:
                expected_fd_values = pd.Series(index=fd_codes, dtype="float64")
            else:
                continue
            for fd_code in fd_codes:
                expected = expected_fd_values.get(fd_code)
                actual = fd_values.get(fd_code)
                if pd.isna(expected) or pd.isna(actual):
                    continue
                _assert_close(f"lower-row fd row={label_text} fd={fd_code}", float(actual), float(expected))


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
    """Choose template alignment strategy for workbook export.

    Returns an effective template path, the ordered product codes, and parent mappings.
    """
    product_order = list(map(str, output_product_codes))
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

    for output_code in product_order:
        parent_candidates = mapped_parents.get(output_code)
        if parent_candidates:
            parents_by_output[output_code] = tuple(parent_candidates)
            continue
        inferred = _base_sector_from_code(output_code, base_codes)
        if inferred is not None:
            parents_by_output[output_code] = (inferred,)

    primary_parent_by_output = {code: parents[0] for code, parents in parents_by_output.items() if parents}
    if template_xlsx is not None and len(primary_parent_by_output) != len(product_order) and verbose:
        print(
            "Template workbook alignment disabled for custom aggregated sectors. "
            f"Could not infer parents for output codes: {[code for code in product_order if code not in primary_parent_by_output]}",
            flush=True,
        )
        return None, product_order, primary_parent_by_output, parents_by_output
    return template_xlsx, product_order, primary_parent_by_output, parents_by_output


def _build_template_parent_total_audit(
    *,
    variant_value: str | None,
    source_parent_by_input: dict[str, str],
    parents_by_output: dict[str, tuple[str, ...]],
    base_product_codes: Sequence[str],
    Y_before: pd.DataFrame,
    Y_after: pd.DataFrame,
    fd_codes_order_no_total: Sequence[str],
) -> pd.DataFrame:
    base_codes = [str(code) for code in base_product_codes]
    before_parent_index = [
        source_parent_by_input.get(str(code)) or _base_sector_from_code(str(code), base_codes)
        for code in Y_before.index
    ]
    valid_before_rows = [parent is not None for parent in before_parent_index]
    Y_before_valid = Y_before.loc[valid_before_rows, :].copy()
    Y_before_valid.index = [parent for parent in before_parent_index if parent is not None]
    Y_before_base = Y_before_valid.groupby(level=0).sum().reindex(index=base_codes).fillna(0.0).astype("float64")

    rows: list[dict[str, object]] = []
    product_codes = [str(code) for code in Y_after.index]
    for parent_code in base_codes:
        child_codes = [code for code in product_codes if parent_code in parents_by_output.get(code, ())]
        for fd_code in fd_codes_order_no_total:
            if fd_code == "TU":
                continue
            if fd_code == "TFU":
                if {"P3", "P5", "P6"}.issubset(set(Y_before_base.columns.astype(str))):
                    before_parent_total = float(Y_before_base.loc[parent_code, ["P3", "P5", "P6"]].sum())
                elif "TFU" in Y_before_base.columns:
                    before_parent_total = float(Y_before_base.loc[parent_code, "TFU"])
                else:
                    before_parent_total = 0.0

                if {"P3", "P5", "P6"}.issubset(set(Y_after.columns.astype(str))):
                    after_child_sum = float(
                        Y_after.reindex(index=child_codes)[["P3", "P5", "P6"]].fillna(0.0).sum().sum()
                    )
                elif "TFU" in Y_after.columns:
                    after_child_sum = float(Y_after.reindex(index=child_codes)["TFU"].fillna(0.0).sum())
                else:
                    after_child_sum = 0.0
            else:
                before_parent_total = float(Y_before_base.reindex(columns=[fd_code]).fillna(0.0).loc[parent_code, fd_code])
                after_child_sum = float(
                    Y_after.reindex(index=child_codes, columns=[fd_code]).fillna(0.0)[fd_code].sum()
                )
            rows.append(
                {
                    "variant": "" if variant_value is None else str(variant_value),
                    "parent_code": parent_code,
                    "fd_code": fd_code,
                    "before_parent_total": before_parent_total,
                    "after_child_sum": after_child_sum,
                    "difference": after_child_sum - before_parent_total,
                    "n_children": len(child_codes),
                    "child_codes": "|".join(child_codes),
                    "status": "ok" if np.isclose(after_child_sum, before_parent_total, rtol=1e-9, atol=1e-6) else "drift",
                }
            )
    return pd.DataFrame(rows)


def _write_sfc_workbook(
    *,
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
    validate_template_parent_totals: bool = True,
) -> None:
    """
    Write an Eurostat-export-shaped workbook for downstream R/SFC scripts.

    - If template_xlsx is provided: load it, overwrite 'Sheet 1' table, blank out the
      energy-balance mapping sheets, and add a sheet with our aggregation mapping.
    - Otherwise: create a minimal workbook with the same 'Sheet 1' table structure.
    """
    import openpyxl

    from ecolecon_mrio.eurostat_iot import load_eurostat_iot

    number_format = "0.00"

    def _write_num(ws, *, row: int, col: int, value: float) -> None:
        cell = ws.cell(row=row, column=col)
        cell.value = float(value)
        cell.number_format = number_format

    filt = dict(eurostat_filters)
    if variant_value is not None:
        filt[variant_dim] = str(variant_value)

    parsed = load_eurostat_iot(
        dataset=eurostat_dataset,
        filters=filt,
        row_dim=eurostat_row_dim,
        col_dim=eurostat_col_dim,
        storage_folder=eurostat_storage_folder,
        force_download=False,
        force_reparse=False,
    )
    dims = parsed.meta.get("dimensions", {})
    row_labels = dims.get(eurostat_row_dim, {})
    col_labels = dims.get(eurostat_col_dim, {})
    base_product_codes: list[str] = list(map(str, parsed.meta.get("identified_industry_columns", [])))
    base_fd_codes: list[str] = list(map(str, parsed.meta.get("identified_final_demand_columns", [])))

    if not base_product_codes:
        raise ValueError("Eurostat meta did not provide an industry/product code order.")

    target_label_by_output: dict[str, str] = {}
    if "target_label" in mapping_df.columns and mapping_to_col in mapping_df.columns:
        for _, row in mapping_df.iterrows():
            output_code = str(row[mapping_to_col])
            label = str(row.get("target_label", "")).strip()
            if output_code and label and output_code not in target_label_by_output:
                target_label_by_output[output_code] = label

    template_xlsx, product_codes_order, primary_parent_by_output, parents_by_output = _resolve_product_template_alignment(
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

    # TOTAL column is moved to the left in the Eurostat XLSX export; the remaining final-demand columns
    # appear after the product columns.
    fd_codes_order = list(base_fd_codes)
    if "TOTAL" in fd_codes_order:
        fd_codes_order_no_total = [c for c in fd_codes_order if c != "TOTAL"]
    else:
        fd_codes_order_no_total = fd_codes_order
        _vprint(verbose, "Note: Eurostat meta did not include a 'TOTAL' column; template alignment may be off.")

    # Eurostat IOT exports use a fixed ordering in the final-demand block that does not always match
    # the SDMX category index order (notably for the P5 aggregates and TU/TFU).
    eurostat_fd_canonical = [
        "P3",
        "P3_S13",
        "P3_S14",
        "P3_S15",
        "P5",
        "P5M",
        "P6",
        "P6_B0",
        "P6_D0",
        "P6_U2",
        "P6_U3",
        "P51G",
        "P52",
        "P53",
        "TU",
        "TFU",
    ]
    eurostat_xlsx_template_fd_display = [
        # Eurostat XLSX export layout (as in tmp/IOT_Austria_2021_Eurostat.xlsx):
        # includes P5 total, then P5M as a subcomponent, then exports, then P51G/P52/P53 breakdown,
        # and finally TU/TFU summary columns.
        "P3",
        "P3_S13",
        "P3_S14",
        "P3_S15",
        "P5",
        "P5M",
        "P6",
        "P6_B0",
        "P6_D0",
        "P6_U2",
        "P6_U3",
        "P51G",
        "P52",
        "P53",
        "TU",
        "TFU",
    ]
    y_cols = set(map(str, Y_after.columns))
    if template_xlsx is not None and set(eurostat_xlsx_template_fd_display).issubset(y_cols):
        fd_codes_order_no_total = eurostat_xlsx_template_fd_display
    elif set(eurostat_fd_canonical).issubset(y_cols):
        fd_codes_order_no_total = eurostat_fd_canonical

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
    x_after = (Z_after.sum(axis=1) + Y_after.sum(axis=1)).astype("float64")

    # Workbook creation/load
    if template_xlsx is not None:
        wb = openpyxl.load_workbook(template_xlsx, data_only=False)
        if "Sheet 1" not in wb.sheetnames:
            raise ValueError(f"Template workbook missing required sheet 'Sheet 1': {template_xlsx}")
        ws = wb["Sheet 1"]
        layout = _template_table_layout(ws)
        if int(layout["n_products"]) != len(product_codes_order):
            layout = _reshape_template_table(ws, target_n_products=len(product_codes_order))
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet 1"
        # Emulate the template's start position for the main table.
        layout = {
            "header_row": 11,
            "total_row": 12,
            "first_product_row": 13,
            "n_products": len(product_codes_order),
            "total_col": 2,
            "first_product_col": 3,
            "first_fd_col": 3 + len(product_codes_order),
        }
        layout["last_product_col"] = layout["first_product_col"] + layout["n_products"] - 1
        layout["header_last_col"] = layout["first_fd_col"] + len(fd_codes_order_no_total) - 1
        layout["last_fd_col"] = layout["header_last_col"]
        layout["n_fd"] = len(fd_codes_order_no_total)
        layout["last_product_row"] = layout["first_product_row"] + layout["n_products"] - 1
        layout["va_row"] = layout["last_product_row"] + 1

        # Minimal metadata + additional sheets
        wb.create_sheet("Summary")
        wb.create_sheet("Structure")
        wb.create_sheet("Mapping")
        wb.create_sheet("Aggregation")

    hr = int(layout["header_row"])
    tr = int(layout["total_row"])
    r0 = int(layout["first_product_row"])
    c_total = int(layout["total_col"])
    c_prod0 = int(layout["first_product_col"])
    c_fd0 = int(layout["first_fd_col"])

    # Header labels (row hr)
    ws.cell(row=hr, column=1).value = ws.cell(row=hr, column=1).value or f"{str(eurostat_col_dim).upper()} (Labels)"
    ws.cell(row=hr, column=c_total).value = "Total"

    for j, code in enumerate(product_codes_order):
        label = target_label_by_output.get(code) or _pretty_sector_label(row_labels.get(code, str(code)))
        ws.cell(row=hr, column=c_prod0 + j).value = label

    for j, code in enumerate(fd_codes_order_no_total):
        label = str(col_labels.get(code, code))
        ws.cell(row=hr, column=c_fd0 + j).value = label

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
    current_fd_totals = pd.Series(
        {
            code: (
                tu_total
                if code == "TU"
                else tfu_total
                if code == "TFU"
                else float(col_sums_Y.get(code, 0.0))
            )
            for code in fd_codes_order_no_total
        },
        dtype="float64",
    )
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
    supplementary_row_by_label: dict[str, str] = {}
    if V_ord is not None:
        for row_code in V_ord.index.astype(str):
            supplementary_row_by_label[row_code.strip()] = row_code
            row_label = str(row_labels.get(row_code, "")).strip()
            if row_label:
                supplementary_row_by_label[row_label] = row_code
                supplementary_row_by_label[_pretty_sector_label(row_label)] = row_code

    # Template-only: rebuild lower rows from identities and parent-sector shares, then remove
    # template-only tabs that would otherwise survive as blank sheets.
    if template_xlsx is not None:
        # Drop template-only mapping sheets instead of blanking them, so downstream consumers
        # do not receive empty workbook tabs.
        for sheet_name in ("Mapping", "Aggregation"):
            if sheet_name in wb.sheetnames:
                wb.remove(wb[sheet_name])

        # Update Summary / Structure "Stock or flow" label if present.
        if "Summary" in wb.sheetnames:
            ws_sum = wb["Summary"]
            if ws_sum.cell(row=16, column=2).value == "Sheet 1" and variant_value is not None:
                human = {"TOTAL": "Total", "DOM": "Domestic", "IMP": "Imports"}.get(str(variant_value), str(variant_value))
                ws_sum.cell(row=16, column=5).value = human
        if "Structure" in wb.sheetnames:
            ws_struct = wb["Structure"]
            if ws_struct.cell(row=6, column=2).value == "Stock or flow" and variant_value is not None:
                human = {"TOTAL": "Total", "DOM": "Domestic", "IMP": "Imports"}.get(str(variant_value), str(variant_value))
                ws_struct.cell(row=6, column=3).value = human

        tpl_wb = openpyxl.load_workbook(template_xlsx, data_only=False)
        tpl_ws = tpl_wb["Sheet 1"]
        tpl_layout = _template_table_layout(tpl_ws)
        tpl_product_cols = {
            code: int(tpl_layout["first_product_col"]) + idx for idx, code in enumerate(base_product_codes)
        }
        tpl_fd_cols = {
            code: int(tpl_layout["first_fd_col"]) + idx for idx, code in enumerate(fd_codes_order_no_total)
        }

        x_before = (Z_before.sum(axis=1) + Y_before.sum(axis=1)).astype("float64")
        output_contrib_by_parent: dict[str, dict[str, float]] = {code: {} for code in product_codes_order}
        if mapping_to_col in mapping_df.columns and mapping_from_col in mapping_df.columns:
            for _, row in mapping_df.iterrows():
                from_sector = str(row[mapping_from_col])
                to_sector = str(row[mapping_to_col])
                parent_code = source_parent_by_input.get(from_sector) or _base_sector_from_code(from_sector, base_product_codes)
                if parent_code is None or to_sector not in output_contrib_by_parent or from_sector not in x_before.index:
                    continue
                output_contrib_by_parent[to_sector][parent_code] = output_contrib_by_parent[to_sector].get(parent_code, 0.0) + float(
                    x_before.loc[from_sector]
                )

        shares_new_given_old: dict[str, dict[str, float]] = {code: {} for code in product_codes_order}
        for parent_code in base_product_codes:
            parent_outputs = [code for code in product_codes_order if parent_code in parents_by_output.get(code, ())]
            if not parent_outputs:
                continue
            total_parent = float(sum(output_contrib_by_parent.get(code, {}).get(parent_code, 0.0) for code in parent_outputs))
            if total_parent <= 0:
                total_parent = float(x_after.reindex(parent_outputs).fillna(0.0).sum())
                for code in parent_outputs:
                    value = float(x_after.get(code, 0.0))
                    shares_new_given_old.setdefault(code, {})[parent_code] = 0.0 if total_parent <= 0 else value / total_parent
            else:
                for code in parent_outputs:
                    value = float(output_contrib_by_parent.get(code, {}).get(parent_code, 0.0))
                    shares_new_given_old.setdefault(code, {})[parent_code] = value / total_parent

        source_lower_rows = list(range(int(tpl_layout["va_row"]), tpl_ws.max_row + 1))
        for offset, source_r in enumerate(source_lower_rows):
            target_r = int(layout["va_row"]) + offset
            label = tpl_ws.cell(row=source_r, column=1).value
            if label is None or str(label).strip() == "":
                continue
            label_text = str(label).strip()
            identity_kind = supplementary_accounting_identity_kind(label_text)
            v_row_code = supplementary_row_by_label.get(label_text)

            if identity_kind == "value_added":
                product_vector = (x_after.reindex(product_codes_order).fillna(0.0) - col_sums_Z.reindex(product_codes_order).fillna(0.0)).astype("float64")
            elif identity_kind == "output":
                product_vector = x_after.reindex(product_codes_order).fillna(0.0).astype("float64")
            elif identity_kind == "intermediate_use":
                product_vector = col_sums_Z.reindex(product_codes_order).fillna(0.0).astype("float64")
            elif identity_kind == "total_supply":
                imports_vector = find_imports_vector(V_ord, product_codes_order)
                if imports_vector is None:
                    imports_vector = find_imports_vector(pd.DataFrame(lower_row_vectors).T, product_codes_order)
                if imports_vector is None:
                    imports_vector = pd.Series(0.0, index=product_codes_order, dtype="float64")
                product_vector = (x_after.reindex(product_codes_order).fillna(0.0) + imports_vector.reindex(product_codes_order).fillna(0.0)).astype("float64")
            elif v_row_code is not None and V_ord is not None:
                product_vector = V_ord.loc[v_row_code].reindex(product_codes_order).fillna(0.0).astype("float64")
            else:
                product_values: dict[str, float] = {}
                for output_code in product_codes_order:
                    total_value = 0.0
                    for parent_code in parents_by_output.get(output_code, ()):
                        cell_value = tpl_ws.cell(row=source_r, column=tpl_product_cols[parent_code]).value
                        if not isinstance(cell_value, (int, float)):
                            continue
                        total_value += float(cell_value) * float(shares_new_given_old.get(output_code, {}).get(parent_code, 0.0))
                    product_values[output_code] = total_value
                product_vector = pd.Series(product_values, dtype="float64").reindex(product_codes_order).fillna(0.0)

            for j, output_code in enumerate(product_codes_order):
                _write_num(ws, row=target_r, col=c_prod0 + j, value=float(product_vector.loc[output_code]))
            lower_row_vectors[label_text] = product_vector

            template_fd_values = pd.Series(
                {code: tpl_ws.cell(row=source_r, column=tpl_fd_cols[code]).value if code in tpl_fd_cols else None for code in fd_codes_order_no_total},
                index=fd_codes_order_no_total,
                dtype="object",
            )
            numeric_template_fd = pd.to_numeric(template_fd_values, errors="coerce")
            if identity_kind == "intermediate_use":
                fd_vector = current_fd_totals.reindex(fd_codes_order_no_total).astype("float64")
            elif identity_kind in {"value_added", "output", "total_supply"}:
                fd_vector = pd.Series(index=fd_codes_order_no_total, dtype="float64")
            elif v_row_code is not None and M_ord is not None and v_row_code in M_ord.index:
                fd_vector = M_ord.loc[v_row_code].reindex(fd_codes_order_no_total).astype("float64")
            elif numeric_template_fd.notna().any():
                fd_vector = _supplementary_fd_row_from_current_shares(
                    template_fd_values=numeric_template_fd,
                    current_fd_totals=current_fd_totals,
                    product_total=float(product_vector.sum()),
                ).reindex(fd_codes_order_no_total)
            else:
                fd_vector = pd.Series(index=fd_codes_order_no_total, dtype="float64")

            for j, fd_code in enumerate(fd_codes_order_no_total):
                value = fd_vector.get(fd_code)
                if pd.isna(value):
                    continue
                _write_num(ws, row=target_r, col=c_fd0 + j, value=float(value))
            lower_fd_vectors[label_text] = fd_vector

        audit_df = _build_template_parent_total_audit(
            variant_value=variant_value,
            source_parent_by_input=source_parent_by_input,
            parents_by_output=parents_by_output,
            base_product_codes=base_product_codes,
            Y_before=Y_before,
            Y_after=Y_ord,
            fd_codes_order_no_total=fd_codes_order_no_total,
        )
        audit_path = out_path.with_name("template_parent_total_audit.csv")
        audit_df.to_csv(audit_path, index=False)
        _vprint(verbose, f"Wrote: {audit_path.resolve()}")

        _validate_template_reaggregation(
            product_codes_order=product_codes_order,
            base_product_codes=base_product_codes,
            parents_by_output=parents_by_output,
            fd_codes_order_no_total=fd_codes_order_no_total,
            Y_before=Y_before,
            tpl_ws=tpl_ws,
            tpl_layout=tpl_layout,
            workbook_ws=ws,
            workbook_layout=layout,
            Z_ord=Z_ord,
            Y_ord=Y_ord,
            lower_row_vectors=lower_row_vectors,
            lower_fd_vectors=lower_fd_vectors,
            source_parent_by_input=source_parent_by_input,
            validate_template_parent_totals=validate_template_parent_totals
            and (variant_value is None or str(variant_value).upper() == "TOTAL"),
            rtol=1e-9,
            atol=1e-6,
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
        allow_missing_sfc_co2=(
            str(variant_dim).strip() == "stk_flow"
            and str(variant_value or "").strip().upper() in {"IMP", "IMPORT", "IMPORTS"}
        ),
        verbose=verbose,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    _vprint(verbose, f"Wrote: {out_path.resolve()}")


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
    inp.add_argument("--z-filename", default="Z_split_balanced.csv")
    inp.add_argument("--y-filename", default="Y_split_balanced.csv")
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
        "--unsafe-skip-template-parent-totals",
        action="store_true",
        help="Unsafe debug-only option: skip template parent-total validation when writing the SFC workbook. Do not use in canonical pipelines.",
    )
    out.add_argument(
        "--sfc-template-xlsx",
        default=None,
        help="Optional Eurostat XLSX export to use as a template (recommended for full fidelity).",
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
        V = None
        M = None
        for v_name in ("V_split_balanced.csv", "V_split.csv"):
            v_path = z_dir / v_name
            if v_path.exists():
                _vprint(args.verbose, f"Reading: {v_path.resolve()}")
                V = _read_table(v_path).astype("float64")
                V.index = V.index.astype(str)
                V.columns = V.columns.astype(str)
                break
        for m_name in ("M_split_balanced.csv", "M_split.csv"):
            m_path = z_dir / m_name
            if m_path.exists():
                _vprint(args.verbose, f"Reading: {m_path.resolve()}")
                M = _read_table(m_path).astype("float64")
                M.index = M.index.astype(str)
                M.columns = M.columns.astype(str)
                break

        edge_metadata_path = z_dir / "split_child_edges.csv"
        mapping_for_run = dict(mapping)
        mapping_df_for_run = mapping_df.copy()
        edge_mapping_audit = pd.DataFrame()
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

        input_sectors = set(Z.index.astype(str))
        mapped_sources = set(map(str, mapping_for_run.keys()))
        unmapped = sorted(input_sectors - mapped_sources)
        if args.fail_on_unmapped and unmapped:
            raise SystemExit(
                "Aggregation mapping leaves unmapped input sectors: "
                f"{unmapped[:20]}"
            )

        if args.fail_on_unmapped_split_children:
            selected_json = (
                Path(args.split_selected_sectors_json)
                if args.split_selected_sectors_json
                else z_dir / "exio_selected_sectors.json"
            )
            selected_children = _load_split_selected_children(selected_json)
            missing_selected = sorted(selected_children - mapped_sources)
            if missing_selected:
                raise SystemExit(
                    "Aggregation mapping leaves selected split children unmapped: "
                    f"{missing_selected[:20]}"
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

        # Try to load disaggregated sector CO2/emissions intensity if they exist
        disaggregated_emissions = None
        emissions_path = z_dir / "disaggregated_sector_co2_intensity.csv"
        if emissions_path.exists():
            _vprint(args.verbose, f"Reading: {emissions_path.resolve()}")
            disaggregated_emissions = pd.read_csv(emissions_path)
            required_cols = {
                "disaggregated_sector",
                "extension",
                "stressor",
                "unit",
                "emission",
                "monetary_output",
            }
            missing_cols = required_cols - set(disaggregated_emissions.columns)
            if missing_cols:
                _vprint(
                    args.verbose,
                    "Skipping sector emissions-intensity aggregation; missing columns in "
                    f"{emissions_path.name}: {sorted(missing_cols)!r}",
                )
                disaggregated_emissions = None

        # Try to load full Eurostat-sector CO2 intensities (from EXIO concordance)
        full_sector_emissions = None
        eurostat_emissions_path = z_dir / "eurostat_sector_co2_intensity.csv"
        if eurostat_emissions_path.exists():
            _vprint(args.verbose, f"Reading: {eurostat_emissions_path.resolve()}")
            full_sector_emissions = pd.read_csv(eurostat_emissions_path)

        # Try to load EXIO final-demand direct CO2 intensity (F_Y) if it exists
        exio_final_demand_emissions = None
        fd_path = z_dir / "exio_final_demand_co2_intensity.csv"
        if fd_path.exists():
            _vprint(args.verbose, f"Reading: {fd_path.resolve()}")
            exio_final_demand_emissions = pd.read_csv(fd_path)
            required_cols = {
                "extension",
                "stressor",
                "unit",
                "category",
                "emission",
                "monetary_final_demand",
                "final_demand_emission_intensity",
            }
            missing_cols = required_cols - set(exio_final_demand_emissions.columns)
            if missing_cols:
                _vprint(
                    args.verbose,
                    "Skipping EXIO final-demand emissions-intensity; missing columns in "
                    f"{fd_path.name}: {sorted(missing_cols)!r}",
                )
                exio_final_demand_emissions = None

        _vprint(args.verbose, "Calling: aggregate_iot(...)")
        t = _Timer()
        res = aggregate_iot(
            Z,
            Y,
            mapping_for_run,
            V=V,
            M=M,
            keep_unmapped=keep_unmapped,
            create_io_system=args.create_io_system,
            compute_sector_value_per_physical_input=disaggregated_values is not None,
            disaggregated_sector_value_per_physical_input=disaggregated_values,
            compute_sector_emission_intensity=disaggregated_emissions is not None,
            disaggregated_sector_emission_intensity=disaggregated_emissions,
        )
        _vprint(args.verbose, f"Returned: aggregate_iot ({t.elapsed_s():.2f}s)")

        if args.expect_n_sectors is not None and len(res.Z.index) != int(args.expect_n_sectors):
            raise SystemExit(
                f"Aggregated output has {len(res.Z.index)} sectors, expected {int(args.expect_n_sectors)}."
            )

        combined_emissions = res.aggregated_sector_emission_intensity
        if full_sector_emissions is not None and not full_sector_emissions.empty:
            if {"aggregated_sector", "extension", "stressor", "unit"}.issubset(full_sector_emissions.columns):
                primary = combined_emissions.copy() if combined_emissions is not None else pd.DataFrame()
                fallback = full_sector_emissions.copy()
                if not primary.empty:
                    for col in ("aggregated_sector", "extension", "stressor", "unit"):
                        primary[col] = primary[col].astype(str)
                for col in ("aggregated_sector", "extension", "stressor", "unit"):
                    fallback[col] = fallback[col].astype(str)

                if not primary.empty:
                    key_cols = ["aggregated_sector", "extension", "stressor", "unit"]
                    primary_keys = set(map(tuple, primary[key_cols].to_numpy()))
                    keep_mask = [tuple(row) not in primary_keys for row in fallback[key_cols].to_numpy()]
                    fallback = fallback.loc[keep_mask]
                    combined_emissions = pd.concat([primary, fallback], ignore_index=True)
                else:
                    combined_emissions = fallback

        _write_df_csv(res.Z, out_base / "Z_agg.csv", verbose=args.verbose)
        _write_df_csv(res.Y, out_base / "Y_agg.csv", verbose=args.verbose)
        _write_series_csv(res.x, out_base / "x_agg.csv", verbose=args.verbose)
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

        # Output aggregated sector emissions intensity if available
        if combined_emissions is not None:
            _write_df_csv(
                combined_emissions,
                out_base / "aggregated_sector_co2_intensity.csv",
                verbose=args.verbose,
            )

        mapping_for_sheet = None
        if full_sector_emissions is not None and not full_sector_emissions.empty:
            mapping_for_sheet = full_sector_emissions.copy()
            if combined_emissions is not None and not combined_emissions.empty:
                key_cols = ["aggregated_sector", "extension", "stressor", "unit"]
                if set(key_cols).issubset(mapping_for_sheet.columns) and set(key_cols).issubset(
                    combined_emissions.columns
                ):
                    note_cols = ["method", "emission_intensity"]
                    cols = [c for c in note_cols if c in combined_emissions.columns]
                    merged = mapping_for_sheet.merge(
                        combined_emissions[key_cols + cols],
                        on=key_cols,
                        how="left",
                        suffixes=("", "_combined"),
                    )
                    for c in cols:
                        if c not in mapping_for_sheet.columns and c in merged.columns:
                            merged[c] = merged[c]
                    mapping_for_sheet = merged

                    mapping_keys = set(map(tuple, mapping_for_sheet[key_cols].to_numpy()))
                    extra = combined_emissions.copy()
                    extra_keys = set(map(tuple, extra[key_cols].to_numpy()))
                    missing_keys = [k for k in extra_keys if k not in mapping_keys]
                    if missing_keys:
                        extra = extra[[*key_cols, *[c for c in extra.columns if c not in key_cols]]]
                        keep_mask = [tuple(row) in missing_keys for row in extra[key_cols].to_numpy()]
                        extra = extra.loc[keep_mask].copy()
                        if "exio_sectors" not in extra.columns:
                            extra["exio_sectors"] = ""
                        if "n_exio_sectors" not in extra.columns:
                            extra["n_exio_sectors"] = 0
                        mapping_for_sheet = pd.concat([mapping_for_sheet, extra], ignore_index=True)

        if exio_final_demand_emissions is not None:
            _write_df_csv(
                exio_final_demand_emissions,
                out_base / "exio_final_demand_co2_intensity.csv",
                verbose=args.verbose,
            )

        if not args.no_xlsx:
            out_xlsx = out_base / "aggregation_report.xlsx"
            write_aggregation_report_xlsx(out_xlsx, Z_before=Z, Y_before=Y, result=res, mapping_df=mapping_df_for_run)
            _vprint(args.verbose, f"Wrote: {out_xlsx.resolve()}")

        if args.write_sfc_workbook:
            if not (args.eurostat_dataset and args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim):
                raise SystemExit(
                    "--write-sfc-workbook requires --eurostat-dataset, --eurostat-filters, "
                    "--eurostat-row-dim, and --eurostat-col-dim."
                )
            template = Path(args.sfc_template_xlsx) if args.sfc_template_xlsx else None

            eurostat_filters = _parse_json_arg(str(args.eurostat_filters))
            variant = _infer_variant_value(
                out_base=out_base,
                source_dir=z_dir,
                variant_dim=str(args.variant_dim),
                eurostat_filters=eurostat_filters,
            )
            out_name = str(args.sfc_outname).format(variant=(variant or "single"))
            out_path = out_base / out_name

            _vprint(args.verbose, "Calling: write SFC workbook ...")
            t2 = _Timer()
            _write_sfc_workbook(
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
                validate_template_parent_totals=not bool(args.unsafe_skip_template_parent_totals),
                disaggregated_sector_value_per_physical_input=disaggregated_values,
                aggregated_sector_value_per_physical_input=res.aggregated_sector_value_per_physical_input,
                disaggregated_sector_emission_intensity=disaggregated_emissions,
                aggregated_sector_emission_intensity=combined_emissions,
                exio_final_demand_emission_intensity=exio_final_demand_emissions,
                eurostat_sector_co2_mapping=mapping_for_sheet,
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

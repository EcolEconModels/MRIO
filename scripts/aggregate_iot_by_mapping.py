#!/usr/bin/env python3

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
    return s


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

    # Add CO2 monetary output + production-based intensity columns for disaggregated sectors
    if disaggregated_sector_emission_intensity is None or disaggregated_sector_emission_intensity.empty:
        raise ValueError("disaggregated_sector_emission_intensity is required for production-based CO2 intensity.")
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
    if exio_final_demand_emission_intensity is None or exio_final_demand_emission_intensity.empty:
        raise ValueError("exio_final_demand_emission_intensity is required for use-phase CO2 intensity.")
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

    # Add aggregated sector summary with domestic/exports/stocks + use-phase intensity (F_Y)
    if aggregated_Y is not None and not aggregated_Y.empty:
        if aggregated_sector_emission_intensity is None or aggregated_sector_emission_intensity.empty:
            raise ValueError("aggregated_sector_emission_intensity is required for the summary sheet.")
        if fy_dom_intensity_scalar is None:
            raise ValueError("use-phase CO2 intensity is required for the summary sheet.")
        agg_domestic_fd, agg_exports, agg_stocks = _split_final_demand(aggregated_Y)

        sheet_title = "Re-aggregated Summary"
        if sheet_title in wb.sheetnames:
            wb.remove(wb[sheet_title])
        ws_s = wb.create_sheet(sheet_title)

        prod_unit_label = "mixed units"
        if "unit" in aggregated_sector_emission_intensity.columns:
            uvals = aggregated_sector_emission_intensity["unit"].dropna().unique().tolist()
            if len(uvals) == 1:
                prod_unit_label = str(uvals[0]).strip()

        ws_s.append(
            [
                mapping_to_col.replace("_", " ").title(),
                "Monetary output (M€)",
                f"Mean CO2 intensity ({prod_unit_label}/M€)",
                "Domestic final demand (M€)",
                "Exports (M€)",
                "Stocks & valuables (M€)",
                f"Mean CO2 intensity for domestic demand ({fy_unit_label}/M€)",
            ]
        )
        agg_emis_df = aggregated_sector_emission_intensity.copy()
        if "aggregated_sector" in agg_emis_df.columns:
            agg_emis_df["aggregated_sector"] = agg_emis_df["aggregated_sector"].astype(str)
        intensity_by_sector = None
        output_by_sector = None
        if "aggregated_sector" in agg_emis_df.columns and "emission_intensity" in agg_emis_df.columns:
            intensity_by_sector = agg_emis_df.set_index("aggregated_sector")["emission_intensity"]

        if aggregated_x is not None and not aggregated_x.empty:
            output_by_sector = aggregated_x.copy()
            output_by_sector.index = output_by_sector.index.astype(str)
        elif "aggregated_sector" in agg_emis_df.columns and "monetary_output" in agg_emis_df.columns:
            output_by_sector = agg_emis_df.set_index("aggregated_sector")["monetary_output"]

        if aggregated_sector_order:
            sector_index = pd.Index(list(map(str, aggregated_sector_order)))
        elif output_by_sector is not None:
            sector_index = output_by_sector.index
        else:
            sector_index = aggregated_Y.index.astype(str)

        for agg_sector in sector_index:
            agg_sector_s = str(agg_sector)
            display_sector = (
                aggregated_sector_labels.get(agg_sector_s)
                if aggregated_sector_labels is not None
                else agg_sector_s
            )
            ws_s.append(
                [
                    display_sector,
                    output_by_sector.get(agg_sector_s) if output_by_sector is not None else None,
                    intensity_by_sector.get(agg_sector_s) if intensity_by_sector is not None else None,
                    agg_domestic_fd.get(agg_sector_s) if agg_domestic_fd is not None else None,
                    agg_exports.get(agg_sector_s) if agg_exports is not None else None,
                    agg_stocks.get(agg_sector_s) if agg_stocks is not None else None,
                    fy_dom_intensity_scalar,
                ]
            )
        for row in ws_s.iter_rows(min_row=2, min_col=2, max_col=7):
            for cell in row:
                if cell.value is None or pd.isna(cell.value):
                    continue
                if cell.column in (2, 4, 5, 6):
                    cell.number_format = "0.00"
                else:
                    cell.number_format = "0.0000"
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


def _resolve_product_template_alignment(
    *,
    template_xlsx: Path | None,
    base_product_codes: list[str],
    output_product_codes: list[str],
    verbose: bool,
) -> tuple[Path | None, dict[str, str], list[str]]:
    """Choose template alignment strategy for workbook export.

    Returns an effective template path, a replacement map, and the ordered product codes.
    If the output sectors no longer match the Eurostat template order and no explicit
    replacement heuristic applies, template mode is disabled and the caller should
    write a generic workbook instead.
    """
    have = set(map(str, output_product_codes))
    missing = [c for c in base_product_codes if c not in have]
    extra = [c for c in list(map(str, output_product_codes)) if c not in set(base_product_codes)]

    replace_map: dict[str, str] = {}
    if not missing and not extra:
        return template_xlsx, replace_map, list(base_product_codes)

    missing_set = set(missing)
    extra_set = set(extra)
    if missing_set == {"CPA_C19", "CPA_D"} and extra_set == {"ENERGY_GREEN", "ENERGY_BROWN"}:
        replace_map = {"CPA_C19": "ENERGY_BROWN", "CPA_D": "ENERGY_GREEN"}
        return template_xlsx, replace_map, [replace_map.get(c, c) for c in base_product_codes]

    if template_xlsx is not None and verbose:
        print(
            "Template workbook alignment disabled for custom aggregated sectors. "
            f"Missing base codes: {missing}; extra output codes: {extra}",
            flush=True,
        )
    return None, {}, list(map(str, output_product_codes))


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

    template_xlsx, replace_map, product_codes_order = _resolve_product_template_alignment(
        template_xlsx=template_xlsx,
        base_product_codes=base_product_codes,
        output_product_codes=Z_after.index.astype(str).tolist(),
        verbose=verbose,
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
    x_after = (Z_after.sum(axis=1) + Y_after.sum(axis=1)).astype("float64")

    # Derive output-share-based allocation shares to split template account rows from old sectors
    # (e.g. CPA_C19/CPA_D) into the new aggregates (e.g. ENERGY_*).
    #
    # Important: we need shares that partition each old sector across new ones, i.e. share(new|old),
    # not share(old|new), otherwise old-sector totals get double-counted.
    shares_new_given_old: dict[str, dict[str, float]] = {}
    if replace_map and mapping_to_col in mapping_df.columns and mapping_from_col in mapping_df.columns:
        x_before = (Z_before.sum(axis=1) + Y_before.sum(axis=1)).astype("float64")
        old_codes = list(replace_map.keys())
        new_codes = sorted(set(replace_map.values()))

        contrib: dict[str, dict[str, float]] = {old: {new: 0.0 for new in new_codes} for old in old_codes}
        for _, row in mapping_df.iterrows():
            from_s = str(row[mapping_from_col])
            to_s = str(row[mapping_to_col])
            if to_s not in new_codes:
                continue
            for old in old_codes:
                if from_s.startswith(f"{old}::"):
                    if from_s in x_before.index:
                        contrib[old][to_s] += float(x_before.loc[from_s])
                    break

        for old in old_codes:
            total_old = float(sum(contrib[old].values()))
            for new in new_codes:
                if total_old > 0:
                    shares_new_given_old.setdefault(new, {})[old] = float(contrib[old][new]) / total_old
                else:
                    shares_new_given_old.setdefault(new, {})[old] = 1.0 / max(1, len(new_codes))

    # Workbook creation/load
    if template_xlsx is not None:
        wb = openpyxl.load_workbook(template_xlsx, data_only=False)
        if "Sheet 1" not in wb.sheetnames:
            raise ValueError(f"Template workbook missing required sheet 'Sheet 1': {template_xlsx}")
        ws = wb["Sheet 1"]
        layout = _template_table_layout(ws)
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

    # Validate product count matches template layout in template mode.
    if template_xlsx is not None and int(layout["n_products"]) != len(product_codes_order):
        raise ValueError(
            f"Template has {layout['n_products']} product rows, but aggregated result has {len(product_codes_order)} "
            "sectors. This script currently expects these to match."
        )

    hr = int(layout["header_row"])
    tr = int(layout["total_row"])
    r0 = int(layout["first_product_row"])
    c_total = int(layout["total_col"])
    c_prod0 = int(layout["first_product_col"])
    c_fd0 = int(layout["first_fd_col"])
    n_prod = len(product_codes_order)
    n_fd = len(fd_codes_order_no_total)

    # Header labels (row hr)
    ws.cell(row=hr, column=1).value = ws.cell(row=hr, column=1).value or f"{str(eurostat_col_dim).upper()} (Labels)"
    ws.cell(row=hr, column=c_total).value = "Total"

    if template_xlsx is None:
        for j, code in enumerate(product_codes_order):
            label = _pretty_sector_label(row_labels.get(code, str(code)))
            ws.cell(row=hr, column=c_prod0 + j).value = label

        for j, code in enumerate(fd_codes_order_no_total):
            label = str(col_labels.get(code, code))
            ws.cell(row=hr, column=c_fd0 + j).value = label
    else:
        # Template mode: preserve template header labels (only rename new aggregates).
        for j, code in enumerate(product_codes_order):
            if str(code).startswith("ENERGY_"):
                label = _pretty_sector_label(code)
                ws.cell(row=hr, column=c_prod0 + j).value = label

    sector_label_map = {
        str(code): _pretty_sector_label(row_labels.get(str(code), str(code)))
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
        if template_xlsx is None:
            ws.cell(row=excel_r, column=1).value = _pretty_sector_label(row_labels.get(row_code, str(row_code)))
        else:
            if str(row_code).startswith("ENERGY_"):
                ws.cell(row=excel_r, column=1).value = _pretty_sector_label(row_code)

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

    # Template-only: update account rows for the new energy sectors and remove template-only sheets
    # that would otherwise survive as blank tabs in the exported workbook.
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

        if replace_map:
            # Use the original template (before overwrite) to read the old energy sector columns.
            tpl_ws = openpyxl.load_workbook(template_xlsx, data_only=False)["Sheet 1"]
            tpl_layout = _template_table_layout(tpl_ws)
            tpl_hr = int(tpl_layout["header_row"])
            tpl_c_prod0 = int(tpl_layout["first_product_col"])
            tpl_n_prod = int(tpl_layout["n_products"])

            def _tpl_find_col_by_label(label: str) -> int | None:
                for j in range(tpl_n_prod):
                    if tpl_ws.cell(row=tpl_hr, column=tpl_c_prod0 + j).value == label:
                        return tpl_c_prod0 + j
                return None

            old_cols = {
                old_code: _tpl_find_col_by_label(str(row_labels.get(old_code, old_code)))
                for old_code in replace_map.keys()
            }

            new_cols: dict[str, int] = {}
            for new_code in set(replace_map.values()):
                new_label = _pretty_sector_label(row_labels.get(new_code, new_code))
                for j in range(n_prod):
                    if ws.cell(row=hr, column=c_prod0 + j).value == new_label:
                        new_cols[new_code] = c_prod0 + j
                        break

            for old_code, col in old_cols.items():
                if col is None:
                    raise ValueError(f"Could not locate template column for old sector {old_code!r}.")
            for new_code, col in new_cols.items():
                if col is None:
                    raise ValueError(f"Could not locate output column for new sector {new_code!r}.")

            # Update account rows by splitting old CPA_C19 and CPA_D columns into new ones using output shares.
            for r in range(int(layout["va_row"]), ws.max_row + 1):
                label = ws.cell(row=r, column=1).value
                if label is None or str(label).strip() == "":
                    continue

                v_c19 = tpl_ws.cell(row=r, column=old_cols["CPA_C19"]).value
                v_d = tpl_ws.cell(row=r, column=old_cols["CPA_D"]).value
                if not isinstance(v_c19, (int, float)) or not isinstance(v_d, (int, float)):
                    continue

                for new_code, new_col in new_cols.items():
                    share = shares_new_given_old.get(new_code, {})
                    ws.cell(row=r, column=new_col).value = float(v_c19) * float(share.get("CPA_C19", 0.5)) + float(v_d) * float(
                        share.get("CPA_D", 0.5)
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
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    _vprint(verbose, f"Wrote: {out_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Aggregate (re-group) an IO table (Z,Y) using the canonical annotated mapping CSV format.")
    p.add_argument("--verbose", action="store_true")

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

        # Try to load disaggregated sector value-per-physical-input if they exist
        disaggregated_values = None
        z_dir = Z_path.parent
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
            mapping,
            keep_unmapped=True,
            create_io_system=args.create_io_system,
            compute_sector_value_per_physical_input=disaggregated_values is not None,
            disaggregated_sector_value_per_physical_input=disaggregated_values,
            compute_sector_emission_intensity=disaggregated_emissions is not None,
            disaggregated_sector_emission_intensity=disaggregated_emissions,
        )
        _vprint(args.verbose, f"Returned: aggregate_iot ({t.elapsed_s():.2f}s)")

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
        _write_text(
            out_base / "aggregation_info.json",
            json.dumps(
                {
                    "mapping_csv": str(Path(args.mapping_csv).resolve()),
                    "n_map_entries": len(mapping),
                    "n_sectors_before": int(len(Z.index)),
                    "n_sectors_after": int(len(res.Z.index)),
                },
                indent=2,
            ),
            verbose=args.verbose,
        )

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
            write_aggregation_report_xlsx(out_xlsx, Z_before=Z, Y_before=Y, result=res, mapping_df=mapping_df)
            _vprint(args.verbose, f"Wrote: {out_xlsx.resolve()}")

        if args.write_sfc_workbook:
            if not (args.eurostat_dataset and args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim):
                raise SystemExit(
                    "--write-sfc-workbook requires --eurostat-dataset, --eurostat-filters, "
                    "--eurostat-row-dim, and --eurostat-col-dim."
                )
            template = Path(args.sfc_template_xlsx) if args.sfc_template_xlsx else None

            # Variant name: derived from output folder name when batch-processing (e.g. stk_flow=DOM).
            variant = None
            if out_base.name.startswith(f"{args.variant_dim}="):
                variant = out_base.name.split("=", 1)[1]
            out_name = str(args.sfc_outname).format(variant=(variant or "single"))
            out_path = out_base / out_name

            _vprint(args.verbose, "Calling: write SFC workbook ...")
            t2 = _Timer()
            _write_sfc_workbook(
                out_path=out_path,
                template_xlsx=template,
                eurostat_dataset=str(args.eurostat_dataset),
                eurostat_filters=_parse_json_arg(str(args.eurostat_filters)),
                eurostat_row_dim=str(args.eurostat_row_dim),
                eurostat_col_dim=str(args.eurostat_col_dim),
                eurostat_storage_folder=Path(args.eurostat_storage_folder),
                variant_dim=str(args.variant_dim),
                variant_value=variant,
                Z_before=Z,
                Y_before=Y,
                Z_after=res.Z,
                Y_after=res.Y,
                mapping_df=mapping_df,
                mapping_from_col="from_sector",
                mapping_to_col="to_sector",
                verbose=args.verbose,
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

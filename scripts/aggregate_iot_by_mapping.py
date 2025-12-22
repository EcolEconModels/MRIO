#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from ecolecon_mrio.iot_aggregation import (
    aggregate_iot,
    read_sector_aggregation_mapping_csv,
    write_aggregation_report_xlsx,
)


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


def _write_excel_mapping_sheet(wb, *, sheet_name: str, mapping_df: pd.DataFrame) -> None:
    import openpyxl

    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        wb.remove(ws)
    ws = wb.create_sheet(sheet_name)

    headers = list(map(str, mapping_df.columns))
    ws.append(headers)
    for row in mapping_df.itertuples(index=False):
        ws.append([("" if v is None else str(v)) for v in row])

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{openpyxl.utils.get_column_letter(len(headers))}{len(mapping_df) + 1}"


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

    # Determine how to place new aggregates (e.g. ENERGY_GREEN/BROWN) into the Eurostat order.
    # For generality, we infer missing/extra codes and use a small heuristic for the common
    # CPA_C19/CPA_D -> ENERGY_* case.
    have = set(map(str, Z_after.index))
    missing = [c for c in base_product_codes if c not in have]
    extra = [c for c in Z_after.index.astype(str).tolist() if c not in set(base_product_codes)]

    replace_map: dict[str, str] = {}
    if missing or extra:
        missing_set = set(missing)
        extra_set = set(extra)
        if missing_set == {"CPA_C19", "CPA_D"} and extra_set == {"ENERGY_GREEN", "ENERGY_BROWN"}:
            replace_map = {"CPA_C19": "ENERGY_BROWN", "CPA_D": "ENERGY_GREEN"}
        else:
            raise ValueError(
                "Cannot align aggregated sector codes to Eurostat template order.\n"
                f"Missing (in aggregated result, but present in Eurostat order): {missing}\n"
                f"Extra (in aggregated result, but not present in Eurostat order): {extra}\n"
                "Provide a template workbook and/or ensure your aggregation produces exactly the Eurostat sector codes, "
                "or extend the script with an explicit replacement mapping for your use case."
            )

    product_codes_order = [replace_map.get(c, c) for c in base_product_codes]

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

    # Template-only: update account rows for the new energy sectors and blank out energy-balance sheets.
    if template_xlsx is not None:
        # Blank mapping sheets as requested (energy balances mapping left empty).
        for sheet_name in ("Mapping", "Aggregation"):
            if sheet_name in wb.sheetnames:
                _clear_worksheet(wb[sheet_name])

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

    _write_excel_mapping_sheet(wb, sheet_name="IO Aggregation Mapping", mapping_df=mapping_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    _vprint(verbose, f"Wrote: {out_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Aggregate (re-group) an IO table (Z,Y) using a CSV mapping.")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--mapping-csv", required=True, help="CSV with columns from_sector,to_sector[,enabled,notes].")
    p.add_argument("--mapping-from-col", default="from_sector")
    p.add_argument("--mapping-to-col", default="to_sector")
    p.add_argument("--mapping-enabled-col", default="enabled")

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

    mapping, mapping_df = read_sector_aggregation_mapping_csv(
        args.mapping_csv,
        from_col=args.mapping_from_col,
        to_col=args.mapping_to_col,
        enabled_col=args.mapping_enabled_col,
    )

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

        _vprint(args.verbose, "Calling: aggregate_iot(...)")
        t = _Timer()
        res = aggregate_iot(Z, Y, mapping, keep_unmapped=True, create_io_system=args.create_io_system)
        _vprint(args.verbose, f"Returned: aggregate_iot ({t.elapsed_s():.2f}s)")

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
                mapping_from_col=str(args.mapping_from_col),
                mapping_to_col=str(args.mapping_to_col),
                verbose=args.verbose,
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

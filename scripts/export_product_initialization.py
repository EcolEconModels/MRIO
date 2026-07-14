#!/usr/bin/env python3
"""Render official product-parent initialization data on the final IOT sectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
import pandas as pd

from ecolecon_mrio.sector_allocation import allocate_parent_matrix_by_output


def _parse_column_specs(specs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in specs:
        name, sep, column = spec.partition(":")
        if not sep or not name or not column.isdigit() or int(column) < 1:
            raise ValueError(f"Expected COLUMN:EXCEL_COLUMN, got {spec!r}.")
        out[name] = int(column)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-input-csv", required=True, type=Path)
    parser.add_argument("--source-sector-column", default="sector")
    parser.add_argument("--iot-output-csv", required=True, type=Path, help="Final x_agg.csv from the fuels pipeline.")
    parser.add_argument("--template-xlsx", required=True, type=Path)
    parser.add_argument("--output-xlsx", required=True, type=Path)
    parser.add_argument("--sheet", default="Sheet 1")
    parser.add_argument("--first-data-row", type=int, default=2)
    parser.add_argument("--sector-column", type=int, default=1)
    parser.add_argument("--value-column", action="append", default=[], metavar="CSV_COLUMN:EXCEL_COLUMN")
    parser.add_argument("--share-column", action="append", default=[], metavar="CSV_COLUMN:EXCEL_COLUMN")
    parser.add_argument("--diagnostic-csv", type=Path)
    args = parser.parse_args()

    columns = _parse_column_specs(args.value_column + args.share_column)
    if not columns:
        parser.error("Provide at least one --value-column or --share-column.")
    source = pd.read_csv(args.official_input_csv).set_index(args.source_sector_column)
    missing_columns = sorted(set(columns) - set(source.columns))
    if missing_columns:
        parser.error(f"Official input is missing requested column(s): {missing_columns!r}.")
    final_output = pd.read_csv(args.iot_output_csv, index_col=0).iloc[:, 0]
    allocated, audit = allocate_parent_matrix_by_output(source[list(columns)], final_output=final_output)

    share_columns = {spec.partition(":")[0] for spec in args.share_column}
    for column in share_columns:
        total = float(allocated[column].sum())
        if total <= 0:
            raise ValueError(f"Cannot normalize non-positive allocated share source {column!r}.")
        allocated[column] /= total

    workbook = openpyxl.load_workbook(args.template_xlsx)
    if args.sheet not in workbook.sheetnames:
        raise ValueError(f"Template has no sheet {args.sheet!r}.")
    sheet = workbook[args.sheet]
    for offset, (sector, row) in enumerate(allocated.iterrows()):
        excel_row = args.first_data_row + offset
        sheet.cell(excel_row, args.sector_column).value = sector
        for name, excel_column in columns.items():
            sheet.cell(excel_row, excel_column).value = float(row[name])
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output_xlsx)

    if args.diagnostic_csv:
        args.diagnostic_csv.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.diagnostic_csv, index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecolecon_mrio.transition_checks import percent_deviation, summarize_scope_totals


def _parse_columns(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        out = json.loads(text)
        if not isinstance(out, list) or not all(isinstance(item, str) for item in out):
            raise ValueError("Column list JSON must be an array of strings.")
        return out
    return [item.strip() for item in text.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check workbook energy-group totals by scope.")
    parser.add_argument("--workbook", required=True, help="Path to the workbook to inspect.")
    parser.add_argument("--sheet", required=True, help="Worksheet name.")
    parser.add_argument("--group-name", required=True, help="Label used in the output JSON.")
    parser.add_argument("--group-cols", required=True, help="Comma-separated or JSON list of column names.")
    parser.add_argument("--reference-total", type=float, default=None, help="Optional reference total to compare against.")
    parser.add_argument("--sector-col", default="Sector", help="Sector label column.")
    parser.add_argument("--household-label", default="Households", help="Label used for the household row.")
    parser.add_argument("--sheet-header", type=int, default=0, help="Header row for pandas.read_excel.")
    parser.add_argument("--scale-divisor", type=float, default=1000.0, help="Divisor applied to totals, e.g. 1000 for GWh->TWh.")
    args = parser.parse_args(argv)

    frame = pd.read_excel(Path(args.workbook), sheet_name=str(args.sheet), header=int(args.sheet_header))
    columns = _parse_columns(str(args.group_cols))
    totals = summarize_scope_totals(
        frame,
        columns,
        sector_col=str(args.sector_col),
        household_label=str(args.household_label),
        scale_divisor=float(args.scale_divisor),
    )
    payload: dict[str, object] = {
        "group_name": str(args.group_name),
        "group_columns": columns,
        **totals.to_dict(),
    }
    if args.reference_total is not None:
        payload["reference_total"] = float(args.reference_total)
        payload["all_total_pct_deviation"] = percent_deviation(totals.all_total, float(args.reference_total))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
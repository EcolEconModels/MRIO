#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _flat_diagnostics_common import (
    build_eurostat_product_use_reference,
    build_signed_value_summary,
    default_output_dir,
    eurostat_order_map,
    load_eurostat_iot_bundle,
    load_exio_slice,
    load_weighted_mapping,
    mapping_label_maps,
    repo_root_from_file,
    sector_total,
    write_csv,
)


FLOW_SPECS = {
    "dom": ("DOM", "domestic", "at2020_dom_mapping_detail.csv"),
    "imp": ("IMP", "imports", "at2020_imp_mapping_detail.csv"),
    "total": ("TOTAL", "total", "at2020_total_mapping_detail.csv"),
}


def build_totals_row(df: pd.DataFrame, *, flow_code: str) -> pd.DataFrame:
    unique_sources = df.drop_duplicates(subset=["exio_source_code"])
    unique_targets = df.drop_duplicates(subset=["eurostat_sector_code"])
    euro_total = sector_total(unique_targets["eurostat_sector_value_meur_2020"])
    mapped_total = sector_total(unique_targets["mapped_exio_sector_aggregate_meur_2020"])
    gap = mapped_total - euro_total
    totals = {
        "flow": flow_code,
        "eurostat_order": np.nan,
        "eurostat_sector_code": "TOTAL",
        "eurostat_sector_name": "All sectors",
        "exio_source_code": "",
        "exio_source_name": "",
        "mapping_weight": np.nan,
        "exio_sector_value_meur_2020": sector_total(unique_sources["exio_sector_value_meur_2020"]),
        "mapped_exio_split_value_meur_2020": sector_total(df["mapped_exio_split_value_meur_2020"]),
        "eurostat_sector_value_meur_2020": euro_total,
        "mapped_exio_sector_aggregate_meur_2020": mapped_total,
        "gap_meur_2020": gap,
        "gap_pct_of_eurostat_2020": (100.0 * gap / euro_total) if euro_total else np.nan,
        "provenance": "",
        "certainty": "",
        "notes": "Totals row: source-sector totals are summed over unique EXIO source codes; Eurostat and mapped sector totals are summed over unique Eurostat sectors. Eurostat reference values are built from atomic final-demand columns only, with P6 excluded, so the comparison stays on domestic absorption and avoids double counting Eurostat subtotal columns.",
    }
    return pd.DataFrame([totals])


def generate_flow_detail(repo_root: Path, *, geo: str, region: str, year: int, flow_key: str) -> pd.DataFrame:
    flow_code, exio_flow, _ = FLOW_SPECS[flow_key]
    mapping_df = load_weighted_mapping(repo_root)
    target_labels, source_labels = mapping_label_maps(mapping_df)
    parsed_iot, _eurostat_output, _ = load_eurostat_iot_bundle(repo_root, geo=geo, year=year, flow_code=flow_code)
    eurostat_output = build_eurostat_product_use_reference(parsed_iot, exclude_final_demand_codes=("P6",))
    exio_slice = load_exio_slice(repo_root, region=region, year=year, flow_name=exio_flow)
    detail, summary = build_signed_value_summary(
        repo_root,
        source_values=exio_slice.x_scope,
        reference_values=eurostat_output,
        mapping_df=mapping_df,
    )
    order_map = eurostat_order_map(eurostat_output.index)
    detail = detail.merge(summary, left_on="aggregate_sector", right_on="target_sector", how="left")
    detail["flow"] = flow_code
    detail["eurostat_order"] = detail["aggregate_sector"].map(order_map)
    detail["eurostat_sector_code"] = detail["aggregate_sector"]
    detail["eurostat_sector_name"] = detail["aggregate_sector"].map(target_labels)
    detail["exio_source_code"] = detail["exio_sector"]
    detail["exio_source_name"] = detail["exio_sector"].map(source_labels)
    detail["mapping_weight"] = pd.to_numeric(detail["weight"], errors="coerce")
    detail["exio_sector_value_meur_2020"] = pd.to_numeric(detail["source_value"], errors="coerce")
    detail["mapped_exio_split_value_meur_2020"] = pd.to_numeric(detail["allocated_value"], errors="coerce")
    detail["eurostat_sector_value_meur_2020"] = pd.to_numeric(detail["reference_total"], errors="coerce")
    detail["mapped_exio_sector_aggregate_meur_2020"] = pd.to_numeric(detail["mapped_total"], errors="coerce")
    detail["gap_meur_2020"] = pd.to_numeric(detail["gap"], errors="coerce")
    detail["gap_pct_of_eurostat_2020"] = pd.to_numeric(detail["gap_pct"], errors="coerce")
    detail["certainty"] = detail["certainty"].fillna("")
    detail = detail.sort_values(["eurostat_order", "eurostat_sector_code", "mapped_exio_split_value_meur_2020"], ascending=[True, True, False])
    out = detail[
        [
            "flow",
            "eurostat_order",
            "eurostat_sector_code",
            "eurostat_sector_name",
            "exio_source_code",
            "exio_source_name",
            "mapping_weight",
            "exio_sector_value_meur_2020",
            "mapped_exio_split_value_meur_2020",
            "eurostat_sector_value_meur_2020",
            "mapped_exio_sector_aggregate_meur_2020",
            "gap_meur_2020",
            "gap_pct_of_eurostat_2020",
            "provenance",
            "certainty",
            "notes",
        ]
    ].copy()
    return pd.concat([out, build_totals_row(out, flow_code=flow_code)], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the 2020 DOM/IMP/TOTAL mapping-detail CSVs.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--region", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = repo_root_from_file(__file__)
    output_dir = args.output_dir or default_output_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for flow_key, (_, _, filename) in FLOW_SPECS.items():
        df = generate_flow_detail(repo_root, geo=args.geo, region=args.region, year=args.year, flow_key=flow_key)
        write_csv(df, output_dir / filename)

    print(f"Wrote flow mapping details to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook

from ecolecon_mrio.eurostat_air_emissions import load_eurostat_air_emissions_by_nace
from ecolecon_mrio.eurostat_energy_balances import (
    PRIMARY_SOURCES,
    DEFAULT_PRIMARY_SOURCE_BUCKETS_PATH,
    build_postgrowth_energy_balance,
    build_tabelle1_from_eurostat,
    compute_mapping_shares_from_outputs,
    disaggregate_unrolled_sources,
    validate_energy_balance_closure,
)
from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import build_eurostat_product_use_reference, load_eurostat_iot
from ecolecon_mrio.mapping_csv import read_category_aggregate_mapping, read_category_mapping_frame, read_source_bucket_groups


def _norm(s: str) -> str:
    return (
        str(s)
        .strip()
        .casefold()
        .replace("&", "and")
        .replace("\u00a0", " ")
    )


def _xlsx_sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Excel does not receive NaN/Inf values (Excel may "repair" files containing NaN literals).
    """
    out = df.replace([np.inf, -np.inf], np.nan)
    # Cast to object so missing values can become real blanks (None).
    return out.astype(object).where(pd.notna(out), None)


def _write_xlsx(path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> None:
    # Use normal (non-write_only) mode so Excel receives full worksheet metadata
    # (notably the <dimension ref="..."> element), avoiding "repair" warnings.
    wb = Workbook(write_only=False)
    if wb.worksheets:
        wb.remove(wb.worksheets[0])

    for name, df in sheets:
        ws = wb.create_sheet(title=str(name)[:31])
        ws.append([str(c) for c in df.columns.tolist()])
        for row in df.itertuples(index=False, name=None):
            ws.append(list(row))

    wb.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="AT", help="Eurostat geo code (default AT)")
    ap.add_argument("--year", type=int, default=2021, help="Year (default 2021)")
    ap.add_argument(
        "--mapping",
        default="scripts/mappings/energy_balance_to_nace2_mapping.csv",
        help="Category -> 65-sector mapping CSV",
    )
    ap.add_argument(
        "--category-aggregate-mapping",
        default="scripts/mappings/energy_balance_category_aggregates.csv",
        help="Energy-balance category harmonization CSV",
    )
    ap.add_argument(
        "--primary-source-buckets",
        default=str(DEFAULT_PRIMARY_SOURCE_BUCKETS_PATH),
        help="Primary energy source bucket registry CSV",
    )
    ap.add_argument("--out", default="outputs/AT_2021_energy_balances_65sectors_primary_sources.xlsx")
    ap.add_argument(
        "--storage-folder",
        default=str((default_data_root() / "eurostat").resolve()),
        help="Eurostat cache folder",
    )
    ap.add_argument(
        "--with-diagnostics",
        action="store_true",
        help="Also write Diagnostics and Mapping_weights sheets",
    )
    ap.add_argument("--energy-unit", default="GWH", help="Energy unit in Eurostat downloads (default GWH)")
    ap.add_argument("--force-download", action="store_true", help="Force re-download Eurostat payloads")
    ap.add_argument("--force-reparse", action="store_true", help="Force re-parse cached payloads")
    ap.add_argument(
        "--no-category-aggregates",
        action="store_true",
        help="Disable declared category harmonization aggregates",
    )
    ap.add_argument(
        "--postgrowth-sheet-name",
        default="Energy_1.0",
        help="Sheet name for the PostGrowth-compatible 65-sector energy balance export",
    )
    ap.add_argument(
        "--split-energy-green-brown",
        action="store_true",
        help="Replace the two aggregate energy rows with Energy Brown and Energy Green using declared source groups",
    )
    args = ap.parse_args()

    mapping_path = Path(args.mapping)
    aggregate_mapping_path = Path(args.category_aggregate_mapping)
    primary_source_buckets_path = Path(args.primary_source_buckets)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(out_path.stem + ".__writing__.xlsx")

    mapping_df = read_category_mapping_frame(mapping_path)
    categories = mapping_df["Category"].drop_duplicates().astype(str).tolist()
    composites = {} if args.no_category_aggregates else read_category_aggregate_mapping(aggregate_mapping_path)

    # 1) Energy balances -> Tabelle1 + unrolled primary sources by Category
    eb = build_tabelle1_from_eurostat(
        geo=args.geo,
        time=args.year,
        categories=categories,
        composite_categories=composites,
        unit=args.energy_unit,
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )
    validate_energy_balance_closure(eb.diagnostics, tol_abs=1e-6)

    # 2) Eurostat IOT -> total output by CPA product code (monetary weights)
    iot_filters: dict[str, Any] = {
        "geo": str(args.geo),
        "time": str(args.year),
        "freq": "A",
        "unit": "MIO_EUR",
        "stk_flow": "TOTAL",
    }
    parsed = load_eurostat_iot(
        dataset="naio_10_cp1700",
        filters=iot_filters,
        row_dim="prd_ava",
        col_dim="prd_use",
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )
    x = build_eurostat_product_use_reference(parsed).astype("float64").fillna(0.0)
    x.index = x.index.astype(str)

    targets = mapping_df["TargetSector"].drop_duplicates().astype(str).tolist()
    target_codes = mapping_df[["TargetSector", "TargetCode"]].drop_duplicates().copy()
    dup_targets = target_codes.groupby("TargetSector")["TargetCode"].nunique()
    dup_targets = dup_targets[dup_targets > 1]
    if not dup_targets.empty:
        example = dup_targets.index.tolist()[0]
        raise SystemExit(f"TargetSector maps to multiple target codes: {example}")
    target_to_cpa = dict(zip(target_codes["TargetSector"].astype(str), target_codes["TargetCode"].astype(str), strict=False))
    target_output = {t: float(x.get(target_to_cpa.get(t, ""), 0.0)) for t in targets}

    # 3) Recompute split ratios using IOT outputs and disaggregate to 65 sectors
    mapping_w = compute_mapping_shares_from_outputs(
        mapping_df,
        target_output_by_target=target_output,
        use_fallback_shares=False,
        equal_split_if_no_output=False,
        require_output_weights=True,
    )
    sector_energy = disaggregate_unrolled_sources(eb.unrolled_sources, mapping_w)
    # Preserve TargetSector order from the mapping file (and allow extra rows like "Households" at the end).
    sector_energy = sector_energy.reindex(targets).fillna(0.0)

    # 4) Air emissions accounts (CO2) and intensities
    co2 = load_eurostat_air_emissions_by_nace(
        geo=args.geo,
        time=args.year,
        airpol="CO2",
        unit="THS_T",
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )
    co2_by_nace = co2.emissions_tonnes.copy()
    co2_by_nace.index = co2_by_nace.index.astype(str)

    def _cpa_to_nace(code: str) -> str:
        s = str(code)
        return s[len("CPA_") :] if s.startswith("CPA_") else s

    rows: list[dict[str, Any]] = []
    for target_label, energy_row in sector_energy.iterrows():
        cpa_code = target_to_cpa.get(str(target_label))
        # Some TargetSectors (e.g. Households) are not CPA products and therefore have no IOT output.
        # Keep these as missing (blank in Excel) rather than zero.
        out_mio_eur = float(x.get(str(cpa_code), 0.0)) if cpa_code is not None else np.nan
        nace_code = _cpa_to_nace(str(cpa_code)) if cpa_code is not None else ""
        co2_t = float(co2_by_nace.get(nace_code, np.nan)) if nace_code else np.nan

        total_energy = float(pd.to_numeric(energy_row, errors="coerce").fillna(0.0).sum())
        row: dict[str, Any] = {
            "TargetSector": str(target_label),
            "CPA_code": str(cpa_code) if cpa_code is not None else "",
            "NACE_code": nace_code,
            "Total_primary_energy_GWh": total_energy,
            "Total_output_MIO_EUR": out_mio_eur,
            "CO2_tonnes": co2_t,
            "CO2_intensity_t_per_GWh": (co2_t / total_energy) if np.isfinite(co2_t) and total_energy > 0 else np.nan,
            "CO2_intensity_t_per_MIO_EUR": (co2_t / out_mio_eur) if np.isfinite(co2_t) and out_mio_eur > 0 else np.nan,
            "Energy_intensity_GWh_per_MIO_EUR": (total_energy / out_mio_eur) if out_mio_eur > 0 else np.nan,
        }
        for src in PRIMARY_SOURCES:
            row[src] = float(energy_row.get(src, 0.0))
        rows.append(row)

    mapped_df = pd.DataFrame(rows)
    # Put source columns first for readability
    front = ["TargetSector", "CPA_code", "NACE_code"] + list(PRIMARY_SOURCES)
    tail = [c for c in mapped_df.columns if c not in set(front)]
    mapped_df = mapped_df[front + tail]

    source_group_by_source = read_source_bucket_groups(primary_source_buckets_path)
    postgrowth_df = build_postgrowth_energy_balance(
        mapped_df,
        source_group_by_source=source_group_by_source,
        split_energy_green_brown=args.split_energy_green_brown,
    )

    # 5) Write single Excel output (atomically), using openpyxl directly for maximum Excel compatibility.
    sheets: list[tuple[str, pd.DataFrame]] = [
        (args.postgrowth_sheet_name, _xlsx_sanitize_df(postgrowth_df)),
        ("Tabelle1", _xlsx_sanitize_df(eb.tabelle1)),
        ("Mapped_65", _xlsx_sanitize_df(mapped_df)),
    ]
    if args.with_diagnostics:
        sheets.extend(
            [
                ("Diagnostics", _xlsx_sanitize_df(eb.diagnostics)),
                ("Mapping_weights", _xlsx_sanitize_df(mapping_w)),
            ]
        )

    _write_xlsx(tmp_out, sheets)
    tmp_out.replace(out_path)
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()

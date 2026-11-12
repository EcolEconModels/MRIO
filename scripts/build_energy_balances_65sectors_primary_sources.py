#!/usr/bin/env python3
"""Export an approximation of gross energy by source, alongside actual activity inputs.

Energy_1.0 includes feedstocks, net transformation and attributed net electricity.
Mapped_65 retains actual activity inputs for combustion and fuel-price calculations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook

from ecolecon_mrio.eurostat_air_emissions import load_eurostat_air_emissions_by_nace
from ecolecon_mrio.eurostat_energy_balances import (
    PRIMARY_SOURCES,
    PRODUCT_SOURCE_MAPPING,
    DEFAULT_PRIMARY_SOURCE_BUCKETS_PATH,
    build_postgrowth_energy_balance,
    build_tabelle1_from_eurostat,
    compute_mapping_shares_from_outputs,
    disaggregate_unrolled_sources,
    load_eurostat_electricity_heat_mix,
    load_eurostat_energy_balance_slice,
    validate_energy_balance_closure,
)
from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import (
    build_eurostat_intermediate_use_reference_from_rows,
    build_eurostat_product_use_reference,
    load_eurostat_iot,
)
from ecolecon_mrio.mapping_csv import read_category_mapping_frame, read_source_bucket_groups


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


def _resolve_energy_weighting_target_codes(mapping_df: pd.DataFrame) -> list[str]:
    mask = mapping_df["TargetSectorRole"].astype(str).eq("transformation_sector")
    mask = mask & mapping_df["IncludeInDirectEnergy"].astype(bool)

    target_codes = [
        str(code).strip()
        for code in mapping_df.loc[mask, "TargetCode"].tolist()
        if str(code).strip() != ""
    ]
    if not target_codes:
        raise ValueError(
            "Category mapping must declare at least one direct-energy transformation-sector target code."
        )
    return list(dict.fromkeys(target_codes))


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
        default="scripts/mappings/energy_balance_to_cpa65_products_mapping.csv",
        help="Category -> project 65-product CPA-based target mapping CSV (legacy nace2 filename retained for compatibility; source row definitions may still be activity-based)",
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
    primary_source_buckets_path = Path(args.primary_source_buckets)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(out_path.stem + ".__writing__.xlsx")

    mapping_df = read_category_mapping_frame(mapping_path)
    categories = mapping_df["Category"].drop_duplicates().astype(str).tolist()

    # 1) Energy balances -> Tabelle1 + unrolled primary sources by Category
    eb = build_tabelle1_from_eurostat(
        geo=args.geo,
        time=args.year,
        categories=categories + ["Gross inland consumption"],
        unit=args.energy_unit,
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )
    validate_energy_balance_closure(eb.diagnostics, tol_abs=1e-6)

    # This supply row is a reference, not an additional sector or combustion input.
    gross = eb.tabelle1.set_index("Category").loc["Gross inland consumption"]
    if not np.isclose(gross[list(PRIMARY_SOURCES)].sum() + gross["Electricity:"] + gross["Heat"],
                      gross["Total"], rtol=0.0, atol=1e-2):
        raise ValueError("Gross inland sources and net electricity/heat do not reproduce TOTAL.")

    # 2) Eurostat IOT -> energy-product intermediate use by CPA product code (monetary weights)
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
    total_output_reference = build_eurostat_product_use_reference(parsed).astype("float64").fillna(0.0)
    total_output_reference.index = total_output_reference.index.astype(str)
    energy_weighting_target_codes = _resolve_energy_weighting_target_codes(mapping_df)
    energy_use_reference = build_eurostat_intermediate_use_reference_from_rows(
        parsed,
        source_row_codes=energy_weighting_target_codes,
    ).astype("float64").fillna(0.0)
    energy_use_reference.index = energy_use_reference.index.astype(str)

    targets = mapping_df["TargetSector"].drop_duplicates().astype(str).tolist()
    target_codes = mapping_df[["TargetSector", "TargetCode"]].drop_duplicates().copy()
    dup_targets = target_codes.groupby("TargetSector")["TargetCode"].nunique()
    dup_targets = dup_targets[dup_targets > 1]
    if not dup_targets.empty:
        example = dup_targets.index.tolist()[0]
        raise SystemExit(f"TargetSector maps to multiple target codes: {example}")
    target_to_cpa = dict(zip(target_codes["TargetSector"].astype(str), target_codes["TargetCode"].astype(str), strict=False))

    # 3) Recompute split ratios using IOT outputs and disaggregate to 65 sectors
    mapping_w = compute_mapping_shares_from_outputs(
        mapping_df,
        target_output_by_code=energy_use_reference.to_dict(),
    )
    # Actual inputs exclude feedstocks, losses and transformations between fuels.
    # Generation fuel remains here; users' electricity/heat is not added again.
    direct_sources = eb.tabelle1.loc[:, ["Category", *PRIMARY_SOURCES]]
    sector_energy = disaggregate_unrolled_sources(
        direct_sources, mapping_w.loc[mapping_w["IncludeInDirectEnergy"]]
    )
    # Preserve TargetSector order from the mapping file (and allow extra rows like "Households" at the end).
    sector_energy = sector_energy.reindex(targets).fillna(0.0)

    # Reuse the same allocation, subtracting outputs of fuel transformation once.
    gross_sources = direct_sources.copy()
    output_categories = mapping_w.loc[
        mapping_w["SourceBalanceRole"].eq("transformation_output"), "Category"
    ]
    gross_sources.loc[gross_sources["Category"].isin(output_categories), list(PRIMARY_SOURCES)] *= -1
    gross_energy = disaggregate_unrolled_sources(gross_sources, mapping_w).reindex(targets).fillna(0.0)
    label_by_code = target_codes.set_index("TargetCode")["TargetSector"].to_dict()

    # Approximate non-energy oil/gas use by chemicals, but coal by basic metals.
    nonenergy_categories = mapping_w.loc[
        mapping_w["SourceBalanceRole"].eq("non_energy_use"), "Category"
    ]
    coal_nonenergy = float(direct_sources.loc[
        direct_sources["Category"].isin(nonenergy_categories), "Solid fossil fuels"
    ].sum())
    if coal_nonenergy != 0:
        gross_energy.loc[label_by_code["CPA_C20"], "Solid fossil fuels"] -= coal_nonenergy
        gross_energy.loc[label_by_code["CPA_C24"], "Solid fossil fuels"] += coal_nonenergy

    # Net imported electricity uses the domestic generation mix as an approximation,
    # in electricity units, without a fuel-efficiency multiplier or Austrian combustion.
    if abs(float(gross["Heat"])) > 1e-2:
        raise ValueError("Nonzero net heat trade requires an explicit source allocation.")
    elec_mix, _, _ = load_eurostat_electricity_heat_mix(
        geo=args.geo, time=args.year, unit=args.energy_unit,
        storage_folder=args.storage_folder, force_download=args.force_download,
        force_reparse=args.force_reparse,
    )
    if not np.isclose(elec_mix.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("Electricity generation shares must sum to one.")
    gross_energy.loc[label_by_code["CPA_D"], list(PRIMARY_SOURCES)] += elec_mix * float(gross["Electricity:"])
    expected_sources = gross[list(PRIMARY_SOURCES)] + elec_mix * float(gross["Electricity:"])
    source_gap = expected_sources - gross_energy.sum()
    # Independently identify the two agreed omissions by source. A match in the
    # grand total must not conceal compensating fuel-classification errors.
    balance, _ = load_eurostat_energy_balance_slice(
        geo=args.geo, time=args.year, unit=args.energy_unit, storage_folder=args.storage_folder,
        force_download=args.force_download, force_reparse=args.force_reparse,
    )
    if not np.isfinite(balance.loc["STATDIFF"].to_numpy()).all():
        raise ValueError("Missing observations in statistical differences.")
    omitted = pd.Series({src: balance.loc["STATDIFF", [c for c in codes if c in balance.columns]].sum()
                         for src, codes in PRODUCT_SOURCE_MAPPING.items()})
    industry_children = balance.index.str.startswith("FC_IND_") & (balance.index != "FC_IND_E")
    omitted["Other Renewables"] += (balance.at["FC_OTH_NSP_E", "RA600"]
        + balance.at["FC_IND_E", "RA600"] - balance.loc[industry_children, "RA600"].sum())
    if (not np.isfinite(omitted.to_numpy()).all()
            or omitted.abs().sum() > abs(float(gross["Total"])) * 0.001
            or not np.allclose(source_gap, omitted, rtol=0.0, atol=0.02)):
        raise ValueError(f"Source gaps do not match ambient heat and statistical differences: {(source_gap - omitted).to_dict()}")

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
        out_mio_eur = float(total_output_reference.get(str(cpa_code), 0.0)) if cpa_code is not None else np.nan
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
        gross_energy.rename_axis("TargetSector").reset_index(),
        source_group_by_source=source_group_by_source,
        split_energy_green_brown=args.split_energy_green_brown,
    )
    if (postgrowth_df[list(PRIMARY_SOURCES)].to_numpy() < -1e-6).any():
        raise ValueError("The gross-energy allocation has a negative sector/source quantity.")
    if not np.allclose(postgrowth_df[list(PRIMARY_SOURCES)].sum(), gross_energy.sum(), rtol=0.0, atol=1e-7):
        raise ValueError("The green/brown allocation did not conserve energy sources.")
    print(f"Activity inputs: {sector_energy.to_numpy().sum():.3f} {args.energy_unit}; "
          f"allocated gross energy: {postgrowth_df['Total_energy_GWh'].sum():.3f}; "
          f"Eurostat GIC: {float(gross['Total']):.3f}.")

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

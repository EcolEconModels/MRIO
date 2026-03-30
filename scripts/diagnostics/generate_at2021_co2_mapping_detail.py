#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _flat_diagnostics_common import (
    compute_weighted_energy_balance_factor,
    default_output_dir,
    default_postgrowth_root,
    eurostat_order_map,
    import_repo_module,
    load_eurostat_iot_bundle,
    load_exio_slice,
    load_ghg_protocol_factors,
    load_weighted_mapping,
    mapping_label_maps,
    read_energy_balance_table,
    repo_root_from_file,
    sector_total,
    write_csv,
)


OFFICIAL_AIR_EMISSIONS_SOURCE = (
    "Eurostat air emissions accounts by NACE Rev. 2, pollutant CO2, unit THS_T converted to tonnes"
)


def build_totals_row(df: pd.DataFrame) -> pd.DataFrame:
    unique_sources = df.drop_duplicates(subset=["exio_source_code"])
    unique_targets = df.drop_duplicates(subset=["eurostat_sector_code"])

    domestic_final_demand_total = sector_total(unique_targets["eurostat_domestic_final_demand_meur_2021"])
    domestic_final_demand_dom_total = sector_total(unique_targets["eurostat_domestic_final_demand_dom_meur_2021"])
    domestic_final_demand_imp_total = sector_total(unique_targets["eurostat_domestic_final_demand_imp_meur_2021"])
    total_final_demand_total = sector_total(unique_targets["eurostat_total_final_demand_meur_2021"])
    exports_total = sector_total(unique_targets["eurostat_exports_meur_2021"])
    gross_total = sector_total(unique_targets["eurostat_gross_output_meur_2021"])
    domestic_total = sector_total(unique_targets["eurostat_domestic_output_meur_2021"])
    export_output_total = sector_total(unique_targets["eurostat_export_output_meur_2021"])
    energy_total = sector_total(unique_targets["eurostat_energy_balance_total_gwh_2021"])
    energy_balance_emissions_total = sector_total(unique_targets["energy_balance_emissions_tco2_2021"])
    mapped_output_total = sector_total(unique_targets["mapped_mrio_sector_output_meur_2020"])
    mapped_energy_total = sector_total(unique_targets["mapped_mrio_sector_energy_gwh_2020"])
    mapped_co2_total = sector_total(unique_targets["mapped_mrio_sector_co2_tco2_2020"])
    official_total = sector_total(unique_targets["eurostat_air_emissions_tco2_2021"])
    mrio_domestic_total = sector_total(unique_targets["mrio_domestic_output_emissions_tco2_2021"])
    mrio_gross_total = sector_total(unique_targets["mrio_gross_output_emissions_tco2_2021"])
    mrio_export_total = sector_total(unique_targets["mrio_export_output_emissions_tco2_2021"])
    gap = energy_balance_emissions_total - mrio_domestic_total
    gross_gap = energy_balance_emissions_total - mrio_gross_total

    totals = {
        "eurostat_order": np.nan,
        "eurostat_sector_code": "TOTAL",
        "eurostat_sector_name": "All sectors",
        "energy_balance_sector_name": "All sectors",
        "exio_source_code": "",
        "exio_source_name": "",
        "mapping_weight": np.nan,
        "mrio_exio_output_meur_2020": sector_total(unique_sources["mrio_exio_output_meur_2020"]),
        "mapped_mrio_output_meur_2020": sector_total(df["mapped_mrio_output_meur_2020"]),
        "mrio_exio_energy_gwh_2020": sector_total(unique_sources["mrio_exio_energy_gwh_2020"]),
        "mapped_mrio_energy_gwh_2020": sector_total(df["mapped_mrio_energy_gwh_2020"]),
        "mrio_exio_co2_tco2_2020": sector_total(unique_sources["mrio_exio_co2_tco2_2020"]),
        "mapped_mrio_co2_tco2_2020": sector_total(df["mapped_mrio_co2_tco2_2020"]),
        "eurostat_domestic_final_demand_meur_2021": domestic_final_demand_total,
        "eurostat_domestic_final_demand_dom_meur_2021": domestic_final_demand_dom_total,
        "eurostat_domestic_final_demand_imp_meur_2021": domestic_final_demand_imp_total,
        "eurostat_total_final_demand_meur_2021": total_final_demand_total,
        "eurostat_exports_meur_2021": exports_total,
        "eurostat_domestic_final_demand_share_2021": (domestic_final_demand_total / total_final_demand_total) if total_final_demand_total else np.nan,
        "eurostat_import_share_of_domestic_final_demand_2021": (domestic_final_demand_imp_total / domestic_final_demand_total) if domestic_final_demand_total else np.nan,
        "eurostat_export_share_of_final_demand_2021": (exports_total / total_final_demand_total) if total_final_demand_total else np.nan,
        "eurostat_gross_output_meur_2021": gross_total,
        "eurostat_domestic_output_meur_2021": domestic_total,
        "eurostat_export_output_meur_2021": export_output_total,
        "eurostat_energy_balance_total_gwh_2021": energy_total,
        "energy_balance_weighted_co2_factor_tco2_per_gwh_2021": (energy_balance_emissions_total / energy_total) if energy_total else np.nan,
        "energy_balance_emissions_tco2_2021": energy_balance_emissions_total,
        "energy_balance_output_per_tco2_meur_2021": (domestic_total / energy_balance_emissions_total) if energy_balance_emissions_total else np.nan,
        "mapped_mrio_sector_output_meur_2020": mapped_output_total,
        "mapped_mrio_sector_energy_gwh_2020": mapped_energy_total,
        "mapped_mrio_sector_co2_tco2_2020": mapped_co2_total,
        "mapped_mrio_sector_energy_intensity_gwh_per_meur_2020": (mapped_energy_total / mapped_output_total) if mapped_output_total else np.nan,
        "mapped_mrio_sector_implied_factor_tco2_per_gwh_2020": (mapped_co2_total / mapped_energy_total) if mapped_energy_total else np.nan,
        "mapped_mrio_sector_co2_intensity_tco2_per_meur_2020": (mapped_co2_total / mapped_output_total) if mapped_output_total else np.nan,
        "mrio_domestic_output_emissions_tco2_2021": mrio_domestic_total,
        "mrio_domestic_output_per_tco2_meur_2021": (domestic_total / mrio_domestic_total) if mrio_domestic_total else np.nan,
        "mrio_gross_output_emissions_tco2_2021": mrio_gross_total,
        "mrio_gross_output_per_tco2_meur_2021": (gross_total / mrio_gross_total) if mrio_gross_total else np.nan,
        "mrio_export_output_emissions_tco2_2021": mrio_export_total,
        "energy_balance_over_mrio_multiple_2021": (energy_balance_emissions_total / mrio_domestic_total) if mrio_domestic_total else np.nan,
        "emissions_gap_tco2_2021": gap,
        "emissions_gap_pct_of_mrio_2021": (100.0 * gap / mrio_domestic_total) if mrio_domestic_total else np.nan,
        "energy_balance_over_mrio_gross_multiple_2021": (energy_balance_emissions_total / mrio_gross_total) if mrio_gross_total else np.nan,
        "emissions_gap_vs_mrio_gross_tco2_2021": gross_gap,
        "emissions_gap_vs_mrio_gross_pct_of_mrio_2021": (100.0 * gross_gap / mrio_gross_total) if mrio_gross_total else np.nan,
        "eurostat_air_emissions_tco2_2021": official_total,
        "official_air_emissions_source": OFFICIAL_AIR_EMISSIONS_SOURCE,
        "provenance": "",
        "certainty": "",
        "notes": "Totals row: source columns are summed over unique EXIO source codes; Eurostat, energy-balance, MRIO sector aggregates, and official air-emissions totals are summed over unique Eurostat sectors.",
    }
    return pd.DataFrame([totals])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the 2021 CO2 mapping-detail CSV.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--region", default="AT")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--exio-year", type=int, default=2020)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--postgrowth-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = repo_root_from_file(__file__)
    postgrowth_root = args.postgrowth_root or default_postgrowth_root(repo_root)
    output_dir = args.output_dir or default_output_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_df = load_weighted_mapping(repo_root)
    target_labels, source_labels = mapping_label_maps(mapping_df)

    concordance = import_repo_module(repo_root, "ecolecon_mrio.diagnostic_concordance")
    air_emissions_mod = import_repo_module(repo_root, "ecolecon_mrio.eurostat_air_emissions")
    exio_ext = import_repo_module(repo_root, "ecolecon_mrio.exiobase3_extensions")
    parsed_total, _gross_output_sumcheck, _ = load_eurostat_iot_bundle(repo_root, geo=args.geo, year=args.year, flow_code="TOTAL")
    parsed_dom, _, _ = load_eurostat_iot_bundle(repo_root, geo=args.geo, year=args.year, flow_code="DOM")
    parsed_imp, _, _ = load_eurostat_iot_bundle(repo_root, geo=args.geo, year=args.year, flow_code="IMP")
    exio_slice = load_exio_slice(repo_root, region=args.region, year=args.exio_year, flow_name="total")

    z_total = parsed_total.Z.astype("float64")
    y_total = parsed_total.Y.astype("float64")
    gross_output = y_total["TU"].astype("float64")
    gross_with_zero_fix = gross_output.replace(0.0, np.nan)
    a_matrix = z_total.divide(gross_with_zero_fix, axis=1).fillna(0.0)
    leontief_inverse = np.linalg.inv(np.eye(len(a_matrix)) - a_matrix.to_numpy())
    domestic_final_demand = (y_total["P3"] + y_total["P5"]).astype("float64")
    domestic_final_demand_dom = (parsed_dom.Y["P3"] + parsed_dom.Y["P5"]).astype("float64")
    domestic_final_demand_imp = (parsed_imp.Y["P3"] + parsed_imp.Y["P5"]).astype("float64")
    total_final_demand = y_total["TFU"].astype("float64")
    exports_final_demand = y_total["P6"].astype("float64")
    domestic_output = pd.Series(leontief_inverse @ domestic_final_demand.to_numpy(), index=gross_output.index, dtype="float64")
    export_output = gross_output - domestic_output

    exio_energy = exio_ext.extract_quantity_and_price(
        container=exio_slice.source_zip,
        region=args.region,
        sectors=list(map(str, exio_slice.sectors)),
        monetary_output=exio_slice.x_region,
        extensions=("energy",),
        stressor_keywords=("Energy use",),
        cache_dir=repo_root.parent / "MRIO" / "data" / "_ext_cache",
    )
    energy_rows = exio_energy[exio_energy["stressor"].astype(str) == "Energy use - Final"].copy()
    if energy_rows.empty:
        energy_rows = exio_energy[exio_energy["stressor"].astype(str) == "Energy use - Gross"].copy()
    exio_energy_by_sector = (energy_rows.groupby("exio_sector")["quantity"].sum().astype("float64") / 3.6)

    exio_co2 = exio_ext.extract_emissions_and_intensity(
        container=exio_slice.source_zip,
        region=args.region,
        sectors=list(map(str, exio_slice.sectors)),
        monetary_output=exio_slice.x_region,
        extension="air_emissions",
        stressor_keywords=("CO2 - combustion - air",),
        cache_dir=repo_root.parent / "MRIO" / "data" / "_ext_cache",
    )
    exio_co2_by_sector = (exio_co2.groupby("exio_sector")["emission"].sum().astype("float64") / 1000.0)

    output_alloc = concordance.allocate_weighted_values(exio_slice.x_region, mapping_df)
    energy_alloc = concordance.allocate_weighted_values(exio_energy_by_sector, mapping_df)
    co2_alloc = concordance.allocate_weighted_values(exio_co2_by_sector, mapping_df)

    mapped_output_by_sector = output_alloc.aggregate_totals.rename("mapped_mrio_sector_output_meur_2020")
    mapped_energy_by_sector = energy_alloc.aggregate_totals.rename("mapped_mrio_sector_energy_gwh_2020")
    mapped_co2_by_sector = co2_alloc.aggregate_totals.rename("mapped_mrio_sector_co2_tco2_2020")

    energy_balance = read_energy_balance_table(postgrowth_root, year=args.year)
    source_factors = load_ghg_protocol_factors(postgrowth_root)
    energy_balance["energy_balance_weighted_co2_factor_tco2_per_gwh_2021"] = compute_weighted_energy_balance_factor(
        energy_balance,
        source_factors,
    )
    energy_balance["energy_balance_emissions_tco2_2021"] = (
        energy_balance["energy_balance_total_energy_gwh_2021"]
        * energy_balance["energy_balance_weighted_co2_factor_tco2_per_gwh_2021"]
    )

    air = air_emissions_mod.load_eurostat_air_emissions_by_nace(
        geo=args.geo,
        time=args.year,
        airpol="CO2",
        unit="THS_T",
        storage_folder=repo_root.parent / "MRIO" / "data" / "eurostat",
        force_download=False,
        force_reparse=False,
    ).emissions_tonnes
    air.index = air.index.astype(str)

    sector_df = pd.DataFrame(index=pd.Index(sorted(set(mapping_df["aggregate_sector"].astype(str))), name="eurostat_sector_code"))
    order_lookup = eurostat_order_map(gross_output.index)
    sector_df["eurostat_order"] = sector_df.index.map(order_lookup)
    sector_df["eurostat_sector_name"] = sector_df.index.map(target_labels)
    sector_df["energy_balance_sector_name"] = sector_df["eurostat_sector_name"]
    sector_df["eurostat_domestic_final_demand_meur_2021"] = sector_df.index.map(domestic_final_demand)
    sector_df["eurostat_domestic_final_demand_dom_meur_2021"] = sector_df.index.map(domestic_final_demand_dom)
    sector_df["eurostat_domestic_final_demand_imp_meur_2021"] = sector_df.index.map(domestic_final_demand_imp)
    sector_df["eurostat_total_final_demand_meur_2021"] = sector_df.index.map(total_final_demand)
    sector_df["eurostat_exports_meur_2021"] = sector_df.index.map(exports_final_demand)
    sector_df["eurostat_domestic_final_demand_share_2021"] = np.where(
        sector_df["eurostat_total_final_demand_meur_2021"] > 0,
        sector_df["eurostat_domestic_final_demand_meur_2021"] / sector_df["eurostat_total_final_demand_meur_2021"],
        np.nan,
    )
    sector_df["eurostat_import_share_of_domestic_final_demand_2021"] = np.where(
        sector_df["eurostat_domestic_final_demand_meur_2021"] > 0,
        sector_df["eurostat_domestic_final_demand_imp_meur_2021"] / sector_df["eurostat_domestic_final_demand_meur_2021"],
        np.nan,
    )
    sector_df["eurostat_export_share_of_final_demand_2021"] = np.where(
        sector_df["eurostat_total_final_demand_meur_2021"] > 0,
        sector_df["eurostat_exports_meur_2021"] / sector_df["eurostat_total_final_demand_meur_2021"],
        np.nan,
    )
    sector_df["eurostat_gross_output_meur_2021"] = sector_df.index.map(gross_output)
    sector_df["eurostat_domestic_output_meur_2021"] = sector_df.index.map(domestic_output)
    sector_df["eurostat_export_output_meur_2021"] = sector_df.index.map(export_output)
    energy_balance_lookup = energy_balance.set_index("energy_balance_sector_name")
    sector_df["eurostat_energy_balance_total_gwh_2021"] = sector_df["energy_balance_sector_name"].map(
        energy_balance_lookup["energy_balance_total_energy_gwh_2021"]
    )
    sector_df["energy_balance_weighted_co2_factor_tco2_per_gwh_2021"] = sector_df["energy_balance_sector_name"].map(
        energy_balance_lookup["energy_balance_weighted_co2_factor_tco2_per_gwh_2021"]
    )
    sector_df["energy_balance_emissions_tco2_2021"] = sector_df["energy_balance_sector_name"].map(
        energy_balance_lookup["energy_balance_emissions_tco2_2021"]
    )
    sector_df["mapped_mrio_sector_output_meur_2020"] = mapped_output_by_sector
    sector_df["mapped_mrio_sector_energy_gwh_2020"] = mapped_energy_by_sector
    sector_df["mapped_mrio_sector_co2_tco2_2020"] = mapped_co2_by_sector
    sector_df["mapped_mrio_sector_energy_intensity_gwh_per_meur_2020"] = np.where(
        sector_df["mapped_mrio_sector_output_meur_2020"] > 0,
        sector_df["mapped_mrio_sector_energy_gwh_2020"] / sector_df["mapped_mrio_sector_output_meur_2020"],
        np.nan,
    )
    sector_df["mapped_mrio_sector_implied_factor_tco2_per_gwh_2020"] = np.where(
        sector_df["mapped_mrio_sector_energy_gwh_2020"] > 0,
        sector_df["mapped_mrio_sector_co2_tco2_2020"] / sector_df["mapped_mrio_sector_energy_gwh_2020"],
        np.nan,
    )
    sector_df["mapped_mrio_sector_co2_intensity_tco2_per_meur_2020"] = np.where(
        sector_df["mapped_mrio_sector_output_meur_2020"] > 0,
        sector_df["mapped_mrio_sector_co2_tco2_2020"] / sector_df["mapped_mrio_sector_output_meur_2020"],
        np.nan,
    )
    sector_df["energy_balance_output_per_tco2_meur_2021"] = np.where(
        sector_df["energy_balance_emissions_tco2_2021"] > 0,
        sector_df["eurostat_domestic_output_meur_2021"] / sector_df["energy_balance_emissions_tco2_2021"],
        np.nan,
    )
    sector_df["mrio_domestic_output_emissions_tco2_2021"] = (
        sector_df["eurostat_domestic_output_meur_2021"] * sector_df["mapped_mrio_sector_co2_intensity_tco2_per_meur_2020"]
    )
    sector_df["mrio_domestic_output_per_tco2_meur_2021"] = np.where(
        sector_df["mrio_domestic_output_emissions_tco2_2021"] > 0,
        sector_df["eurostat_domestic_output_meur_2021"] / sector_df["mrio_domestic_output_emissions_tco2_2021"],
        np.nan,
    )
    sector_df["mrio_gross_output_emissions_tco2_2021"] = (
        sector_df["eurostat_gross_output_meur_2021"] * sector_df["mapped_mrio_sector_co2_intensity_tco2_per_meur_2020"]
    )
    sector_df["mrio_gross_output_per_tco2_meur_2021"] = np.where(
        sector_df["mrio_gross_output_emissions_tco2_2021"] > 0,
        sector_df["eurostat_gross_output_meur_2021"] / sector_df["mrio_gross_output_emissions_tco2_2021"],
        np.nan,
    )
    sector_df["mrio_export_output_emissions_tco2_2021"] = (
        sector_df["mrio_gross_output_emissions_tco2_2021"] - sector_df["mrio_domestic_output_emissions_tco2_2021"]
    )
    sector_df["energy_balance_over_mrio_multiple_2021"] = np.where(
        sector_df["mrio_domestic_output_emissions_tco2_2021"] > 0,
        sector_df["energy_balance_emissions_tco2_2021"] / sector_df["mrio_domestic_output_emissions_tco2_2021"],
        np.nan,
    )
    sector_df["emissions_gap_tco2_2021"] = (
        sector_df["energy_balance_emissions_tco2_2021"] - sector_df["mrio_domestic_output_emissions_tco2_2021"]
    )
    sector_df["emissions_gap_pct_of_mrio_2021"] = np.where(
        sector_df["mrio_domestic_output_emissions_tco2_2021"] > 0,
        100.0 * sector_df["emissions_gap_tco2_2021"] / sector_df["mrio_domestic_output_emissions_tco2_2021"],
        np.nan,
    )
    sector_df["energy_balance_over_mrio_gross_multiple_2021"] = np.where(
        sector_df["mrio_gross_output_emissions_tco2_2021"] > 0,
        sector_df["energy_balance_emissions_tco2_2021"] / sector_df["mrio_gross_output_emissions_tco2_2021"],
        np.nan,
    )
    sector_df["emissions_gap_vs_mrio_gross_tco2_2021"] = (
        sector_df["energy_balance_emissions_tco2_2021"] - sector_df["mrio_gross_output_emissions_tco2_2021"]
    )
    sector_df["emissions_gap_vs_mrio_gross_pct_of_mrio_2021"] = np.where(
        sector_df["mrio_gross_output_emissions_tco2_2021"] > 0,
        100.0 * sector_df["emissions_gap_vs_mrio_gross_tco2_2021"] / sector_df["mrio_gross_output_emissions_tco2_2021"],
        np.nan,
    )
    sector_df["eurostat_air_emissions_tco2_2021"] = [air.get(code.removeprefix("CPA_"), np.nan) for code in sector_df.index]
    sector_df["official_air_emissions_source"] = OFFICIAL_AIR_EMISSIONS_SOURCE

    detail = output_alloc.detail.copy()
    detail = detail.rename(columns={
        "aggregate_sector": "eurostat_sector_code",
        "exio_sector": "exio_source_code",
        "weight": "mapping_weight",
        "source_value": "mrio_exio_output_meur_2020",
        "allocated_value": "mapped_mrio_output_meur_2020",
    })
    detail["mrio_exio_energy_gwh_2020"] = detail["exio_source_code"].map(exio_energy_by_sector)
    detail["mapped_mrio_energy_gwh_2020"] = detail["mapping_weight"] * detail["mrio_exio_energy_gwh_2020"]
    detail["mrio_exio_co2_tco2_2020"] = detail["exio_source_code"].map(exio_co2_by_sector)
    detail["mapped_mrio_co2_tco2_2020"] = detail["mapping_weight"] * detail["mrio_exio_co2_tco2_2020"]
    detail["eurostat_order"] = detail["eurostat_sector_code"].map(order_lookup)
    detail["eurostat_sector_name"] = detail["eurostat_sector_code"].map(target_labels)
    detail["energy_balance_sector_name"] = detail["eurostat_sector_name"]
    detail["exio_source_name"] = detail["exio_source_code"].map(source_labels)
    detail = detail.drop(columns=[col for col in [
        "eurostat_sector_name",
        "energy_balance_sector_name",
        "eurostat_order",
    ] if col in detail.columns])
    detail = detail.merge(sector_df.reset_index(), on="eurostat_sector_code", how="left")
    detail = detail.sort_values(["eurostat_order", "eurostat_sector_code", "mapped_mrio_output_meur_2020"], ascending=[True, True, False])

    out = detail[
        [
            "eurostat_order",
            "eurostat_sector_code",
            "eurostat_sector_name",
            "energy_balance_sector_name",
            "exio_source_code",
            "exio_source_name",
            "mapping_weight",
            "mrio_exio_output_meur_2020",
            "mapped_mrio_output_meur_2020",
            "mrio_exio_energy_gwh_2020",
            "mapped_mrio_energy_gwh_2020",
            "mrio_exio_co2_tco2_2020",
            "mapped_mrio_co2_tco2_2020",
            "eurostat_domestic_final_demand_meur_2021",
            "eurostat_domestic_final_demand_dom_meur_2021",
            "eurostat_domestic_final_demand_imp_meur_2021",
            "eurostat_total_final_demand_meur_2021",
            "eurostat_exports_meur_2021",
            "eurostat_domestic_final_demand_share_2021",
            "eurostat_import_share_of_domestic_final_demand_2021",
            "eurostat_export_share_of_final_demand_2021",
            "eurostat_gross_output_meur_2021",
            "eurostat_domestic_output_meur_2021",
            "eurostat_export_output_meur_2021",
            "eurostat_energy_balance_total_gwh_2021",
            "energy_balance_weighted_co2_factor_tco2_per_gwh_2021",
            "energy_balance_emissions_tco2_2021",
            "energy_balance_output_per_tco2_meur_2021",
            "mapped_mrio_sector_output_meur_2020",
            "mapped_mrio_sector_energy_gwh_2020",
            "mapped_mrio_sector_co2_tco2_2020",
            "mapped_mrio_sector_energy_intensity_gwh_per_meur_2020",
            "mapped_mrio_sector_implied_factor_tco2_per_gwh_2020",
            "mapped_mrio_sector_co2_intensity_tco2_per_meur_2020",
            "mrio_domestic_output_emissions_tco2_2021",
            "mrio_domestic_output_per_tco2_meur_2021",
            "mrio_gross_output_emissions_tco2_2021",
            "mrio_gross_output_per_tco2_meur_2021",
            "mrio_export_output_emissions_tco2_2021",
            "energy_balance_over_mrio_multiple_2021",
            "emissions_gap_tco2_2021",
            "emissions_gap_pct_of_mrio_2021",
            "energy_balance_over_mrio_gross_multiple_2021",
            "emissions_gap_vs_mrio_gross_tco2_2021",
            "emissions_gap_vs_mrio_gross_pct_of_mrio_2021",
            "eurostat_air_emissions_tco2_2021",
            "official_air_emissions_source",
            "provenance",
            "certainty",
            "notes",
        ]
    ].copy()
    out = pd.concat([out, build_totals_row(out)], ignore_index=True)
    write_csv(out, output_dir / "at2021_co2_mapping_detail.csv")
    print(f"Wrote CO2 mapping detail to {output_dir / 'at2021_co2_mapping_detail.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
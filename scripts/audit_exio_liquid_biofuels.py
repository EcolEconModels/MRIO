#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecolecon_mrio.data import default_data_root  # noqa: E402
from ecolecon_mrio.eurostat_iot import load_eurostat_iot, load_eurostat_table_2d  # noqa: E402
from ecolecon_mrio.exiobase3 import load_exiobase3_region_slice  # noqa: E402


REGION = "AT"
EXIO_YEAR = 2020
EUROSTAT_YEARS = (2020, 2021)
BIOFUELS = ("p24.g", "p24.h", "p24.i")
WHOLESALE = "p51"
EUROSTAT_C20 = "CPA_C20"
EUROSTAT_G46 = "CPA_G46"
FLOWS: tuple[tuple[str, str], ...] = (("domestic", "DOM"), ("imports", "IMP"), ("total", "TOTAL"))
CENTRAL_FLOW = "total"

OUTDIR = REPO_ROOT / "outputs" / "audits" / "liquid_biofuels_exio_vs_eurostat"


@dataclass(frozen=True)
class LabelMaps:
    exio_products: dict[str, str]
    exio_industries: dict[str, str]
    eurostat_products: dict[str, str]


def _read_code_label(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str), strict=False))


def labels() -> LabelMaps:
    mapping_dir = REPO_ROOT / "scripts" / "mappings"
    return LabelMaps(
        exio_products=_read_code_label(mapping_dir / "exiobase_products_dictionary.csv"),
        exio_industries=_read_code_label(mapping_dir / "exiobase_industries_dictionary.csv"),
        eurostat_products=_read_code_label(mapping_dir / "eurostat_cpa65_products_dictionary.csv"),
    )


def label(mapping: dict[str, str], code: str) -> str:
    return mapping.get(str(code), str(code))


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def exio_slice(year: int, flow: str):
    return load_exiobase3_region_slice(
        year=year,
        region=REGION,
        system="pxp",
        flow=flow,
        cache_root=default_data_root(),
        force_download=False,
        force_rebuild_slice=False,
    )


def write_csv(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def c20_products_from_mapping() -> list[str]:
    path = REPO_ROOT / "scripts" / "mappings" / "exio_to_eurostat_cpa_products.csv"
    df = pd.read_csv(path)
    enabled = pd.to_numeric(df.get("enabled", 1), errors="coerce").fillna(0).astype(int) == 1
    c20 = df.loc[enabled & (df["target_code"].astype(str) == EUROSTAT_C20), "source_code"]
    return sorted(c20.astype(str).unique().tolist())


def transport_codes(exio_labels: dict[str, str]) -> list[str]:
    fixed = {"p60.1", "p60.2", "p60.3", "p61.1", "p61.2", "p62", "p63"}
    out: set[str] = set()
    for code, lab in exio_labels.items():
        text = str(lab).lower()
        if (
            "transportation service" in text
            or "transport services" in text
            or "air transport" in text
            or "travel agency" in text
            or code in fixed
        ):
            out.add(str(code))
    return sorted(out)


def selected_destination_codes(exio_labels: dict[str, str]) -> list[str]:
    wanted = {"p51", "p52", "p55", "p60.1", "p60.2", "p60.3", "p61.1", "p61.2", "p62", "p63"}
    wanted.update(transport_codes(exio_labels))
    return sorted(code for code in wanted if code in exio_labels)


def audit_exio_z_y(lab: LabelMaps) -> tuple[pd.DataFrame, pd.DataFrame]:
    sl = exio_slice(EXIO_YEAR, CENTRAL_FLOW)
    transport = transport_codes(lab.exio_products)
    selected = selected_destination_codes(lab.exio_products)
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    for fuel in BIOFUELS:
        z = pd.to_numeric(sl.Z_domestic.loc[fuel], errors="coerce").fillna(0.0)
        y = pd.to_numeric(sl.Y_region.loc[fuel], errors="coerce").fillna(0.0)
        z_total = float(z.sum())
        y_total = float(y.sum())
        total = z_total + y_total
        wholesale = float(z.get(WHOLESALE, 0.0))
        transport_total = float(z.reindex(transport).fillna(0.0).sum())
        red_flags: list[str] = []
        if fuel == "p24.g" and wholesale > y_total:
            red_flags.append("Z[p24.g,p51] > Y_total[p24.g]")
        if fuel == "p24.g" and safe_ratio(wholesale, total) > 0.25:
            red_flags.append("Z[p24.g,p51] / total_use[p24.g] > 0.25")
        if fuel == "p24.g" and safe_ratio(transport_total, z_total) < 0.05:
            red_flags.append("transport_share_of_Z[p24.g] < 0.05")
        summary_rows.append(
            {
                "year": EXIO_YEAR,
                "region": REGION,
                "exio_table_type": "pxp",
                "exio_flow_scope": CENTRAL_FLOW,
                "fuel_code": fuel,
                "fuel_label": label(lab.exio_products, fuel),
                "Z_row_total_intermediate_use": z_total,
                "Y_row_total_final_demand": y_total,
                "total_use": total,
                "share_intermediate": safe_ratio(z_total, total),
                "share_final": safe_ratio(y_total, total),
                "wholesale_value_meur": wholesale,
                "wholesale_share_of_Z": safe_ratio(wholesale, z_total),
                "wholesale_share_of_total_use": safe_ratio(wholesale, total),
                "transport_value_meur": transport_total,
                "transport_share_of_Z": safe_ratio(transport_total, z_total),
                "transport_share_of_total_use": safe_ratio(transport_total, total),
                "red_flags": "; ".join(red_flags),
            }
        )

        top = z.sort_values(ascending=False).head(30)
        for rank, (dest, value) in enumerate(top.items(), start=1):
            top_rows.append(
                {
                    "fuel_code": fuel,
                    "fuel_label": label(lab.exio_products, fuel),
                    "destination_code": str(dest),
                    "destination_label": label(lab.exio_products, str(dest)),
                    "Z_value_meur": float(value),
                    "share_of_fuel_Z": safe_ratio(float(value), z_total),
                    "share_of_fuel_total_use": safe_ratio(float(value), total),
                    "rank": rank,
                }
            )

        for dest in selected:
            if dest not in top.index and dest in z.index:
                value = float(z.get(dest, 0.0))
                top_rows.append(
                    {
                        "fuel_code": fuel,
                        "fuel_label": label(lab.exio_products, fuel),
                        "destination_code": dest,
                        "destination_label": label(lab.exio_products, dest),
                        "Z_value_meur": value,
                        "share_of_fuel_Z": safe_ratio(value, z_total),
                        "share_of_fuel_total_use": safe_ratio(value, total),
                        "rank": np.nan,
                    }
                )

    out_summary = pd.DataFrame(summary_rows)
    out_top = pd.DataFrame(top_rows)
    write_csv(out_summary, "biofuel_exio_z_y_summary.csv")
    write_csv(out_top, "biofuel_exio_top_destinations.csv")
    del sl
    gc.collect()
    return out_summary, out_top


def audit_exio_c20_parent(lab: LabelMaps, c20_products: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exio_flow, _euro_flow in FLOWS:
        sl = exio_slice(EXIO_YEAR, exio_flow)
        z = sl.Z_domestic
        c20_available = [p for p in c20_products if p in z.index]
        exio_c20 = float(z.loc[c20_available, WHOLESALE].sum())
        bio = {fuel: float(z.loc[fuel, WHOLESALE]) for fuel in BIOFUELS}
        liquid_total = float(sum(bio.values()))
        rows.append(
            {
                "year": EXIO_YEAR,
                "region": REGION,
                "exio_table_type": "pxp",
                "exio_flow_scope": exio_flow,
                "destination_code": WHOLESALE,
                "destination_label": label(lab.exio_products, WHOLESALE),
                "exio_c20_to_destination_meur": exio_c20,
                "biogasoline_to_destination_meur": bio["p24.g"],
                "biodiesels_to_destination_meur": bio["p24.h"],
                "other_liquid_biofuels_to_destination_meur": bio["p24.i"],
                "liquid_biofuels_total_to_destination_meur": liquid_total,
                "biogasoline_share_of_exio_c20": safe_ratio(bio["p24.g"], exio_c20),
                "liquid_biofuels_share_of_exio_c20": safe_ratio(liquid_total, exio_c20),
                "n_exio_c20_products": len(c20_available),
            }
        )
        del sl, z
        gc.collect()
    out = pd.DataFrame(rows)
    write_csv(out, "biofuel_exio_c20_parent_check.csv")
    return out


def load_eurostat_cell(year: int, flow: str) -> tuple[float, dict[str, str]]:
    parsed = load_eurostat_iot(
        dataset="naio_10_cp1700",
        filters={"geo": REGION, "time": str(year), "freq": "A", "unit": "MIO_EUR", "stk_flow": flow},
        row_dim="prd_ava",
        col_dim="prd_use",
        storage_folder=default_data_root() / "eurostat",
        force_download=False,
        force_reparse=False,
    )
    dims = parsed.meta.get("dimensions", {})
    meta = {
        "row_label": str(dims.get("prd_ava", {}).get(EUROSTAT_C20, EUROSTAT_C20)),
        "column_label": str(dims.get("prd_use", {}).get(EUROSTAT_G46, EUROSTAT_G46)),
    }
    value = float(parsed.Z.loc[EUROSTAT_C20, EUROSTAT_G46])
    del parsed
    gc.collect()
    return value, meta


def audit_exio_vs_eurostat(parent: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parent_by_flow = {str(r.exio_flow_scope): r for r in parent.itertuples(index=False)}
    flow_map = {"domestic": "DOM", "imports": "IMP", "total": "TOTAL"}
    for year in EUROSTAT_YEARS:
        for exio_flow, euro_flow in flow_map.items():
            euro_value, meta = load_eurostat_cell(year, euro_flow)
            p = parent_by_flow[exio_flow]
            c20 = float(p.exio_c20_to_destination_meur)
            biogas = float(p.biogasoline_to_destination_meur)
            liquid = float(p.liquid_biofuels_total_to_destination_meur)
            rows.append(
                {
                    "exio_year": EXIO_YEAR,
                    "eurostat_year": year,
                    "region": REGION,
                    "exio_table_type": "pxp",
                    "exio_flow_scope": exio_flow,
                    "eurostat_table": "naio_10_cp1700",
                    "eurostat_stk_flow": euro_flow,
                    "valuation": "basic prices, per Eurostat product-by-product IOT table metadata/convention",
                    "unit": "MIO_EUR",
                    "exio_destination_code": WHOLESALE,
                    "eurostat_row_code": EUROSTAT_C20,
                    "eurostat_row_label": meta["row_label"],
                    "eurostat_column_code": EUROSTAT_G46,
                    "eurostat_column_label": meta["column_label"],
                    "eurostat_c20_to_g46_meur": euro_value,
                    "exio_c20_to_p51_meur": c20,
                    "exio_biogasoline_to_p51_meur": biogas,
                    "exio_liquid_biofuels_to_p51_meur": liquid,
                    "exio_c20_to_eurostat_c20_ratio": safe_ratio(c20, euro_value),
                    "exio_biogasoline_to_eurostat_c20_ratio": safe_ratio(biogas, euro_value),
                    "exio_liquid_biofuels_to_eurostat_c20_ratio": safe_ratio(liquid, euro_value),
                    "biogasoline_exceeds_parent": bool(biogas > euro_value),
                    "liquid_biofuels_exceed_parent": bool(liquid > euro_value),
                    "exio_c20_very_different": bool(safe_ratio(abs(c20 - euro_value), euro_value) > 0.5),
                    "comparison_note": "Like-for-like on region, flow scope, row/column concept, unit, and table year only for eurostat_year == exio_year.",
                }
            )
    out = pd.DataFrame(rows)
    write_csv(out, "biofuel_exio_vs_eurostat_parent_check.csv")
    return out


def audit_pxp_ixi(lab: LabelMaps) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ixi_zip = default_data_root() / f"exiobase3_{EXIO_YEAR}" / f"IOT_{EXIO_YEAR}_ixi.zip"
    ixi_slice = default_data_root() / f"exiobase3_{EXIO_YEAR}" / "slices" / REGION / "ixi"
    sl = exio_slice(EXIO_YEAR, CENTRAL_FLOW)
    for fuel in BIOFUELS:
        z = pd.to_numeric(sl.Z_domestic.loc[fuel], errors="coerce").fillna(0.0)
        value = float(z.get(WHOLESALE, 0.0))
        rows.append(
            {
                "table_type": "pxp",
                "fuel_or_activity_code": fuel,
                "fuel_or_activity_label": label(lab.exio_products, fuel),
                "destination_code": WHOLESALE,
                "destination_label": label(lab.exio_products, WHOLESALE),
                "value_meur": value,
                "share_of_row_total": safe_ratio(value, float(z.sum())),
                "notes": "Product-by-product EXIOBASE destination-use slice.",
            }
        )
    del sl
    gc.collect()
    for fuel in BIOFUELS:
        rows.append(
            {
                "table_type": "ixi",
                "fuel_or_activity_code": fuel.replace("p", "i", 1),
                "fuel_or_activity_label": f"Activity analogue for {label(lab.exio_products, fuel)}",
                "destination_code": "i51",
                "destination_label": label(lab.exio_industries, "i51"),
                "value_meur": np.nan,
                "share_of_row_total": np.nan,
                "notes": f"IXI comparison unavailable: no local {ixi_zip.name} or cached IXI slice found at {ixi_slice}.",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "biofuel_pxp_vs_ixi_check.csv")
    return out


def audit_timeseries(lab: LabelMaps) -> pd.DataFrame:
    transport = transport_codes(lab.exio_products)
    rows: list[dict[str, Any]] = []
    for year_dir in sorted(default_data_root().glob("exiobase3_*")):
        try:
            year = int(year_dir.name.split("_", 1)[1])
        except Exception:
            continue
        slice_path = year_dir / "slices" / REGION / "pxp" / "Z_total.pkl"
        if not slice_path.exists():
            continue
        sl = exio_slice(year, CENTRAL_FLOW)
        z = pd.to_numeric(sl.Z_domestic.loc["p24.g"], errors="coerce").fillna(0.0)
        y = pd.to_numeric(sl.Y_region.loc["p24.g"], errors="coerce").fillna(0.0)
        total = float(z.sum() + y.sum())
        rows.append(
            {
                "year": year,
                "region": REGION,
                "fuel_code": "p24.g",
                "fuel_label": label(lab.exio_products, "p24.g"),
                "share_wholesale_biogasoline": safe_ratio(float(z.get(WHOLESALE, 0.0)), total),
                "share_transport_biogasoline": safe_ratio(float(z.reindex(transport).fillna(0.0).sum()), total),
                "transport_share_of_Z_biogasoline": safe_ratio(float(z.reindex(transport).fillna(0.0).sum()), float(z.sum())),
                "final_demand_share_biogasoline": safe_ratio(float(y.sum()), total),
                "total_use_meur": total,
                "notes": "Computed only for locally cached AT total-use slices; no multi-country EXIO slices were available locally.",
            }
        )
        del sl, z, y
        gc.collect()
    out = pd.DataFrame(rows)
    write_csv(out, "biofuel_timeseries_country_sanity.csv")
    return out


def audit_energy_balance() -> pd.DataFrame:
    nrg_bal_codes = [
        "FC_IND_CON_E",
        "FC_IND_CPC_E",
        "FC_IND_FBT_E",
        "FC_IND_IS_E",
        "FC_IND_MAC_E",
        "FC_IND_MQ_E",
        "FC_IND_NFM_E",
        "FC_IND_NMM_E",
        "FC_IND_NSP_E",
        "FC_IND_PPP_E",
        "FC_IND_TE_E",
        "FC_IND_TL_E",
        "FC_IND_WP_E",
        "FC_OTH_AF_E",
        "FC_OTH_CP_E",
        "FC_OTH_FISH_E",
        "FC_OTH_HH_E",
        "FC_TRA_DAVI_E",
        "FC_TRA_DNAVI_E",
        "FC_TRA_RAIL_E",
        "FC_TRA_ROAD_E",
    ]
    df, meta = load_eurostat_table_2d(
        dataset="nrg_bal_c",
        filters={"geo": REGION, "time": str(EXIO_YEAR), "unit": "GWH", "freq": "A", "nrg_bal": nrg_bal_codes},
        row_dim="nrg_bal",
        col_dim="siec",
        storage_folder=default_data_root() / "eurostat",
        force_download=False,
        force_reparse=False,
    )
    siec_labels = meta.get("dimensions", {}).get("siec", {})
    nrg_labels = meta.get("dimensions", {}).get("nrg_bal", {})
    liquid_codes = ["R5210P", "R5210B", "R5220P", "R5220B", "R5230P", "R5230B", "R5290"]
    available = [c for c in liquid_codes if c in df.columns]
    rows: list[dict[str, Any]] = []
    for flow_code, row in df.iterrows():
        values = pd.to_numeric(row.reindex(available), errors="coerce").fillna(0.0)
        rows.append(
            {
                "year": EXIO_YEAR,
                "region": REGION,
                "nrg_bal_code": str(flow_code),
                "nrg_bal_label": str(nrg_labels.get(str(flow_code), flow_code)),
                "liquid_biofuels_gwh": float(values.sum()),
                "biogasoline_gwh": float(values.reindex(["R5210P", "R5210B"]).fillna(0.0).sum()),
                "biodiesels_gwh": float(values.reindex(["R5220P", "R5220B"]).fillna(0.0).sum()),
                "other_liquid_biofuels_gwh": float(values.reindex(["R5230P", "R5230B", "R5290"]).fillna(0.0).sum()),
                "included_siec_codes": "; ".join(f"{c}: {siec_labels.get(c, c)}" for c in available),
            }
        )
    out = pd.DataFrame(rows)
    total_final = float(out["liquid_biofuels_gwh"].sum())
    road = float(out.loc[out["nrg_bal_code"] == "FC_TRA_ROAD_E", "liquid_biofuels_gwh"].sum())
    out["share_of_liquid_biofuels_final_use"] = out["liquid_biofuels_gwh"].map(lambda v: safe_ratio(float(v), total_final))
    out["energy_balance_road_transport_share"] = safe_ratio(road, total_final)
    write_csv(out, "biofuel_energy_balance_plausibility.csv")
    del df
    gc.collect()
    return out


def write_settings_report() -> None:
    text = f"""# Liquid biofuels audit settings

- country / region: {REGION}
- EXIOBASE year: {EXIO_YEAR}
- Eurostat years: {', '.join(map(str, EUROSTAT_YEARS))}
- EXIO table type: pxp
- EXIO flow scopes: domestic, imports, total when cached slices are available
- Eurostat table: product-by-product IOT `naio_10_cp1700`
- Eurostat stk_flow scopes: DOM, IMP, TOTAL
- valuation: Eurostat product-by-product IOT at basic prices; EXIOBASE monetary IOT values are treated as million EUR current-price monetary flows from the cached EXIOBASE table
- unit: million EUR for IOT checks; GWh for energy-balance plausibility
- row dimensions: EXIO product rows `p24.g`, `p24.h`, `p24.i`; Eurostat row `CPA_C20`
- column dimensions: EXIO product destination columns, especially `p51`; Eurostat column `CPA_G46`
- central comparison: EXIO 2020 total-use slice versus Eurostat 2020 `TOTAL`
- sensitivity checks: EXIO 2020 domestic/import scopes and Eurostat 2021 DOM/IMP/TOTAL

The audit labels any year or scope mismatch explicitly. It does not compare EXIO `total` to Eurostat `DOM` as a central result.
"""
    (OUTDIR / "biofuel_year_scope_valuation_check.md").write_text(text, encoding="utf-8")


def fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):,.{digits}f}"


def write_report(
    z_y: pd.DataFrame,
    parent: pd.DataFrame,
    vs: pd.DataFrame,
    pxi: pd.DataFrame,
    ts: pd.DataFrame,
    energy: pd.DataFrame,
) -> None:
    central = vs[(vs["eurostat_year"] == EXIO_YEAR) & (vs["exio_flow_scope"] == CENTRAL_FLOW)].iloc[0]
    zy_bio = z_y[z_y["fuel_code"] == "p24.g"].iloc[0]
    parent_total = parent[parent["exio_flow_scope"] == CENTRAL_FLOW].iloc[0]
    road_share = float(energy["energy_balance_road_transport_share"].dropna().iloc[0]) if not energy.empty else np.nan
    pxi_unavailable = pxi[pxi["table_type"] == "ixi"]["value_meur"].isna().all()
    sut_available = False

    strong_red_flag = bool(central["biogasoline_exceeds_parent"] or central["liquid_biofuels_exceed_parent"])
    if pxi_unavailable or not sut_available:
        resolution = "not resolved"
    else:
        resolution = "not resolved"
    recommendation = (
        "For transition-direction analysis, p24.g/h/i should be classified statistically under CPA_C20 and "
        "analytically as ENERGY_GREEN where appropriate, but their EXIOBASE monetary PXP destination columns "
        "should not be interpreted as physical sectoral combustion. Physical use/emissions should be handled "
        "via a separate Eurostat Energy Balance satellite account."
    )

    text = f"""# Liquid biofuels audit: EXIOBASE vs Eurostat

## Executive summary

- Main finding: **{'strong red flag' if strong_red_flag else 'red flag'}**. EXIOBASE Austria 2020 PXP places {fmt(float(central['exio_biogasoline_to_p51_meur']))} MEUR of `p24.g` Biogasoline into `p51` wholesale, while Eurostat Austria 2020 `CPA_C20 -> CPA_G46` is {fmt(float(central['eurostat_c20_to_g46_meur']))} MEUR.
- EXIO `p24.g -> p51` exceeds Eurostat `CPA_C20 -> CPA_G46`: **{bool(central['biogasoline_exceeds_parent'])}**. EXIO liquid biofuels total `p24.g/h/i -> p51` is {fmt(float(central['exio_liquid_biofuels_to_p51_meur']))} MEUR and also exceeds the Eurostat parent cell: **{bool(central['liquid_biofuels_exceed_parent'])}**.
- Scope/year/valuation do not explain the central red flag: the central comparison is AT, 2020, total flow, product-by-product IOT, million EUR. DOM/IMP and 2021 sensitivity rows are written separately in `biofuel_exio_vs_eurostat_parent_check.csv`.
- Recommended treatment: {recommendation}

## Data and comparability

- EXIOBASE: country/region `{REGION}`, year `{EXIO_YEAR}`, table type `pxp`, flow scopes `domestic`, `imports`, and `total`.
- Eurostat: `naio_10_cp1700`, years `{EUROSTAT_YEARS[0]}` and `{EUROSTAT_YEARS[1]}`, `stk_flow` values `DOM`, `IMP`, and `TOTAL`; row dimension `prd_ava`, column dimension `prd_use`.
- Unit: million EUR for EXIO and Eurostat IOT checks. Energy-balance plausibility uses GWh and is qualitative, not converted to monetary terms.
- Valuation: Eurostat product-by-product IOT is treated as basic-price product-by-product data. EXIO values are current-price monetary IOT flows from cached EXIOBASE PXP slices.
- Row/column mappings: EXIO `p24.g`, `p24.h`, and `p24.i` map statistically to `CPA_C20`; EXIO `p51` maps to `CPA_G46`.

## EXIOBASE Z/Y allocation

- `p24.g` Biogasoline total use in the AT 2020 total-use slice is {fmt(float(zy_bio['total_use']))} MEUR: intermediate use {fmt(float(zy_bio['Z_row_total_intermediate_use']))} MEUR and final demand {fmt(float(zy_bio['Y_row_total_final_demand']))} MEUR.
- `p24.g -> p51` wholesale is {fmt(float(zy_bio['wholesale_value_meur']))} MEUR, equal to {fmt(100 * float(zy_bio['wholesale_share_of_total_use']), 1)}% of total biogasoline use and {fmt(100 * float(zy_bio['wholesale_share_of_Z']), 1)}% of intermediate use.
- Transport-like EXIO destination columns receive {fmt(float(zy_bio['transport_value_meur']))} MEUR of `p24.g`, or {fmt(100 * float(zy_bio['transport_share_of_Z']), 2)}% of intermediate use.
- The top-destination table is written to `biofuel_exio_top_destinations.csv`.

## EXIO C20 parent consistency check

- EXIO products mapped to `CPA_C20`: {int(parent_total['n_exio_c20_products'])}.
- EXIO `CPA_C20`-mapped products to `p51`: {fmt(float(parent_total['exio_c20_to_destination_meur']))} MEUR.
- Liquid biofuels to `p51`: {fmt(float(parent_total['liquid_biofuels_total_to_destination_meur']))} MEUR, {fmt(100 * float(parent_total['liquid_biofuels_share_of_exio_c20']), 1)}% of EXIO's own C20-to-wholesale parent aggregate.
- Biogasoline alone is {fmt(100 * float(parent_total['biogasoline_share_of_exio_c20']), 1)}% of EXIO's own C20-to-wholesale parent aggregate.

## EXIO vs Eurostat CPA_C20->G46 check

- Central ratio `EXIO C20 -> p51 / Eurostat C20 -> G46`: {fmt(float(central['exio_c20_to_eurostat_c20_ratio']), 2)}.
- Central ratio `EXIO p24.g -> p51 / Eurostat C20 -> G46`: {fmt(float(central['exio_biogasoline_to_eurostat_c20_ratio']), 2)}.
- Central ratio `EXIO p24.g/h/i -> p51 / Eurostat C20 -> G46`: {fmt(float(central['exio_liquid_biofuels_to_eurostat_c20_ratio']), 2)}.
- Decision rule result: **strong red flag**, because a subset product (`p24.g`) and the liquid-biofuel subset (`p24.g/h/i`) exceed the Eurostat parent aggregate after the like-for-like 2020 total-flow check.

## PXP vs IXI / SUT check

- PXP shows the high `p24.g -> p51` allocation.
- IXI comparison: unavailable in this local cache. No `IOT_2020_ixi.zip` or cached `slices/AT/ixi` object was found, so the audit cannot distinguish a PXP transformation artefact from a deeper EXIO allocation issue using IXI.
- Underlying SUT/product-by-industry check: no EXIO SUT or use-table object was found in the repo/data cache. The decisive upstream object is not available in this repo.
- Decision rule result: **{resolution}**. Even in this unresolved case, EXIO `p24.g` destination columns should not be treated as physical combustion allocation.

## Energy-balance plausibility

- Eurostat `nrg_bal_c` Austria 2020 liquid-biofuel final-use rows show road transport receives {fmt(100 * road_share, 1)}% of liquid-biofuel final energy use across the selected final-use categories.
- This is a physical energy-balance concept, not a monetary PXP concept. The qualitative mismatch is nevertheless material: physical liquid-biofuel use is concentrated in road transport, while EXIO PXP biogasoline is concentrated in non-transport monetary destination columns including wholesale, R&D, and real estate.
- If EXIO is used as an embodied-input monetary trace, this can be defensible as a monetary construction. If it is used as direct combustion allocation, the mismatch invalidates that interpretation.

## Interpretation

- Wholesale and retail output are trade-margin services, not the full value of goods resold.
- Goods purchased for resale should not be read as wholesale/retail intermediate consumption in the same way as own fuel use.
- If `Z[p24.g,p51]` is interpreted literally, it means wholesale's own intermediate use of biogasoline, not biogasoline being resold to households.
- Therefore, a large `p24.g -> p51` cell is credible as sectoral fuel use only if wholesale plausibly burns that fuel itself. The available physical energy-balance evidence does not support wholesale as the major direct user.
- This audit does not claim EXIOBASE violated SNA/ESA accounting. The upstream SUT/use table needed to prove that is unavailable.

## Recommendation

- Quarantine EXIO liquid-biofuel PXP destination columns for direct combustion/emissions allocation.
- Use EXIO liquid-biofuel monetary PXP data only as an embodied-input allocation unless separately validated from upstream EXIO SUT/use-table evidence.
- {recommendation}

## Output files

- `biofuel_exio_z_y_summary.csv`
- `biofuel_exio_top_destinations.csv`
- `biofuel_exio_c20_parent_check.csv`
- `biofuel_exio_vs_eurostat_parent_check.csv`
- `biofuel_pxp_vs_ixi_check.csv`
- `biofuel_year_scope_valuation_check.md`
- `biofuel_timeseries_country_sanity.csv`
- `biofuel_energy_balance_plausibility.csv`
"""
    (OUTDIR / "biofuel_audit_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lab = labels()
    write_settings_report()
    c20 = c20_products_from_mapping()
    z_y, _top = audit_exio_z_y(lab)
    parent = audit_exio_c20_parent(lab, c20)
    vs = audit_exio_vs_eurostat(parent)
    pxi = audit_pxp_ixi(lab)
    ts = audit_timeseries(lab)
    energy = audit_energy_balance()
    write_report(z_y, parent, vs, pxi, ts, energy)
    print(f"Wrote liquid-biofuels audit outputs to {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

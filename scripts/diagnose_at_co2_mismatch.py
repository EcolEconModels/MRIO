#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.diagnostic_concordance import (
    allocate_category_matrix_by_share,
    build_exio_nace_audit,
    build_mapping_audit,
    build_weighted_value_comparison,
    load_exiobase_nace_concordance,
    write_canonical_mapping_csv,
)
from ecolecon_mrio.eurostat_energy_balances import (
    PRIMARY_SOURCES,
    build_tabelle1_from_eurostat,
    compute_mapping_shares_from_outputs,
)
from ecolecon_mrio.eurostat_air_emissions import load_eurostat_air_emissions_by_nace
from ecolecon_mrio.eurostat_iot import load_eurostat_iot
from ecolecon_mrio.exiobase3 import load_exiobase3_region_slice
from ecolecon_mrio.exiobase3_extensions import extract_emissions_and_intensity, extract_quantity_and_price
from ecolecon_mrio.mapping_csv import load_category_mapping_frame, load_weighted_mapping_frame


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _meta_dimension_labels(meta: Mapping[str, Any], dimension: str) -> dict[str, str]:
    raw_dimensions = meta.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        return {}
    dimensions = cast(Mapping[str, object], raw_dimensions)

    raw_labels = dimensions.get(dimension)
    if not isinstance(raw_labels, Mapping):
        return {}
    labels = cast(Mapping[object, object], raw_labels)

    return {str(key): str(value) for key, value in labels.items()}


def _float_scalar(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _load_eurostat_outputs(*, geo: str, year: int, storage_folder: Path) -> pd.Series:
    parsed = load_eurostat_iot(
        dataset="naio_10_cp1700",
        filters={"geo": str(geo), "time": str(year), "freq": "A", "unit": "MIO_EUR", "stk_flow": "TOTAL"},
        row_dim="prd_ava",
        col_dim="prd_use",
        storage_folder=storage_folder,
        force_download=False,
        force_reparse=False,
    )
    x = (parsed.Z.sum(axis=1) + parsed.Y.sum(axis=1)).astype("float64")
    x.index = x.index.astype(str)
    x.name = "eurostat_output"
    return x


def _load_exio_outputs(*, region: str, year: int, data_root: Path) -> pd.Series:
    exio_slice = load_exiobase3_region_slice(
        year=year,
        region=region,
        system="pxp",
        flow="total",
        cache_root=data_root,
    )
    x = exio_slice.x_region.astype("float64")
    x.index = x.index.astype(str)
    x.name = "exio_output"
    return x


def _load_exio_slice(*, region: str, year: int, data_root: Path):
    return load_exiobase3_region_slice(
        year=year,
        region=region,
        system="pxp",
        flow="total",
        cache_root=data_root,
    )


def _load_eurostat_iot_bundle(*, geo: str, year: int, storage_folder: Path):
    parsed = load_eurostat_iot(
        dataset="naio_10_cp1700",
        filters={"geo": str(geo), "time": str(year), "freq": "A", "unit": "MIO_EUR", "stk_flow": "TOTAL"},
        row_dim="prd_ava",
        col_dim="prd_use",
        storage_folder=storage_folder,
        force_download=False,
        force_reparse=False,
    )
    y_cols: list[str] = []
    parsed_y_cols = [str(col) for col in parsed.Y.columns]
    p3_detail = [code for code in ("P3_S13", "P3_S14", "P3_S15") if code in parsed_y_cols]
    if p3_detail:
        y_cols.extend(p3_detail)
    elif "P3" in parsed_y_cols:
        y_cols.append("P3")
    p5_detail = [code for code in ("P51G", "P52", "P53") if code in parsed_y_cols]
    if p5_detail:
        y_cols.extend(p5_detail)
    elif "P5" in parsed_y_cols:
        y_cols.append("P5")
    if "P6" in parsed_y_cols:
        y_cols.append("P6")
    else:
        y_cols.extend(sorted(code for code in parsed_y_cols if code.startswith("P6_")))
    y_block = parsed.Y.loc[:, y_cols].astype("float64") if y_cols else parsed.Y.iloc[:, 0:0].copy()
    x = (parsed.Z.sum(axis=1) + y_block.sum(axis=1)).astype("float64")
    x.index = x.index.astype(str)
    return parsed, x


def _match_target_sectors_to_cpa_codes(target_labels: list[str], *, cpa_code_to_label: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    import difflib

    norm_to_code: dict[str, list[str]] = {}
    for code, label in cpa_code_to_label.items():
        norm_to_code.setdefault(" ".join(str(label).strip().casefold().split()), []).append(str(code))

    out: dict[str, str] = {}
    missing: list[str] = []
    for label in target_labels:
        key = " ".join(str(label).strip().casefold().split())
        codes = norm_to_code.get(key, [])
        if len(codes) == 1:
            out[str(label)] = codes[0]
            continue
        if not codes:
            close = difflib.get_close_matches(key, list(norm_to_code.keys()), n=3, cutoff=0.88)
            if len(close) == 1 and len(norm_to_code[close[0]]) == 1:
                out[str(label)] = norm_to_code[close[0]][0]
                continue
        missing.append(str(label))
    return out, missing


def _cpa_to_nace(code: str) -> str:
    s = str(code).strip()
    return s[len("CPA_") :] if s.startswith("CPA_") else s


def _run_energy_diagnostics(
    *,
    geo: str,
    year: int,
    exio_year: int,
    region: str,
    data_root: Path,
    weighted_mapping: pd.DataFrame,
    energy_mapping: pd.DataFrame,
    outdir: Path,
) -> None:
    parsed, eurostat_output = _load_eurostat_iot_bundle(geo=geo, year=year, storage_folder=data_root / "eurostat")
    energy_mapping = energy_mapping.copy()
    targets = energy_mapping["TargetSector"].drop_duplicates().astype(str).tolist()
    row_labels = _meta_dimension_labels(parsed.meta, "prd_ava")
    target_to_cpa, _missing_targets = _match_target_sectors_to_cpa_codes(targets, cpa_code_to_label=row_labels)
    target_output = {target: float(eurostat_output.get(target_to_cpa.get(target, ""), 0.0)) for target in targets}

    mapping_w = compute_mapping_shares_from_outputs(energy_mapping, target_output_by_target=target_output)
    eb = build_tabelle1_from_eurostat(
        geo=geo,
        time=year,
        categories=energy_mapping["Category"].drop_duplicates().astype(str).tolist(),
        unit="GWH",
        storage_folder=data_root / "eurostat",
        force_download=False,
        force_reparse=False,
    )
    unrolled_sources = eb.unrolled_sources.set_index("Category").loc[:, list(PRIMARY_SOURCES)].astype("float64")

    euro_detail = allocate_category_matrix_by_share(
        unrolled_sources,
        mapping_w,
        share_col="Share_used",
    )
    euro_target = euro_detail.target_matrix.reindex(targets).fillna(0.0)
    euro_target["total_energy_gwh"] = euro_target[list(PRIMARY_SOURCES)].sum(axis=1)
    euro_target_by_cpa = pd.Series(
        {
            target_to_cpa[target]: _float_scalar(euro_target.at[target, "total_energy_gwh"])
            for target in targets
            if target in target_to_cpa
        },
        dtype="float64",
    )

    exio_slice = _load_exio_slice(region=region, year=exio_year, data_root=data_root)
    exio_qp = extract_quantity_and_price(
        container=exio_slice.source_zip,
        region=region,
        sectors=list(map(str, exio_slice.sectors)),
        monetary_output=exio_slice.x_region,
        extensions=("energy",),
        stressor_keywords=("Energy use",),
        cache_dir=data_root / "_ext_cache",
    )
    if exio_qp is None or exio_qp.empty:
        raise RuntimeError("No EXIO energy quantities could be extracted for diagnostics.")
    preferred_stressor = "Energy use - Final"
    stressor_df = exio_qp[exio_qp["stressor"].astype(str) == preferred_stressor].copy()
    if stressor_df.empty:
        preferred_stressor = "Energy use - Gross"
        stressor_df = exio_qp[exio_qp["stressor"].astype(str) == preferred_stressor].copy()
    if stressor_df.empty:
        raise RuntimeError("Could not find a usable EXIO energy stressor for total-energy comparison.")
    exio_total_energy = stressor_df.groupby("exio_sector")["quantity"].sum().astype("float64")

    energy_cmp = build_weighted_value_comparison(
        exio_total_energy,
        euro_target_by_cpa,
        weighted_mapping,
        scale_to_reference=True,
    )
    energy_cmp.insert(1, "exio_energy_stressor", preferred_stressor)
    cpa_to_target_label = {cpa: target for target, cpa in target_to_cpa.items()}
    energy_cmp.insert(1, "target_label", energy_cmp["target_sector"].map(cpa_to_target_label))
    _write_csv(energy_cmp, outdir / "energy_mismatches_summary.csv")

    fuel_rows: list[pd.DataFrame] = []
    for fuel in PRIMARY_SOURCES:
        source_by_category = unrolled_sources[fuel].astype("float64")
        reconstructed_by_category = euro_detail.category_reconstructed[fuel].astype("float64")
        closure = pd.DataFrame(
            {
                "fuel": fuel,
                "category": source_by_category.index.astype(str),
                "source_total_gwh": source_by_category.values,
                "reconstructed_total_gwh": reconstructed_by_category.reindex(source_by_category.index).fillna(0.0).values,
            }
        )
        closure["abs_gap"] = (closure["source_total_gwh"] - closure["reconstructed_total_gwh"]).abs()
        closure["pct_gap"] = 0.0
        nonzero = closure["source_total_gwh"] != 0.0
        closure.loc[nonzero, "pct_gap"] = (
            closure.loc[nonzero, "abs_gap"] / closure.loc[nonzero, "source_total_gwh"].abs()
        )
        closure["comparison_type"] = "eurostat-disaggregation-reconstruction"
        fuel_rows.append(closure)
    _write_csv(pd.concat(fuel_rows, ignore_index=True), outdir / "energy_mismatches_by_fuel.csv")

    closure_rows: list[dict[str, object]] = []
    euro_reconstructed = euro_detail.category_reconstructed.reindex(unrolled_sources.index).fillna(0.0).astype("float64")
    euro_source = unrolled_sources
    for category in euro_source.index:
        src_row = cast(pd.Series, euro_source.loc[category])
        rec_row = cast(pd.Series, euro_reconstructed.loc[category])
        src_total = float(src_row.sum())
        rec_total = float(rec_row.sum())
        closure_rows.append(
            {
                "category": category,
                "source_total_gwh": src_total,
                "reconstructed_total_gwh": rec_total,
                "abs_gap": abs(src_total - rec_total),
                "pct_gap": abs(src_total - rec_total) / abs(src_total) if src_total else 0.0,
            }
        )
    _write_csv(pd.DataFrame(closure_rows), outdir / "energy_closure_detail.csv")


def _run_co2_diagnostics(
    *,
    geo: str,
    year: int,
    exio_year: int,
    region: str,
    data_root: Path,
    weighted_mapping: pd.DataFrame,
    energy_mapping: pd.DataFrame,
    outdir: Path,
) -> None:
    parsed, _eurostat_output = _load_eurostat_iot_bundle(geo=geo, year=year, storage_folder=data_root / "eurostat")
    energy_mapping = energy_mapping.copy()
    targets = energy_mapping["TargetSector"].drop_duplicates().astype(str).tolist()
    row_labels = _meta_dimension_labels(parsed.meta, "prd_ava")
    target_to_cpa, _missing_targets = _match_target_sectors_to_cpa_codes(targets, cpa_code_to_label=row_labels)

    eurostat_co2 = load_eurostat_air_emissions_by_nace(
        geo=geo,
        time=year,
        airpol="CO2",
        unit="THS_T",
        storage_folder=data_root / "eurostat",
        force_download=False,
        force_reparse=False,
    ).emissions_tonnes
    eurostat_co2.index = eurostat_co2.index.astype(str)
    eurostat_co2_by_cpa = pd.Series(
        {
            target_to_cpa[target]: float(eurostat_co2.get(_cpa_to_nace(target_to_cpa[target]), float("nan")))
            for target in targets
            if target in target_to_cpa
        },
        dtype="float64",
    )

    exio_slice = _load_exio_slice(region=region, year=exio_year, data_root=data_root)
    exio_emissions = extract_emissions_and_intensity(
        container=exio_slice.source_zip,
        region=region,
        sectors=list(map(str, exio_slice.sectors)),
        monetary_output=exio_slice.x_region,
        extension="air_emissions",
        stressor_keywords=("CO2 - combustion - air",),
        cache_dir=data_root / "_ext_cache",
    )
    if exio_emissions is None or exio_emissions.empty:
        raise RuntimeError("No EXIO air emissions could be extracted for CO2 diagnostics.")
    exio_emissions_by_sector = exio_emissions.groupby("exio_sector")["emission"].sum().astype("float64")
    stressor_names = "|".join(sorted(set(map(str, exio_emissions["stressor"]))))

    co2_cmp = build_weighted_value_comparison(
        exio_emissions_by_sector,
        eurostat_co2_by_cpa,
        weighted_mapping,
        scale_to_reference=True,
    )
    cpa_to_target_label = {cpa: target for target, cpa in target_to_cpa.items()}
    co2_cmp.insert(1, "target_label", co2_cmp["target_sector"].map(cpa_to_target_label))
    co2_cmp.insert(2, "exio_stressor", stressor_names)
    _write_csv(co2_cmp, outdir / "co2_intensity_comparison.csv")

    special_cases = co2_cmp[
        co2_cmp["target_sector"].astype(str).isin(["CPA_D", "CPA_E37-39"])
        | co2_cmp["target_label"].astype(str).str.contains("household", case=False, na=False)
    ].copy()
    _write_csv(special_cases, outdir / "co2_special_cases.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit concordances and compare Eurostat vs EXIO monetary outputs.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--exio-year", type=int, default=None)
    parser.add_argument("--region", default="AT")
    parser.add_argument(
        "--weighted-mapping-csv",
        default="scripts/mappings/exio_to_eurostat_cpa.csv",
        help="Canonical weighted EXIO->Eurostat mapping CSV to audit and compare.",
    )
    parser.add_argument(
        "--energy-mapping-csv",
        default="scripts/mappings/energy_balance_to_cpa65_products_mapping.csv",
        help="Canonical energy-balance mapping CSV for energy diagnostics; despite the legacy filename, it maps activity-defined Eurostat energy-balance rows into the repo's 65-product CPA-based IO targets.",
    )
    parser.add_argument(
        "--exio-nace-concordance-xlsx",
        default="data/exiobase_concordance_sources/table_correspondance_nace-gics-exiobase_20230816.xlsx",
        help="Vendored EXIOBASE to NACE concordance workbook to audit.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Diagnostics output directory. Defaults to outputs/diagnostics/at_<year>.",
    )
    parser.add_argument(
        "--write-canonical-mapping",
        action="store_true",
        help="Also write the normalized canonical weighted mapping CSV to the output directory.",
    )
    args = parser.parse_args()

    data_root = default_data_root()
    exio_year = args.exio_year if args.exio_year is not None else args.year
    outdir = Path(args.outdir) if args.outdir else Path("outputs") / "diagnostics" / f"at_{args.year}"
    outdir.mkdir(parents=True, exist_ok=True)

    mapping_df = load_weighted_mapping_frame(Path(args.weighted_mapping_csv))
    energy_mapping_df = load_category_mapping_frame(Path(args.energy_mapping_csv))
    audit_df = build_mapping_audit(mapping_df, mapping_family="weighted-exio-to-eurostat")
    _write_csv(audit_df, outdir / "concordance_audit.csv")

    if args.write_canonical_mapping:
        write_canonical_mapping_csv(mapping_df, outdir / "canonical_weighted_mapping.csv")

    exio_nace_df = load_exiobase_nace_concordance(Path(args.exio_nace_concordance_xlsx))
    exio_nace_audit = build_exio_nace_audit(exio_nace_df)
    _write_csv(exio_nace_audit, outdir / "exio_nace_concordance_audit.csv")

    eurostat_output = _load_eurostat_outputs(geo=args.geo, year=args.year, storage_folder=data_root / "eurostat")
    exio_output = _load_exio_outputs(region=args.region, year=exio_year, data_root=data_root)
    comparison_df = build_weighted_value_comparison(
        exio_output,
        eurostat_output,
        mapping_df,
        scale_to_reference=True,
    )
    comparison_df.insert(1, "eurostat_year", args.year)
    comparison_df.insert(2, "exio_year", exio_year)
    _write_csv(comparison_df, outdir / "monetary_mismatches_detail.csv")

    summary_df = pd.DataFrame(
        {
            "eurostat_year": [args.year],
            "exio_year": [exio_year],
            "target_sector_count": [int(comparison_df["target_sector"].nunique())],
            "max_abs_gap": [float(comparison_df["abs_gap"].max())],
            "mean_abs_gap": [float(comparison_df["abs_gap"].mean())],
            "max_pct_gap": [float(comparison_df["pct_gap"].max(skipna=True))],
            "mean_pct_gap": [float(comparison_df["pct_gap"].mean(skipna=True))],
        }
    )
    _write_csv(summary_df, outdir / "monetary_mismatches_summary.csv")

    _run_energy_diagnostics(
        geo=args.geo,
        year=args.year,
        exio_year=exio_year,
        region=args.region,
        data_root=data_root,
        weighted_mapping=mapping_df,
        energy_mapping=energy_mapping_df,
        outdir=outdir,
    )
    _run_co2_diagnostics(
        geo=args.geo,
        year=args.year,
        exio_year=exio_year,
        region=args.region,
        data_root=data_root,
        weighted_mapping=mapping_df,
        energy_mapping=energy_mapping_df,
        outdir=outdir,
    )
    print(f"Wrote diagnostics to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
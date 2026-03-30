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
    build_tabelle1_from_eurostat,
    compute_mapping_shares_from_outputs,
    disaggregate_unrolled_sources,
    read_category_target_mapping_csv,
    validate_energy_balance_closure,
)
from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import build_eurostat_product_use_reference, load_eurostat_iot


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


def _default_composites() -> dict[str, list[str]]:
    return {
        # Custom aggregates used in scripts/mappings/energy_balance_to_nace2_mapping.csv
        "metal and other minerals": [
            "Final consumption - industry sector - iron and steel - energy use",
            "Final consumption - industry sector - non-ferrous metals - energy use",
            "Final consumption - industry sector - non-metallic minerals - energy use",
        ],
        "transport sector - land": [
            "Final consumption - transport sector - road - energy use",
            "Final consumption - transport sector - rail - energy use",
        ],
    }


def _match_target_sectors_to_cpa_codes(
    target_labels: list[str],
    *,
    cpa_code_to_label: dict[str, str],
    allow_missing: bool = True,
    ignore_targets: set[str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    import difflib

    ignore = set(ignore_targets or set())

    # Normalized exact match first
    norm_to_code: dict[str, list[str]] = {}
    for code, lab in cpa_code_to_label.items():
        norm_to_code.setdefault(_norm(lab), []).append(str(code))

    out: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for lab in target_labels:
        if str(lab) in ignore:
            missing.append(str(lab))
            continue
        codes = norm_to_code.get(_norm(lab), [])
        if not codes:
            # Try substring containment in either direction (helps with minor wording differences)
            key = _norm(lab)
            cand = [c for k, cs in norm_to_code.items() if key in k or k in key for c in cs]
            cand = sorted(set(cand))
            if len(cand) == 1:
                out[lab] = cand[0]
                continue
            if len(cand) > 1:
                ambiguous[lab] = cand
                continue

            # Finally, fuzzy match on normalized strings.
            close = difflib.get_close_matches(key, list(norm_to_code.keys()), n=3, cutoff=0.88)
            if len(close) == 1 and len(norm_to_code.get(close[0], [])) == 1:
                out[lab] = norm_to_code[close[0]][0]
                continue
            missing.append(lab)
            continue
        if len(codes) > 1:
            ambiguous[lab] = codes
            continue
        out[lab] = codes[0]

    if ambiguous:
        first = next(iter(ambiguous.items()))
        raise SystemExit(f"Ambiguous CPA label matches. Example: {first[0]} -> {first[1]}")
    if missing and not allow_missing:
        show = ", ".join(missing[:10])
        raise SystemExit(f"Could not match some TargetSector labels to Eurostat CPA codes. Examples: {show}")
    return out, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="AT", help="Eurostat geo code (default AT)")
    ap.add_argument("--year", type=int, default=2021, help="Year (default 2021)")
    ap.add_argument(
        "--mapping",
        default="scripts/mappings/energy_balance_to_nace2_mapping.csv",
        help="Category -> 65-sector mapping CSV",
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
    ap.add_argument("--no-composites", action="store_true", help="Disable built-in composite categories")
    args = ap.parse_args()

    mapping_path = Path(args.mapping)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(out_path.stem + ".__writing__.xlsx")

    mapping_df = read_category_target_mapping_csv(mapping_path)
    categories = mapping_df["Category"].drop_duplicates().astype(str).tolist()
    composites = {} if args.no_composites else _default_composites()

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

    row_labels = (parsed.meta.get("dimensions", {}) or {}).get("prd_ava", {}) or {}
    row_labels = {str(k): str(v) for k, v in row_labels.items()}

    targets = mapping_df["TargetSector"].drop_duplicates().astype(str).tolist()
    target_to_cpa, _missing_targets = _match_target_sectors_to_cpa_codes(
        targets,
        cpa_code_to_label=row_labels,
        allow_missing=True,
        ignore_targets={"Households"},
    )
    target_output = {t: float(x.get(target_to_cpa.get(t, ""), 0.0)) for t in targets}

    # 3) Recompute split ratios using IOT outputs and disaggregate to 65 sectors
    mapping_w = compute_mapping_shares_from_outputs(mapping_df, target_output_by_target=target_output)
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

    # 5) Write single Excel output (atomically), using openpyxl directly for maximum Excel compatibility.
    sheets: list[tuple[str, pd.DataFrame]] = [
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

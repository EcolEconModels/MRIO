#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate an EXIOBASE-sector mapping CSV for splitting Eurostat energy sectors using "
            "pymrio's packaged EXIOBASE 3 sector concordance (no network required)."
        )
    )
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument(
        "--system",
        default="pxp",
        choices=["pxp", "ixi"],
        help="EXIOBASE system type for concordance (pxp recommended).",
    )
    p.add_argument(
        "--electricity-by-source",
        action="store_true",
        help="Keep electricity generation separated by source (adds more subsectors).",
    )
    p.add_argument(
        "--no-biofuels",
        action="store_true",
        help="Exclude biofuels from the CPA_C19 mapping.",
    )
    p.add_argument(
        "--coke-oven-gas-to",
        default="CPA_C19",
        choices=["CPA_C19", "CPA_D"],
        help="Assign 'Coke oven gas' to CPA_C19 or CPA_D (default: CPA_C19).",
    )
    p.add_argument(
        "--drop-nuclear-fuel-row",
        action="store_true",
        help="Do not emit the (enabled=0) 'Nuclear fuel' row in CPA_C19 mapping.",
    )
    p.add_argument(
        "--no-patent-fuel",
        action="store_true",
        help="Exclude 'Patent Fuel' from the CPA_C19 mapping.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_path = Path(args.out)

    include_biofuels = not bool(args.no_biofuels)
    include_patent_fuel = not bool(args.no_patent_fuel)

    concordance = load_pymrio_exio3_sectors(system=args.system).sectors
    concordance["ExioNumber"] = pd.to_numeric(concordance["ExioNumber"], errors="coerce")

    def _pick(*, codes: list[str] | None = None, code_prefixes: list[str] | None = None) -> pd.DataFrame:
        df = concordance
        m = pd.Series(False, index=df.index)
        if codes:
            m = m | df["ExioCode"].isin(list(map(str, codes)))
        if code_prefixes:
            for pref in code_prefixes:
                m = m | df["ExioCode"].astype(str).str.startswith(str(pref))
        return df[m].copy()

    rows: list[dict[str, str]] = []

    # CPA_C19: coke & refined petroleum products (plus optional biofuels).
    c19 = _pick(code_prefixes=["p23.1.", "p23.20."])
    # Keep the option visible by default (enabled=0, constituent=0).
    c19 = pd.concat([c19, _pick(codes=["p23.3"])], ignore_index=True)
    if include_biofuels:
        c19 = pd.concat([c19, _pick(codes=["p24.g", "p24.h", "p24.i"])], ignore_index=True)
    if include_patent_fuel:
        c19 = pd.concat([c19, _pick(codes=["p10.e"])], ignore_index=True)
    if args.coke_oven_gas_to == "CPA_C19":
        c19 = pd.concat([c19, _pick(codes=["p40.2.a"])], ignore_index=True)

    # Keep stable ordering by ExioNumber when available.
    c19 = c19.sort_values(["ExioNumber", "ExioCode", "ExioName"], na_position="last").drop_duplicates(
        subset=["ExioCode"], keep="first"
    )
    for _, r in c19.iterrows():
        exio_name = str(r["ExioName"])
        exio_code = str(r["ExioCode"])
        enabled = "1"
        notes = ""
        subsector = exio_name
        constituent = "1"
        if exio_code == "p23.3":
            if args.drop_nuclear_fuel_row:
                continue
            enabled = "0"
            constituent = "0"
            notes = "Optional: nuclear fuel is often not part of CPA_C19 in Eurostat IOTs"
        rows.append(
            {
                "aggregate_sector": "CPA_C19",
                "subsector": subsector,
                "exio_sector": exio_name,
                "exio_code": exio_code,
                "exio_label": str(r.get("ExioLabel", "")),
                "constituent": constituent,
                "enabled": enabled,
                "notes": notes,
            }
        )

    # CPA_D: electricity, gas, steam/hot water, distribution services.
    # Core electricity generation set (p40.11.*) and related services.
    d_parts: list[pd.DataFrame] = [
        _pick(code_prefixes=["p40.11."]),
        _pick(codes=["p40.12", "p40.13", "p40.2.1", "p40.2.d", "p40.2.e", "p40.3"]),
    ]
    if args.coke_oven_gas_to == "CPA_D":
        d_parts.append(_pick(codes=["p40.2.a"]))
    d = pd.concat(d_parts, ignore_index=True)
    d = d.sort_values(["ExioNumber", "ExioCode", "ExioName"], na_position="last").drop_duplicates(
        subset=["ExioCode"], keep="first"
    )

    for _, r in d.iterrows():
        exio_name = str(r["ExioName"])
        exio_code = str(r["ExioCode"])
        if exio_code.startswith("p40.11.") and not args.electricity_by_source:
            subsector = "Electricity generation"
            notes = "Grouped to avoid electricity-by-source split"
        else:
            # Provide stable, human-oriented subsector labels for commonly-used items.
            if exio_code == "p40.2.1":
                subsector = "Gas distribution"
            elif exio_code == "p40.2.d":
                subsector = "Manufactured gas"
            else:
                subsector = exio_name
            notes = ""
        rows.append(
            {
                "aggregate_sector": "CPA_D",
                "subsector": subsector,
                "exio_sector": exio_name,
                "exio_code": exio_code,
                "exio_label": str(r.get("ExioLabel", "")),
                "constituent": "1",
                "enabled": "1",
                "notes": notes,
            }
        )

    df_out = pd.DataFrame(rows)
    df_out.insert(0, "aggregate_sector", df_out.pop("aggregate_sector"))
    df_out = df_out[
        [
            "aggregate_sector",
            "subsector",
            "exio_sector",
            "exio_code",
            "exio_label",
            "constituent",
            "enabled",
            "notes",
        ]
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

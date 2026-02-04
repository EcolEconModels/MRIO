#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.exio_share_split import (
    load_local_iot,
    read_exio_sector_mapping_csv,
    split_iot_using_exiobase_shares,
    write_split_report_xlsx,
    map_exio_value_per_physical_input,
    map_exio_emission_intensity,
)
from ecolecon_mrio.exiobase3 import load_exiobase3_region_slice
from ecolecon_mrio.exiobase3_extensions import (
    extract_emissions_and_intensity as extract_exio_emissions_and_intensity,
    extract_final_demand_emissions_and_intensity as extract_exio_final_demand_emissions_and_intensity,
)
from ecolecon_mrio.eurostat_iot import load_eurostat_iot, load_eurostat_iot_variants


DEFAULT_PATTERNS_19: tuple[str, ...] = ("coke", "petroleum", "refined")
DEFAULT_PATTERNS_35: tuple[str, ...] = ("electricity", "gas", "steam", "hot water", "air conditioning")
DEFAULT_ELECTRICITY_BY_SOURCE_PATTERNS: tuple[str, ...] = (
    "electricity from",
    "electricity by",
    "nuclear",
    "hydro",
    "wind",
    "solar",
    "biomass",
    "geothermal",
)


def _vprint(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


class _Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


def _write_text(path: Path, content: str, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_df_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_series_csv(series: pd.Series, path: Path, *, verbose: bool, header: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, header=header)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_rows_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _vprint(verbose, f"Wrote: {path.resolve()}")
    
def _aggregate_exio_emissions_to_eurostat(
    exio_emissions: pd.DataFrame,
    sector_to_exio_sectors: dict[str, list[str]] | dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    from ecolecon_mrio.sector_intensity import aggregate_intensity_by_mapping

    return aggregate_intensity_by_mapping(
        exio_emissions,
        sector_to_exio_sectors,
        group_cols=("extension", "stressor", "unit"),
        value_col="emission",
        output_col="monetary_output",
        intensity_col="emission_intensity",
        method_label="ratio-of-sums (EXIO)",
        include_exio_list=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    def _normalize_stk_flow(v: str) -> str:
        # Eurostat uses "TOTAL" (not "TOT") for stk_flow.
        if str(v).upper() == "TOT":
            return "TOTAL"
        return str(v)

    def _normalize_eurostat_filters(filters: dict) -> dict:
        if args.eurostat_stk_flow_dim == "stk_flow" and "stk_flow" in filters:
            cur = filters["stk_flow"]
            if isinstance(cur, (list, tuple)):
                filters["stk_flow"] = [_normalize_stk_flow(x) for x in cur]
            else:
                filters["stk_flow"] = _normalize_stk_flow(cur)
        return filters

    def _exio_flow_for_eurostat_flow_value(flow_value: str) -> str:
        if args.exio_share_flow != "auto":
            return str(args.exio_share_flow)
        if args.eurostat_stk_flow_dim == "stk_flow":
            f = str(flow_value).upper()
            if f == "DOM":
                return "domestic"
            if f == "IMP":
                return "imports"
            if f in {"TOTAL", "TOT"}:
                return "total"
        return "domestic"

    p = argparse.ArgumentParser(
        description="Split Eurostat AT IOT energy sectors (NACE 19/35) using EXIOBASE shares and RAS/GRAS-balance."
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Base folder for cached datasets (defaults to repo-root `data/`, overridable via env var MRIO_DATA_ROOT).",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress and full paths for outputs.")

    euro = p.add_argument_group("Eurostat input")
    euro.add_argument("--eurostat-z", help="Path to Eurostat Z (square).")
    euro.add_argument("--eurostat-y", help="Path to Eurostat Y (same index as Z).")
    euro.add_argument("--eurostat-dataset", help="Eurostat dataset code (e.g. naio_10_cp1700).")
    euro.add_argument("--eurostat-filters", help="JSON dict of Eurostat dimension filters.")
    euro.add_argument("--eurostat-row-dim", help="Row dimension id in the Eurostat payload.")
    euro.add_argument("--eurostat-col-dim", help="Column dimension id in the Eurostat payload.")
    euro.add_argument("--eurostat-force-download", action="store_true")
    euro.add_argument("--eurostat-force-reparse", action="store_true")
    euro.add_argument(
        "--eurostat-stk-flows",
        help="JSON list of stk_flow values to run in one go (e.g. '[\"DOM\",\"IMP\"]'). Requires --eurostat-dataset mode.",
    )
    euro.add_argument("--eurostat-stk-flow-dim", default="stk_flow")

    p.add_argument("--exio-year", type=int, default=2020)
    p.add_argument("--exio-region", default="AT")
    p.add_argument(
        "--exio-share-flow",
        default="auto",
        choices=["auto", "domestic", "imports", "total"],
        help="Which EXIO use-slice to derive split shares from. "
        "auto maps Eurostat stk_flow DOM/IMP/TOTAL -> EXIO domestic/imports/total.",
    )
    p.add_argument(
        "--balancing-scope",
        default="local",
        choices=["local", "global"],
        help="How to apply RAS/GRAS balancing after splitting. "
        "local keeps the untouched parts of the IO table fixed and only balances cells that touch the new subsectors; "
        "global allows all cells to change.",
    )

    p.add_argument(
        "--dump-exio-sectors-csv",
        default=None,
        help="Write the available EXIO sector labels for the selected region slice to CSV and exit.",
    )
    p.add_argument(
        "--exio-mapping-csv",
        default=None,
        help="CSV mapping file with columns aggregate_sector, exio_sector[,subsector,enabled]. "
        "When provided, pattern matching is disabled (recommended).",
    )
    p.add_argument(
        "--require-mapping-coverage",
        action="store_true",
        help=(
            "Require a constituent coverage check before splitting. When enabled, the mapping CSV must list "
            "the full constituent set for each aggregate (including any disabled rows). The split will error "
            "if the enabled subset covers less than (1 - --coverage-tol) of the constituent total, as measured "
            "in the selected EXIO flow."
        ),
    )
    p.add_argument(
        "--coverage-tol",
        type=float,
        default=0.0,
        help=(
            "Tolerance for the coverage check (default 0.0). Example: 1e-6 allows up to 0.0001%% of EXIO Z+Y row totals "
            "to be omitted/disabled."
        ),
    )
    p.add_argument(
        "--allow-normalized-exio-match",
        action="store_true",
        help="Allow whitespace/case normalized matching of EXIO sector labels (default: strict exact match).",
    )
    p.add_argument(
        "--write-mapping-template-csv",
        default=None,
        help="Write a mapping CSV template generated by the current pattern selection and exit.",
    )

    # Energy-specific defaults belong to the CLI (module code is generic).
    p.add_argument("--sector-19", default="C19", help="Aggregate sector code for refined products.")
    p.add_argument("--sector-35", default="D35", help="Aggregate sector code for electricity/gas/steam.")
    p.add_argument("--patterns-19", default=None, help="JSON list of EXIO patterns for sector-19 subsectors.")
    p.add_argument("--patterns-35", default=None, help="JSON list of EXIO patterns for sector-35 subsectors.")
    p.add_argument("--electricity-by-source", action="store_true", help="Add more granular electricity patterns.")

    p.add_argument(
        "--no-exio-quantity-price",
        action="store_true",
        help="Skip extracting physical quantities and value-per-physical-input proxies from EXIO extensions.",
    )
    p.add_argument(
        "--qp-extensions",
        default='["energy"]',
        help='JSON list of EXIO extension names to read for quantities (default: ["energy"]).',
    )
    p.add_argument(
        "--qp-stressor-keywords",
        default='["Energy use -"]',
        help='JSON list of stressor substring keywords to select quantity rows (default: ["Energy use -"]).',
    )

    p.add_argument(
        "--no-exio-co2-intensity",
        action="store_true",
        help="Skip extracting direct CO2 intensity (emissions per monetary output) from EXIO air_emissions extension.",
    )
    p.add_argument(
        "--co2-extension",
        default="air_emissions",
        help="EXIO extension to read for CO2 (default: air_emissions).",
    )
    p.add_argument(
        "--co2-stressor-keywords",
        default='["CO2 - combustion - air"]',
        help='JSON list of stressor substring keywords to select CO2 rows (default: ["CO2 - combustion - air"]).',
    )
    p.add_argument(
        "--exio-to-eurostat-mapping-csv",
        default=None,
        help=(
            "Optional mapping CSV (aggregate_sector, exio_sector) to map EXIO CO2 intensities to Eurostat sectors. "
            "If provided, writes eurostat_sector_co2_intensity.csv."
        ),
    )

    p.add_argument("--outdir", default=None, help="Output directory (required unless using a dump/template mode).")
    p.add_argument("--no-xlsx-report", action="store_true", help="Skip writing the XLSX report.")
    args = p.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root is not None else default_data_root()
    eurostat_storage = data_root / "eurostat"
    _vprint(args.verbose, f"Using data root: {data_root.resolve()}")

    if (args.eurostat_z or args.eurostat_y) and args.eurostat_dataset:
        raise SystemExit("Provide either --eurostat-z/--eurostat-y OR --eurostat-dataset (not both).")

    patterns_19 = DEFAULT_PATTERNS_19 if args.patterns_19 is None else tuple(json.loads(args.patterns_19))
    patterns_35 = DEFAULT_PATTERNS_35 if args.patterns_35 is None else tuple(json.loads(args.patterns_35))
    if args.electricity_by_source:
        patterns_35 = tuple(patterns_35) + DEFAULT_ELECTRICITY_BY_SOURCE_PATTERNS
    sector_patterns = {str(args.sector_19): patterns_19, str(args.sector_35): patterns_35}

    if args.outdir is None and not (
        args.dump_exio_sectors_csv is not None or args.write_mapping_template_csv is not None
    ):
        raise SystemExit("Provide --outdir (or use --dump-exio-sectors-csv / --write-mapping-template-csv).")

    outdir = Path(args.outdir) if args.outdir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    extract_qp = not args.no_exio_quantity_price
    qp_extensions = tuple(json.loads(args.qp_extensions))
    qp_keywords = tuple(json.loads(args.qp_stressor_keywords))
    extract_co2 = not args.no_exio_co2_intensity
    co2_extension = str(args.co2_extension)
    co2_keywords = tuple(json.loads(args.co2_stressor_keywords))
    full_sector_mapping = None
    if args.exio_to_eurostat_mapping_csv:
        full_sector_mapping = read_exio_sector_mapping_csv(args.exio_to_eurostat_mapping_csv)
    qp_cache_dir = (
        data_root / f"exiobase3_{args.exio_year}" / "slices" / args.exio_region / "pxp" / "extensions"
    )

    if args.dump_exio_sectors_csv is not None or args.write_mapping_template_csv is not None:
        _vprint(args.verbose, "Calling: load_exiobase3_region_slice(...)")
        t = _Timer()
        exio_slice = load_exiobase3_region_slice(
            year=args.exio_year,
            region=args.exio_region,
            system="pxp",
            flow="domestic",
            cache_root=data_root,
            verbose=args.verbose,
        )
        _vprint(args.verbose, f"Returned: load_exiobase3_region_slice ({t.elapsed_s():.2f}s)")

        if args.dump_exio_sectors_csv is not None:
            out = Path(args.dump_exio_sectors_csv)
            _write_rows_csv(
                pd.DataFrame({"exio_sector": list(map(str, exio_slice.sectors))}),
                out,
                verbose=args.verbose,
            )
            return 0

        # Template from pattern selection (not robust; intended to be edited into an explicit mapping).
        assert args.write_mapping_template_csv is not None
        rows: list[dict[str, str]] = []
        for agg, pats in sector_patterns.items():
            matched = []
            for s in map(str, exio_slice.sectors):
                s_l = s.casefold()
                if any(p.casefold() in s_l for p in pats):
                    matched.append(s)
            for s in matched:
                rows.append(
                    {
                        "aggregate_sector": str(agg),
                        "subsector": str(s),
                        "exio_sector": str(s),
                        "enabled": "1",
                    }
                )
        out = Path(args.write_mapping_template_csv)
        _write_rows_csv(pd.DataFrame(rows), out, verbose=args.verbose)
        return 0

    sector_to_exio_sectors = None
    coverage_expected_exio = None
    if args.exio_mapping_csv is not None:
        _vprint(args.verbose, "Calling: read_exio_sector_mapping_csv(...)")
        t = _Timer()
        sector_to_exio_sectors = read_exio_sector_mapping_csv(args.exio_mapping_csv)
        _vprint(args.verbose, f"Returned: read_exio_sector_mapping_csv ({t.elapsed_s():.2f}s)")
        if args.verbose and (args.patterns_19 is not None or args.patterns_35 is not None or args.electricity_by_source):
            _vprint(args.verbose, "NOTE: --exio-mapping-csv provided; ignoring pattern selection flags.")
        if args.require_mapping_coverage:
            df_all = pd.read_csv(args.exio_mapping_csv, dtype=str).fillna("")
            needed = {"aggregate_sector", "exio_sector"}
            missing = sorted(needed - set(df_all.columns))
            if missing:
                raise SystemExit(
                    f"--require-mapping-coverage requires mapping CSV columns {sorted(needed)}; missing: {missing}"
                )
            df_all["aggregate_sector"] = df_all["aggregate_sector"].astype(str).str.strip()
            df_all["exio_sector"] = df_all["exio_sector"].astype(str).str.strip()
            df_all = df_all[(df_all["aggregate_sector"] != "") & (df_all["exio_sector"] != "")]
            if "constituent" in df_all.columns:
                def _boolish(v: object) -> bool:
                    s = str(v).strip().lower()
                    if s == "" or s in {"1", "true", "t", "yes", "y", "on"}:
                        return True
                    if s in {"0", "false", "f", "no", "n", "off"}:
                        return False
                    raise SystemExit(f"Could not parse boolean-ish mapping CSV constituent value: {v!r}")

                df_all = df_all[df_all["constituent"].map(_boolish)].copy()
            coverage_expected_exio = {
                str(agg): list(map(str, g["exio_sector"].tolist()))
                for agg, g in df_all.groupby("aggregate_sector", sort=False)
            }
    report_aggregates = (
        tuple(sector_to_exio_sectors.keys())
        if sector_to_exio_sectors is not None
        else (args.sector_19, args.sector_35)
    )

    if args.eurostat_dataset:
        if not (args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim):
            raise SystemExit(
                "When using --eurostat-dataset, also provide --eurostat-filters, --eurostat-row-dim, --eurostat-col-dim."
            )
        base_filters = _normalize_eurostat_filters(json.loads(args.eurostat_filters))

        if args.eurostat_stk_flows is not None:
            flows = list(map(str, json.loads(args.eurostat_stk_flows)))
            if args.eurostat_stk_flow_dim == "stk_flow":
                flows = [_normalize_stk_flow(f) for f in flows]
            _vprint(args.verbose, "Calling: load_eurostat_iot_variants(...)")
            t = _Timer()
            variants = load_eurostat_iot_variants(
                dataset=args.eurostat_dataset,
                base_filters=base_filters,
                variant_dim=args.eurostat_stk_flow_dim,
                variant_values=flows,
                row_dim=args.eurostat_row_dim,
                col_dim=args.eurostat_col_dim,
                storage_folder=eurostat_storage,
                force_download=args.eurostat_force_download,
                force_reparse=args.eurostat_force_reparse,
            )
            _vprint(args.verbose, f"Returned: load_eurostat_iot_variants ({t.elapsed_s():.2f}s)")

            exio_slices: dict[str, object] = {}
            for flow in flows:
                exio_flow = _exio_flow_for_eurostat_flow_value(flow)
                if exio_flow in exio_slices:
                    continue
                _vprint(args.verbose, f"Calling: load_exiobase3_region_slice(..., flow={exio_flow})")
                t = _Timer()
                exio_slices[exio_flow] = load_exiobase3_region_slice(
                    year=args.exio_year,
                    region=args.exio_region,
                    system="pxp",
                    flow=exio_flow,
                    cache_root=data_root,
                    verbose=args.verbose,
                )
                _vprint(args.verbose, f"Returned: load_exiobase3_region_slice ({t.elapsed_s():.2f}s)")

            for flow, parsed in variants.items():
                flow_dir = outdir / f"{args.eurostat_stk_flow_dim}={flow}"
                flow_dir.mkdir(parents=True, exist_ok=True)

                _vprint(args.verbose, f"Calling: split_iot_using_exiobase_split_profiles(...) [{flow}]")
                t = _Timer()
                exio_flow = _exio_flow_for_eurostat_flow_value(flow)
                exio_slice = exio_slices[exio_flow]
                res = split_iot_using_exiobase_shares(
                    eurostat_Z=parsed.Z,
                    eurostat_Y=parsed.Y,
                    exio_year=args.exio_year,
                    exio_region=args.exio_region,
                    exio_slice=exio_slice,
                    sector_to_exio_sectors=sector_to_exio_sectors,
                    sector_patterns=None if sector_to_exio_sectors is not None else sector_patterns,
                    allow_normalized_exio_match=args.allow_normalized_exio_match,
                    exio_cache_root=data_root,
                    balancing_scope=args.balancing_scope,
                    coverage_expected_exio_sectors=coverage_expected_exio,
                    coverage_tol=args.coverage_tol,
                    extract_quantity_and_price=extract_qp,
                    quantity_price_extensions=qp_extensions if extract_qp else None,
                    quantity_price_stressor_keywords=qp_keywords if extract_qp else None,
                    quantity_price_cache_dir=qp_cache_dir if extract_qp else None,
                )
                _vprint(args.verbose, f"Returned: split_iot_using_exiobase_shares ({t.elapsed_s():.2f}s)")

                _write_df_csv(res.Z, flow_dir / "Z_split_balanced.csv", verbose=args.verbose)
                _write_df_csv(res.Y, flow_dir / "Y_split_balanced.csv", verbose=args.verbose)
                _write_series_csv(res.x, flow_dir / "x_split_balanced.csv", verbose=args.verbose, header=True)
                _write_df_csv(res.A, flow_dir / "A_split_balanced.csv", verbose=args.verbose)
                _write_text(
                    flow_dir / "exio_selected_sectors.json",
                    json.dumps(res.exio_selected_sectors, indent=2),
                    verbose=args.verbose,
                )
                _write_text(
                    flow_dir / "balance_info.json",
                    json.dumps(
                        {
                            "method": res.balancing_method,
                            "iterations": res.balancing_stats.iterations,
                            "max_abs_row_gap": res.balancing_stats.max_abs_row_gap,
                            "max_abs_col_gap": res.balancing_stats.max_abs_col_gap,
                            "variant_dim": args.eurostat_stk_flow_dim,
                            "variant_value": flow,
                        },
                        indent=2,
                    ),
                    verbose=args.verbose,
                )
                if res.exio_quantity_and_price is not None:
                    out = flow_dir / "exio_quantity_and_price.csv"
                    res.exio_quantity_and_price.to_csv(out, index=False)
                    _vprint(args.verbose, f"Wrote: {out.resolve()}")
                    
                    # Compute and output disaggregated sector values per physical input
                    mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
                    disagg_values = map_exio_value_per_physical_input(
                        exio_quantity_and_price=res.exio_quantity_and_price,
                        sector_to_exio_sectors=mapping_for_values,
                        split_profiles=res.split_profiles,
                    )
                    if disagg_values is not None:
                        out = flow_dir / "disaggregated_sector_value_per_physical_input.csv"
                        disagg_values.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")

                # Compute and output disaggregated sector CO2 intensities (direct emissions per monetary output)
                if extract_co2:
                    exio_sectors: list[str] = []
                    for agg in res.split_profiles.keys():
                        exio_sectors.extend(res.exio_selected_sectors.get(str(agg), []))
                    exio_sectors = list(dict.fromkeys(map(str, exio_sectors)))

                    # Use the same monetary-output convention as the slice (Eurostat-style Z+Y totals).
                    exio_out = (exio_slice.Z_domestic.sum(axis=1) + exio_slice.Y_region.sum(axis=1)).astype(
                        "float64"
                    )

                    # Final-demand monetary totals by EXIO category (EUR)
                    y_by_category = exio_slice.Y_region.sum(axis=0)
                    if isinstance(y_by_category, pd.DataFrame):
                        y_by_category = y_by_category.iloc[:, 0]
                    y_by_category = y_by_category.astype("float64")

                    exio_co2 = extract_exio_emissions_and_intensity(
                        container=exio_slice.source_zip,
                        region=exio_slice.region,
                        sectors=exio_sectors,
                        monetary_output=exio_out,
                        extension=co2_extension,
                        stressor_keywords=co2_keywords,
                        cache_dir=qp_cache_dir,
                    )
                    if exio_co2 is not None:
                        out = flow_dir / "exio_co2_intensity.csv"
                        exio_co2.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")

                        mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
                        disagg_co2 = map_exio_emission_intensity(
                            exio_emissions_and_intensity=exio_co2,
                            sector_to_exio_sectors=mapping_for_values,
                            split_profiles=res.split_profiles,
                        )
                        if disagg_co2 is not None:
                            out = flow_dir / "disaggregated_sector_co2_intensity.csv"
                            disagg_co2.to_csv(out, index=False)
                            _vprint(args.verbose, f"Wrote: {out.resolve()}")

                    # Also extract direct final-demand CO2 (F_Y), so users can report F_Y (or intensity * y)
                    exio_fd_co2 = extract_exio_final_demand_emissions_and_intensity(
                        container=exio_slice.source_zip,
                        region=exio_slice.region,
                        monetary_final_demand=y_by_category,
                        extension=co2_extension,
                        stressor_keywords=co2_keywords,
                        cache_dir=qp_cache_dir,
                    )
                    if exio_fd_co2 is not None:
                        out = flow_dir / "exio_final_demand_co2_intensity.csv"
                        exio_fd_co2.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")

                if extract_co2 and full_sector_mapping is not None:
                    exio_sectors_full: list[str] = []
                    for members in full_sector_mapping.values():
                        if isinstance(members, dict):
                            for v in members.values():
                                exio_sectors_full.extend(list(map(str, v)))
                        else:
                            exio_sectors_full.extend(list(map(str, members)))
                    exio_sectors_full = list(dict.fromkeys(exio_sectors_full))

                    available = set(map(str, exio_slice.sectors))
                    exio_sectors_full = [s for s in exio_sectors_full if s in available]

                    exio_out_full = (
                        exio_slice.Z_domestic.sum(axis=1) + exio_slice.Y_region.sum(axis=1)
                    ).astype("float64")
                    exio_co2_full = extract_exio_emissions_and_intensity(
                        container=exio_slice.source_zip,
                        region=exio_slice.region,
                        sectors=exio_sectors_full,
                        monetary_output=exio_out_full,
                        extension=co2_extension,
                        stressor_keywords=co2_keywords,
                        cache_dir=qp_cache_dir,
                    )
                    if exio_co2_full is not None:
                        eurostat_co2 = _aggregate_exio_emissions_to_eurostat(
                            exio_co2_full, full_sector_mapping
                        )
                        if not eurostat_co2.empty:
                            out = flow_dir / "eurostat_sector_co2_intensity.csv"
                            eurostat_co2.to_csv(out, index=False)
                            _vprint(args.verbose, f"Wrote: {out.resolve()}")
                if not args.no_xlsx_report:
                    out = flow_dir / "energy_split_report.xlsx"
                    write_split_report_xlsx(
                        out,
                        result=res,
                        aggregates=report_aggregates,
                    )
                    _vprint(args.verbose, f"Wrote: {out.resolve()}")
            return 0

        _vprint(args.verbose, "Calling: load_eurostat_iot(...)")
        t = _Timer()
        parsed = load_eurostat_iot(
            dataset=args.eurostat_dataset,
            filters=base_filters,
            row_dim=args.eurostat_row_dim,
            col_dim=args.eurostat_col_dim,
            storage_folder=eurostat_storage,
            force_download=args.eurostat_force_download,
            force_reparse=args.eurostat_force_reparse,
        )
        _vprint(args.verbose, f"Returned: load_eurostat_iot ({t.elapsed_s():.2f}s)")
        Z, Y = parsed.Z, parsed.Y
        exio_flow_single = _exio_flow_for_eurostat_flow_value(
            str(base_filters.get(args.eurostat_stk_flow_dim, "DOM"))
        )
    else:
        if not (args.eurostat_z and args.eurostat_y):
            raise SystemExit("Provide --eurostat-z and --eurostat-y, or use --eurostat-dataset mode.")
        _vprint(args.verbose, "Calling: load_local_iot(...)")
        t = _Timer()
        Z, Y = load_local_iot(args.eurostat_z, args.eurostat_y)
        _vprint(args.verbose, f"Returned: load_local_iot ({t.elapsed_s():.2f}s)")
        exio_flow_single = "domestic" if args.exio_share_flow == "auto" else str(args.exio_share_flow)

    _vprint(args.verbose, "Calling: split_iot_using_exiobase_shares(...)")
    t = _Timer()
    res = split_iot_using_exiobase_shares(
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_year=args.exio_year,
        exio_region=args.exio_region,
        exio_slice=load_exiobase3_region_slice(
            year=args.exio_year,
            region=args.exio_region,
            system="pxp",
            flow=exio_flow_single,
            cache_root=data_root,
            verbose=args.verbose,
        ),
        sector_to_exio_sectors=sector_to_exio_sectors,
        sector_patterns=None if sector_to_exio_sectors is not None else sector_patterns,
        allow_normalized_exio_match=args.allow_normalized_exio_match,
        exio_cache_root=data_root,
        balancing_scope=args.balancing_scope,
        coverage_expected_exio_sectors=coverage_expected_exio,
        coverage_tol=args.coverage_tol,
        extract_quantity_and_price=extract_qp,
        quantity_price_extensions=qp_extensions if extract_qp else None,
        quantity_price_stressor_keywords=qp_keywords if extract_qp else None,
        quantity_price_cache_dir=qp_cache_dir if extract_qp else None,
    )
    _vprint(args.verbose, f"Returned: split_iot_using_exiobase_shares ({t.elapsed_s():.2f}s)")

    assert outdir is not None
    _write_df_csv(res.Z, outdir / "Z_split_balanced.csv", verbose=args.verbose)
    _write_df_csv(res.Y, outdir / "Y_split_balanced.csv", verbose=args.verbose)
    _write_series_csv(res.x, outdir / "x_split_balanced.csv", verbose=args.verbose, header=True)
    _write_df_csv(res.A, outdir / "A_split_balanced.csv", verbose=args.verbose)
    _write_text(
        outdir / "exio_selected_sectors.json",
        json.dumps(res.exio_selected_sectors, indent=2),
        verbose=args.verbose,
    )
    _write_text(
        outdir / "ras_info.json",
        json.dumps(
            {
                "method": res.balancing_method,
                "iterations": res.balancing_stats.iterations,
                "max_abs_row_gap": res.balancing_stats.max_abs_row_gap,
                "max_abs_col_gap": res.balancing_stats.max_abs_col_gap,
            },
            indent=2,
        ),
        verbose=args.verbose,
    )
    if res.exio_quantity_and_price is not None:
        out = outdir / "exio_quantity_and_price_proxies.csv"
        res.exio_quantity_and_price.to_csv(out, index=False)
        _vprint(args.verbose, f"Wrote: {out.resolve()}")
        
        # Compute and output disaggregated sector values per physical input
        mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
        disagg_values = map_exio_value_per_physical_input(
            exio_quantity_and_price=res.exio_quantity_and_price,
            sector_to_exio_sectors=mapping_for_values,
            split_profiles=res.split_profiles,
        )
        if disagg_values is not None:
            out = outdir / "disaggregated_sector_value_per_physical_input.csv"
            disagg_values.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")

    # Compute and output disaggregated sector CO2 intensities (direct emissions per monetary output)
    exio_slice_single = None
    if extract_co2:
        exio_sectors: list[str] = []
        for agg in res.split_profiles.keys():
            exio_sectors.extend(res.exio_selected_sectors.get(str(agg), []))
        exio_sectors = list(dict.fromkeys(map(str, exio_sectors)))

        exio_slice_single = load_exiobase3_region_slice(
            year=args.exio_year,
            region=args.exio_region,
            system="pxp",
            flow=exio_flow_single,
            cache_root=data_root,
            verbose=args.verbose,
        )
        exio_out = (exio_slice_single.Z_domestic.sum(axis=1) + exio_slice_single.Y_region.sum(axis=1)).astype(
            "float64"
        )
        exio_co2 = extract_exio_emissions_and_intensity(
            container=exio_slice_single.source_zip,
            region=exio_slice_single.region,
            sectors=exio_sectors,
            monetary_output=exio_out,
            extension=co2_extension,
            stressor_keywords=co2_keywords,
            cache_dir=qp_cache_dir,
        )
        if exio_co2 is not None:
            out = outdir / "exio_co2_intensity.csv"
            exio_co2.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")

            mapping_for_values = sector_to_exio_sectors or res.exio_selected_sectors
            disagg_co2 = map_exio_emission_intensity(
                exio_emissions_and_intensity=exio_co2,
                sector_to_exio_sectors=mapping_for_values,
                split_profiles=res.split_profiles,
            )
            if disagg_co2 is not None:
                out = outdir / "disaggregated_sector_co2_intensity.csv"
                disagg_co2.to_csv(out, index=False)
                _vprint(args.verbose, f"Wrote: {out.resolve()}")

    if extract_co2 and full_sector_mapping is not None:
        exio_sectors_full: list[str] = []
        for members in full_sector_mapping.values():
            if isinstance(members, dict):
                for v in members.values():
                    exio_sectors_full.extend(list(map(str, v)))
            else:
                exio_sectors_full.extend(list(map(str, members)))
        exio_sectors_full = list(dict.fromkeys(exio_sectors_full))

        if exio_slice_single is None:
            exio_slice_single = load_exiobase3_region_slice(
                year=args.exio_year,
                region=args.exio_region,
                system="pxp",
                flow=exio_flow_single,
                cache_root=data_root,
                verbose=args.verbose,
            )
        available = set(map(str, exio_slice_single.sectors))
        exio_sectors_full = [s for s in exio_sectors_full if s in available]

        exio_out_full = (
            exio_slice_single.Z_domestic.sum(axis=1) + exio_slice_single.Y_region.sum(axis=1)
        ).astype("float64")
        exio_co2_full = extract_exio_emissions_and_intensity(
            container=exio_slice_single.source_zip,
            region=exio_slice_single.region,
            sectors=exio_sectors_full,
            monetary_output=exio_out_full,
            extension=co2_extension,
            stressor_keywords=co2_keywords,
            cache_dir=qp_cache_dir,
        )
        if exio_co2_full is not None:
            eurostat_co2 = _aggregate_exio_emissions_to_eurostat(exio_co2_full, full_sector_mapping)
            if not eurostat_co2.empty:
                out = outdir / "eurostat_sector_co2_intensity.csv"
                eurostat_co2.to_csv(out, index=False)
                _vprint(args.verbose, f"Wrote: {out.resolve()}")
    if not args.no_xlsx_report:
        out = outdir / "energy_split_report.xlsx"
        write_split_report_xlsx(out, result=res, aggregates=report_aggregates)
        _vprint(args.verbose, f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

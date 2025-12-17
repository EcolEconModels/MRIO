#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_energy_split import (
    load_local_eurostat_iot,
    split_iot_using_exiobase_shares,
    write_energy_split_report_xlsx,
)
from ecolecon_mrio.exiobase3 import load_exiobase3_region_slice
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


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

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

    # Energy-specific defaults belong to the CLI.
    p.add_argument("--sector-19", default="C19", help="Aggregate sector code for refined products.")
    p.add_argument("--sector-35", default="D35", help="Aggregate sector code for electricity/gas/steam.")
    p.add_argument("--patterns-19", default=None, help="JSON list of EXIO patterns for sector-19 subsectors.")
    p.add_argument("--patterns-35", default=None, help="JSON list of EXIO patterns for sector-35 subsectors.")
    p.add_argument("--electricity-by-source", action="store_true", help="Add more granular electricity patterns.")

    p.add_argument(
        "--no-exio-quantity-price",
        action="store_true",
        help="Skip extracting physical quantities and implied price proxies from EXIO extensions.",
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

    p.add_argument("--outdir", required=True, help="Output directory.")
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extract_qp = not args.no_exio_quantity_price
    qp_extensions = tuple(json.loads(args.qp_extensions))
    qp_keywords = tuple(json.loads(args.qp_stressor_keywords))
    qp_cache_dir = (
        data_root / f"exiobase3_{args.exio_year}" / "slices" / args.exio_region / "pxp" / "extensions"
    )

    if args.eurostat_dataset:
        if not (args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim):
            raise SystemExit(
                "When using --eurostat-dataset, also provide --eurostat-filters, --eurostat-row-dim, --eurostat-col-dim."
            )
        base_filters = json.loads(args.eurostat_filters)

        if args.eurostat_stk_flows is not None:
            flows = list(map(str, json.loads(args.eurostat_stk_flows)))
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

            _vprint(args.verbose, "Calling: load_exiobase3_region_slice(...)")
            t = _Timer()
            exio_slice = load_exiobase3_region_slice(
                year=args.exio_year,
                region=args.exio_region,
                system="pxp",
                cache_root=data_root,
                verbose=False,
            )
            _vprint(args.verbose, f"Returned: load_exiobase3_region_slice ({t.elapsed_s():.2f}s)")

            for flow, parsed in variants.items():
                flow_dir = outdir / f"{args.eurostat_stk_flow_dim}={flow}"
                flow_dir.mkdir(parents=True, exist_ok=True)

                _vprint(args.verbose, f"Calling: split_iot_using_exiobase_split_profiles(...) [{flow}]")
                t = _Timer()
                res = split_iot_using_exiobase_shares(
                    eurostat_Z=parsed.Z,
                    eurostat_Y=parsed.Y,
                    exio_year=args.exio_year,
                    exio_region=args.exio_region,
                    exio_slice=exio_slice,
                    sector_patterns=sector_patterns,
                    exio_cache_root=data_root,
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
                if not args.no_xlsx_report:
                    out = flow_dir / "energy_split_report.xlsx"
                    write_energy_split_report_xlsx(
                        out,
                        result=res,
                        aggregates=(args.sector_19, args.sector_35),
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
    else:
        if not (args.eurostat_z and args.eurostat_y):
            raise SystemExit("Provide --eurostat-z and --eurostat-y, or use --eurostat-dataset mode.")
        _vprint(args.verbose, "Calling: load_local_eurostat_iot(...)")
        t = _Timer()
        Z, Y = load_local_eurostat_iot(args.eurostat_z, args.eurostat_y)
        _vprint(args.verbose, f"Returned: load_local_eurostat_iot ({t.elapsed_s():.2f}s)")

    _vprint(args.verbose, "Calling: split_iot_using_exiobase_shares(...)")
    t = _Timer()
    res = split_iot_using_exiobase_shares(
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_year=args.exio_year,
        exio_region=args.exio_region,
        sector_patterns=sector_patterns,
        exio_cache_root=data_root,
        extract_quantity_and_price=extract_qp,
        quantity_price_extensions=qp_extensions if extract_qp else None,
        quantity_price_stressor_keywords=qp_keywords if extract_qp else None,
        quantity_price_cache_dir=qp_cache_dir if extract_qp else None,
    )
    _vprint(args.verbose, f"Returned: split_iot_using_exiobase_shares ({t.elapsed_s():.2f}s)")

    _write_df_csv(res.Z, outdir / "Z_split_balanced.csv", verbose=args.verbose)
    _write_df_csv(res.Y, outdir / "Y_split_balanced.csv", verbose=args.verbose)
    _write_series_csv(res.x, outdir / "x_split_balanced.csv", verbose=args.verbose, header=True)
    _write_df_csv(res.A, outdir / "A_split_balanced.csv", verbose=args.verbose)
    _write_text(outdir / "exio_selected_sectors.json", json.dumps(res.exio_selected_sectors, indent=2), verbose=args.verbose)
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
    if not args.no_xlsx_report:
        out = outdir / "energy_split_report.xlsx"
        write_energy_split_report_xlsx(out, result=res, aggregates=(args.sector_19, args.sector_35))
        _vprint(args.verbose, f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

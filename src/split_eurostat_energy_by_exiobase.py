"""
Disaggregate Eurostat Austria IOT (64x64) energy sectors (NACE C19, D35)
using monetary split shares from EXIOBASE 3 (Austria block), then balance
with an unconstrained RAS.

This module is intentionally standalone (in `src/`) but reuses repo utilities.
Run with `PYTHONPATH=src` so imports like `mrio_pyramids.*` resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd

from mrio_pyramids.balancing import BalanceStats
from mrio_pyramids.disaggregation import SplitProfiles, disaggregate_and_balance
from mrio_pyramids.exiobase3 import (
    ExioRegionSlice,
    build_split_profiles_from_exio_domestic_block,
    load_exiobase3_region_slice,
)
from mrio_pyramids.exiobase3_extensions import (
    extract_quantity_and_price as extract_exio_quantity_and_price,
)


DEFAULT_EUROSTAT_SECTOR_19 = "C19"
DEFAULT_EUROSTAT_SECTOR_35 = "D35"

DEFAULT_EXIO_PATTERNS_19: tuple[str, ...] = (
    "coke",
    "petroleum",
    "refined",
)

DEFAULT_EXIO_PATTERNS_35: tuple[str, ...] = (
    "electricity",
    "gas",
    "steam",
    "hot water",
    "air conditioning",
)

DEFAULT_EXIO_ELECTRICITY_BY_SOURCE_PATTERNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class SplitResult:
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    split_profiles: dict[str, SplitProfiles]
    balancing_method: str
    balancing_stats: BalanceStats
    exio_selected_sectors: dict[str, list[str]]
    exio_quantity_and_price: pd.DataFrame | None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, index_col=0)
    raise ValueError(f"Unsupported table format: {path}")


def load_eurostat_iot(
    z_path: str | Path,
    y_path: str | Path,
    *,
    dtype: str | np.dtype = "float64",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a Eurostat national IOT provided as two tables (Z and Y).

    This loader is file-format agnostic (csv/parquet/pickle/xlsx).
    """
    Z = _read_table(Path(z_path)).astype(dtype)
    Y = _read_table(Path(y_path)).astype(dtype)

    Z.index = Z.index.astype(str)
    Z.columns = Z.columns.astype(str)
    Y.index = Y.index.astype(str)
    Y.columns = Y.columns.astype(str)

    if not Z.index.equals(Z.columns):
        raise ValueError("Eurostat Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Eurostat Y must have the same index as Z.")

    return Z, Y


def split_eurostat_energy_sectors_using_exiobase(
    *,
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_year: int = 2020,
    exio_region: str = "AT",
    exio_slice: ExioRegionSlice | None = None,
    eurostat_sector_19: str = DEFAULT_EUROSTAT_SECTOR_19,
    eurostat_sector_35: str = DEFAULT_EUROSTAT_SECTOR_35,
    exio_patterns_19: Sequence[str] = DEFAULT_EXIO_PATTERNS_19,
    exio_patterns_35: Sequence[str] = DEFAULT_EXIO_PATTERNS_35,
    electricity_by_source: bool = False,
    ras_max_iter: int = 50_000,
    ras_tol: float = 1e-10,
    prefix_subsectors: bool = True,
    extract_quantity_and_price: bool = True,
    verbose: bool = False,
) -> SplitResult:
    """
    Disaggregate Eurostat energy sectors (C19, D35) using EXIOBASE 2020 shares and balance via RAS.

    Assumptions:
      - Eurostat Z is industry×industry and Y is industry×final-demand.
      - EXIOBASE patterns identify subsectors within the target energy aggregates.
      - Splitting uses domestic Austria block totals for shares (not partner-specific).
    """
    Z = eurostat_Z.copy()
    Y = eurostat_Y.copy()

    Z.index = Z.index.astype(str)
    Z.columns = Z.columns.astype(str)
    Y.index = Y.index.astype(str)
    Y.columns = Y.columns.astype(str)

    if not Z.index.equals(Z.columns):
        raise ValueError("Eurostat Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Eurostat Y must share the same index as Z.")

    if exio_slice is None:
        _vprint(verbose, f"Calling: load_exiobase3_region_slice(region={exio_region}, year={exio_year})")
        t = _Timer()
        exio_slice = load_exiobase3_region_slice(
            year=exio_year, region=exio_region, system="pxp", verbose=verbose
        )
        _vprint(verbose, f"Returned: load_exiobase3_region_slice ({t.elapsed_s():.2f}s)")

    _vprint(verbose, f"Calling: build_split_profiles_from_exio_domestic_block(aggregate={eurostat_sector_19})")
    t = _Timer()
    prof_19, exio_19 = build_split_profiles_from_exio_domestic_block(
        exio_slice,
        aggregate_sector=eurostat_sector_19,
        exio_sector_patterns=exio_patterns_19,
        prefix_labels_with=f"{eurostat_sector_19}::" if prefix_subsectors else None,
    )
    _vprint(
        verbose,
        f"Returned: build_split_profiles_from_exio_domestic_block ({t.elapsed_s():.2f}s) "
        f"matched={len(exio_19)} subsectors={len(prof_19.subsectors)}",
    )

    patterns_35 = tuple(exio_patterns_35)
    if electricity_by_source:
        patterns_35 = tuple(patterns_35) + tuple(DEFAULT_EXIO_ELECTRICITY_BY_SOURCE_PATTERNS)

    _vprint(verbose, f"Calling: build_split_profiles_from_exio_domestic_block(aggregate={eurostat_sector_35})")
    t = _Timer()
    prof_35, exio_35 = build_split_profiles_from_exio_domestic_block(
        exio_slice,
        aggregate_sector=eurostat_sector_35,
        exio_sector_patterns=patterns_35,
        prefix_labels_with=f"{eurostat_sector_35}::" if prefix_subsectors else None,
    )
    _vprint(
        verbose,
        f"Returned: build_split_profiles_from_exio_domestic_block ({t.elapsed_s():.2f}s) "
        f"matched={len(exio_35)} subsectors={len(prof_35.subsectors)}",
    )

    split_profiles: dict[str, SplitProfiles] = {
        eurostat_sector_19: prof_19,
        eurostat_sector_35: prof_35,
    }

    if verbose:
        print(
            f"Splitting sectors {eurostat_sector_19} and {eurostat_sector_35} "
            f"using EXIOBASE {exio_year} region {exio_region} shares; running RAS..."
        )
    t = _Timer()
    res = disaggregate_and_balance(Z, Y, split_profiles=split_profiles, max_iter=ras_max_iter, tol=ras_tol)
    _vprint(verbose, f"Returned: disaggregate_and_balance ({t.elapsed_s():.2f}s) method={res.balancing_method}")

    exio_qp = None
    if extract_quantity_and_price:
        _vprint(verbose, "Calling: extract_exio_quantity_and_price(...)")
        t = _Timer()
        exio_sectors = list(dict.fromkeys(exio_19 + exio_35))
        monetary_output = (exio_slice.Z_domestic.sum(axis=1) + exio_slice.Y_region.sum(axis=1)).astype(
            "float64"
        )
        exio_qp = extract_exio_quantity_and_price(
            container=exio_slice.source_zip,
            region=exio_region,
            sectors=exio_sectors,
            monetary_output=monetary_output,
            extensions=("energy",),
            stressor_keywords=("Energy use -",),
            cache_dir=exio_slice.source_zip.parent / "slices" / exio_region / "pxp" / "extensions",
        )
        _vprint(verbose, f"Returned: extract_exio_quantity_and_price ({t.elapsed_s():.2f}s) rows={len(exio_qp)}")

    return SplitResult(
        Z=res.Z,
        Y=res.Y,
        x=res.x,
        A=res.A,
        split_profiles=res.split_profiles,
        balancing_method=res.balancing_method,
        balancing_stats=res.balancing_stats,
        exio_selected_sectors={eurostat_sector_19: exio_19, eurostat_sector_35: exio_35},
        exio_quantity_and_price=exio_qp,
    )


def write_energy_split_report_xlsx(
    out_path: str | Path,
    *,
    result: SplitResult,
    aggregates: Sequence[str],
    verbose: bool = False,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _vprint(verbose, f"Writing XLSX report: {out_path.resolve()}")

    energy_subsectors: list[str] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        energy_subsectors.extend(list(map(str, prof.subsectors)))
    energy_subsectors = list(dict.fromkeys(energy_subsectors))

    def _safe_sheet(name: str) -> str:
        return str(name)[:31]

    rows: list[pd.DataFrame] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        subs = pd.Index(prof.subsectors, dtype=str)
        df = pd.DataFrame(
            {
                "aggregate_sector": str(agg),
                "subsector": subs,
                "output_share": prof.output_share.reindex(subs).to_numpy(),
                "intermediate_sales_share": prof.intermediate_sales_share.reindex(subs).to_numpy(),
                "intermediate_input_share": prof.intermediate_input_share.reindex(subs).to_numpy(),
                "final_demand_share": prof.final_demand_share.reindex(subs).to_numpy(),
            }
        )
        rows.append(df)
    weights = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for agg in aggregates:
        prof = result.split_profiles.get(str(agg))
        if prof is None:
            continue
        subs = list(map(str, prof.subsectors))
        row_total = float(result.Z.loc[subs, :].sum(axis=1).sum() + result.Y.loc[subs, :].sum(axis=1).sum())
        col_total = float(result.Z.loc[:, subs].sum(axis=0).sum())
        summary_rows.append(
            {
                "aggregate_sector": str(agg),
                "num_subsectors": len(subs),
                "sum_output_share": float(prof.output_share.sum()),
                "sum_intermediate_sales_share": float(prof.intermediate_sales_share.sum()),
                "sum_intermediate_input_share": float(prof.intermediate_input_share.sum()),
                "sum_final_demand_share": float(prof.final_demand_share.sum()),
                "total_row_output_in_split_tables": row_total,
                "total_col_intermediate_input_in_split_tables": col_total,
            }
        )
    summary = pd.DataFrame(summary_rows)

    Z_rows = result.Z.loc[energy_subsectors, :] if energy_subsectors else pd.DataFrame()
    Z_cols = result.Z.loc[:, energy_subsectors] if energy_subsectors else pd.DataFrame()
    Z_block = result.Z.loc[energy_subsectors, energy_subsectors] if energy_subsectors else pd.DataFrame()
    Y_rows = result.Y.loc[energy_subsectors, :] if energy_subsectors else pd.DataFrame()

    exio_sel = (
        pd.DataFrame(
            [
                {"aggregate_sector": agg, "exio_sector": s}
                for agg, sectors in result.exio_selected_sectors.items()
                for s in sectors
            ]
        )
        if result.exio_selected_sectors
        else pd.DataFrame()
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name=_safe_sheet("summary"), index=False)
        weights.to_excel(xw, sheet_name=_safe_sheet("split_weights"), index=False)
        exio_sel.to_excel(xw, sheet_name=_safe_sheet("exio_selected"), index=False)
        Z_rows.to_excel(xw, sheet_name=_safe_sheet("Z_rows_energy"))
        Z_cols.to_excel(xw, sheet_name=_safe_sheet("Z_cols_energy"))
        Z_block.to_excel(xw, sheet_name=_safe_sheet("Z_energy_block"))
        Y_rows.to_excel(xw, sheet_name=_safe_sheet("Y_rows_energy"))
        if result.exio_quantity_and_price is not None:
            result.exio_quantity_and_price.to_excel(
                xw, sheet_name=_safe_sheet("exio_qty_price"), index=False
            )

    _vprint(verbose, f"Wrote: {out_path.resolve()}")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        description="Split Eurostat AT IOT energy sectors (C19, D35) using EXIOBASE 2020 shares and RAS-balance."
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Base folder for cached datasets (defaults to repo-root `data/`, overridable via env var MRIO_DATA_ROOT).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each major step, including full paths for outputs.",
    )
    euro = p.add_argument_group("Eurostat input")
    euro.add_argument("--eurostat-z", help="Path to Eurostat Z (square).")
    euro.add_argument("--eurostat-y", help="Path to Eurostat Y (same index as Z).")
    euro.add_argument(
        "--eurostat-dataset",
        help="Eurostat dataset code (e.g. naio_10_cp1700) to download+parse into Z/Y.",
    )
    euro.add_argument(
        "--eurostat-filters",
        help="JSON dict of Eurostat dimension filters (e.g. '{\"geo\":\"AT\",\"time\":\"2021\",\"stk_flow\":\"DOM\"}').",
    )
    euro.add_argument("--eurostat-row-dim", help="Row dimension id in the Eurostat payload.")
    euro.add_argument("--eurostat-col-dim", help="Column dimension id in the Eurostat payload.")
    euro.add_argument("--eurostat-force-download", action="store_true")
    euro.add_argument("--eurostat-force-reparse", action="store_true")
    euro.add_argument(
        "--eurostat-stk-flows",
        help="JSON list of stk_flow values to run in one go (e.g. '[\"DOM\",\"IMP\"]'). Requires --eurostat-dataset mode.",
    )
    euro.add_argument(
        "--eurostat-stk-flow-dim",
        default="stk_flow",
        help="Dimension name for domestic/import variants (default: stk_flow).",
    )
    p.add_argument("--exio-year", type=int, default=2020)
    p.add_argument("--exio-region", default="AT")
    p.add_argument("--sector-19", default=DEFAULT_EUROSTAT_SECTOR_19)
    p.add_argument("--sector-35", default=DEFAULT_EUROSTAT_SECTOR_35)
    p.add_argument("--patterns-19", default=None, help="JSON list of patterns for C19 subsectors.")
    p.add_argument("--patterns-35", default=None, help="JSON list of patterns for D35 subsectors.")
    p.add_argument("--electricity-by-source", action="store_true")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument(
        "--no-xlsx-report",
        action="store_true",
        help="Skip writing the energy split XLSX report (CSV outputs are still written).",
    )
    args = p.parse_args(argv)

    from mrio_pyramids.data import default_data_root

    data_root = Path(args.data_root) if args.data_root is not None else default_data_root()
    eurostat_storage = data_root / "eurostat"
    _vprint(args.verbose, f"Using data root: {data_root.resolve()}")

    patterns_19 = DEFAULT_EXIO_PATTERNS_19
    patterns_35 = DEFAULT_EXIO_PATTERNS_35
    if args.patterns_19 is not None:
        patterns_19 = tuple(json.loads(args.patterns_19))
    if args.patterns_35 is not None:
        patterns_35 = tuple(json.loads(args.patterns_35))

    if (args.eurostat_z or args.eurostat_y) and args.eurostat_dataset:
        raise SystemExit("Provide either --eurostat-z/--eurostat-y OR --eurostat-dataset (not both).")

    if args.eurostat_dataset:
        if not (args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim):
            raise SystemExit(
                "When using --eurostat-dataset, also provide --eurostat-filters, --eurostat-row-dim, --eurostat-col-dim."
            )
        base_filters = json.loads(args.eurostat_filters)
        _vprint(args.verbose, f"Calling: load_eurostat_iot(dataset={args.eurostat_dataset}, storage={eurostat_storage})")

        if args.eurostat_stk_flows is not None:
            from mrio_pyramids.eurostat_iot import load_eurostat_iot_variants

            flows = list(map(str, json.loads(args.eurostat_stk_flows)))
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

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            _vprint(args.verbose, f"Calling: load_exiobase3_region_slice(region={args.exio_region}, year={args.exio_year})")
            t = _Timer()
            exio_slice = load_exiobase3_region_slice(
                year=args.exio_year,
                region=args.exio_region,
                system="pxp",
                cache_root=data_root,
                verbose=args.verbose,
            )
            _vprint(args.verbose, f"Returned: load_exiobase3_region_slice ({t.elapsed_s():.2f}s)")

            for flow, parsed in variants.items():
                _vprint(args.verbose, f"Variant: {args.eurostat_stk_flow_dim}={flow}")
                _vprint(args.verbose, "Calling: split_eurostat_energy_sectors_using_exiobase(...)")
                t = _Timer()
                res = split_eurostat_energy_sectors_using_exiobase(
                    eurostat_Z=parsed.Z,
                    eurostat_Y=parsed.Y,
                    exio_year=args.exio_year,
                    exio_region=args.exio_region,
                    exio_slice=exio_slice,
                    eurostat_sector_19=args.sector_19,
                    eurostat_sector_35=args.sector_35,
                    exio_patterns_19=patterns_19,
                    exio_patterns_35=patterns_35,
                    electricity_by_source=args.electricity_by_source,
                    verbose=args.verbose,
                )
                _vprint(args.verbose, f"Returned: split_eurostat_energy_sectors_using_exiobase ({t.elapsed_s():.2f}s)")

                flow_dir = outdir / f"{args.eurostat_stk_flow_dim}={flow}"
                flow_dir.mkdir(parents=True, exist_ok=True)
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
                    _vprint(args.verbose, "Calling: write_energy_split_report_xlsx(...)")
                    write_energy_split_report_xlsx(
                        flow_dir / "energy_split_report.xlsx",
                        result=res,
                        aggregates=(args.sector_19, args.sector_35),
                        verbose=args.verbose,
                    )
            return 0

        from mrio_pyramids.eurostat_iot import load_eurostat_iot as load_eurostat_iot_api

        t = _Timer()
        parsed = load_eurostat_iot_api(
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
        _vprint(args.verbose, f"Calling: load_eurostat_iot(z={args.eurostat_z}, y={args.eurostat_y})")
        t = _Timer()
        Z, Y = load_eurostat_iot(args.eurostat_z, args.eurostat_y)
        _vprint(args.verbose, f"Returned: load_eurostat_iot ({t.elapsed_s():.2f}s)")
    _vprint(args.verbose, "Calling: split_eurostat_energy_sectors_using_exiobase(...)")
    t = _Timer()
    res = split_eurostat_energy_sectors_using_exiobase(
        eurostat_Z=Z,
        eurostat_Y=Y,
        exio_year=args.exio_year,
        exio_region=args.exio_region,
        eurostat_sector_19=args.sector_19,
        eurostat_sector_35=args.sector_35,
        exio_patterns_19=patterns_19,
        exio_patterns_35=patterns_35,
        electricity_by_source=args.electricity_by_source,
        verbose=args.verbose,
    )
    _vprint(args.verbose, f"Returned: split_eurostat_energy_sectors_using_exiobase ({t.elapsed_s():.2f}s)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
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
    if not args.no_xlsx_report:
        _vprint(args.verbose, "Calling: write_energy_split_report_xlsx(...)")
        write_energy_split_report_xlsx(
            outdir / "energy_split_report.xlsx",
            result=res,
            aggregates=(args.sector_19, args.sector_35),
            verbose=args.verbose,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

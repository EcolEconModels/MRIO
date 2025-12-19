#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from ecolecon_mrio.iot_aggregation import (
    aggregate_iot,
    read_sector_aggregation_mapping_csv,
    write_aggregation_report_xlsx,
)


def _vprint(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


class _Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


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


def _write_df_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_series_csv(series: pd.Series, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, header=True)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_text(path: Path, content: str, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _vprint(verbose, f"Wrote: {path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Aggregate (re-group) an IO table (Z,Y) using a CSV mapping.")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--mapping-csv", required=True, help="CSV with columns from_sector,to_sector[,enabled,notes].")
    p.add_argument("--mapping-from-col", default="from_sector")
    p.add_argument("--mapping-to-col", default="to_sector")
    p.add_argument("--mapping-enabled-col", default="enabled")

    p.add_argument(
        "--require-from-contains",
        default=None,
        help="Safety check (CLI-only): require every from_sector in the mapping CSV to contain this substring "
        "(e.g. '::' to ensure mapping only targets disaggregated subsectors).",
    )

    inp = p.add_argument_group("Input")
    inp.add_argument("--Z", help="Path to Z table.")
    inp.add_argument("--Y", help="Path to Y table.")
    inp.add_argument(
        "--indir",
        help="Input directory. If it contains subfolders like 'stk_flow=DOM', all variants are processed.",
    )
    inp.add_argument("--variant-dim", default="stk_flow")
    inp.add_argument("--z-filename", default="Z_split_balanced.csv")
    inp.add_argument("--y-filename", default="Y_split_balanced.csv")

    out = p.add_argument_group("Output")
    out.add_argument("--outdir", required=True, help="Output directory.")
    out.add_argument("--no-xlsx", action="store_true", help="Skip XLSX report.")
    out.add_argument("--create-io-system", action="store_true", help="Create pymrio.IOSystem and calc_system().")

    args = p.parse_args(argv)

    mapping, mapping_df = read_sector_aggregation_mapping_csv(
        args.mapping_csv,
        from_col=args.mapping_from_col,
        to_col=args.mapping_to_col,
        enabled_col=args.mapping_enabled_col,
    )

    if args.require_from_contains is not None:
        bad = [k for k in mapping.keys() if args.require_from_contains not in str(k)]
        if bad:
            raise SystemExit(
                f"--require-from-contains {args.require_from_contains!r} failed for mapping rows: {bad[:10]}"
                + (f" (and {len(bad)-10} more)" if len(bad) > 10 else "")
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _process_one(Z_path: Path, Y_path: Path, out_base: Path) -> None:
        _vprint(args.verbose, f"Reading: {Z_path.resolve()}")
        Z = _read_table(Z_path).astype("float64")
        _vprint(args.verbose, f"Reading: {Y_path.resolve()}")
        Y = _read_table(Y_path).astype("float64")

        Z.index = Z.index.astype(str)
        Z.columns = Z.columns.astype(str)
        Y.index = Y.index.astype(str)
        Y.columns = Y.columns.astype(str)

        _vprint(args.verbose, "Calling: aggregate_iot(...)")
        t = _Timer()
        res = aggregate_iot(Z, Y, mapping, keep_unmapped=True, create_io_system=args.create_io_system)
        _vprint(args.verbose, f"Returned: aggregate_iot ({t.elapsed_s():.2f}s)")

        _write_df_csv(res.Z, out_base / "Z_agg.csv", verbose=args.verbose)
        _write_df_csv(res.Y, out_base / "Y_agg.csv", verbose=args.verbose)
        _write_series_csv(res.x, out_base / "x_agg.csv", verbose=args.verbose)
        _write_df_csv(res.A, out_base / "A_agg.csv", verbose=args.verbose)
        _write_text(
            out_base / "aggregation_info.json",
            json.dumps(
                {
                    "mapping_csv": str(Path(args.mapping_csv).resolve()),
                    "n_map_entries": len(mapping),
                    "n_sectors_before": int(len(Z.index)),
                    "n_sectors_after": int(len(res.Z.index)),
                },
                indent=2,
            ),
            verbose=args.verbose,
        )

        if not args.no_xlsx:
            out_xlsx = out_base / "aggregation_report.xlsx"
            write_aggregation_report_xlsx(out_xlsx, Z_before=Z, Y_before=Y, result=res, mapping_df=mapping_df)
            _vprint(args.verbose, f"Wrote: {out_xlsx.resolve()}")

    if args.indir:
        indir = Path(args.indir)
        if not indir.exists():
            raise SystemExit(f"--indir not found: {indir}")

        variant_prefix = f"{args.variant_dim}="
        variant_dirs = sorted([p for p in indir.iterdir() if p.is_dir() and p.name.startswith(variant_prefix)])
        if variant_dirs:
            for vd in variant_dirs:
                Z_path = vd / args.z_filename
                Y_path = vd / args.y_filename
                if not (Z_path.exists() and Y_path.exists()):
                    raise SystemExit(f"Missing {args.z_filename} or {args.y_filename} under {vd}")
                out_base = outdir / vd.name
                out_base.mkdir(parents=True, exist_ok=True)
                _process_one(Z_path, Y_path, out_base)
            return 0

        Z_path = indir / args.z_filename
        Y_path = indir / args.y_filename
        if not (Z_path.exists() and Y_path.exists()):
            raise SystemExit(
                f"--indir provided but could not find {args.z_filename} and {args.y_filename} in {indir} "
                f"and no '{variant_prefix}*' subfolders were present."
            )
        _process_one(Z_path, Y_path, outdir)
        return 0

    if not (args.Z and args.Y):
        raise SystemExit("Provide either --indir OR both --Z and --Y.")
    _process_one(Path(args.Z), Path(args.Y), outdir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


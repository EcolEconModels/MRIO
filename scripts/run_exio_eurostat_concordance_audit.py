#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import resource
from pathlib import Path

import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import download_eurostat_dataset, parse_eurostat_iot, read_eurostat_raw
from ecolecon_mrio.exio_eurostat_concordance_audit import (
    build_audit_metadata,
    build_concordance_audit,
    load_default_label_maps,
    write_concordance_audit_outputs,
)
from ecolecon_mrio.exiobase3 import load_exiobase3_region_slice


REPO_ROOT = Path(__file__).resolve().parents[1]


def _rss_mb() -> float:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    value_kb = float(parts[1])
                    unit = parts[2].lower() if len(parts) >= 3 else "kb"
                    if unit == "kb":
                        return value_kb / 1024.0
                    if unit == "mb":
                        return value_kb
                    if unit == "gb":
                        return value_kb * 1024.0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage) / 1024.0


def _log_stage(stage: str, message: str, started: float) -> None:
    elapsed = time.monotonic() - started
    print(f"[{stage}] {message} | elapsed={elapsed:.1f}s | rss={_rss_mb():.1f} MB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an EXIOBASE–Eurostat concordance audit.")
    parser.add_argument("--country", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--exio-year", type=int, default=None)
    parser.add_argument("--mapping-csv", default=str(REPO_ROOT / "scripts" / "mappings" / "exio_to_eurostat_cpa_products.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rel-threshold", type=float, default=0.20)
    parser.add_argument("--min-abs-meur", type=float, default=5.0)
    parser.add_argument("--identity-mode", choices=("strict", "diagnostic"), default="diagnostic")
    parser.add_argument("--eurostat-dataset", default="naio_10_cp1700")
    parser.add_argument("--eurostat-unit", default="MIO_EUR")
    parser.add_argument("--eurostat-flow", default="TOTAL")
    parser.add_argument("--exio-flow", default="total")
    parser.add_argument("--exio-system", default="pxp")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed audit-stage logging.")
    parser.add_argument("--full-subcomponent-z-scan", action="store_true", help="Evaluate Z-cell subcomponent exceedances by scanning every Z cell.")
    args = parser.parse_args()

    data_root = default_data_root()
    exio_year = args.exio_year if args.exio_year is not None else args.year
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "audits" / "exio_eurostat_concordance" / args.country / str(args.year)

    started = time.monotonic()
    _log_stage("1/5", f"Preparing Eurostat slice for {args.country} {args.year}...", started)

    euro_filters = {
        "geo": args.country,
        "time": str(args.year),
        "freq": "A",
        "unit": args.eurostat_unit,
        "stk_flow": args.eurostat_flow,
    }
    euro_dataset = download_eurostat_dataset(
        args.eurostat_dataset,
        filters=euro_filters,
        storage_folder=data_root / "eurostat",
        force_download=False,
    )
    _log_stage("2/5", "Parsing Eurostat IOT from cached raw payload...", started)
    eurostat = parse_eurostat_iot(read_eurostat_raw(euro_dataset.raw_path), row_dim="prd_ava", col_dim="prd_use")
    _log_stage("3/5", f"Loading EXIOBASE region slice for {args.country} {exio_year}...", started)
    exio = load_exiobase3_region_slice(
        year=exio_year,
        region=args.country,
        system=args.exio_system,
        flow=args.exio_flow,
        cache_root=data_root,
        force_download=False,
        force_rebuild_slice=False,
    )

    _log_stage("4/5", "Building concordance audit tables...", started)
    exio_labels, euro_labels = load_default_label_maps(REPO_ROOT)
    metadata = build_audit_metadata(
        country=args.country,
        year=args.year,
        exio_year=exio_year,
        eurostat_dataset=args.eurostat_dataset,
        eurostat_request_url=euro_dataset.request_url,
        mapping_file=args.mapping_csv,
        exio_flow=args.exio_flow,
        eurostat_unit=args.eurostat_unit,
    )

    result = build_concordance_audit(
        country=args.country,
        year=args.year,
        exio_x=exio.x_region,
        exio_Z=exio.Z_domestic,
        exio_Y=exio.Y_region,
        eurostat_Z=eurostat.Z,
        eurostat_Y=eurostat.Y,
        mapping_df=pd.read_csv(args.mapping_csv, dtype=str).fillna(""),
        exio_labels=exio_labels,
        eurostat_labels=euro_labels,
        metadata=metadata,
        rel_threshold=args.rel_threshold,
        min_abs_meur=args.min_abs_meur,
        identity_mode=args.identity_mode,
        verbose=args.verbose,
        full_subcomponent_z_scan=args.full_subcomponent_z_scan,
    )

    _log_stage("4/5", "Finished concordance audit tables.", started)

    _log_stage("5/5", "Writing audit outputs...", started)
    write_concordance_audit_outputs(result, output_dir)
    elapsed = time.monotonic() - started
    print(f"Wrote audit outputs to {output_dir.resolve()} in {elapsed:.1f}s | rss={_rss_mb():.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

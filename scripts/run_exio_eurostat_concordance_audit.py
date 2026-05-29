#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ecolecon_mrio.concordance_audit_flows import resolve_comparison_flows, run_single_flow_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an EXIOBASE–Eurostat concordance audit.")
    parser.add_argument("--country", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--exio-year", type=int, default=None)
    parser.add_argument("--mapping-csv", default=str(REPO_ROOT / "scripts" / "mappings" / "exio_to_eurostat_cpa_products.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--comparison-mode", choices=("domestic", "imports", "total"), default=None, help="Flow pairing to audit: domestic=EXIO domestic vs Eurostat DOM; imports=EXIO imports vs Eurostat IMP; total=EXIO total vs Eurostat TOTAL.")
    parser.add_argument("--rel-threshold", type=float, default=0.20)
    parser.add_argument("--min-abs-meur", type=float, default=5.0)
    parser.add_argument("--identity-mode", choices=("strict", "diagnostic"), default="diagnostic")
    parser.add_argument("--eurostat-dataset", default="naio_10_cp1700")
    parser.add_argument("--eurostat-unit", default="MIO_EUR")
    parser.add_argument("--eurostat-flow", default="DOM")
    parser.add_argument("--exio-flow", default="domestic")
    parser.add_argument("--allow-flow-mismatch", action="store_true", help="Allow inconsistent EXIO/Eurostat flow pairings. Use only for debugging.")
    parser.add_argument("--exio-system", default="pxp")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed audit-stage logging.")
    parser.add_argument("--full-subcomponent-z-scan", action="store_true", help="Evaluate Z-cell subcomponent exceedances by scanning every Z cell.")
    args = parser.parse_args()

    comparison_mode, exio_flow, eurostat_flow = resolve_comparison_flows(
        comparison_mode=args.comparison_mode,
        exio_flow=args.exio_flow,
        eurostat_flow=args.eurostat_flow,
        allow_flow_mismatch=args.allow_flow_mismatch,
    )
    args.exio_flow = exio_flow
    args.eurostat_flow = eurostat_flow

    if args.full_subcomponent_z_scan:
        print("Warning: --full-subcomponent-z-scan enables a slower full Z-cell scan.", flush=True)

    resolved_output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "audits" / "exio_eurostat_concordance" / args.country / str(args.year) / comparison_mode
    run_command = shlex.join(["PYTHONPATH=src", ".venv/bin/python", str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]])

    run_single_flow_audit(
        country=args.country,
        year=args.year,
        exio_year=args.exio_year,
        comparison_mode=comparison_mode,
        exio_flow=exio_flow,
        eurostat_flow=eurostat_flow,
        mapping_csv=Path(args.mapping_csv),
        output_dir=resolved_output_dir,
        rel_threshold=args.rel_threshold,
        min_abs_meur=args.min_abs_meur,
        identity_mode=args.identity_mode,
        eurostat_dataset=args.eurostat_dataset,
        eurostat_unit=args.eurostat_unit,
        exio_system=args.exio_system,
        verbose=args.verbose,
        full_subcomponent_z_scan=args.full_subcomponent_z_scan,
        run_command=run_command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

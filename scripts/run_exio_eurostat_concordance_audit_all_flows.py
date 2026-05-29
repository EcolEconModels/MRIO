#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ecolecon_mrio.concordance_audit_flows import FlowAuditRun, run_single_flow_audit, write_flow_consolidated_outputs
from ecolecon_mrio.exio_eurostat_concordance_audit import load_default_label_maps


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run domestic, imports, and total concordance audits and consolidate the results.")
    parser.add_argument("--country", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--exio-year", type=int, default=None)
    parser.add_argument("--mapping-csv", default=str(REPO_ROOT / "scripts" / "mappings" / "exio_to_eurostat_cpa_products.csv"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--rel-threshold", type=float, default=0.20)
    parser.add_argument("--min-abs-meur", type=float, default=5.0)
    parser.add_argument("--identity-mode", choices=("strict", "diagnostic"), default="diagnostic")
    parser.add_argument("--eurostat-dataset", default="naio_10_cp1700")
    parser.add_argument("--eurostat-unit", default="MIO_EUR")
    parser.add_argument("--exio-system", default="pxp")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed audit-stage logging.")
    parser.add_argument("--full-subcomponent-z-scan", action="store_true", help="Evaluate Z-cell subcomponent exceedances by scanning every Z cell.")
    args = parser.parse_args()

    if args.full_subcomponent_z_scan:
        print("Warning: --full-subcomponent-z-scan enables a slower full Z-cell scan.", flush=True)

    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs" / "audits" / "exio_eurostat_concordance" / args.country / str(args.year)
    flow_specs = [
        ("domestic", "domestic", "DOM"),
        ("imports", "imports", "IMP"),
        ("total", "total", "TOTAL"),
    ]
    run_command = shlex.join(["PYTHONPATH=src", ".venv/bin/python", str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]])

    runs: dict[str, FlowAuditRun] = {}
    for comparison_mode, exio_flow, eurostat_flow in flow_specs:
        result = run_single_flow_audit(
            country=args.country,
            year=args.year,
            exio_year=args.exio_year,
            comparison_mode=comparison_mode,
            exio_flow=exio_flow,
            eurostat_flow=eurostat_flow,
            mapping_csv=Path(args.mapping_csv),
            output_dir=output_root / comparison_mode,
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
        runs[comparison_mode] = FlowAuditRun(mode=comparison_mode, exio_flow=exio_flow, eurostat_flow=eurostat_flow, result=result)

    _, euro_labels = load_default_label_maps(REPO_ROOT)
    consolidated_dir = output_root / "consolidated"
    write_flow_consolidated_outputs(
        domestic=runs["domestic"],
        imports=runs["imports"],
        total=runs["total"],
        output_dir=consolidated_dir,
        eurostat_labels=euro_labels,
        rel_threshold=args.rel_threshold,
        min_abs_meur=args.min_abs_meur,
    )
    print(f"Wrote consolidated outputs to {consolidated_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

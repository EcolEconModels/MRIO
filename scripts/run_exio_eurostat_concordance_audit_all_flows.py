#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import cast

import pandas as pd

from ecolecon_mrio.concordance_audit_flows import FlowAuditRun, run_single_flow_audit, write_flow_consolidated_outputs
from ecolecon_mrio.exio_eurostat_concordance_audit import ConcordanceAuditResult, load_default_label_maps


REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_additive_audit_fields(
    result: ConcordanceAuditResult,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    required = {
        "mapped_z": result.mapped_z,
        "mapped_y": result.mapped_y,
        "mapped_x": result.mapped_x,
        "eurostat_z": result.eurostat_z,
        "eurostat_y_total": result.eurostat_y_total,
        "eurostat_x": result.eurostat_x,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Cannot construct additive mapped TOTAL; "
            f"missing audit result fields: {missing}"
        )
    return (
        cast(pd.DataFrame, result.mapped_z),
        cast(pd.Series, result.mapped_y),
        cast(pd.Series, result.mapped_x),
        cast(pd.DataFrame, result.eurostat_z),
        cast(pd.Series, result.eurostat_y_total),
        cast(pd.Series, result.eurostat_x),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run domestic and import concordance audits; optionally run an independently mapped TOTAL sensitivity.")
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
    parser.add_argument("--include-independent-total", action="store_true", help="Also run the legacy independently mapped TOTAL sensitivity and consolidated three-flow reconciliation.")
    args = parser.parse_args()

    if args.full_subcomponent_z_scan:
        print("Warning: --full-subcomponent-z-scan enables a slower full Z-cell scan.", flush=True)

    output_root = Path(args.output_root) if args.output_root else REPO_ROOT / "outputs" / "audits" / "exio_eurostat_concordance" / args.country / str(args.year)
    flow_specs = [
        ("domestic", "domestic", "DOM"),
        ("imports", "imports", "IMP"),
    ]
    # Namespace callers predating the opt-in flag retain the historical three-flow audit.
    if getattr(args, "include_independent_total", True):
        flow_specs.append(("total", "total", "TOTAL"))
    script = str(Path(__file__).relative_to(REPO_ROOT))
    run_command = "PYTHONPATH=src " + shlex.join([".venv/bin/python", script, *sys.argv[1:]])

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

    additive_dir = output_root / "additive_total"
    additive_dir.mkdir(parents=True, exist_ok=True)
    dom_z, dom_y, dom_x, dom_euro_z, dom_euro_y, dom_euro_x = _require_additive_audit_fields(
        runs["domestic"].result
    )
    imp_z, imp_y, imp_x, imp_euro_z, imp_euro_y, imp_euro_x = _require_additive_audit_fields(
        runs["imports"].result
    )
    rows: list[dict[str, object]] = []
    for metric, dom_values, imp_values, dom_euro, imp_euro in (
        ("TOTAL_USE_Z_PLUS_Y", dom_x, imp_x, dom_euro_x, imp_euro_x),
        ("Y_FINAL_DEMAND_TOTAL", dom_y, imp_y, dom_euro_y, imp_euro_y),
        ("Z_ROW_INTERMEDIATE_SALES", dom_z.sum(axis=1), imp_z.sum(axis=1), dom_euro_z.sum(axis=1), imp_euro_z.sum(axis=1)),
    ):
        codes = sorted(set(map(str, dom_values.index)).union(map(str, imp_values.index)))
        exio = dom_values.reindex(codes).fillna(0.0) + imp_values.reindex(codes).fillna(0.0)
        euro = dom_euro.reindex(codes).fillna(0.0) + imp_euro.reindex(codes).fillna(0.0)
        for code in codes:
            gap = float(exio.loc[code] - euro.loc[code])
            rows.append({
                "metric": metric,
                "target_code": code,
                "exio_additive_total": float(exio.loc[code]),
                "eurostat_additive_total": float(euro.loc[code]),
                "signed_gap": gap,
                "abs_gap": abs(gap),
                "rel_gap": abs(gap) / max(abs(float(euro.loc[code])), 1e-12),
            })
    pd.DataFrame(rows).to_csv(additive_dir / "additive_total_comparison.csv", index=False)

    if not getattr(args, "include_independent_total", True):
        print("Wrote domestic, import, and additive-total audits; independently mapped TOTAL was skipped.", flush=True)
        return 0

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

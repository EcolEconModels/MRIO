#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ecolecon_mrio.energy_balance_pxp import (
    compile_energy_balance_pxp_split_plan,
    load_energy_balance_dictionary_frame,
    load_energy_balance_pxp_input_technology_frame,
    load_energy_balance_pxp_prices_frame,
    load_energy_balance_pxp_reaggregation_frame,
    load_energy_balance_pxp_split_frame,
    run_energy_balance_pxp_split,
    write_energy_balance_pxp_diagnostics,
)
from ecolecon_mrio.eurostat_iot import load_eurostat_iot_variants
from ecolecon_mrio.iot_aggregation import aggregate_iot

try:  # pragma: no cover - optional workbook helper
    from aggregate_iot_by_mapping import write_aggregation_report_xlsx
except ImportError:  # pragma: no cover
    write_aggregation_report_xlsx = None


def _parse_json_list(value: str) -> list[str]:
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list.")
    return [str(item) for item in data]


def _write_split_outputs(outdir: Path, result: Any) -> None:
    pd.DataFrame(result.Z).to_csv(outdir / "Z_split.csv")
    pd.DataFrame(result.Y).to_csv(outdir / "Y_split.csv")
    pd.Series(result.x).to_csv(outdir / "x_split.csv", header=True)
    pd.DataFrame(result.A).to_csv(outdir / "A_split.csv")


def _write_reaggregated_outputs(outdir: Path, result: Any) -> None:
    pd.DataFrame(result.Z).to_csv(outdir / "Z_reaggregated.csv")
    pd.DataFrame(result.Y).to_csv(outdir / "Y_reaggregated.csv")
    pd.Series(result.x).to_csv(outdir / "x_reaggregated.csv", header=True)
    pd.DataFrame(result.A).to_csv(outdir / "A_reaggregated.csv")


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Split Eurostat pxp IOT using the Energy Balance pxp recipe.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--eurostat-dataset", default="naio_10_cp1700")
    parser.add_argument("--eurostat-row-dim", default="prd_ava")
    parser.add_argument("--eurostat-col-dim", default="prd_use")
    parser.add_argument("--eurostat-stk-flow-dim", default="stk_flow")
    parser.add_argument("--eurostat-stk-flows", default='["DOM","IMP","TOTAL"]')
    parser.add_argument(
        "--split-csv",
        default=str(repo_root / "scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_split.csv"),
    )
    parser.add_argument(
        "--reaggregation-csv",
        default=str(repo_root / "scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_reaggregation.csv"),
    )
    parser.add_argument(
        "--input-technology-csv",
        default=str(repo_root / "scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_input_technology.csv"),
    )
    parser.add_argument(
        "--prices-csv",
        default=str(repo_root / "scripts/pipelines/eurostat_energybalances_split/energy_balance_pxp_prices.csv"),
    )
    parser.add_argument(
        "--energy-balance-dictionary-csv",
        default=str(repo_root / "scripts/mappings/eurostat_energy_balances_dictionary.csv"),
    )
    parser.add_argument("--outdir", default=str(repo_root / "outputs/AT_2021_energy_balance_pxp_split"))
    parser.add_argument("--storage-folder", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--balancing-scope", default="local", choices=["local", "global"])
    parser.add_argument("--write-reaggregated", action="store_true")
    parser.add_argument("--write-sfc-workbook", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    base_filters = {
        "geo": args.geo,
        "time": str(args.year),
        "freq": "A",
        "unit": "MIO_EUR",
    }
    flows = _parse_json_list(args.eurostat_stk_flows)
    variants = load_eurostat_iot_variants(
        dataset=args.eurostat_dataset,
        base_filters=base_filters,
        variant_dim=args.eurostat_stk_flow_dim,
        variant_values=flows,
        row_dim=args.eurostat_row_dim,
        col_dim=args.eurostat_col_dim,
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )

    split_frame = load_energy_balance_pxp_split_frame(args.split_csv)
    reagg_frame = load_energy_balance_pxp_reaggregation_frame(args.reaggregation_csv)
    tech_frame = load_energy_balance_pxp_input_technology_frame(args.input_technology_csv)
    prices_frame = load_energy_balance_pxp_prices_frame(args.prices_csv)
    eb_dictionary = load_energy_balance_dictionary_frame(args.energy_balance_dictionary_csv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    total_variant = variants.get("TOTAL")
    if total_variant is None:
        total_variant = variants[str(flows[0])]

    plan = compile_energy_balance_pxp_split_plan(
        Z=total_variant.Z,
        Y=total_variant.Y,
        split_frame=split_frame,
        reaggregation_frame=reagg_frame,
        technology_frame=tech_frame,
        prices_frame=prices_frame,
        energy_balance_dictionary_frame=eb_dictionary,
        geo=args.geo,
        year=args.year,
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )

    write_energy_balance_pxp_diagnostics(
        outdir / "plan_diagnostics",
        result={
            "component_values": plan.component_values,
            "split_shares": plan.split_shares,
            "selector_residuals": plan.selector_residuals,
            "parent_residuals": plan.parent_residuals,
            "price_resolution": plan.price_resolution,
            "diagnostics": plan.diagnostics,
        },
        missing_or_zero_selectors=plan.diagnostics.get("missing_or_zero_selectors"),
    )

    run_summaries: list[dict[str, object]] = []
    for flow in flows:
        parsed = variants[str(flow)]
        flow_outdir = outdir / f"{args.eurostat_stk_flow_dim}={flow}"
        flow_outdir.mkdir(parents=True, exist_ok=True)

        split_result = run_energy_balance_pxp_split(
            parsed.Z,
            parsed.Y,
            plan,
            balancing_scope=args.balancing_scope,
            create_io_system=False,
        )
        _write_split_outputs(flow_outdir, split_result)

        reaggregated_rows = None
        if args.write_reaggregated:
            aggregated = aggregate_iot(
                split_result.Z,
                split_result.Y,
                plan.reaggregation_mapping,
                keep_unmapped=True,
                create_io_system=False,
            )
            _write_reaggregated_outputs(flow_outdir, aggregated)
            reaggregated_rows = int(len(aggregated.Z))

            if args.write_sfc_workbook:
                if write_aggregation_report_xlsx is None:
                    raise RuntimeError("--write-sfc-workbook requested but aggregate_iot_by_mapping helper is unavailable.")
                write_aggregation_report_xlsx(
                    flow_outdir / "aggregation_report.xlsx",
                    Z_before=split_result.Z,
                    Y_before=split_result.Y,
                    result=aggregated,
                    mapping_df=reagg_frame,
                )

        run_summaries.append(
            {
                "variant_dim": args.eurostat_stk_flow_dim,
                "variant_value": flow,
                "n_split_rows": int(len(split_result.Z)),
                "n_reaggregated_rows": reaggregated_rows,
                "n_components": int(len(plan.component_values)),
                "n_zero_or_missing_selectors": int(len(plan.diagnostics.get("missing_or_zero_selectors", pd.DataFrame()))),
            }
        )

    pd.DataFrame(run_summaries).to_csv(outdir / "run_summary.csv", index=False)
    if args.verbose:
        print(f"Wrote Energy Balance pxp split outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

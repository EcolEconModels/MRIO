#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecolecon_mrio.transition_checks import (
    annualized_growth_rate,
    compound_endpoint,
    compute_closure_growth,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check transition endpoints and closure target consistency.")
    parser.add_argument("--weights-csv", required=True, help="CSV with sector weights and block labels.")
    parser.add_argument("--weight-col", default="weight", help="Column containing closure weights.")
    parser.add_argument("--block-col", default="block", help="Column containing block labels.")
    parser.add_argument("--renewable-label", default="ren", help="Block label used for renewable sectors.")
    parser.add_argument("--nonrenewable-label", default="nren", help="Block label used for non-renewable sectors.")
    parser.add_argument("--years", type=int, required=True, help="Number of years including the base year.")
    parser.add_argument("--ren-start", type=float, required=True)
    parser.add_argument("--ren-end", type=float, required=True)
    parser.add_argument("--nren-start", type=float, required=True)
    parser.add_argument("--nren-end", type=float, required=True)
    parser.add_argument("--aggregate-growth", type=float, required=True, help="Aggregate growth target g.")
    parser.add_argument(
        "--closure-option",
        default="residual-others",
        choices=["residual-others", "fixed-others", "uniform-demand", "eps-only"],
    )
    args = parser.parse_args(argv)

    weights = pd.read_csv(Path(args.weights_csv))
    if args.weight_col not in weights.columns or args.block_col not in weights.columns:
        raise ValueError(
            f"weights CSV must contain {args.weight_col!r} and {args.block_col!r}; columns={list(weights.columns)!r}"
        )
    weight_vec = pd.to_numeric(weights[args.weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    block_series = weights[args.block_col].astype(str)
    ren_idx = [int(idx) for idx, block in enumerate(block_series) if block == str(args.renewable_label)]
    nren_idx = [int(idx) for idx, block in enumerate(block_series) if block == str(args.nonrenewable_label)]

    transitions = int(args.years) - 1
    eps_ren = annualized_growth_rate(float(args.ren_start), float(args.ren_end), transitions)
    eps_nren = annualized_growth_rate(float(args.nren_start), float(args.nren_end), transitions)
    closure = compute_closure_growth(
        option=str(args.closure_option),
        eps_ren=eps_ren,
        eps_nren=eps_nren,
        aggregate_growth=float(args.aggregate_growth),
        weights=weight_vec,
        renewable_indices=ren_idx,
        nonrenewable_indices=nren_idx,
    )

    print(
        json.dumps(
            {
                "years": int(args.years),
                "closure_option": str(args.closure_option),
                "eps_ren": eps_ren,
                "eps_nren": eps_nren,
                "projected_ren_end": compound_endpoint(float(args.ren_start), eps_ren, int(args.years)),
                "projected_nren_end": compound_endpoint(float(args.nren_start), eps_nren, int(args.years)),
                **closure.to_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
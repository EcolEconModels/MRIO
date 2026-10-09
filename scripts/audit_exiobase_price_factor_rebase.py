#!/usr/bin/env python3
"""Audit a monetary rebase against the complete previously certified factor panel.

The public MRIO factor builder retains ownership of GDP/FX observations, geography
and conversion formulas. This audit compares every region/year against frozen
numerical evidence before checking the change of euro gauge. It adds no MRIO cache.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecolecon_mrio import build_exiobase3_eur_price_factors  # noqa: E402

REFERENCE = ROOT / "tests/reference_data/exiobase3_common_eur_2022_2012_2022.csv"
CERTIFIED_SHA = "11827838f4fd1594ec4a7410eb5ab30fb2135b8bdba4881625cac21f0698f8f2"


def compare_reference(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    """Reject omissions, metadata drift or any numeric revision in the frozen panel.

    Comparison is exact after CSV round-trip parsing; it protects all 539 certified
    observations rather than only identity factors or a few conversion fixtures.
    """
    keys = ["year", "exio_region"]
    if reference.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("factor panel has duplicate region/year keys")
    # Select exactly the reference domain, then align labels before comparing values.
    years = reference["year"].astype(int).unique().tolist()
    selected = candidate.loc[candidate["year"].isin(years)].sort_values(keys).reset_index(drop=True)
    expected = reference.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, selected, check_exact=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Verify full old-domain equivalence and the region-wise euro-gauge identity."""
    reference_bytes = Path(args.reference_panel).read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != CERTIFIED_SHA:
        raise ValueError("reference panel no longer matches its previously certified SHA-256")
    reference = pd.read_csv(args.reference_panel, float_precision="round_trip")
    if len(reference) != 539:
        raise ValueError("certified reference requires all 49 regions in 2012--2022")
    # Reconstruct using the older cached UNSD query vintage, then compare the extended
    # query vintage independently. A source revision remains a blocking finding.
    old, old_meta = build_exiobase3_eur_price_factors(
        years=range(2012, 2023), base_year=2022, storage_folder=args.storage_folder,
    )
    compare_reference(reference, old)
    extended, extended_meta = build_exiobase3_eur_price_factors(
        years=range(2012, int(args.end_year) + 1), base_year=2022,
        storage_folder=args.storage_folder,
    )
    compare_reference(reference, extended)
    rebased, rebased_meta = build_exiobase3_eur_price_factors(
        years=range(2012, int(args.end_year) + 1), base_year=int(args.base_year),
        storage_folder=args.storage_folder,
    )
    # A change of base is a time-invariant region diagonal: eta[t;b] =
    # eta[t;2022]/eta[b;2022]. This must hold for every admitted observation.
    anchor = extended.loc[extended["year"].eq(args.base_year)].set_index("exio_region")["price_adjustment_factor"]
    predicted = extended["price_adjustment_factor"].to_numpy(float) / extended["exio_region"].map(anchor).to_numpy(float)
    np.testing.assert_allclose(rebased["price_adjustment_factor"], predicted, rtol=1e-14, atol=0.0)
    identities = rebased.loc[rebased["year"].eq(args.base_year), "price_adjustment_factor"].to_numpy(float)
    np.testing.assert_array_equal(identities, np.ones(49))
    payload: dict[str, Any] = {
        "schema_version": "exiobase_price_factor_rebase_audit_v1",
        "certified_panel_sha256": CERTIFIED_SHA, "certified_factor_count": 539,
        "all_certified_factors_exact": True, "base_year": int(args.base_year),
        "end_year": int(args.end_year), "base_identity_count": len(identities),
        "gauge_identity_passed": True, "old_panel": old_meta,
        "extended_panel": extended_meta, "rebased_panel": rebased_meta,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: all 539 certified factors exact; {args.base_year} EUR gauge identity and 49 base identities", flush=True)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the source cache and caller-selected monetary base for the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-panel", type=Path, default=REFERENCE)
    parser.add_argument("--storage-folder", type=Path, default=None)
    parser.add_argument("--base-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.end_year < 2022 or not 2012 <= args.base_year <= args.end_year:
        parser.error("require end-year >= 2022 and base-year inside the full 2012--end window")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ecolecon_mrio.eurostat_energy_balances import write_eurostat_energy_balance_dictionary_csv


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build Eurostat Energy Balance dictionary from official SDMX metadata.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument(
        "--out",
        default=str(repo_root / "scripts/mappings/eurostat_energy_balances_dictionary.csv"),
    )
    parser.add_argument("--storage-folder", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-reparse", action="store_true")
    args = parser.parse_args(argv)

    df = write_eurostat_energy_balance_dictionary_csv(
        args.out,
        geo=args.geo,
        time=args.year,
        storage_folder=args.storage_folder,
        force_download=args.force_download,
        force_reparse=args.force_reparse,
    )
    print(f"Wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ecolecon_mrio.eurostat_iot import load_eurostat_iot
from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors
from ecolecon_mrio.exio_to_eurostat_mapping import build_exio_to_eurostat_mapping


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


def _normalize_cpa_codes(codes: Iterable[str]) -> list[str]:
    out = []
    for c in codes:
        s = str(c).strip()
        if not s:
            continue
        out.append(s)
    return list(dict.fromkeys(out))


def main() -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        description=(
            "Generate an EXIOBASE -> Eurostat CPA mapping CSV. "
            "Uses Eurostat labels with rule-based overrides for energy sectors."
        )
    )
    p.add_argument(
        "--eurostat-z",
        default=None,
        help="Path to a Eurostat Z table (used to infer CPA sector codes).",
    )
    p.add_argument("--eurostat-dataset", default=None, help="Eurostat dataset id (e.g. naio_10_cp1700).")
    p.add_argument("--eurostat-filters", default=None, help="Eurostat filters as JSON.")
    p.add_argument("--eurostat-row-dim", default=None, help="Eurostat row dimension id (e.g. prd_ava).")
    p.add_argument("--eurostat-col-dim", default=None, help="Eurostat col dimension id (e.g. prd_use).")
    p.add_argument("--eurostat-stk-flow-dim", default="stk_flow", help="Eurostat stock/flow dimension id.")
    p.add_argument("--eurostat-stk-flow-value", default="TOTAL", help="Eurostat stock/flow value.")
    p.add_argument("--eurostat-storage-folder", default="data/eurostat", help="Eurostat cache folder.")
    p.add_argument(
        "--min-label-similarity",
        type=float,
        default=0.25,
        help="Minimum token overlap (Jaccard) between EXIO and CPA labels for auto-matching.",
    )
    p.add_argument("--out", required=True, help="Output CSV mapping file.")
    args = p.parse_args()

    row_labels = None
    if args.eurostat_z:
        z = _read_table(Path(args.eurostat_z))
    else:
        if not (args.eurostat_dataset and args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim):
            raise SystemExit(
                "Provide --eurostat-z OR all of --eurostat-dataset, --eurostat-filters, "
                "--eurostat-row-dim, --eurostat-col-dim."
            )
        filters = json.loads(str(args.eurostat_filters))
        if args.eurostat_stk_flow_dim and args.eurostat_stk_flow_dim not in filters:
            filters[args.eurostat_stk_flow_dim] = str(args.eurostat_stk_flow_value)

        parsed = load_eurostat_iot(
            dataset=str(args.eurostat_dataset),
            filters=filters,
            row_dim=str(args.eurostat_row_dim),
            col_dim=str(args.eurostat_col_dim),
            storage_folder=Path(args.eurostat_storage_folder),
            force_download=False,
            force_reparse=False,
        )
        z = parsed.Z
        row_labels = parsed.meta.get("dimensions", {}).get(str(args.eurostat_row_dim), {})

    cpa_codes = _normalize_cpa_codes(z.index.astype(str))

    exio_sectors = load_pymrio_exio3_sectors("pxp").sectors
    valid_exio = exio_sectors["ExioName"].astype(str).str.strip().tolist()

    if row_labels is None:
        row_labels = {code: code for code in cpa_codes}
    cpa_label_map = {code: str(row_labels.get(code, code)) for code in cpa_codes}

    df_out = build_exio_to_eurostat_mapping(
        valid_exio,
        cpa_codes,
        cpa_label_map,
        min_similarity=float(args.min_label_similarity),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)

    mapped_exio = df_out["exio_sector"].nunique() if not df_out.empty else 0
    mapped_cpa = df_out["aggregate_sector"].nunique() if not df_out.empty else 0
    total_exio = exio_sectors["ExioName"].nunique()
    total_cpa = len(cpa_codes)
    print(
        "Mapping coverage: "
        f"EXIO sectors mapped {mapped_exio}/{total_exio}, "
        f"Eurostat CPA sectors covered {mapped_cpa}/{total_cpa}"
    )
    print(f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

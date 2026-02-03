#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from ecolecon_mrio.eurostat_iot import load_eurostat_iot
from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors


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


def _division_to_section(div: int) -> str | None:
    if 1 <= div <= 3:
        return "A"
    if 5 <= div <= 9:
        return "B"
    if 10 <= div <= 33:
        return "C"
    if div == 35:
        return "D"
    if 36 <= div <= 39:
        return "E"
    if 41 <= div <= 43:
        return "F"
    if 45 <= div <= 47:
        return "G"
    if 49 <= div <= 53:
        return "H"
    if 55 <= div <= 56:
        return "I"
    if 58 <= div <= 63:
        return "J"
    if 64 <= div <= 66:
        return "K"
    if div == 68:
        return "L"
    if 69 <= div <= 75:
        return "M"
    if 77 <= div <= 82:
        return "N"
    if div == 84:
        return "O"
    if div == 85:
        return "P"
    if 86 <= div <= 88:
        return "Q"
    if 90 <= div <= 93:
        return "R"
    if 94 <= div <= 96:
        return "S"
    if 97 <= div <= 98:
        return "T"
    return None


def _parse_exio_division(exio_code: str) -> int | None:
    if not exio_code:
        return None
    m = re.search(r"(\d{2})", str(exio_code))
    if not m:
        return None
    return int(m.group(1))


def _parse_cpa_code(code: str) -> tuple[str, int | None, int | None] | None:
    s = str(code).strip().upper()
    if s.startswith("CPA_"):
        s = s[4:]
    if not s:
        return None
    # Letter-only section (e.g., B, D, F)
    if re.fullmatch(r"[A-Z]", s):
        return s, None, None

    m = re.fullmatch(r"([A-Z])(\d{1,2})(?:-(\d{1,2}))?", s)
    if not m:
        return None
    letter = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) is not None else start
    return letter, start, end


def _choose_best_match(div: int, candidates: list[tuple[str, int | None, int | None, str]]) -> str | None:
    # Prefer exact division matches over ranges; then smallest range length.
    best = None
    best_span = None
    for letter, start, end, code in candidates:
        if start is None or end is None:
            span = 999
        else:
            span = end - start
        if best is None:
            best = code
            best_span = span
            continue
        if span < best_span:
            best = code
            best_span = span
    return best


def _match_cpa_for_division(division: int, cpa_codes: Iterable[str]) -> str | None:
    section = _division_to_section(division)
    if section is None:
        return None

    candidates: list[tuple[str, int | None, int | None, str]] = []
    for code in cpa_codes:
        parsed_cpa = _parse_cpa_code(code)
        if parsed_cpa is None:
            continue
        c_letter, start, end = parsed_cpa
        if c_letter != section:
            continue
        if start is None or end is None:
            candidates.append((c_letter, start, end, code))
        elif start <= division <= end:
            candidates.append((c_letter, start, end, code))

    if not candidates:
        return None
    return _choose_best_match(division, candidates)


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
            "Generate an EXIOBASE -> Eurostat CPA mapping CSV from a concordance file. "
            "Uses NACE codes to match CPA divisions present in a Eurostat IO table."
        )
    )
    p.add_argument(
        "--concordance-xlsx",
        default="data/exiobase_concordance_sources/table_correspondance_nace-gics-exiobase_20230816.xlsx",
        help="Optional EXIOBASE concordance workbook (not required for code-based mapping).",
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
    p.add_argument("--out", required=True, help="Output CSV mapping file.")
    args = p.parse_args()

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
    cpa_codes = _normalize_cpa_codes(z.index.astype(str))

    rows = []
    sectors = load_pymrio_exio3_sectors("pxp").sectors
    for _, row in sectors.iterrows():
        exio_name = str(row.get("ExioName", "")).strip()
        exio_code = str(row.get("ExioCode", "")).strip()
        if not exio_name or not exio_code:
            continue
        division = _parse_exio_division(exio_code)
        if division is None:
            continue
        target = _match_cpa_for_division(division, cpa_codes)
        if not target:
            continue
        rows.append(
            {
                "aggregate_sector": target,
                "exio_sector": exio_name,
                "enabled": "1",
                "notes": exio_code,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)

    mapped_exio = df_out["exio_sector"].nunique() if not df_out.empty else 0
    mapped_cpa = df_out["aggregate_sector"].nunique() if not df_out.empty else 0
    total_exio = sectors["ExioName"].nunique()
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

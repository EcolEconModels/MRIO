#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ecolecon_mrio.eurostat_iot import load_eurostat_iot
from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors
from ecolecon_mrio.exio_to_eurostat_mapping import (
    build_cpa_division_index,
    build_exio_to_eurostat_mapping,
    candidate_cpa_codes_for_exio,
    filter_cpa_candidates_for_exio,
)


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
            "Uses Eurostat labels plus EXIO code-based heuristics (NACE/CPA divisions)."
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
        "--no-split-multi",
        action="store_true",
        help="Do not split EXIO sectors across multiple CPA codes; keep a single best match.",
    )
    p.add_argument(
        "--no-eurostat-x-weights",
        action="store_true",
        help="Do not weight splits by Eurostat output (x); use equal weights.",
    )
    p.add_argument(
        "--min-label-similarity",
        type=float,
        default=0.6,
        help="Minimum token overlap (Jaccard) between EXIO and CPA labels for auto-matching.",
    )
    p.add_argument(
        "--min-tfidf-similarity",
        type=float,
        default=0.15,
        help="Minimum TF-IDF cosine similarity for label matching fallback.",
    )
    p.add_argument("--out", required=True, help="Output CSV mapping file.")
    args = p.parse_args()

    row_labels = None
    Y = None
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
        Y = parsed.Y
        row_labels = parsed.meta.get("dimensions", {}).get(str(args.eurostat_row_dim), {})

    cpa_codes = _normalize_cpa_codes(z.index.astype(str))

    exio_sectors = load_pymrio_exio3_sectors("pxp").sectors
    valid_exio = exio_sectors["ExioName"].astype(str).str.strip().tolist()
    exio_code_map = dict(
        zip(
            exio_sectors["ExioName"].astype(str).str.strip(),
            exio_sectors["ExioCode"].astype(str).str.strip(),
        )
    )

    if row_labels is None:
        row_labels = {code: code for code in cpa_codes}
    cpa_label_map = {code: str(row_labels.get(code, code)) for code in cpa_codes}

    base_df = build_exio_to_eurostat_mapping(
        valid_exio,
        cpa_codes,
        cpa_label_map,
        exio_code_map=exio_code_map,
        min_similarity=float(args.min_label_similarity),
        min_tfidf_similarity=float(args.min_tfidf_similarity),
    )
    if base_df.empty:
        df_out = base_df.copy()
    else:
        df_out = base_df.copy()

    if "weight" not in df_out.columns:
        df_out["weight"] = 1.0

    split_multi = not args.no_split_multi
    use_x_weights = not args.no_eurostat_x_weights
    x_series = None
    if use_x_weights and Y is not None:
        x_series = (z.sum(axis=1) + Y.sum(axis=1)).astype("float64")
        x_series = x_series.replace([pd.NA, pd.NaT], 0.0)
        x_series = pd.to_numeric(x_series, errors="coerce").fillna(0.0)

    def _weights_for_candidates(candidates: list[str]) -> dict[str, float]:
        if not candidates:
            return {}
        if x_series is None:
            w = 1.0 / len(candidates)
            return {c: w for c in candidates}
        vals = [float(x_series.get(c, 0.0)) for c in candidates]
        vals = [v if v > 0 else 0.0 for v in vals]
        total = float(sum(vals))
        if total <= 0:
            w = 1.0 / len(candidates)
            return {c: w for c in candidates}
        return {c: v / total for c, v in zip(candidates, vals)}

    def _split_note(candidates: list[str], weights: dict[str, float]) -> str:
        if not candidates:
            return ""
        parts = [f"{c}={weights.get(c, 0.0):.4f}" for c in candidates]
        if x_series is None:
            basis = "equal weights (no Eurostat x available)"
        else:
            basis = "Eurostat output shares (x = Z+Y)"
        return (
            "Split mapping across CPA codes using "
            f"{basis}. Weights: " + ", ".join(parts) + "."
        )

    if split_multi:
        div_index = build_cpa_division_index(cpa_codes)
        split_rows: list[dict[str, object]] = []
        split_exio: set[str] = set()
        for exio_name in valid_exio:
            exio_code = exio_code_map.get(exio_name, "")
            candidates = candidate_cpa_codes_for_exio(exio_code, division_index=div_index)
            candidates = filter_cpa_candidates_for_exio(exio_name, candidates)
            candidates = [c for c in candidates if c in cpa_codes]
            if len(candidates) <= 1:
                continue
            split_exio.add(exio_name)
            weights = _weights_for_candidates(candidates)
            note = _split_note(candidates, weights)
            for cpa_code in candidates:
                split_rows.append(
                    {
                        "aggregate_sector": cpa_code,
                        "exio_sector": exio_name,
                        "weight": float(weights.get(cpa_code, 0.0)),
                        "enabled": "1",
                        "notes": note,
                    }
                )
        if split_exio:
            df_out = df_out[~df_out["exio_sector"].isin(split_exio)].copy()
            if split_rows:
                df_out = pd.concat([df_out, pd.DataFrame(split_rows)], ignore_index=True)

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

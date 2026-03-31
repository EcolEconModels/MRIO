#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _flat_diagnostics_common import (
    load_eurostat_iot_bundle,
    load_exio_slice,
    load_weighted_mapping,
    mapping_label_maps,
    repo_root_from_file,
    write_csv,
)


FLOW_SPECS = {
    "dom": ("DOM", "domestic"),
    "imp": ("IMP", "imports"),
    "total": ("TOTAL", "total"),
}


def default_output_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "diagnostics_debug"


def _enabled_weight_one_mapping(mapping_df: pd.DataFrame) -> pd.DataFrame:
    weights = mapping_df.copy()
    weights["aggregate_sector"] = weights["aggregate_sector"].astype(str)
    weights["exio_sector"] = weights["exio_sector"].astype(str)
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    weights["enabled"] = weights["enabled"].astype(str).str.strip().str.lower()
    weights = weights[weights["enabled"].isin({"", "1", "true", "t", "yes", "y", "on"})].copy()
    weights = weights[weights["weight"] == 1.0].copy()
    weights = weights[(weights["aggregate_sector"] != "") & (weights["exio_sector"] != "")].copy()
    return weights


def _strict_pairs(mapping_df: pd.DataFrame) -> pd.DataFrame:
    strict = _enabled_weight_one_mapping(mapping_df)
    target_counts = strict.groupby("aggregate_sector")["exio_sector"].nunique()
    source_counts = strict.groupby("exio_sector")["aggregate_sector"].nunique()
    strict = strict[strict["aggregate_sector"].map(target_counts) == 1].copy()
    strict = strict[strict["exio_sector"].map(source_counts) == 1].copy()
    return strict[["aggregate_sector", "exio_sector"]].drop_duplicates().sort_values("aggregate_sector")


def _build_cell_outliers(
    *,
    strict_pairs: pd.DataFrame,
    z_exio: pd.DataFrame,
    z_euro: pd.DataFrame,
    target_labels: dict[str, str],
    source_labels: dict[str, str],
    flow_code: str,
    material_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left in strict_pairs.itertuples(index=False):
        for right in strict_pairs.itertuples(index=False):
            target_i = str(left.aggregate_sector)
            target_j = str(right.aggregate_sector)
            source_i = str(left.exio_sector)
            source_j = str(right.exio_sector)
            exio_value = float(z_exio.loc[source_i, source_j])
            euro_value = float(z_euro.loc[target_i, target_j])
            if exio_value == 0.0 and euro_value == 0.0:
                continue
            gap = exio_value - euro_value
            rel_gap = gap / euro_value if euro_value else np.nan
            if abs(euro_value) < material_threshold:
                continue
            rows.append(
                {
                    "flow": flow_code,
                    "target_i": target_i,
                    "target_i_name": target_labels.get(target_i, target_i),
                    "target_j": target_j,
                    "target_j_name": target_labels.get(target_j, target_j),
                    "source_i": source_i,
                    "source_i_name": source_labels.get(source_i, source_i),
                    "source_j": source_j,
                    "source_j_name": source_labels.get(source_j, source_j),
                    "exio_zij_meur_2020": exio_value,
                    "eurostat_zij_meur_2020": euro_value,
                    "gap_meur_2020": gap,
                    "gap_pct_of_eurostat_2020": rel_gap * 100.0 if np.isfinite(rel_gap) else np.nan,
                    "abs_gap_meur_2020": abs(gap),
                    "comparison_basis": (
                        "Strict 1:1 sectors only. Eurostat benchmark = Austria intermediate-use block Z only. "
                        "EXIO benchmark = Austria-destination intermediate-use block Z for the selected flow."
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_gap_meur_2020", "target_i", "target_j"], ascending=[False, True, True], kind="stable")


def _build_row_outliers(
    *,
    strict_pairs: pd.DataFrame,
    z_exio: pd.DataFrame,
    z_euro: pd.DataFrame,
    target_labels: dict[str, str],
    source_labels: dict[str, str],
    flow_code: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in strict_pairs.itertuples(index=False):
        target = str(item.aggregate_sector)
        source = str(item.exio_sector)
        exio_total = float(z_exio.loc[source].sum())
        euro_total = float(z_euro.loc[target].sum())
        gap = exio_total - euro_total
        rows.append(
            {
                "flow": flow_code,
                "target_sector": target,
                "target_sector_name": target_labels.get(target, target),
                "source_sector": source,
                "source_sector_name": source_labels.get(source, source),
                "exio_row_total_meur_2020": exio_total,
                "eurostat_row_total_meur_2020": euro_total,
                "gap_meur_2020": gap,
                "gap_pct_of_eurostat_2020": (100.0 * gap / euro_total) if euro_total else np.nan,
                "ratio_exio_to_eurostat": (exio_total / euro_total) if euro_total else np.nan,
                "comparison_basis": (
                    "Strict 1:1 sectors only. Row totals are summed from the Austria intermediate-use block Z for the selected flow."
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_gap_meur_2020"] = out["gap_meur_2020"].abs()
    return out.sort_values(["abs_gap_meur_2020", "target_sector"], ascending=[False, True], kind="stable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strict 1:1 z_ij outlier diagnostics for Austria 2020.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--region", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--flow", choices=sorted(FLOW_SPECS), default="dom")
    parser.add_argument("--material-threshold", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = repo_root_from_file(__file__)
    output_dir = args.output_dir or default_output_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_code, exio_flow = FLOW_SPECS[args.flow]
    mapping_df = load_weighted_mapping(repo_root)
    target_labels, source_labels = mapping_label_maps(mapping_df)
    strict_pairs = _strict_pairs(mapping_df)

    parsed_iot, _, _ = load_eurostat_iot_bundle(repo_root, geo=args.geo, year=args.year, flow_code=flow_code)
    exio_slice = load_exio_slice(repo_root, region=args.region, year=args.year, flow_name=exio_flow)
    z_euro = parsed_iot.Z.astype("float64")
    z_exio = exio_slice.Z_domestic.astype("float64")

    cell_outliers = _build_cell_outliers(
        strict_pairs=strict_pairs,
        z_exio=z_exio,
        z_euro=z_euro,
        target_labels=target_labels,
        source_labels=source_labels,
        flow_code=flow_code,
        material_threshold=args.material_threshold,
    )
    row_outliers = _build_row_outliers(
        strict_pairs=strict_pairs,
        z_exio=z_exio,
        z_euro=z_euro,
        target_labels=target_labels,
        source_labels=source_labels,
        flow_code=flow_code,
    )

    cell_path = output_dir / f"at2020_{args.flow}_strict_zij_outliers.csv"
    row_path = output_dir / f"at2020_{args.flow}_strict_row_outliers.csv"
    write_csv(cell_outliers, cell_path)
    write_csv(row_outliers, row_path)

    print(f"Wrote strict z_ij outliers to {cell_path}")
    print(f"Wrote strict row outliers to {row_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ecolecon_mrio.mapping_csv import validate_mapping_weights

def _flatten_exio_list(exio_sectors: list[str] | dict[str, list[str]] | dict[str, float]) -> list[str]:
    if isinstance(exio_sectors, dict):
        if exio_sectors and any(isinstance(v, list) for v in exio_sectors.values()):
            out: list[str] = []
            for members in exio_sectors.values():
                if isinstance(members, list):
                    out.extend(list(map(str, members)))
            return out
        return list(map(str, exio_sectors.keys()))
    return list(map(str, exio_sectors))


def aggregate_intensity_by_mapping(
    exio_df: pd.DataFrame,
    sector_to_exio_sectors: dict[str, list[str]] | dict[str, dict[str, list[str]]] | dict[str, dict[str, float]],
    *,
    exio_sector_col: str = "exio_sector",
    sector_col: str = "aggregated_sector",
    group_cols: Iterable[str] = ("extension", "stressor", "unit"),
    value_col: str = "emission",
    output_col: str = "monetary_output",
    intensity_col: str = "emission_intensity",
    method_label: str = "ratio-of-sums",
    include_exio_list: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = list(group_cols)

    weighted_detected = False
    non_weighted_detected = False
    for v in sector_to_exio_sectors.values():
        if isinstance(v, dict):
            if v and any(isinstance(x, (list, dict)) for x in v.values()):
                non_weighted_detected = True
            else:
                weighted_detected = True
        else:
            non_weighted_detected = True
    if weighted_detected and non_weighted_detected:
        raise ValueError("Cannot mix weighted and unweighted mappings in aggregate_intensity_by_mapping.")
    if weighted_detected:
        validate_mapping_weights(sector_to_exio_sectors)  # strict

    for agg_sector, exio_sectors in sector_to_exio_sectors.items():
        weight_map: dict[str, float] | None = None
        if isinstance(exio_sectors, dict) and exio_sectors and all(
            not isinstance(v, (list, dict)) for v in exio_sectors.values()
        ):
            weight_map = {str(k): float(v) for k, v in exio_sectors.items()}
            exio_list = list(weight_map.keys())
        else:
            exio_list = _flatten_exio_list(exio_sectors)
            weight_map = {str(s): 1.0 for s in exio_list}

        subset = exio_df[exio_df[exio_sector_col].isin(exio_list)].copy()
        if subset.empty:
            continue
        subset["_weight"] = subset[exio_sector_col].map(weight_map).astype(float)
        grouped = subset.groupby(group_cols, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            total_value = float((group[value_col] * group["_weight"]).sum())
            total_out = float((group[output_col] * group["_weight"]).sum())
            if not pd.notna(total_value) or not pd.notna(total_out) or total_out == 0:
                continue
            row = {sector_col: str(agg_sector)}
            for col, key in zip(group_cols, keys):
                row[col] = key
            row[value_col] = total_value
            row[output_col] = total_out
            row[intensity_col] = total_value / total_out
            row["method"] = method_label
            if include_exio_list:
                unique_exio = sorted(set(map(str, exio_list)))
                row["exio_sectors"] = " | ".join(unique_exio)
                row["n_exio_sectors"] = len(unique_exio)
                if any(abs(weight_map.get(s, 1.0) - 1.0) > 1e-12 for s in unique_exio):
                    row["exio_weights"] = " | ".join(
                        f"{s}={weight_map.get(s, 1.0):.6g}" for s in unique_exio
                    )
            rows.append(row)

    return pd.DataFrame(rows)

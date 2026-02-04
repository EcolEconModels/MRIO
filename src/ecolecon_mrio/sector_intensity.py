from __future__ import annotations

from typing import Iterable

import pandas as pd


def _flatten_exio_list(exio_sectors: list[str] | dict[str, list[str]]) -> list[str]:
    if isinstance(exio_sectors, dict):
        out: list[str] = []
        for members in exio_sectors.values():
            out.extend(list(map(str, members)))
        return out
    return list(map(str, exio_sectors))


def aggregate_intensity_by_mapping(
    exio_df: pd.DataFrame,
    sector_to_exio_sectors: dict[str, list[str]] | dict[str, dict[str, list[str]]],
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

    for agg_sector, exio_sectors in sector_to_exio_sectors.items():
        exio_list = _flatten_exio_list(exio_sectors)
        subset = exio_df[exio_df[exio_sector_col].isin(exio_list)]
        if subset.empty:
            continue
        grouped = subset.groupby(group_cols, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            total_value = float(group[value_col].sum())
            total_out = float(group[output_col].sum())
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
            rows.append(row)

    return pd.DataFrame(rows)

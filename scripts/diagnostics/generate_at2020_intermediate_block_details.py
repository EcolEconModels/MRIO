#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _flat_diagnostics_common import (
    default_output_dir,
    eurostat_order_map,
    import_repo_module,
    load_eurostat_iot_bundle,
    load_exio_slice,
    load_weighted_mapping,
    repo_root_from_file,
    write_csv,
)


FLOW_SPECS = {
    "dom": (
        "DOM",
        "domestic",
        "at2020_dom_intermediate_cell_detail.csv",
        "at2020_dom_intermediate_compact_summary.csv",
        "at2020_dom_intermediate_split_trace.csv",
    ),
    "imp": (
        "IMP",
        "imports",
        "at2020_imp_intermediate_cell_detail.csv",
        "at2020_imp_intermediate_compact_summary.csv",
        "at2020_imp_intermediate_split_trace.csv",
    ),
    "total": (
        "TOTAL",
        "total",
        "at2020_total_intermediate_cell_detail.csv",
        "at2020_total_intermediate_compact_summary.csv",
        "at2020_total_intermediate_split_trace.csv",
    ),
}

COMPARISON_BASIS = (
    "Eurostat benchmark = Austria intermediate-use block Z only; "
    "EXIO benchmark = Austria-destination intermediate-use block Z for the selected flow; "
    "rows and columns are both mapped to Eurostat CPA sectors using the weighted concordance."
)

SUMMARY_NOTES = (
    "This compact file ranks the largest row-total, column-total, and cell-level intermediate-block gaps "
    "from the full matrix detail file for the same flow."
)

TRACE_NOTES = (
    "This trace file ranks the EXIO source-cell contributions behind the worst target-cell mismatches, "
    "focusing on source pairs where at least one EXIO sector is split across multiple Eurostat targets."
)

DEFAULT_SUMMARY_TOP_N = 15
DEFAULT_TRACE_CELL_COUNT = 12
DEFAULT_TRACE_CONTRIBUTORS_PER_CELL = 15


def _parse_boolish(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return True
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def _build_weight_matrix(
    mapping_df: pd.DataFrame,
    *,
    exio_index: pd.Index,
    target_index: pd.Index,
) -> pd.DataFrame:
    weights = mapping_df.copy()
    weights = weights[weights["enabled"].map(_parse_boolish)].copy()
    weights["aggregate_sector"] = weights["aggregate_sector"].astype(str)
    weights["exio_sector"] = weights["exio_sector"].astype(str)
    weights["weight"] = pd.to_numeric(weights["weight"], errors="raise")
    matrix = weights.pivot_table(
        index="exio_sector",
        columns="aggregate_sector",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix.reindex(index=exio_index.astype(str), columns=target_index.astype(str), fill_value=0.0)


def _enabled_mapping(mapping_df: pd.DataFrame) -> pd.DataFrame:
    enabled = mapping_df.copy()
    enabled = enabled[enabled["enabled"].map(_parse_boolish)].copy()
    enabled["aggregate_sector"] = enabled["aggregate_sector"].astype(str)
    enabled["exio_sector"] = enabled["exio_sector"].astype(str)
    enabled["weight"] = pd.to_numeric(enabled["weight"], errors="raise")
    return enabled


def _build_source_metadata(enabled_mapping: pd.DataFrame) -> pd.DataFrame:
    positive = enabled_mapping[enabled_mapping["weight"].abs() > 0].copy()
    if positive.empty:
        return pd.DataFrame(
            columns=[
                "source_label",
                "split_target_count",
                "is_split_source",
                "split_targets",
            ]
        )

    grouped = positive.groupby("exio_sector", sort=False)
    metadata = grouped.agg(
        source_label=("source_label", "first"),
        split_target_count=("aggregate_sector", "nunique"),
    )
    metadata["is_split_source"] = metadata["split_target_count"] > 1
    metadata["split_targets"] = grouped.apply(
        lambda group: "; ".join(
            f"{row.aggregate_sector}={row.weight:.6f}"
            for row in group.sort_values(["weight", "aggregate_sector"], ascending=[False, True]).itertuples(index=False)
        ),
        include_groups=False,
    )
    return metadata


def _map_square_matrix(
    source_matrix: pd.DataFrame,
    weight_matrix: pd.DataFrame,
    *,
    target_index: pd.Index,
) -> pd.DataFrame:
    source = source_matrix.copy().astype("float64")
    source.index = source.index.astype(str)
    source.columns = source.columns.astype(str)
    row_weights = weight_matrix.reindex(index=source.index, columns=target_index.astype(str), fill_value=0.0)
    col_weights = weight_matrix.reindex(index=source.columns, columns=target_index.astype(str), fill_value=0.0)
    mapped_values = row_weights.to_numpy(dtype="float64").T @ source.to_numpy(dtype="float64") @ col_weights.to_numpy(dtype="float64")
    return pd.DataFrame(mapped_values, index=target_index.astype(str), columns=target_index.astype(str))


def _rank_gap_block(
    block: pd.DataFrame,
    *,
    gap_column: str,
    top_n: int,
) -> pd.DataFrame:
    ranked = block.copy()
    ranked["abs_gap_meur_2020"] = ranked[gap_column].abs()
    total_abs_gap = float(ranked["abs_gap_meur_2020"].sum())
    ranked["abs_gap_pct_of_all_abs_gap_2020"] = np.where(
        total_abs_gap > 0,
        100.0 * ranked["abs_gap_meur_2020"] / total_abs_gap,
        np.nan,
    )
    ranked = ranked.sort_values(
        ["abs_gap_meur_2020", "eurostat_row_order", "eurostat_col_order"],
        ascending=[False, True, True],
        kind="stable",
    ).head(top_n)
    ranked["rank_by_abs_gap"] = np.arange(1, len(ranked) + 1, dtype="int64")
    return ranked


def _build_compact_summary(detail: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    row_block = detail[
        [
            "flow",
            "eurostat_row_order",
            "eurostat_row_sector_code",
            "eurostat_row_sector_name",
            "mapped_exio_row_total_meur_2020",
            "eurostat_row_total_meur_2020",
            "row_gap_meur_2020",
            "row_gap_pct_of_eurostat_2020",
            "comparison_basis",
            "notes",
        ]
    ].drop_duplicates(subset=["eurostat_row_sector_code"])
    row_block = row_block.assign(
        item_type="row_total",
        eurostat_col_order=np.nan,
        eurostat_col_sector_code="",
        eurostat_col_sector_name="",
        mapped_value_meur_2020=row_block["mapped_exio_row_total_meur_2020"],
        eurostat_value_meur_2020=row_block["eurostat_row_total_meur_2020"],
        gap_meur_2020=row_block["row_gap_meur_2020"],
        gap_pct_of_eurostat_2020=row_block["row_gap_pct_of_eurostat_2020"],
    )
    row_block = _rank_gap_block(row_block, gap_column="gap_meur_2020", top_n=top_n)

    col_block = detail[
        [
            "flow",
            "eurostat_col_order",
            "eurostat_col_sector_code",
            "eurostat_col_sector_name",
            "mapped_exio_col_total_meur_2020",
            "eurostat_col_total_meur_2020",
            "col_gap_meur_2020",
            "col_gap_pct_of_eurostat_2020",
            "comparison_basis",
            "notes",
        ]
    ].drop_duplicates(subset=["eurostat_col_sector_code"])
    col_block = col_block.assign(
        item_type="col_total",
        eurostat_row_order=np.nan,
        eurostat_row_sector_code="",
        eurostat_row_sector_name="",
        mapped_value_meur_2020=col_block["mapped_exio_col_total_meur_2020"],
        eurostat_value_meur_2020=col_block["eurostat_col_total_meur_2020"],
        gap_meur_2020=col_block["col_gap_meur_2020"],
        gap_pct_of_eurostat_2020=col_block["col_gap_pct_of_eurostat_2020"],
    )
    col_block = _rank_gap_block(col_block, gap_column="gap_meur_2020", top_n=top_n)

    cell_block = detail[
        [
            "flow",
            "eurostat_row_order",
            "eurostat_row_sector_code",
            "eurostat_row_sector_name",
            "eurostat_col_order",
            "eurostat_col_sector_code",
            "eurostat_col_sector_name",
            "mapped_exio_cell_meur_2020",
            "eurostat_cell_meur_2020",
            "cell_gap_meur_2020",
            "cell_gap_pct_of_eurostat_2020",
            "comparison_basis",
            "notes",
        ]
    ].copy()
    cell_block = cell_block.assign(
        item_type="cell",
        mapped_value_meur_2020=cell_block["mapped_exio_cell_meur_2020"],
        eurostat_value_meur_2020=cell_block["eurostat_cell_meur_2020"],
        gap_meur_2020=cell_block["cell_gap_meur_2020"],
        gap_pct_of_eurostat_2020=cell_block["cell_gap_pct_of_eurostat_2020"],
    )
    cell_block = _rank_gap_block(cell_block, gap_column="gap_meur_2020", top_n=top_n)

    summary = pd.concat([row_block, col_block, cell_block], ignore_index=True)
    summary["summary_notes"] = SUMMARY_NOTES
    return summary[
        [
            "flow",
            "item_type",
            "rank_by_abs_gap",
            "eurostat_row_order",
            "eurostat_row_sector_code",
            "eurostat_row_sector_name",
            "eurostat_col_order",
            "eurostat_col_sector_code",
            "eurostat_col_sector_name",
            "mapped_value_meur_2020",
            "eurostat_value_meur_2020",
            "gap_meur_2020",
            "gap_pct_of_eurostat_2020",
            "abs_gap_meur_2020",
            "abs_gap_pct_of_all_abs_gap_2020",
            "comparison_basis",
            "summary_notes",
        ]
    ].sort_values(["item_type", "rank_by_abs_gap"], kind="stable")


def _empty_trace_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "flow",
            "target_cell_rank_by_abs_gap",
            "eurostat_row_order",
            "eurostat_row_sector_code",
            "eurostat_row_sector_name",
            "eurostat_col_order",
            "eurostat_col_sector_code",
            "eurostat_col_sector_name",
            "mapped_target_cell_meur_2020",
            "eurostat_target_cell_meur_2020",
            "target_cell_gap_meur_2020",
            "target_cell_gap_pct_of_eurostat_2020",
            "trace_basis",
            "contribution_rank_within_target_cell",
            "exio_row_sector_code",
            "exio_row_sector_name",
            "row_weight_to_target",
            "row_split_target_count",
            "row_is_split_source",
            "row_split_targets",
            "exio_col_sector_code",
            "exio_col_sector_name",
            "col_weight_to_target",
            "col_split_target_count",
            "col_is_split_source",
            "col_split_targets",
            "exio_source_cell_meur_2020",
            "mapped_cell_contribution_meur_2020",
            "contribution_pct_of_mapped_target_cell_2020",
            "contribution_pct_of_target_gap_2020",
            "comparison_basis",
            "trace_notes",
        ]
    )


def _build_split_trace(
    detail: pd.DataFrame,
    *,
    exio_z: pd.DataFrame,
    weight_matrix: pd.DataFrame,
    source_metadata: pd.DataFrame,
    top_cell_count: int,
    contributors_per_cell: int,
) -> pd.DataFrame:
    top_cells = detail.copy()
    top_cells["abs_gap_meur_2020"] = top_cells["cell_gap_meur_2020"].abs()
    top_cells = top_cells.sort_values(
        ["abs_gap_meur_2020", "eurostat_row_order", "eurostat_col_order"],
        ascending=[False, True, True],
        kind="stable",
    ).head(top_cell_count)

    if top_cells.empty:
        return _empty_trace_frame()

    source_long = (
        exio_z.rename_axis(index="exio_row_sector_code", columns="exio_col_sector_code")
        .stack(future_stack=True)
        .rename("exio_source_cell_meur_2020")
        .reset_index()
    )

    row_labels = source_metadata["source_label"].to_dict() if "source_label" in source_metadata.columns else {}
    split_counts = source_metadata["split_target_count"].to_dict() if "split_target_count" in source_metadata.columns else {}
    split_flags = source_metadata["is_split_source"].to_dict() if "is_split_source" in source_metadata.columns else {}
    split_targets = source_metadata["split_targets"].to_dict() if "split_targets" in source_metadata.columns else {}

    trace_rows: list[pd.DataFrame] = []
    for target_rank, cell in enumerate(top_cells.itertuples(index=False), start=1):
        row_target = str(cell.eurostat_row_sector_code)
        col_target = str(cell.eurostat_col_sector_code)
        row_weights = weight_matrix.loc[:, row_target]
        col_weights = weight_matrix.loc[:, col_target]

        contribution_long = (
            exio_z.mul(row_weights, axis=0)
            .mul(col_weights, axis=1)
            .rename_axis(index="exio_row_sector_code", columns="exio_col_sector_code")
            .stack(future_stack=True)
            .rename("mapped_cell_contribution_meur_2020")
            .reset_index()
        )
        contribution_long = contribution_long.merge(
            source_long,
            on=["exio_row_sector_code", "exio_col_sector_code"],
            how="left",
        )
        contribution_long = contribution_long[contribution_long["mapped_cell_contribution_meur_2020"].abs() > 0].copy()

        if contribution_long.empty:
            empty_row = _empty_trace_frame().iloc[0:0].copy()
            empty_row = pd.DataFrame(
                [
                    {
                        "flow": cell.flow,
                        "target_cell_rank_by_abs_gap": target_rank,
                        "eurostat_row_order": cell.eurostat_row_order,
                        "eurostat_row_sector_code": row_target,
                        "eurostat_row_sector_name": cell.eurostat_row_sector_name,
                        "eurostat_col_order": cell.eurostat_col_order,
                        "eurostat_col_sector_code": col_target,
                        "eurostat_col_sector_name": cell.eurostat_col_sector_name,
                        "mapped_target_cell_meur_2020": cell.mapped_exio_cell_meur_2020,
                        "eurostat_target_cell_meur_2020": cell.eurostat_cell_meur_2020,
                        "target_cell_gap_meur_2020": cell.cell_gap_meur_2020,
                        "target_cell_gap_pct_of_eurostat_2020": cell.cell_gap_pct_of_eurostat_2020,
                        "trace_basis": "no_mapped_source_pair",
                        "contribution_rank_within_target_cell": np.nan,
                        "exio_row_sector_code": "",
                        "exio_row_sector_name": "",
                        "row_weight_to_target": np.nan,
                        "row_split_target_count": np.nan,
                        "row_is_split_source": np.nan,
                        "row_split_targets": "",
                        "exio_col_sector_code": "",
                        "exio_col_sector_name": "",
                        "col_weight_to_target": np.nan,
                        "col_split_target_count": np.nan,
                        "col_is_split_source": np.nan,
                        "col_split_targets": "",
                        "exio_source_cell_meur_2020": 0.0,
                        "mapped_cell_contribution_meur_2020": 0.0,
                        "contribution_pct_of_mapped_target_cell_2020": np.nan,
                        "contribution_pct_of_target_gap_2020": np.nan,
                        "comparison_basis": COMPARISON_BASIS,
                        "trace_notes": TRACE_NOTES,
                    }
                ]
            )
            trace_rows.append(empty_row)
            continue

        contribution_long["row_weight_to_target"] = contribution_long["exio_row_sector_code"].map(row_weights)
        contribution_long["col_weight_to_target"] = contribution_long["exio_col_sector_code"].map(col_weights)
        contribution_long["exio_row_sector_name"] = contribution_long["exio_row_sector_code"].map(row_labels)
        contribution_long["exio_col_sector_name"] = contribution_long["exio_col_sector_code"].map(row_labels)
        contribution_long["row_split_target_count"] = contribution_long["exio_row_sector_code"].map(split_counts).fillna(0).astype(int)
        contribution_long["col_split_target_count"] = contribution_long["exio_col_sector_code"].map(split_counts).fillna(0).astype(int)
        contribution_long["row_is_split_source"] = contribution_long["exio_row_sector_code"].map(split_flags).fillna(False)
        contribution_long["col_is_split_source"] = contribution_long["exio_col_sector_code"].map(split_flags).fillna(False)
        contribution_long["row_split_targets"] = contribution_long["exio_row_sector_code"].map(split_targets).fillna("")
        contribution_long["col_split_targets"] = contribution_long["exio_col_sector_code"].map(split_targets).fillna("")
        contribution_long["abs_contribution_meur_2020"] = contribution_long["mapped_cell_contribution_meur_2020"].abs()

        split_only = contribution_long[
            contribution_long["row_is_split_source"] | contribution_long["col_is_split_source"]
        ].copy()
        if split_only.empty:
            selected = contribution_long.copy()
            trace_basis = "fallback_any_source_pair"
        else:
            selected = split_only
            trace_basis = "split_source_pair"

        selected = selected.sort_values(
            ["abs_contribution_meur_2020", "exio_row_sector_code", "exio_col_sector_code"],
            ascending=[False, True, True],
            kind="stable",
        ).head(contributors_per_cell)
        selected["target_cell_rank_by_abs_gap"] = target_rank
        selected["trace_basis"] = trace_basis
        selected["contribution_rank_within_target_cell"] = np.arange(1, len(selected) + 1, dtype="int64")
        selected["flow"] = cell.flow
        selected["eurostat_row_order"] = cell.eurostat_row_order
        selected["eurostat_row_sector_code"] = row_target
        selected["eurostat_row_sector_name"] = cell.eurostat_row_sector_name
        selected["eurostat_col_order"] = cell.eurostat_col_order
        selected["eurostat_col_sector_code"] = col_target
        selected["eurostat_col_sector_name"] = cell.eurostat_col_sector_name
        selected["mapped_target_cell_meur_2020"] = cell.mapped_exio_cell_meur_2020
        selected["eurostat_target_cell_meur_2020"] = cell.eurostat_cell_meur_2020
        selected["target_cell_gap_meur_2020"] = cell.cell_gap_meur_2020
        selected["target_cell_gap_pct_of_eurostat_2020"] = cell.cell_gap_pct_of_eurostat_2020
        selected["contribution_pct_of_mapped_target_cell_2020"] = np.where(
            float(cell.mapped_exio_cell_meur_2020) != 0.0,
            100.0 * selected["mapped_cell_contribution_meur_2020"] / float(cell.mapped_exio_cell_meur_2020),
            np.nan,
        )
        selected["contribution_pct_of_target_gap_2020"] = np.where(
            float(cell.cell_gap_meur_2020) != 0.0,
            100.0 * selected["mapped_cell_contribution_meur_2020"] / float(cell.cell_gap_meur_2020),
            np.nan,
        )
        selected["comparison_basis"] = COMPARISON_BASIS
        selected["trace_notes"] = TRACE_NOTES
        trace_rows.append(selected)

    if not trace_rows:
        return _empty_trace_frame()

    trace = pd.concat(trace_rows, ignore_index=True)
    return trace[
        [
            "flow",
            "target_cell_rank_by_abs_gap",
            "eurostat_row_order",
            "eurostat_row_sector_code",
            "eurostat_row_sector_name",
            "eurostat_col_order",
            "eurostat_col_sector_code",
            "eurostat_col_sector_name",
            "mapped_target_cell_meur_2020",
            "eurostat_target_cell_meur_2020",
            "target_cell_gap_meur_2020",
            "target_cell_gap_pct_of_eurostat_2020",
            "trace_basis",
            "contribution_rank_within_target_cell",
            "exio_row_sector_code",
            "exio_row_sector_name",
            "row_weight_to_target",
            "row_split_target_count",
            "row_is_split_source",
            "row_split_targets",
            "exio_col_sector_code",
            "exio_col_sector_name",
            "col_weight_to_target",
            "col_split_target_count",
            "col_is_split_source",
            "col_split_targets",
            "exio_source_cell_meur_2020",
            "mapped_cell_contribution_meur_2020",
            "contribution_pct_of_mapped_target_cell_2020",
            "contribution_pct_of_target_gap_2020",
            "comparison_basis",
            "trace_notes",
        ]
    ].sort_values(
        ["target_cell_rank_by_abs_gap", "contribution_rank_within_target_cell"],
        kind="stable",
    )


def generate_flow_outputs(
    repo_root: Path,
    *,
    geo: str,
    region: str,
    year: int,
    flow_key: str,
    summary_top_n: int,
    trace_cell_count: int,
    trace_contributors_per_cell: int,
) -> dict[str, pd.DataFrame]:
    flow_code, exio_flow, _, _, _ = FLOW_SPECS[flow_key]
    mapping_df = load_weighted_mapping(repo_root)
    enabled_mapping = _enabled_mapping(mapping_df)
    concordance = import_repo_module(repo_root, "ecolecon_mrio.diagnostic_concordance")
    concordance.validate_weighted_mapping(mapping_df)

    parsed_iot, _eurostat_output, target_labels = load_eurostat_iot_bundle(repo_root, geo=geo, year=year, flow_code=flow_code)
    eurostat_z = parsed_iot.Z.copy().astype("float64")
    eurostat_z.index = eurostat_z.index.astype(str)
    eurostat_z.columns = eurostat_z.columns.astype(str)
    target_index = pd.Index(eurostat_z.index.astype(str))
    order_lookup = eurostat_order_map(target_index)

    exio_slice = load_exio_slice(repo_root, region=region, year=year, flow_name=exio_flow)
    exio_z = exio_slice.Z_domestic.copy().astype("float64")
    exio_z.index = exio_z.index.astype(str)
    exio_z.columns = exio_z.columns.astype(str)

    exio_index = pd.Index(sorted(set(exio_z.index.astype(str)).union(set(exio_z.columns.astype(str)))))
    weight_matrix = _build_weight_matrix(enabled_mapping, exio_index=exio_index, target_index=target_index)
    mapped_z = _map_square_matrix(exio_z, weight_matrix, target_index=target_index)

    mapped_row_totals = mapped_z.sum(axis=1)
    eurostat_row_totals = eurostat_z.sum(axis=1)
    mapped_col_totals = mapped_z.sum(axis=0)
    eurostat_col_totals = eurostat_z.sum(axis=0)

    mapped_total = float(mapped_z.to_numpy(dtype="float64").sum())
    eurostat_total = float(eurostat_z.to_numpy(dtype="float64").sum())
    total_gap = mapped_total - eurostat_total

    mapped_long = (
        mapped_z.rename_axis(index="eurostat_row_sector_code", columns="eurostat_col_sector_code")
        .stack(future_stack=True)
        .rename("mapped_exio_cell_meur_2020")
        .reset_index()
    )
    eurostat_long = (
        eurostat_z.rename_axis(index="eurostat_row_sector_code", columns="eurostat_col_sector_code")
        .stack(future_stack=True)
        .rename("eurostat_cell_meur_2020")
        .reset_index()
    )
    detail = mapped_long.merge(
        eurostat_long,
        on=["eurostat_row_sector_code", "eurostat_col_sector_code"],
        how="inner",
    )

    detail["flow"] = flow_code
    detail["eurostat_row_order"] = detail["eurostat_row_sector_code"].map(order_lookup)
    detail["eurostat_col_order"] = detail["eurostat_col_sector_code"].map(order_lookup)
    detail["eurostat_row_sector_name"] = detail["eurostat_row_sector_code"].map(target_labels)
    detail["eurostat_col_sector_name"] = detail["eurostat_col_sector_code"].map(target_labels)
    detail["cell_gap_meur_2020"] = detail["mapped_exio_cell_meur_2020"] - detail["eurostat_cell_meur_2020"]
    detail["cell_gap_pct_of_eurostat_2020"] = np.where(
        detail["eurostat_cell_meur_2020"].abs() > 0,
        100.0 * detail["cell_gap_meur_2020"] / detail["eurostat_cell_meur_2020"],
        np.nan,
    )
    detail["mapped_exio_row_total_meur_2020"] = detail["eurostat_row_sector_code"].map(mapped_row_totals)
    detail["eurostat_row_total_meur_2020"] = detail["eurostat_row_sector_code"].map(eurostat_row_totals)
    detail["row_gap_meur_2020"] = detail["mapped_exio_row_total_meur_2020"] - detail["eurostat_row_total_meur_2020"]
    detail["row_gap_pct_of_eurostat_2020"] = np.where(
        detail["eurostat_row_total_meur_2020"].abs() > 0,
        100.0 * detail["row_gap_meur_2020"] / detail["eurostat_row_total_meur_2020"],
        np.nan,
    )
    detail["mapped_exio_col_total_meur_2020"] = detail["eurostat_col_sector_code"].map(mapped_col_totals)
    detail["eurostat_col_total_meur_2020"] = detail["eurostat_col_sector_code"].map(eurostat_col_totals)
    detail["col_gap_meur_2020"] = detail["mapped_exio_col_total_meur_2020"] - detail["eurostat_col_total_meur_2020"]
    detail["col_gap_pct_of_eurostat_2020"] = np.where(
        detail["eurostat_col_total_meur_2020"].abs() > 0,
        100.0 * detail["col_gap_meur_2020"] / detail["eurostat_col_total_meur_2020"],
        np.nan,
    )
    detail["mapped_exio_intermediate_total_meur_2020"] = mapped_total
    detail["eurostat_intermediate_total_meur_2020"] = eurostat_total
    detail["intermediate_total_gap_meur_2020"] = total_gap
    detail["intermediate_total_gap_pct_of_eurostat_2020"] = (100.0 * total_gap / eurostat_total) if eurostat_total else np.nan
    detail["comparison_basis"] = COMPARISON_BASIS
    detail["notes"] = (
        "Final-demand columns are excluded from this matrix diagnostic. "
        "Use the mapping-detail CSVs for the broader Z plus final-demand absorption comparison."
    )

    detail = detail[
        [
            "flow",
            "eurostat_row_order",
            "eurostat_row_sector_code",
            "eurostat_row_sector_name",
            "eurostat_col_order",
            "eurostat_col_sector_code",
            "eurostat_col_sector_name",
            "mapped_exio_cell_meur_2020",
            "eurostat_cell_meur_2020",
            "cell_gap_meur_2020",
            "cell_gap_pct_of_eurostat_2020",
            "mapped_exio_row_total_meur_2020",
            "eurostat_row_total_meur_2020",
            "row_gap_meur_2020",
            "row_gap_pct_of_eurostat_2020",
            "mapped_exio_col_total_meur_2020",
            "eurostat_col_total_meur_2020",
            "col_gap_meur_2020",
            "col_gap_pct_of_eurostat_2020",
            "mapped_exio_intermediate_total_meur_2020",
            "eurostat_intermediate_total_meur_2020",
            "intermediate_total_gap_meur_2020",
            "intermediate_total_gap_pct_of_eurostat_2020",
            "comparison_basis",
            "notes",
        ]
    ].sort_values(["eurostat_row_order", "eurostat_col_order"], kind="stable")

    summary = _build_compact_summary(detail, top_n=summary_top_n)
    source_metadata = _build_source_metadata(enabled_mapping)
    trace = _build_split_trace(
        detail,
        exio_z=exio_z,
        weight_matrix=weight_matrix,
        source_metadata=source_metadata,
        top_cell_count=trace_cell_count,
        contributors_per_cell=trace_contributors_per_cell,
    )
    return {"detail": detail, "summary": summary, "trace": trace}


def generate_flow_detail(repo_root: Path, *, geo: str, region: str, year: int, flow_key: str) -> pd.DataFrame:
    outputs = generate_flow_outputs(
        repo_root,
        geo=geo,
        region=region,
        year=year,
        flow_key=flow_key,
        summary_top_n=DEFAULT_SUMMARY_TOP_N,
        trace_cell_count=DEFAULT_TRACE_CELL_COUNT,
        trace_contributors_per_cell=DEFAULT_TRACE_CONTRIBUTORS_PER_CELL,
    )
    return outputs["detail"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Austria 2020 intermediate-block cell comparison CSVs.")
    parser.add_argument("--geo", default="AT")
    parser.add_argument("--region", default="AT")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-top-n", type=int, default=DEFAULT_SUMMARY_TOP_N)
    parser.add_argument("--trace-cell-count", type=int, default=DEFAULT_TRACE_CELL_COUNT)
    parser.add_argument("--trace-contributors-per-cell", type=int, default=DEFAULT_TRACE_CONTRIBUTORS_PER_CELL)
    args = parser.parse_args()

    repo_root = repo_root_from_file(__file__)
    output_dir = args.output_dir or default_output_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for flow_key, (_, _, detail_filename, summary_filename, trace_filename) in FLOW_SPECS.items():
        outputs = generate_flow_outputs(
            repo_root,
            geo=args.geo,
            region=args.region,
            year=args.year,
            flow_key=flow_key,
            summary_top_n=args.summary_top_n,
            trace_cell_count=args.trace_cell_count,
            trace_contributors_per_cell=args.trace_contributors_per_cell,
        )
        write_csv(outputs["detail"], output_dir / detail_filename)
        write_csv(outputs["summary"], output_dir / summary_filename)
        write_csv(outputs["trace"], output_dir / trace_filename)

    print(f"Wrote intermediate block details, summaries, and traces to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
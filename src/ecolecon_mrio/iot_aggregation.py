# pyright: reportAssignmentType=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_flow_semantics import refresh_supplementary_rows, technical_coefficients


@dataclass(frozen=True)
class AggregationResult:
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    V: pd.DataFrame | None = None
    M: pd.DataFrame | None = None
    aggregated_sector_value_per_physical_input: pd.DataFrame | None = None
    aggregated_sector_emission_intensity: pd.DataFrame | None = None
    io_system: object | None = None  # pymrio.IOSystem when available

def _map_index_values(
    idx: pd.Index,
    mapping: Mapping[str, str],
    *,
    keep_unmapped: bool,
) -> tuple[pd.Index, np.ndarray]:
    s = pd.Series(idx.astype(str), index=idx)
    mapped = s.map(lambda v: mapping.get(v, v) if keep_unmapped else mapping.get(v, None))
    keep = mapped.notna().to_numpy()
    return pd.Index(mapped[keep].astype(str)), keep


def _map_multiindex(
    idx: pd.MultiIndex,
    mapping: Mapping[str, str],
    *,
    level: int,
    keep_unmapped: bool,
) -> tuple[pd.MultiIndex, np.ndarray]:
    df = idx.to_frame(index=False)
    level_idx = level if level >= 0 else df.shape[1] + level
    col = df.columns[level_idx]
    before = df[col].astype(str)
    after = before.map(lambda v: mapping.get(v, v) if keep_unmapped else mapping.get(v, None))
    keep = after.notna().to_numpy()
    df = df.loc[keep].copy()
    df[col] = after.loc[keep].astype(str).to_numpy()
    out = pd.MultiIndex.from_frame(df, names=idx.names)
    return out, keep


def aggregate_square_matrix(
    Z: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    level: int = -1,
    keep_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Aggregate a square matrix by summing rows+cols according to a mapping.

    Works for simple Index or MultiIndex (mapping applied to the given `level`).
    """
    Z = Z.copy()
    if not Z.index.equals(Z.columns):
        raise ValueError("Z must be square with matching index/columns.")

    if isinstance(Z.index, pd.MultiIndex):
        new_index, keep_rows = _map_multiindex(Z.index, mapping, level=level, keep_unmapped=keep_unmapped)
        Z = Z.loc[keep_rows, keep_rows]
        Z.index = new_index
        Z.columns = new_index
        Z = Z.groupby(level=list(range(Z.index.nlevels))).sum()
        Z = Z.T.groupby(level=list(range(Z.columns.nlevels))).sum().T
        return Z

    new_index, keep_rows = _map_index_values(Z.index, mapping, keep_unmapped=keep_unmapped)
    Z = Z.loc[keep_rows, keep_rows]
    Z.index = new_index
    Z.columns = new_index
    Z = Z.groupby(level=0).sum()
    Z = Z.T.groupby(level=0).sum().T
    return Z


def aggregate_Y(
    Y: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    level: int = -1,
    keep_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Aggregate an IO final demand matrix by summing sector rows according to a mapping.
    """
    Y = Y.copy()
    if isinstance(Y.index, pd.MultiIndex):
        new_index, keep_rows = _map_multiindex(Y.index, mapping, level=level, keep_unmapped=keep_unmapped)
        Y = Y.loc[keep_rows, :]
        Y.index = new_index
        Y = Y.groupby(level=list(range(Y.index.nlevels))).sum()
        return Y

    new_index, keep_rows = _map_index_values(Y.index, mapping, keep_unmapped=keep_unmapped)
    Y = Y.loc[keep_rows, :]
    Y.index = new_index
    Y = Y.groupby(level=0).sum()
    return Y


def aggregate_V(
    V: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    keep_unmapped: bool = True,
) -> pd.DataFrame:
    """Aggregate supplementary/lower IO rows over product columns."""
    V = V.copy()
    new_columns, keep_cols = _map_index_values(pd.Index(V.columns.astype(str)), mapping, keep_unmapped=keep_unmapped)
    V = V.loc[:, keep_cols]
    V.columns = new_columns
    return V.T.groupby(level=0).sum().T


def _compute_aggregated_sector_ratio_of_sums(
    values: pd.DataFrame,
    aggregation_mapping: Mapping[str, str],
    *,
    numerator_col: str,
    denominator_col: str,
    out_ratio_col: str,
    out_total_numerator_col: str | None = None,
    out_total_denominator_col: str | None = None,
) -> pd.DataFrame | None:
    """Aggregate a disaggregated metric to aggregated sectors via ratio-of-sums."""
    values = values.copy()
    values["aggregated_sector"] = values["disaggregated_sector"].map(aggregation_mapping)
    values = values[values["aggregated_sector"].notna()].copy()
    if values.empty:
        return None

    values[numerator_col] = pd.to_numeric(values[numerator_col], errors="coerce")
    values[denominator_col] = pd.to_numeric(values[denominator_col], errors="coerce")

    rows: list[dict] = []
    grouped = values.groupby(
        ["aggregated_sector", "extension", "stressor", "unit"],
        dropna=False,
    )
    for (agg_sector, ext, stressor, unit), group in grouped:
        total_den = float(group[denominator_col].sum())
        if not np.isfinite(total_den) or total_den == 0:
            continue
        total_num = float(group[numerator_col].sum())
        if not np.isfinite(total_num):
            continue

        row = {
            "aggregated_sector": agg_sector,
            "extension": ext,
            "stressor": stressor,
            "unit": unit,
            out_ratio_col: total_num / total_den,
            "n_components": int(group["disaggregated_sector"].nunique()),
        }
        if out_total_numerator_col is not None:
            row[out_total_numerator_col] = total_num
        if out_total_denominator_col is not None:
            row[out_total_denominator_col] = total_den
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_aggregated_sector_value_per_physical_input(
    disaggregated_sector_value_per_physical_input: pd.DataFrame | None,
    aggregation_mapping: Mapping[str, str],
) -> pd.DataFrame | None:
    """
    Compute value-per-physical-input for aggregated sectors from disaggregated data.

    For each aggregated sector, computes value per physical input as:
        value_per_physical_input = sum(monetary_output) / sum(quantity)

    Args:
        disaggregated_sector_value_per_physical_input: DataFrame with columns:
            disaggregated_sector, extension, stressor, unit, quantity, monetary_output, value_per_physical_input
        aggregation_mapping: Mapping of disaggregated_sector -> aggregated_sector

    Returns:
        DataFrame with columns: aggregated_sector, extension, stressor, unit, value_per_physical_input, n_components
        or None if inputs are insufficient
    """
    if (
        disaggregated_sector_value_per_physical_input is None
        or disaggregated_sector_value_per_physical_input.empty
        or not aggregation_mapping
    ):
        return None

    required_cols = {"disaggregated_sector", "extension", "stressor", "unit", "quantity", "monetary_output"}
    missing_cols = required_cols - set(disaggregated_sector_value_per_physical_input.columns)
    if missing_cols:
        raise ValueError(
            "disaggregated_sector_value_per_physical_input missing required columns: "
            f"{sorted(missing_cols)!r}"
        )

    return _compute_aggregated_sector_ratio_of_sums(
        disaggregated_sector_value_per_physical_input,
        aggregation_mapping,
        numerator_col="monetary_output",
        denominator_col="quantity",
        out_ratio_col="value_per_physical_input",
    )

def compute_aggregated_sector_emission_intensity(
    disaggregated_sector_emission_intensity: pd.DataFrame | None,
    aggregation_mapping: Mapping[str, str],
) -> pd.DataFrame | None:
    """Compute emissions intensity for aggregated sectors from disaggregated data.

    For each aggregated sector, computes direct emissions intensity as:
        emission_intensity = sum(emission) / sum(monetary_output)

    Args:
        disaggregated_sector_emission_intensity: DataFrame with columns:
            disaggregated_sector, extension, stressor, unit, emission, monetary_output, emission_intensity
        aggregation_mapping: Mapping of disaggregated_sector -> aggregated_sector

    Returns:
        DataFrame with columns: aggregated_sector, extension, stressor, unit,
        emission_intensity, n_components; or None if inputs are insufficient
    """
    if (
        disaggregated_sector_emission_intensity is None
        or disaggregated_sector_emission_intensity.empty
        or not aggregation_mapping
    ):
        return None

    required_cols = {"disaggregated_sector", "extension", "stressor", "unit", "emission", "monetary_output"}
    missing_cols = required_cols - set(disaggregated_sector_emission_intensity.columns)
    if missing_cols:
        raise ValueError(
            "disaggregated_sector_emission_intensity missing required columns: "
            f"{sorted(missing_cols)!r}"
        )

    return _compute_aggregated_sector_ratio_of_sums(
        disaggregated_sector_emission_intensity,
        aggregation_mapping,
        numerator_col="emission",
        denominator_col="monetary_output",
        out_ratio_col="emission_intensity",
        out_total_numerator_col="emission",
        out_total_denominator_col="monetary_output",
    )


def aggregate_iot(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    V: pd.DataFrame | None = None,
    M: pd.DataFrame | None = None,
    output: pd.Series | None = None,
    flow_role: str = "TOTAL",
    level: int = -1,
    keep_unmapped: bool = True,
    create_io_system: bool = True,
    compute_sector_value_per_physical_input: bool = False,
    disaggregated_sector_value_per_physical_input: pd.DataFrame | None = None,
    compute_sector_emission_intensity: bool = False,
    disaggregated_sector_emission_intensity: pd.DataFrame | None = None,
) -> AggregationResult:
    """Aggregate one flow using an explicit, already validated output vector."""
    if not Z.index.equals(Z.columns):
        raise ValueError("Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Y must have the same index as Z.")
    if output is None:
        raise ValueError("aggregate_iot requires the canonical domestic output vector; implicit Z+Y output was removed.")
    role = str(flow_role).upper()
    if role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unknown flow_role={flow_role!r}")
    Z_agg = aggregate_square_matrix(Z, mapping, level=level, keep_unmapped=keep_unmapped)
    Y_agg = aggregate_Y(Y, mapping, level=level, keep_unmapped=keep_unmapped)
    if not Y_agg.index.equals(Z_agg.index):
        raise RuntimeError("Aggregation produced inconsistent Z/Y indices.")
    output_df = output.astype("float64").to_frame("x")
    output_agg_df = aggregate_Y(output_df, mapping, level=level, keep_unmapped=keep_unmapped)
    x = output_agg_df["x"].reindex(Z_agg.index).fillna(0.0).astype("float64")
    x.name = "x"
    V_agg = aggregate_V(V, mapping, keep_unmapped=keep_unmapped) if V is not None else None
    V_agg, _ = refresh_supplementary_rows(V_agg, None, Z=Z_agg, Y=Y_agg, output=x, flow_role=role)  # type: ignore[arg-type]
    M_agg = None if M is None else M.copy()
    A = technical_coefficients(Z_agg, x)
    io_system = None
    if create_io_system:
        try:
            import pymrio  # type: ignore
            io_system = pymrio.IOSystem(Z=Z_agg, Y=Y_agg, x=x)
            io_system.calc_system()
        except Exception as exc:
            raise RuntimeError("pymrio aggregation construction failed") from exc
    agg_sector_values = None
    if compute_sector_value_per_physical_input and disaggregated_sector_value_per_physical_input is not None:
        agg_sector_values = compute_aggregated_sector_value_per_physical_input(
            disaggregated_sector_value_per_physical_input=disaggregated_sector_value_per_physical_input,
            aggregation_mapping=mapping,
        )
    agg_sector_emission_intensity = None
    if compute_sector_emission_intensity:
        agg_sector_emission_intensity = compute_aggregated_sector_emission_intensity(
            disaggregated_sector_emission_intensity, mapping
        )
    return AggregationResult(
        Z=Z_agg, Y=Y_agg, x=x, A=A, V=V_agg, M=M_agg,
        aggregated_sector_value_per_physical_input=agg_sector_values,
        aggregated_sector_emission_intensity=agg_sector_emission_intensity,
        io_system=io_system,
    )

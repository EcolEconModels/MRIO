from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_flow_semantics import (
    _validated_nonnegative_output,
    flow_use_total,
    technical_coefficients,
)


@dataclass(frozen=True)
class AggregationResult:
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    V: pd.DataFrame | None = None
    M: pd.DataFrame | None = None
    aggregated_sector_value_per_physical_input: pd.DataFrame | None = None
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

    try:
        values[numerator_col] = pd.to_numeric(values[numerator_col], errors="raise")
        values[denominator_col] = pd.to_numeric(values[denominator_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Aggregated ratio components must be numeric.") from exc
    if not np.isfinite(values[numerator_col].to_numpy(dtype=float)).all() or not np.isfinite(values[denominator_col].to_numpy(dtype=float)).all():
        raise ValueError("Aggregated ratio components must be finite.")

    rows: list[dict] = []
    grouped = values.groupby(
        ["aggregated_sector", "extension", "stressor", "unit"],
        dropna=False,
    )
    for (agg_sector, ext, stressor, unit), group in grouped:
        total_den = float(group[denominator_col].sum())
        if total_den == 0:
            continue
        total_num = float(group[numerator_col].sum())

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

def aggregate_environmental_extensions(
    environmental_extensions: pd.DataFrame | None,
    mapping: Mapping[str, str],
    *,
    output: pd.Series,
    keep_unmapped: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> pd.DataFrame | None:
    """Reaggregate boundary-aware target environmental totals by ratio of sums.

    ``output`` is the current flow's authoritative monetary denominator: DOM
    output/use for ``scope_production``, IMP supply/use for
    ``external_imports_direct``, or TOTAL supply for ``total_supply_direct``.
    The historical argument name is retained because this function is internal to
    the current pipeline, but its meaning is validated by ``denominator_role``.
    """
    if environmental_extensions is None or environmental_extensions.empty:
        return None
    required = {
        "sector", "boundary", "denominator_role", "extension", "stressor",
        "unit", "value", "monetary_output",
    }
    environmental_extensions = environmental_extensions.copy()
    missing = required - set(environmental_extensions.columns)
    if missing:
        raise ValueError(
            "environmental_extensions missing required columns: "
            f"{sorted(missing)!r}"
        )
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("Environmental extension tolerances must be non-negative.")
    frame = environmental_extensions.copy()
    key_cols = [
        "sector", "boundary", "denominator_role", "extension", "stressor", "unit"
    ]
    if frame.duplicated(key_cols).any():
        duplicate_rows = frame.loc[frame.duplicated(key_cols, keep=False), key_cols]
        raise ValueError(
            "environmental_extensions must contain one row per sector/boundary/stressor; "
            f"duplicates={duplicate_rows.head(10).to_dict(orient='records')!r}"
        )
    for column in ("sector", "boundary", "denominator_role", "extension", "stressor", "unit"):
        frame[column] = frame[column].astype(str)
    frame["value"] = np.asarray(pd.to_numeric(frame["value"], errors="raise"), dtype="float64")
    frame["monetary_output"] = np.asarray(
        pd.to_numeric(frame["monetary_output"], errors="raise"), dtype="float64"
    )
    if not np.isfinite(frame["value"].to_numpy(dtype="float64")).all():
        raise ValueError("Environmental extension values must be finite.")

    source_output = output.copy().astype("float64")
    source_output.index = source_output.index.astype(str)
    if not source_output.index.is_unique:
        raise ValueError("Environmental denominator vector index must be unique.")
    if source_output.isna().any() or not np.isfinite(source_output.to_numpy()).all() or bool((source_output < 0.0).any()):
        raise ValueError("Environmental denominator vector must be finite and non-negative.")
    positive_sources_by_target: dict[str, set[str]] = {}
    for source, output_value in source_output.items():
        source_label = str(source)
        if float(output_value) <= 0.0:
            continue
        target = str(mapping.get(source_label, source_label)) if keep_unmapped else mapping.get(source_label)
        if target is not None:
            positive_sources_by_target.setdefault(str(target), set()).add(source_label)
    group_cols = ["boundary", "denominator_role", "extension", "stressor", "unit"]
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        covered = set(group["sector"].astype(str))
        represented_targets = {
            str(mapping.get(source, source)) if keep_unmapped else str(mapping[source])
            for source in covered
            if keep_unmapped or source in mapping
        }
        for target in sorted(represented_targets):
            missing_positive = sorted(positive_sources_by_target.get(target, set()) - covered)
            if missing_positive:
                raise ValueError(
                    "Positive-output environmental extension coverage is incomplete before reaggregation: "
                    f"group={keys!r}, target={target!r}, sectors={missing_positive[:20]!r}."
                )

    observed = frame["monetary_output"].to_numpy(dtype="float64")
    expected_series = frame["sector"].map(source_output)
    if expected_series.isna().any():
        missing_sectors = sorted(frame.loc[expected_series.isna(), "sector"].unique())
        raise ValueError(f"Environmental sectors are absent from the flow denominator: {missing_sectors!r}.")
    expected = expected_series.to_numpy(dtype="float64")
    if not np.allclose(observed, expected, rtol=float(rtol), atol=float(atol), equal_nan=False):
        raise ValueError(
            "Environmental monetary denominator disagrees with the current IOT flow use/supply."
        )

    if keep_unmapped:
        frame["sector"] = frame["sector"].map(
            lambda sector: str(mapping.get(str(sector), str(sector)))
        )
    else:
        frame["sector"] = frame["sector"].map(mapping)
        frame = frame[frame["sector"].notna()].copy()
        frame["sector"] = frame["sector"].astype(str)
    if frame.empty:
        return None
    grouped = (
        frame.groupby(
            ["sector", "boundary", "denominator_role", "extension", "stressor", "unit"],
            dropna=False,
            sort=False,
        )[["value", "monetary_output"]]
        .sum()
        .reset_index()
    )
    grouped["intensity"] = grouped["value"] / grouped["monetary_output"].replace(0.0, np.nan)
    return grouped

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
) -> AggregationResult:
    """Aggregate one flow using an exact, validated source output vector."""
    if not Z.index.equals(Z.columns):
        raise ValueError("Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Y must have the same index as Z.")
    if output is None:
        raise ValueError(
            "aggregate_iot requires the canonical domestic output vector; implicit Z+Y output was removed."
        )
    output_values = _validated_nonnegative_output(
        output,
        expected_index=Z.index,
        context="aggregate_iot output",
    )
    source_output = pd.Series(output_values, index=Z.index, name="x", dtype="float64")
    role = str(flow_role).upper()
    if role not in {"DOM", "IMP", "TOTAL"}:
        raise ValueError(f"Unknown flow_role={flow_role!r}")
    Z_agg = aggregate_square_matrix(Z, mapping, level=level, keep_unmapped=keep_unmapped)
    Y_agg = aggregate_Y(Y, mapping, level=level, keep_unmapped=keep_unmapped)
    if not Y_agg.index.equals(Z_agg.index):
        raise RuntimeError("Aggregation produced inconsistent Z/Y indices.")
    output_agg_df = aggregate_Y(
        source_output.to_frame("x"),
        mapping,
        level=level,
        keep_unmapped=keep_unmapped,
    )
    if not output_agg_df.index.equals(Z_agg.index):
        raise RuntimeError("Aggregation produced an output axis inconsistent with Z/Y.")
    x = cast(pd.Series, output_agg_df["x"].astype("float64")).rename("x")
    V_agg = aggregate_V(V, mapping, keep_unmapped=keep_unmapped) if V is not None else None
    M_agg = None if M is None else M.copy()
    A = technical_coefficients(Z_agg, x)
    io_system = None
    if create_io_system:
        if role != "DOM":
            raise ValueError(
                f"{role} flow is a coefficient decomposition sharing domestic output, "
                "not a standalone balanced Leontief IO system."
            )
        dom_use = flow_use_total(Z_agg, Y_agg)
        residual = x.sub(dom_use, fill_value=0.0)
        max_gap = float(residual.abs().max()) if len(residual) else 0.0
        scale = max(
            float(x.abs().max()) if len(x) else 0.0,
            float(dom_use.abs().max()) if len(dom_use) else 0.0,
        )
        if max_gap > max(1e-6, 1e-9 * scale):
            raise ValueError(
                "DOM analytical IO system is not balanced: "
                f"max_abs_x_minus_use={max_gap:.12g}"
            )
        try:
            import pymrio  # type: ignore

            io_system = pymrio.IOSystem(Z=Z_agg, Y=Y_agg, x=x)
            io_system.calc_system()
        except Exception as exc:
            raise RuntimeError("pymrio aggregation construction failed") from exc
    agg_sector_values = None
    if (
        compute_sector_value_per_physical_input
        and disaggregated_sector_value_per_physical_input is not None
    ):
        agg_sector_values = compute_aggregated_sector_value_per_physical_input(
            disaggregated_sector_value_per_physical_input=disaggregated_sector_value_per_physical_input,
            aggregation_mapping=mapping,
        )
    return AggregationResult(
        Z=Z_agg,
        Y=Y_agg,
        x=x,
        A=A,
        V=V_agg,
        M=M_agg,
        aggregated_sector_value_per_physical_input=agg_sector_values,
        io_system=io_system,
    )

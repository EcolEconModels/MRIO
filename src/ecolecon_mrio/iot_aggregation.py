# pyright: reportAssignmentType=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_flow_semantics import flow_use_total, technical_coefficients


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

def aggregate_environmental_extensions(
    environmental_extensions: pd.DataFrame | None,
    mapping: Mapping[str, str],
    *,
    output: pd.Series,
    keep_unmapped: bool = True,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> pd.DataFrame | None:
    """Reaggregate physical totals against the canonical monetary output vector.

    ``value`` is extensive and additive. ``monetary_output`` in the input
    sidecar is an integrity copy of the current IOT output, not an independent
    denominator: it must match ``output`` sector by sector. Reaggregation sums
    physical values, independently aggregates canonical ``output``, and derives
    intensity only afterwards.
    """
    if environmental_extensions is None or environmental_extensions.empty:
        return None
    required = {
        "sector",
        "extension",
        "stressor",
        "unit",
        "value",
        "monetary_output",
    }
    missing = required - set(environmental_extensions.columns)
    if missing:
        raise ValueError(
            "environmental_extensions missing required columns: "
            f"{sorted(missing)!r}"
        )
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("Environmental extension tolerances must be non-negative.")

    frame = environmental_extensions.copy()
    key_cols = ["sector", "extension", "stressor", "unit"]
    if frame.duplicated(key_cols).any():
        duplicate_rows = frame.loc[frame.duplicated(key_cols, keep=False), key_cols]
        raise ValueError(
            "environmental_extensions must contain one row per sector/stressor; "
            f"duplicates={duplicate_rows.head(10).to_dict(orient='records')!r}"
        )
    frame["sector"] = frame["sector"].astype(str)
    value = cast(pd.Series, pd.to_numeric(frame["value"], errors="raise"))
    monetary_output = cast(
        pd.Series, pd.to_numeric(frame["monetary_output"], errors="raise")
    )
    frame["value"] = value.astype("float64")
    frame["monetary_output"] = monetary_output.astype("float64")
    if not np.isfinite(frame["value"].to_numpy()).all():
        raise ValueError("Environmental extension values must be finite.")

    source_output = output.copy().astype("float64")
    source_output.index = source_output.index.astype(str)
    if not source_output.index.is_unique:
        raise ValueError("Environmental extension output vector index must be unique.")
    if (
        source_output.isna().any()
        or not np.isfinite(source_output.to_numpy()).all()
        or bool((source_output < 0.0).any())
    ):
        raise ValueError("Environmental extension output vector must be finite and non-negative.")
    sidecar_output = frame["sector"].map(source_output)
    if sidecar_output.isna().any():
        missing_sectors = sorted(
            frame.loc[sidecar_output.isna(), "sector"].astype(str).unique().tolist()
        )
        raise ValueError(
            "Environmental extension sectors are absent from canonical output: "
            f"{missing_sectors!r}."
        )
    observed_output = frame["monetary_output"].to_numpy(dtype="float64")
    expected_output = sidecar_output.to_numpy(dtype="float64")
    if not np.allclose(
        observed_output,
        expected_output,
        rtol=float(rtol),
        atol=float(atol),
        equal_nan=False,
    ):
        max_gap = float(np.max(np.abs(observed_output - expected_output)))
        raise ValueError(
            "Environmental extension monetary_output disagrees with canonical IOT output: "
            f"max_abs_gap={max_gap:.12g}."
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
            ["sector", "extension", "stressor", "unit"],
            dropna=False,
            sort=False,
        )[["value"]]
        .sum()
        .reset_index()
    )
    output_frame = aggregate_Y(
        source_output.to_frame("monetary_output"),
        mapping,
        keep_unmapped=keep_unmapped,
    )
    aggregated_output = output_frame["monetary_output"].astype("float64")
    grouped["monetary_output"] = grouped["sector"].map(aggregated_output)
    if grouped["monetary_output"].isna().any():
        raise RuntimeError("Environmental reaggregation lost canonical output coverage.")
    grouped["intensity"] = grouped["value"] / grouped["monetary_output"].replace(
        0.0, np.nan
    )
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
    if compute_sector_value_per_physical_input and disaggregated_sector_value_per_physical_input is not None:
        agg_sector_values = compute_aggregated_sector_value_per_physical_input(
            disaggregated_sector_value_per_physical_input=disaggregated_sector_value_per_physical_input,
            aggregation_mapping=mapping,
        )
    return AggregationResult(
        Z=Z_agg, Y=Y_agg, x=x, A=A, V=V_agg, M=M_agg,
        aggregated_sector_value_per_physical_input=agg_sector_values,
        io_system=io_system,
    )

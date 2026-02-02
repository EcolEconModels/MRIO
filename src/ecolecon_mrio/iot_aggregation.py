from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.errors import ParserError


@dataclass(frozen=True)
class AggregationResult:
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
    aggregated_sector_value_per_physical_input: pd.DataFrame | None = None
    aggregated_sector_emission_intensity: pd.DataFrame | None = None
    io_system: object | None = None  # pymrio.IOSystem when available


def _parse_bool(x: object) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return True
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s == "":
        return True
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {x!r}")


def read_sector_aggregation_mapping_csv(
    path: str | Path,
    *,
    from_col: str = "from_sector",
    to_col: str = "to_sector",
    enabled_col: str = "enabled",
) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Read a sector aggregation mapping CSV.

    Expected columns:
    - from_sector: existing sector label
    - to_sector: new sector label

    Optional:
    - enabled: boolean-ish (default True)
    - notes: ignored by the parser (kept in returned DataFrame)
    """
    path = Path(path)

    def _read_mapping_csv_heuristic() -> pd.DataFrame:
        """
        Heuristic reader for hand-edited CSVs with unquoted commas.

        Assumes the conventional column order: from_sector,to_sector,enabled,notes
        and repairs by parsing from the right (..., to, enabled, notes...).
        """
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header: list[str] | None = None
            for raw in reader:
                if not raw:
                    continue
                if raw[0].lstrip().startswith("#"):
                    continue
                header = [c.strip() for c in raw]
                break
            if header is None:
                raise ValueError(f"Aggregation mapping CSV {path} is empty.")

            if from_col not in header or to_col not in header:
                raise ValueError(
                    f"Aggregation mapping CSV {path} missing required columns: "
                    f"{[c for c in (from_col, to_col) if c not in header]}"
                )

            # Heuristic repair for unquoted commas, assuming the conventional column order:
            # from_sector,to_sector,enabled,notes
            expected = [from_col, to_col, enabled_col, "notes"]
            if header[: len(expected)] != expected[: len(header)]:
                raise ValueError(
                    f"Aggregation mapping CSV {path} parsed unexpectedly; "
                    f"expected header starting with {expected}, got {header}"
                )

            for raw in reader:
                if not raw or all(str(x).strip() == "" for x in raw):
                    continue
                if str(raw[0]).lstrip().startswith("#"):
                    continue

                parts = [str(x).strip() for x in raw]
                enabled_idx = None
                if enabled_col in header:
                    enabled_idx = header.index(enabled_col)

                if enabled_idx is None:
                    if len(parts) < 2:
                        continue
                    from_sector = ",".join(parts[:-1]).strip()
                    to_sector = parts[-1].strip()
                    enabled = "1"
                    notes = ""
                else:
                    # Parse from the right: ... , to_sector, enabled, notes...
                    enabled_pos = None
                    for i in range(len(parts) - 1, -1, -1):
                        try:
                            _parse_bool(parts[i])
                            enabled_pos = i
                            break
                        except Exception:
                            continue
                    if enabled_pos is None or enabled_pos == 0:
                        raise ValueError(
                            f"Aggregation mapping CSV {path} has a row without a parseable "
                            f"{enabled_col}: {raw!r}"
                        )
                    if enabled_pos - 1 < 0:
                        raise ValueError(
                            f"Aggregation mapping CSV {path} has a row without {to_col}: {raw!r}"
                        )
                    to_sector = parts[enabled_pos - 1].strip()
                    enabled = parts[enabled_pos].strip()
                    notes = ",".join(parts[enabled_pos + 1 :]).strip()
                    from_sector = ",".join(parts[: enabled_pos - 1]).strip()

                rows.append(
                    {
                        from_col: from_sector,
                        to_col: to_sector,
                        enabled_col: enabled,
                        "notes": notes,
                    }
                )

        return pd.DataFrame(rows).fillna("")

    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except ParserError:
        # Most commonly: unquoted commas in from_sector or notes.
        df = _read_mapping_csv_heuristic()
    missing = [c for c in (from_col, to_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Aggregation mapping CSV {path} missing required columns: {missing}")

    if enabled_col in df.columns:
        mask = df[enabled_col].map(_parse_bool).astype(bool)
        df = df[mask].copy()

    df[from_col] = df[from_col].str.strip()
    df[to_col] = df[to_col].str.strip()

    bad = df[(df[from_col] == "") | (df[to_col] == "")]
    if not bad.empty:
        raise ValueError(
            f"Aggregation mapping CSV {Path(path)} has empty {from_col} or {to_col} in rows: {bad.index.tolist()}"
        )

    mapping: dict[str, str] = {}
    for f, t in zip(df[from_col], df[to_col], strict=True):
        f_s = str(f)
        t_s = str(t)
        if f_s in mapping and mapping[f_s] != t_s:
            raise ValueError(
                f"Aggregation mapping CSV {Path(path)} maps {f_s!r} to multiple targets: "
                f"{mapping[f_s]!r} and {t_s!r}"
            )
        mapping[f_s] = t_s
    return mapping, df


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
    level: int = -1,
    keep_unmapped: bool = True,
    create_io_system: bool = True,
    compute_sector_value_per_physical_input: bool = False,
    disaggregated_sector_value_per_physical_input: pd.DataFrame | None = None,
    compute_sector_emission_intensity: bool = False,
    disaggregated_sector_emission_intensity: pd.DataFrame | None = None,
) -> AggregationResult:
    """
    Aggregate an IO table (Z,Y) by summing rows/cols according to a mapping.

    Args:
        Z: Intermediate transactions matrix (must be square)
        Y: Final demand matrix (must have same index as Z)
        mapping: Dict mapping aggregated sector names to original sector names
        level: Which level of MultiIndex to aggregate (if applicable)
        keep_unmapped: Whether to keep sectors not in the mapping
        create_io_system: Whether to build a pymrio IOSystem
        compute_sector_value_per_physical_input: If True and disaggregated_sector_value_per_physical_input provided,
            compute aggregated sector value-per-physical-input
        disaggregated_sector_value_per_physical_input: DataFrame with value-per-physical-input data for disaggregated sectors
            (required if compute_sector_value_per_physical_input=True)

    Returns:
        AggregationResult with aggregated Z, Y, x, A, and optionally aggregated_sector_value_per_physical_input
    """
    if not Z.index.equals(Z.columns):
        raise ValueError("Z must be square with matching index/columns.")
    if not Y.index.equals(Z.index):
        raise ValueError("Y must have the same index as Z.")

    Z_agg = aggregate_square_matrix(Z, mapping, level=level, keep_unmapped=keep_unmapped)
    Y_agg = aggregate_Y(Y, mapping, level=level, keep_unmapped=keep_unmapped)

    if not Y_agg.index.equals(Z_agg.index):
        raise RuntimeError("Aggregation produced inconsistent Z/Y indices.")

    x = (Z_agg.sum(axis=1) + Y_agg.sum(axis=1)).astype("float64")

    A = Z_agg.astype("float64").copy()
    denom = x.replace(0.0, np.nan).astype("float64")
    A = A.div(denom, axis=1).fillna(0.0)

    io_system = None
    if create_io_system:
        try:
            import pymrio  # type: ignore

            io_system = pymrio.IOSystem(Z=Z_agg, Y=Y_agg, x=x)
            io_system.calc_system()
        except Exception:
            io_system = None

    # Compute aggregated sector value-per-physical-input if requested
    agg_sector_values = None
    if compute_sector_value_per_physical_input and disaggregated_sector_value_per_physical_input is not None:
        agg_sector_values = compute_aggregated_sector_value_per_physical_input(
            disaggregated_sector_value_per_physical_input=disaggregated_sector_value_per_physical_input,
            aggregation_mapping=mapping,
        )

    agg_sector_emission_intensity = None
    if compute_sector_emission_intensity:
        agg_sector_emission_intensity = compute_aggregated_sector_emission_intensity(
            disaggregated_sector_emission_intensity,
            mapping,
        )

    return AggregationResult(
        Z=Z_agg,
        Y=Y_agg,
        x=x,
        A=A,
        aggregated_sector_value_per_physical_input=agg_sector_values,
        aggregated_sector_emission_intensity=agg_sector_emission_intensity,
        io_system=io_system,
    )


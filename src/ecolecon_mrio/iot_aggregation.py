from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggregationResult:
    Z: pd.DataFrame
    Y: pd.DataFrame
    x: pd.Series
    A: pd.DataFrame
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
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in (from_col, to_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Aggregation mapping CSV {Path(path)} missing required columns: {missing}")

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


def aggregate_iot(
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    level: int = -1,
    keep_unmapped: bool = True,
    create_io_system: bool = True,
) -> AggregationResult:
    """
    Aggregate an IO table (Z,Y) by summing rows/cols according to a mapping.
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

    return AggregationResult(Z=Z_agg, Y=Y_agg, x=x, A=A, io_system=io_system)


def write_aggregation_report_xlsx(
    out_path: str | Path,
    *,
    Z_before: pd.DataFrame,
    Y_before: pd.DataFrame,
    result: AggregationResult,
    mapping_df: pd.DataFrame,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    before_sectors = pd.DataFrame({"sector": list(map(str, Z_before.index))})
    after_sectors = pd.DataFrame({"sector": list(map(str, result.Z.index))})

    counts = mapping_df.copy()
    if "enabled" not in counts.columns:
        counts["enabled"] = "1"

    with pd.ExcelWriter(out_path) as xw:
        mapping_df.to_excel(xw, sheet_name="mapping", index=False)
        before_sectors.to_excel(xw, sheet_name="sectors_before", index=False)
        after_sectors.to_excel(xw, sheet_name="sectors_after", index=False)

        summary = pd.DataFrame(
            {
                "n_sectors_before": [len(Z_before.index)],
                "n_sectors_after": [len(result.Z.index)],
                "Z_total_before": [float(Z_before.to_numpy().sum())],
                "Z_total_after": [float(result.Z.to_numpy().sum())],
                "Y_total_before": [float(Y_before.to_numpy().sum())],
                "Y_total_after": [float(result.Y.to_numpy().sum())],
            }
        )
        summary.to_excel(xw, sheet_name="summary", index=False)

    return out_path


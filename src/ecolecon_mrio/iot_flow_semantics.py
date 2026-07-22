from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_identities import (
    refresh_m_identity_rows,
    supplementary_accounting_identity_kind,
)

FlowRole = Literal["DOM", "IMP", "TOTAL"]

_P3_DETAIL = ("P3_S13", "P3_S14", "P3_S15")
_P5_DETAIL = ("P51G", "P52", "P53")
_P6_DETAIL = ("P6_B0", "P6_D0", "P6_U2", "P6_U3")
_ACCOUNTING_RTOL = 1e-9
_ACCOUNTING_ATOL = 1e-6


def _choose_family(columns: set[str], *, aggregate: str, detail: tuple[str, ...]) -> list[str]:
    present = [code for code in detail if code in columns]
    if len(present) == len(detail):
        return present
    if aggregate in columns:
        return [aggregate]
    if present:
        raise ValueError(f"Incomplete {aggregate} detail partition without aggregate: {present!r}")
    return []


def _choose_p6_family(columns: set[str]) -> list[str]:
    """Choose exports without treating P6 detail columns as additive to P6."""
    if "P6" in columns:
        return ["P6"]
    return _choose_family(columns, aggregate="P6", detail=_P6_DETAIL)


def atomic_final_demand_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return one complete, non-overlapping Eurostat final-demand basis."""
    cols = set(map(str, columns))
    selected = _choose_family(cols, aggregate="P3", detail=_P3_DETAIL)
    selected.extend(_choose_family(cols, aggregate="P5", detail=_P5_DETAIL))
    selected.extend(_choose_p6_family(cols))
    if not selected:
        raise ValueError("Eurostat Y has no recognised non-overlapping final-demand columns.")
    return selected


def _validate_partition_identity(
    Y: pd.DataFrame,
    *,
    aggregate: str,
    detail: tuple[str, ...],
    rtol: float = _ACCOUNTING_RTOL,
    atol: float = _ACCOUNTING_ATOL,
) -> None:
    cols = set(map(str, Y.columns))
    if aggregate not in cols or not set(detail).issubset(cols):
        return
    aggregate_values = Y[aggregate].astype("float64")
    detail_values = Y.loc[:, list(detail)].sum(axis=1).astype("float64")
    if not np.allclose(
        aggregate_values.to_numpy(dtype="float64"),
        detail_values.to_numpy(dtype="float64"),
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        gap = float(np.max(np.abs((aggregate_values - detail_values).to_numpy(dtype="float64"))))
        raise ValueError(
            f"Eurostat {aggregate} aggregate/detail identity failed before output construction: "
            f"max_abs_gap={gap:.12g}."
        )


def atomic_final_demand(Y: pd.DataFrame) -> pd.DataFrame:
    Y_ord = Y.copy().astype("float64")
    Y_ord.columns = Y_ord.columns.astype(str)
    for aggregate, detail in (("P3", _P3_DETAIL), ("P5", _P5_DETAIL)):
        _validate_partition_identity(Y_ord, aggregate=aggregate, detail=detail)
    cols = atomic_final_demand_columns(Y_ord.columns)
    return Y_ord.loc[:, cols].copy()


def flow_use_total(Z: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    use = Z.sum(axis=1).add(atomic_final_demand(Y).sum(axis=1), fill_value=0.0).astype("float64")
    use.name = "use_total"
    return use


def technical_coefficients(Z: pd.DataFrame, output: pd.Series) -> pd.DataFrame:
    missing = [str(code) for code in Z.columns if str(code) not in set(map(str, output.index))]
    if missing:
        raise ValueError(f"Output vector is missing Z columns: {missing[:20]!r}")
    denom = output.copy().astype("float64")
    denom.index = denom.index.astype(str)
    denom = denom.reindex(Z.columns.astype(str))
    denom_index = cast(pd.Index, denom.index)
    values = denom.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        bad = denom_index[~np.isfinite(values)].astype(str).tolist()
        raise ValueError(f"Output vector contains non-finite values: {bad[:20]!r}")
    if (values < 0.0).any():
        bad = denom_index[values < 0.0].astype(str).tolist()
        raise ValueError(f"Output vector contains negative values: {bad[:20]!r}")
    z_col_mass = Z.astype("float64").abs().sum(axis=0).reindex(denom.index).fillna(0.0)
    invalid_zero = (denom == 0.0) & (z_col_mass > 0.0)
    if bool(invalid_zero.any()):
        bad = denom_index[invalid_zero].astype(str).tolist()
        raise ValueError(f"Zero output with non-zero intermediate inputs for sectors: {bad[:20]!r}")
    safe = denom.replace(0.0, np.nan)
    return Z.astype("float64").divide(safe, axis="columns").fillna(0.0)


def refresh_supplementary_rows(
    V: pd.DataFrame | None,
    M: pd.DataFrame | None,
    *,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    output: pd.Series,
    flow_role: FlowRole,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Apply one explicit supplementary-row contract without inventing missing rows."""
    V_out = None if V is None else V.copy().astype("float64")
    if V_out is not None:
        V_out.index = V_out.index.astype(str)
        V_out.columns = V_out.columns.astype(str)
        cols = pd.Index(V_out.columns)
        missing_z = sorted(set(cols.astype(str)) - set(Z.columns.astype(str)))
        if missing_z:
            raise ValueError(f"Supplementary V columns are absent from Z: {missing_z[:20]!r}")
        z_col = Z.reindex(index=cols, columns=cols).fillna(0.0).sum(axis=0).astype("float64")
        x = output.reindex(cols)
        if x.isna().any():
            x_index = cast(pd.Index, x.index)
            missing = x_index[x.isna()].astype(str).tolist()
            raise ValueError(f"Output vector is missing supplementary V columns: {missing[:20]!r}")
        x = x.astype("float64")
        for row_code in list(map(str, V_out.index)):
            kind = supplementary_accounting_identity_kind(row_code)
            if kind == "output":
                V_out.loc[row_code, cols] = x
            elif kind == "intermediate_use":
                V_out.loc[row_code, cols] = z_col
            elif flow_role == "TOTAL" and kind == "value_added":
                V_out.loc[row_code, cols] = x - z_col
            elif flow_role == "TOTAL" and kind == "total_supply":
                imports = pd.Series(0.0, index=cols, dtype="float64")
                for candidate in ("P7", "P7_CIF", "P7_FOB", "Imports", "Imports of goods and services"):
                    if candidate in V_out.index:
                        imports = V_out.loc[candidate].reindex(cols).fillna(0.0).astype("float64")
                        break
                V_out.loc[row_code, cols] = x + imports
    M_out = refresh_m_identity_rows(M, Y=Y, Z=Z)
    return V_out, M_out

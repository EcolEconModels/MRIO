from __future__ import annotations

import numpy as np
import pandas as pd


def normalise_supplementary_row_label(row_code: object) -> str:
    return " ".join(str(row_code).strip().casefold().replace("\\", "/").split())


def supplementary_accounting_identity_kind(row_code: object) -> str | None:
    code = str(row_code).strip().upper()
    label = normalise_supplementary_row_label(row_code)
    if code in {"P1", "OUTPUT"} or label == "output":
        return "output"
    if code in {"P2_ADJ", "P2ADJ"} or label == "total intermediate consumption /final use":
        return "intermediate_use"
    if code in {"B1G", "B1GQ", "VALUE_ADDED_GROSS"} or label == "value added, gross":
        return "value_added"
    if code in {"TS_BP", "TOTAL_SUPPLY_AT_BASIC_PRICES"} or label == "total supply at basic prices":
        return "total_supply"
    return None


def is_imports_row(row_code: object) -> bool:
    code = str(row_code).strip().upper()
    label = normalise_supplementary_row_label(row_code)
    return code in {"P7", "P7_CIF", "P7_FOB", "IMPORTS"} or label in {
        "imports",
        "imports of goods and services",
        "imports cif",
        "imports fob",
    }


def find_imports_vector(V: pd.DataFrame | None, columns: pd.Index | list[str]) -> pd.Series | None:
    if V is None:
        return None

    cols = pd.Index(list(map(str, columns)))
    V_ord = V.copy().astype("float64")
    V_ord.index = V_ord.index.astype(str)
    V_ord.columns = V_ord.columns.astype(str)

    preferred = ["P7", "Imports of goods and services", "P7_CIF", "P7_FOB", "Imports"]
    for candidate in preferred:
        if candidate in V_ord.index:
            return V_ord.loc[candidate].reindex(cols).fillna(0.0).astype("float64")

    for row_code in V_ord.index.astype(str):
        if is_imports_row(row_code):
            return V_ord.loc[row_code].reindex(cols).fillna(0.0).astype("float64")
    return None


def derive_v_identity_rows(
    V: pd.DataFrame | None,
    *,
    Z: pd.DataFrame,
    x: pd.Series,
) -> pd.DataFrame | None:
    if V is None:
        return None

    out = V.copy().astype("float64")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    cols = pd.Index(out.columns.astype(str))
    Z_ord = Z.reindex(index=cols, columns=cols).fillna(0.0).astype("float64")
    x_ord = x.reindex(cols).fillna(0.0).astype("float64")
    intermediate_use = Z_ord.sum(axis=0).astype("float64")
    value_added = (x_ord - intermediate_use).astype("float64")
    imports = find_imports_vector(out, cols)
    if imports is None:
        imports = pd.Series(0.0, index=cols, dtype="float64")

    for row_code in list(map(str, out.index)):
        kind = supplementary_accounting_identity_kind(row_code)
        if kind == "output":
            out.loc[row_code, cols] = x_ord
        elif kind == "intermediate_use":
            out.loc[row_code, cols] = intermediate_use
        elif kind == "value_added":
            out.loc[row_code, cols] = value_added
        elif kind == "total_supply":
            out.loc[row_code, cols] = x_ord + imports
    return out


def refresh_m_identity_rows(
    M: pd.DataFrame | None,
    *,
    Y: pd.DataFrame | None,
    blank_non_applicable: bool = True,
) -> pd.DataFrame | None:
    if M is None:
        return None

    out = M.copy().astype("float64")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)

    fd_totals = None
    if Y is not None:
        fd_totals = Y.reindex(columns=out.columns).fillna(0.0).astype("float64").sum(axis=0)

    for row_code in list(map(str, out.index)):
        kind = supplementary_accounting_identity_kind(row_code)
        if kind == "intermediate_use" and fd_totals is not None:
            out.loc[row_code, out.columns] = fd_totals
        elif blank_non_applicable and kind in {"value_added", "output", "total_supply"}:
            out.loc[row_code, out.columns] = np.nan
    return out

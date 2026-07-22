from __future__ import annotations

import re

import numpy as np
import pandas as pd


def normalise_supplementary_row_label(row_code: object) -> str:
    label = " ".join(str(row_code).strip().casefold().replace("\\", "/").split())
    return re.sub(r"\s*/\s*", " /", label)


def supplementary_accounting_identity_kind(row_code: object) -> str | None:
    code = str(row_code).strip().upper()
    label = normalise_supplementary_row_label(row_code)
    if code in {"P1", "OUTPUT"} or label == "output":
        return "output"
    if code in {"P2_ADJ", "P2ADJ"} or label in {
        "total intermediate consumption /final use",
        "total intermediate consumption /final demand",
        "total intermediate consumption /final uses",
        "total intermediate use /final use",
    }:
        return "intermediate_use"
    if code in {"B1G", "B1GQ", "VALUE_ADDED_GROSS"} or label == "value added, gross":
        return "value_added"
    if code in {"TS_BP", "TOTAL_SUPPLY_AT_BASIC_PRICES"} or label == "total supply at basic prices":
        return "total_supply"
    return None


def final_demand_identity_totals(
    Y: pd.DataFrame,
    columns: pd.Index | list[str],
    *,
    Z: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Return lower-row final-demand identity totals for Eurostat-style FD columns.

    Ordinary final-demand columns are column sums of ``Y``. ``TFU`` is the
    non-overlapping total final use ``P3 + P5 + P6`` when those aggregate
    columns are available, otherwise the existing ``TFU`` column is used.
    ``TU`` is total intermediate use plus ``TFU`` and therefore needs ``Z``;
    without ``Z`` it falls back to an existing ``TU`` column if present.
    """
    cols = pd.Index(list(map(str, columns)))
    Y_ord = Y.copy().astype("float64")
    Y_ord.index = Y_ord.index.astype(str)
    Y_ord.columns = Y_ord.columns.astype(str)
    y_col_sums = Y_ord.sum(axis=0).astype("float64")

    if {"P3", "P5", "P6"}.issubset(set(Y_ord.columns)):
        tfu_total = float((Y_ord["P3"] + Y_ord["P5"] + Y_ord["P6"]).sum())
    elif "TFU" in Y_ord.columns:
        tfu_total = float(Y_ord["TFU"].sum())
    else:
        tfu_total = 0.0

    tu_total = float(y_col_sums.get("TU", np.nan))
    if Z is not None:
        Z_ord = Z.copy().astype("float64")
        Z_ord.index = Z_ord.index.astype(str)
        Z_ord.columns = Z_ord.columns.astype(str)
        tu_total = float(Z_ord.to_numpy(dtype="float64").sum()) + tfu_total
    elif not np.isfinite(tu_total):
        tu_total = tfu_total

    return pd.Series(
        {
            col: (
                tu_total
                if col == "TU"
                else tfu_total
                if col == "TFU"
                else float(y_col_sums.get(col, 0.0))
            )
            for col in cols
        },
        dtype="float64",
    )


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
    """Return one complete finite imports row aligned to the requested products.

    Missing products and non-finite values are accounting defects, not zero imports.
    Multiple recognised imports rows are ambiguous and therefore rejected.
    """
    if V is None:
        return None

    cols = pd.Index(list(map(str, columns)))
    if cols.has_duplicates:
        duplicates = cols[cols.duplicated()].astype(str).tolist()
        raise ValueError(f"Requested imports columns contain duplicates: {duplicates[:20]!r}")

    V_ord = V.copy().astype("float64")
    V_ord.index = V_ord.index.astype(str)
    V_ord.columns = V_ord.columns.astype(str)
    if V_ord.columns.has_duplicates:
        duplicates = V_ord.columns[V_ord.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"Supplementary V contains duplicate product columns: {duplicates[:20]!r}")

    matches = [str(row_code) for row_code in V_ord.index if is_imports_row(row_code)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one imports row, found {matches!r}")
    if V_ord.index.has_duplicates:
        duplicate_matches = [row for row in matches if int((V_ord.index == row).sum()) > 1]
        if duplicate_matches:
            raise ValueError(f"Imports row labels are duplicated: {duplicate_matches!r}")

    missing = [str(code) for code in cols if str(code) not in set(V_ord.columns)]
    if missing:
        raise ValueError(f"Imports row is missing required product columns: {missing[:20]!r}")

    row = V_ord.loc[matches[0], cols]
    if not isinstance(row, pd.Series):
        raise ValueError(f"Imports row {matches[0]!r} is not unique")
    values = row.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        bad = cols[~np.isfinite(values)].astype(str).tolist()
        raise ValueError(f"Imports row contains non-finite values for products: {bad[:20]!r}")
    return row.astype("float64")


def refresh_m_identity_rows(
    M: pd.DataFrame | None,
    *,
    Y: pd.DataFrame | None,
    Z: pd.DataFrame | None = None,
    blank_non_applicable: bool = True,
) -> pd.DataFrame | None:
    if M is None:
        return None

    out = M.copy().astype("float64")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)

    fd_totals = None
    if Y is not None:
        fd_totals = final_demand_identity_totals(Y, out.columns, Z=Z)

    for row_code in list(map(str, out.index)):
        kind = supplementary_accounting_identity_kind(row_code)
        if kind == "intermediate_use" and fd_totals is not None:
            out.loc[row_code, out.columns] = fd_totals
        elif blank_non_applicable and kind in {"value_added", "output", "total_supply"}:
            out.loc[row_code, out.columns] = np.nan
    return out

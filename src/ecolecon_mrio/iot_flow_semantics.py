from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_identities import (
    refresh_m_identity_rows,
    supplementary_accounting_identity_kind,
)

FlowRole = Literal["DOM", "IMP", "TOTAL"]


def atomic_final_demand_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return a non-overlapping Eurostat final-demand basis.

    Prefer detailed household/government/NPISH and capital-formation components,
    while preferring aggregate exports P6 over its geographic/detail columns.
    TU, TFU, TOTAL, P5M and all overlapping aggregate/detail duplicates are excluded.
    """
    cols = list(map(str, columns))
    selected: list[str] = []
    p3_detail = [code for code in ("P3_S13", "P3_S14", "P3_S15") if code in cols]
    selected.extend(p3_detail if p3_detail else (["P3"] if "P3" in cols else []))
    p5_detail = [code for code in ("P51G", "P52", "P53") if code in cols]
    selected.extend(p5_detail if p5_detail else (["P5"] if "P5" in cols else []))
    if "P6" in cols:
        selected.append("P6")
    else:
        selected.extend(sorted(code for code in cols if code.startswith("P6_")))
    return selected


def atomic_final_demand(Y: pd.DataFrame) -> pd.DataFrame:
    cols = atomic_final_demand_columns(Y.columns)
    if not cols:
        raise ValueError("Eurostat Y has no recognised non-overlapping final-demand columns.")
    return Y.loc[:, cols].astype("float64")


def flow_use_total(Z: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    use = Z.sum(axis=1).add(atomic_final_demand(Y).sum(axis=1), fill_value=0.0).astype("float64")
    use.name = "use_total"
    return use


def technical_coefficients(Z: pd.DataFrame, output: pd.Series) -> pd.DataFrame:
    denom = output.reindex(Z.columns).astype("float64").replace(0.0, np.nan)
    return Z.astype("float64").divide(denom, axis="columns").fillna(0.0)


def refresh_supplementary_rows(
    V: pd.DataFrame | None,
    M: pd.DataFrame | None,
    *,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    output: pd.Series,
    flow_role: FlowRole,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Apply one explicit supplementary-row contract.

    TOTAL is the only complete symmetric IOT: output, intermediate use, gross
    value added and total supply are recomputed there. DOM/IMP retain every
    flow-specific supplementary value, including rows whose complete-table
    identities are not applicable to partial use tables. Output is retained as
    the shared activity denominator; intermediate use remains flow-specific.
    """
    V_out = None if V is None else V.copy().astype("float64")
    if V_out is not None:
        V_out.index = V_out.index.astype(str)
        V_out.columns = V_out.columns.astype(str)
        cols = pd.Index(V_out.columns)
        z_col = Z.reindex(index=cols, columns=cols).fillna(0.0).sum(axis=0).astype("float64")
        x = output.reindex(cols).fillna(0.0).astype("float64")
        for row_code in list(map(str, V_out.index)):
            kind = supplementary_accounting_identity_kind(row_code)
            if kind == "output":
                V_out.loc[row_code, cols] = x
            elif kind == "intermediate_use":
                V_out.loc[row_code, cols] = z_col
            elif flow_role == "TOTAL" and kind == "value_added":
                V_out.loc[row_code, cols] = x - z_col
            elif flow_role == "TOTAL" and kind == "total_supply":
                # Preserve any parsed imports row already present; total supply is
                # output plus that explicit imports bridge, not a proxy.
                imports = pd.Series(0.0, index=cols, dtype="float64")
                for candidate in ("P7", "P7_CIF", "P7_FOB", "Imports", "Imports of goods and services"):
                    if candidate in V_out.index:
                        imports = V_out.loc[candidate].reindex(cols).fillna(0.0).astype("float64")
                        break
                V_out.loc[row_code, cols] = x + imports
    M_out = refresh_m_identity_rows(M, Y=Y, Z=Z)
    return V_out, M_out

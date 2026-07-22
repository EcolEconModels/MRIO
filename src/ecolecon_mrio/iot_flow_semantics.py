from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_identities import (
    find_imports_vector,
    refresh_m_identity_rows,
    supplementary_accounting_identity_kind,
)

FlowRole = Literal["DOM", "IMP", "TOTAL"]

_P3_DETAIL = ("P3_S13", "P3_S14", "P3_S15")
_P5_DETAIL = ("P51G", "P52", "P53")
_P6_DETAIL = ("P6_B0", "P6_D0", "P6_U2", "P6_U3")
_ACCOUNTING_RTOL = 1e-9
_ACCOUNTING_ATOL = 1e-6


def rounded_source_control_diagnostic(
    *,
    identity: str,
    published: float,
    derived: float,
    component_term_count: int,
    source_rounding_quantum: float,
    rtol: float = _ACCOUNTING_RTOL,
    atol: float = _ACCOUNTING_ATOL,
) -> dict[str, object]:
    """Validate one independently rounded published control.

    Generated accounting identities remain subject to ordinary ``atol``/``rtol``.
    This helper is only for a published control compared with a sum of independently
    rounded source components.
    """
    if not np.isfinite(source_rounding_quantum) or source_rounding_quantum <= 0.0:
        raise ValueError(f"Invalid source rounding quantum: {source_rounding_quantum!r}")
    if component_term_count < 1:
        raise ValueError(f"{identity}: component_term_count must be positive")
    if not np.isfinite(published) or not np.isfinite(derived):
        raise ValueError(f"{identity}: published and derived values must be finite")

    signed_gap = float(published - derived)
    absolute_gap = abs(signed_gap)
    scale = max(abs(float(published)), abs(float(derived)))
    floating_tolerance = max(float(atol), float(rtol) * scale)
    rounding_bound = (
        0.5 * float(source_rounding_quantum) * (int(component_term_count) + 1)
        + floating_tolerance
    )
    ok = absolute_gap <= rounding_bound
    diagnostic: dict[str, object] = {
        "identity": str(identity),
        "published_value": float(published),
        "derived_value": float(derived),
        "signed_gap": signed_gap,
        "absolute_gap": absolute_gap,
        "source_rounding_quantum": float(source_rounding_quantum),
        "component_term_count": int(component_term_count),
        "rounding_bound": rounding_bound,
        "floating_tolerance": floating_tolerance,
        "ok": ok,
    }
    if not ok:
        raise ValueError(
            f"Published Eurostat {identity} control failed: "
            f"published={published:.12g}, derived={derived:.12g}, "
            f"abs_gap={absolute_gap:.12g}, rounding_bound={rounding_bound:.12g}, "
            f"source_quantum={source_rounding_quantum:.12g}, "
            f"component_terms={component_term_count}."
        )
    return diagnostic

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


def source_atomic_final_demand(Y: pd.DataFrame) -> pd.DataFrame:
    """Select the non-overlapping basis from a published source table.

    Published aggregate controls may differ from their independently rounded
    details.  This source-only selector therefore chooses the same atomic basis
    without applying the strict generated aggregate/detail identity check.
    """
    Y_ord = Y.copy().astype("float64")
    Y_ord.columns = Y_ord.columns.astype(str)
    cols = atomic_final_demand_columns(Y_ord.columns)
    return Y_ord.loc[:, cols].copy()


def derived_final_demand_control_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return display controls that must be derived rather than allocated."""
    cols = set(map(str, columns))
    derived: list[str] = []
    if "P3" in cols and set(_P3_DETAIL).issubset(cols):
        derived.append("P3")
    if "P5" in cols and set(_P5_DETAIL).issubset(cols):
        derived.append("P5")
    derived.extend(code for code in ("TFU", "TU") if code in cols)
    return derived


def enforce_final_demand_display_identities(
    Y: pd.DataFrame,
    *,
    Z: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return Y with aggregate display columns derived from atomic uses.

    Complete P3/P5 detail partitions determine their aggregate display columns; P6
    remains aggregate-authoritative because its detail codes are not assumed to be
    exhaustive. TFU is the non-overlapping final-use total. TU additionally includes
    row-wise intermediate use and therefore requires an aligned Z whenever present.
    """
    out = Y.copy().astype("float64")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    for aggregate, detail in (("P3", _P3_DETAIL), ("P5", _P5_DETAIL)):
        if set(detail).issubset(out.columns):
            out.loc[:, aggregate] = out.loc[:, list(detail)].sum(axis=1).astype("float64")
    atomic = atomic_final_demand(out)
    tfu = atomic.sum(axis=1).astype("float64")
    if "TFU" in out.columns:
        out.loc[:, "TFU"] = tfu
    if "TU" in out.columns:
        if Z is None:
            raise ValueError("TU requires Z for row-wise intermediate use.")
        Z_ord = Z.copy().astype("float64")
        Z_ord.index = Z_ord.index.astype(str)
        if not Z_ord.index.equals(out.index):
            raise ValueError("TU derivation requires Z and Y to have identical row indices.")
        out.loc[:, "TU"] = Z_ord.sum(axis=1).add(tfu, fill_value=0.0).astype("float64")
    return out


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
                required_products = pd.Index(Z.columns.astype(str))
                imports = find_imports_vector(V_out, required_products)
                if imports is None:
                    raise ValueError(
                        f"TOTAL supply row {row_code!r} exists but no imports bridge can be identified."
                    )
                required_output = output.reindex(required_products)
                if required_output.isna().any():
                    required_index = cast(pd.Index, required_output.index)
                    missing = required_index[required_output.isna()].astype(str).tolist()
                    raise ValueError(f"TOTAL supply output is missing products: {missing[:20]!r}")
                V_out.loc[row_code, required_products] = required_output.astype("float64") + imports
    M_out = refresh_m_identity_rows(M, Y=Y, Z=Z)
    return V_out, M_out

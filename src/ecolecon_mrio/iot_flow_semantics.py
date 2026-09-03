from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

_P3_DETAIL = ("P3_S13", "P3_S14", "P3_S15")
_P5_DETAIL = ("P51G", "P52", "P53")
_P5M_DETAIL = ("P52", "P53")
_ACCOUNTING_RTOL = 1e-9
_ACCOUNTING_ATOL = 1e-6


def _normalize_nonnegative_accounting_control(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """Normalize only tolerance-scale negative noise on gross use/supply controls.

    Eurostat generated-table policy treats values below ``-1e-6`` M.EUR as
    genuinely negative while allowing roundoff-scale residuals near zero.
    Environmental target denominators use the same fixed boundary: values in
    ``[-1e-6, 0)`` are numerical zero, whereas a material negative remains an
    accounting failure.
    """
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError(f"{context}: accounting control must be a unique pandas Series.")
    try:
        numeric = cast(pd.Series, pd.to_numeric(values, errors="raise")).astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: accounting control must be numeric.") from exc
    array = numeric.to_numpy(dtype="float64")
    if not np.isfinite(array).all():
        bad = [str(numeric.index[pos]) for pos in np.flatnonzero(~np.isfinite(array))[:20]]
        raise ValueError(f"{context}: accounting control contains non-finite values: {bad!r}")
    material_negative = array < -_ACCOUNTING_ATOL
    if bool(material_negative.any()):
        bad = {
            str(numeric.index[pos]): float(array[pos])
            for pos in np.flatnonzero(material_negative)[:20]
        }
        raise ValueError(
            f"{context}: non-negative accounting control is materially negative "
            f"below tolerance {_ACCOUNTING_ATOL:.12g}: {bad!r}"
        )
    out = numeric.copy()
    roundoff_negative = (array < 0.0) & ~material_negative
    if bool(roundoff_negative.any()):
        out.iloc[np.flatnonzero(roundoff_negative)] = 0.0
    return out


def rounded_source_control_diagnostic(
    *,
    identity: str,
    published: float,
    derived: float,
    component_term_count: int,
    source_rounding_quantum: float,
    rtol: float = _ACCOUNTING_RTOL,
    atol: float = _ACCOUNTING_ATOL,
    raise_on_failure: bool = True,
) -> dict[str, object]:
    """Validate one independently rounded published control.

    Generated accounting identities remain subject to ordinary ``atol``/``rtol``.
    This helper is only for a published control compared with a sum of independently
    rounded source components.
    """
    if not np.isfinite(source_rounding_quantum) or source_rounding_quantum <= 0.0:
        raise ValueError(
            f"Invalid source rounding quantum: {source_rounding_quantum!r}"
        )
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
    if not ok and raise_on_failure:
        raise ValueError(
            f"Published Eurostat {identity} control failed: "
            f"published={published:.12g}, derived={derived:.12g}, "
            f"abs_gap={absolute_gap:.12g}, rounding_bound={rounding_bound:.12g}, "
            f"source_quantum={source_rounding_quantum:.12g}, "
            f"component_terms={component_term_count}."
        )
    return diagnostic


def _choose_family(
    columns: set[str], *, aggregate: str, detail: tuple[str, ...]
) -> list[str]:
    present = [code for code in detail if code in columns]
    if len(present) == len(detail):
        return present
    if aggregate in columns:
        return [aggregate]
    if present:
        raise ValueError(
            f"Incomplete {aggregate} detail partition without aggregate: {present!r}"
        )
    return []


def _choose_p6_family(columns: set[str]) -> list[str]:
    """Choose exports without treating P6 detail columns as additive to P6."""
    if "P6" in columns:
        return ["P6"]
    if {"P6_B0", "P6_D0"}.issubset(columns):
        return ["P6_B0", "P6_D0"]
    if {"P6_U2", "P6_U3", "P6_D0"}.issubset(columns):
        return ["P6_U2", "P6_U3", "P6_D0"]
    present = sorted(columns.intersection({"P6_B0", "P6_D0", "P6_U2", "P6_U3"}))
    if present:
        raise ValueError(f"Incomplete P6 detail partition without aggregate: {present!r}")
    return []


def atomic_final_demand_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return one complete, non-overlapping Eurostat final-demand basis."""
    cols = set(map(str, columns))
    selected = _choose_family(cols, aggregate="P3", detail=_P3_DETAIL)
    selected.extend(_choose_family(cols, aggregate="P5", detail=_P5_DETAIL))
    selected.extend(_choose_p6_family(cols))
    if not selected:
        raise ValueError(
            "Eurostat Y has no recognised non-overlapping final-demand columns."
        )
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
        gap = float(
            np.max(np.abs((aggregate_values - detail_values).to_numpy(dtype="float64")))
        )
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
    if "P5M" in cols and set(_P5M_DETAIL).issubset(cols):
        derived.append("P5M")
    if "P5" in cols and set(_P5_DETAIL).issubset(cols):
        derived.append("P5")
    if {"P6", "P6_B0", "P6_D0"}.issubset(cols):
        derived.append("P6_D0")
    if {"P6_B0", "P6_U2", "P6_U3"}.issubset(cols):
        derived.append("P6_U3")
    derived.extend(code for code in ("TFU", "TU") if code in cols)
    return derived


def enforce_final_demand_display_identities(
    Y: pd.DataFrame,
    *,
    Z: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return Y with aggregate display columns derived from atomic uses.

    Complete P3/P5 detail partitions determine their aggregate display columns.
    P6 is the atomic export total. P6_B0 and P6_U2 are independently rounded
    geographic export controls, while P6_D0 and P6_U3 are rebuilt as exact residual
    display controls and may therefore be slightly signed. TFU is the non-overlapping
    final-use total. TU additionally includes row-wise intermediate use and therefore
    requires an aligned Z whenever present.
    """
    out = Y.copy().astype("float64")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    # Allocate primitive components first, then regenerate published controls in
    # dependency order so no aggregate is treated as an independent donor target.
    if set(_P3_DETAIL).issubset(out.columns):
        out.loc[:, "P3"] = out.loc[:, list(_P3_DETAIL)].sum(axis=1).astype("float64")
    if set(_P5M_DETAIL).issubset(out.columns):
        out.loc[:, "P5M"] = out.loc[:, list(_P5M_DETAIL)].sum(axis=1).astype("float64")
    if set(_P5_DETAIL).issubset(out.columns):
        out.loc[:, "P5"] = (
            out.loc[:, "P51G"].add(out.loc[:, "P5M"], fill_value=0.0).astype("float64")
        )
    # P6 is atomic. Its independently rounded geographic export controls may
    # cross by a source-rounding quantum, so preserve the published controls and
    # put the discrepancy into non-atomic residual display columns rather than
    # clipping or fabricating trade. Domain checks remain owned by the declared
    # source/generated accounting policy rather than this display-identity helper.
    if {"P6", "P6_B0", "P6_D0"}.issubset(out.columns):
        out.loc[:, "P6_D0"] = out.loc[:, "P6"] - out.loc[:, "P6_B0"]
    if {"P6_B0", "P6_U2", "P6_U3"}.issubset(out.columns):
        out.loc[:, "P6_U3"] = out.loc[:, "P6_B0"] - out.loc[:, "P6_U2"]
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
            raise ValueError(
                "TU derivation requires Z and Y to have identical row indices."
            )
        out.loc[:, "TU"] = Z_ord.sum(axis=1).add(tfu, fill_value=0.0).astype("float64")
    return out


def flow_use_total(Z: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    use = (
        Z.sum(axis=1)
        .add(atomic_final_demand(Y).sum(axis=1), fill_value=0.0)
        .astype("float64")
    )
    use.name = "use_total"
    return use


def _validated_nonnegative_output(
    output: pd.Series,
    *,
    expected_index: pd.Index,
    context: str,
) -> np.ndarray:
    """Return aligned finite non-negative gross output without repairing labels."""
    if not isinstance(output, pd.Series):
        raise TypeError(f"{context}: output must be a pandas Series.")
    if not expected_index.is_unique or not output.index.is_unique:
        raise ValueError(f"{context}: output and coefficient axes must be unique.")
    if not output.index.equals(expected_index):
        raise ValueError(f"{context}: output index must exactly match the coefficient column axis.")
    try:
        numeric = cast(pd.Series, pd.to_numeric(output, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: output contains non-numeric values.") from exc
    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        bad = [str(expected_index[pos]) for pos in np.flatnonzero(~np.isfinite(values))[:20]]
        raise ValueError(f"{context}: output contains non-finite values: {bad!r}")
    if bool((values < 0.0).any()):
        bad = [str(expected_index[pos]) for pos in np.flatnonzero(values < 0.0)[:20]]
        raise ValueError(f"{context}: output contains negative values: {bad!r}")
    return values


def _validate_technical_coefficient_denominators(
    output: pd.Series,
    *,
    expected_index: pd.Index,
    column_absolute_mass: np.ndarray,
    context: str,
) -> np.ndarray:
    """Validate x_j before forming A_ij=Z_ij/x_j on one buyer-sector axis."""
    values = _validated_nonnegative_output(
        output,
        expected_index=expected_index,
        context=context,
    )
    mass = np.asarray(column_absolute_mass, dtype="float64")
    if mass.shape != values.shape:
        raise ValueError(
            f"{context}: intermediate-input column mass shape {mass.shape!r} "
            f"does not match output shape {values.shape!r}."
        )
    if not np.isfinite(mass).all() or bool((mass < 0.0).any()):
        raise ValueError(f"{context}: intermediate-input column mass must be finite and non-negative.")
    invalid_zero = (values == 0.0) & (mass > 0.0)
    if bool(invalid_zero.any()):
        bad = [str(expected_index[pos]) for pos in np.flatnonzero(invalid_zero)[:20]]
        raise ValueError(
            f"{context}: zero output with non-zero intermediate inputs for sectors: {bad!r}"
        )
    return values


def technical_coefficients(Z: pd.DataFrame, output: pd.Series) -> pd.DataFrame:
    """Return dense A=Z/x after the same fail-closed denominator checks as sparse A."""
    Z_numeric = Z.astype("float64")
    if not Z_numeric.index.equals(Z_numeric.columns):
        raise ValueError("Technical-coefficient Z must be square with identical row and column axes.")
    if not Z_numeric.columns.is_unique:
        raise ValueError("Technical-coefficient Z columns must be unique.")
    z_values = Z_numeric.to_numpy(dtype="float64", copy=False)
    if not np.isfinite(z_values).all():
        raise ValueError("Technical-coefficient Z contains non-finite monetary values.")
    column_mass = np.abs(z_values).sum(axis=0)
    values = _validate_technical_coefficient_denominators(
        output,
        expected_index=Z_numeric.columns,
        column_absolute_mass=column_mass,
        context="technical_coefficients",
    )
    safe = pd.Series(values, index=Z_numeric.columns, dtype="float64").replace(0.0, np.nan)
    return Z_numeric.divide(safe, axis="columns").fillna(0.0)

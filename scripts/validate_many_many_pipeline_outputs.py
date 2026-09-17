#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportGeneralTypeIssues=false
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_REPO_ROOT, _REPO_ROOT / "src"):
    _extra_text = str(_extra_path)
    if _extra_text not in sys.path:
        sys.path.insert(0, _extra_text)

from ecolecon_mrio.iot_aggregation import aggregate_environmental_extensions, aggregate_iot
from ecolecon_mrio.iot_flow_semantics import flow_use_total
from ecolecon_mrio.mapping_csv import read_aggregation_mapping
from scripts.aggregate_iot_by_mapping import (
    _augment_mapping_from_split_child_edges,
    _load_split_selected_children,
    _partition_environmental_extensions_for_sector_reaggregation,
    _production_environmental_group_domain,
)


_ENVIRONMENTAL_GRID_COLUMNS = [
    "sector_position",
    "sector",
    "boundary",
    "denominator_role",
    "extension",
    "stressor",
    "unit",
    "value",
    "monetary_output",
    "intensity",
]
_REQUIRED_TRANSFORMATION_COHERENCE_CHECKS = {
    ("DOM", "tc.row_child_mass"),
    ("DOM", "tc.parent_z_cell"),
    ("DOM", "tc.zero_target_output_column"),
    ("IMP", "tc.row_child_mass"),
    ("IMP", "tc.parent_z_cell"),
    ("IMP", "tc.zero_target_output_column"),
    ("DOM", "tc.dom_output_margin"),
    ("IMP", "tc.imp_row_use_margin"),
    ("TOTAL", "tc.total_use_additivity"),
}


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, index_col=0)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Expected DataFrame in {path}")
        return obj
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out.astype("float64")


def _comparison_failure(*, reason: str) -> dict[str, Any]:
    """Return a uniform failed comparison diagnostic for structural defects."""
    return {
        "ok": False,
        "max_abs_gap": float("inf"),
        "max_relative_gap": float("inf"),
        "max_normalized_gap": float("inf"),
        "worst_label": None,
        "worst_field": None,
        "worst_actual": None,
        "worst_expected": None,
        "worst_abs_gap": float("inf"),
        "worst_tolerance": None,
        "reason": reason,
    }


def _comparison_label(label: object) -> object:
    if isinstance(label, tuple):
        return [str(item) for item in label]
    return str(label)


def _compare_labelled_numeric_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare aligned labelled extensive values under the symmetric rtol+atol contract."""
    if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
        raise ValueError("rtol and atol must be finite and non-negative")
    if not actual.index.is_unique or not actual.columns.is_unique:
        return _comparison_failure(reason="actual labels are not unique")
    if not expected.index.is_unique or not expected.columns.is_unique:
        return _comparison_failure(reason="expected labels are not unique")
    if set(actual.index) != set(expected.index) or set(actual.columns) != set(expected.columns):
        return _comparison_failure(reason="label domains differ")

    expected_aligned = expected.reindex(index=actual.index, columns=actual.columns)
    actual_values = actual.astype("float64").to_numpy(dtype="float64")
    expected_values = expected_aligned.astype("float64").to_numpy(dtype="float64")
    actual_missing = np.isnan(actual_values)
    expected_missing = np.isnan(expected_values)
    if np.any(actual_missing != expected_missing):
        return _comparison_failure(reason="NaN masks differ")
    jointly_missing = actual_missing & expected_missing
    finite = np.isfinite(actual_values) & np.isfinite(expected_values)
    if np.any(~finite & ~jointly_missing):
        return _comparison_failure(reason="non-finite numerical values")
    if actual_values.size == 0:
        return {
            "ok": True,
            "max_abs_gap": 0.0,
            "max_relative_gap": 0.0,
            "max_normalized_gap": 0.0,
            "worst_label": None,
            "worst_field": None,
            "worst_actual": None,
            "worst_expected": None,
            "worst_abs_gap": 0.0,
            "worst_tolerance": None,
        }

    delta = actual_values - expected_values
    abs_gap = np.where(jointly_missing, 0.0, np.abs(delta))
    scale = np.where(
        jointly_missing, 0.0, np.maximum(np.abs(actual_values), np.abs(expected_values))
    )
    tolerance = atol + rtol * scale
    normalized = np.full_like(abs_gap, np.inf)
    positive_tolerance = tolerance > 0.0
    np.divide(abs_gap, tolerance, out=normalized, where=positive_tolerance)
    normalized[~positive_tolerance & (abs_gap == 0.0)] = 0.0
    relative = np.divide(abs_gap, scale, out=np.zeros_like(abs_gap), where=scale > 0.0)
    worst_position = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    row_position, column_position = worst_position
    return {
        "ok": bool(np.all(abs_gap <= tolerance)),
        "max_abs_gap": float(np.max(abs_gap)),
        "max_relative_gap": float(np.max(relative)),
        "max_normalized_gap": float(np.max(normalized)),
        "worst_label": _comparison_label(actual.index[row_position]),
        "worst_field": str(actual.columns[column_position]),
        "worst_actual": float(actual_values[worst_position]),
        "worst_expected": float(expected_values[worst_position]),
        "worst_abs_gap": float(abs_gap[worst_position]),
        "worst_tolerance": float(tolerance[worst_position]),
    }


def _compare_numeric_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_norm = actual.copy()
    expected_norm = expected.copy()
    actual_norm.index = actual_norm.index.astype(str)
    actual_norm.columns = actual_norm.columns.astype(str)
    expected_norm.index = expected_norm.index.astype(str)
    expected_norm.columns = expected_norm.columns.astype(str)
    if (
        not actual_norm.index.is_unique
        or not actual_norm.columns.is_unique
        or not expected_norm.index.is_unique
        or not expected_norm.columns.is_unique
    ):
        return _comparison_failure(reason="frame labels are not unique")
    if set(actual_norm.index) != set(expected_norm.index):
        return _comparison_failure(reason="frame row domains differ")
    if set(actual_norm.columns) != set(expected_norm.columns):
        return _comparison_failure(reason="frame column domains differ")
    return _compare_labelled_numeric_frames(
        actual_norm, expected_norm, rtol=rtol, atol=atol
    )


def _compare_numeric_series(
    actual: pd.Series,
    expected: pd.Series,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_norm = actual.copy()
    expected_norm = expected.copy()
    actual_norm.index = actual_norm.index.astype(str)
    expected_norm.index = expected_norm.index.astype(str)
    if not actual_norm.index.is_unique or not expected_norm.index.is_unique:
        return _comparison_failure(reason="series labels are not unique")
    if set(actual_norm.index) != set(expected_norm.index):
        return _comparison_failure(reason="series label domains differ")
    return _compare_labelled_numeric_frames(
        actual_norm.to_frame("value"), expected_norm.to_frame("value"), rtol=rtol, atol=atol
    )


def _compare_environmental_extensions(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    key_cols = [
        "sector", "boundary", "denominator_role", "extension", "stressor", "unit"
    ]
    value_cols = ["value", "monetary_output"]
    required = set(key_cols + value_cols)
    if not required.issubset(actual.columns) or not required.issubset(expected.columns):
        return _comparison_failure(reason="environmental columns are missing")
    sort_keys = list(map(str, key_cols))
    actual_keyed = cast(
        pd.DataFrame, actual.loc[:, key_cols + value_cols].copy()
    ).sort_values(sort_keys).reset_index(drop=True)
    expected_keyed = cast(
        pd.DataFrame, expected.loc[:, key_cols + value_cols].copy()
    ).sort_values(sort_keys).reset_index(drop=True)
    actual_keyed = actual_keyed.set_index(key_cols)
    expected_keyed = expected_keyed.set_index(key_cols)
    return _compare_labelled_numeric_frames(
        actual_keyed[value_cols], expected_keyed[value_cols], rtol=rtol, atol=atol
    )

def _check_ratio_file(
    path: Path,
    *,
    numerator_col: str,
    denominator_col: str,
    ratio_col: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": True, "max_abs_gap": 0.0}
    df = pd.read_csv(path)
    required = {numerator_col, denominator_col, ratio_col}
    missing = sorted(required - set(df.columns))
    if missing:
        return {"path": str(path), "exists": True, "ok": False, "missing_columns": missing}
    num = pd.to_numeric(df[numerator_col], errors="coerce").astype("float64")
    den = pd.to_numeric(df[denominator_col], errors="coerce").astype("float64")
    observed = pd.to_numeric(df[ratio_col], errors="coerce").astype("float64")
    valid = den.replace(0.0, np.nan).notna()
    if not bool(valid.any()):
        return {"path": str(path), "exists": True, "ok": True, "max_abs_gap": 0.0, "n_checked": 0}
    expected = num.loc[valid] / den.loc[valid]
    comparison = _compare_numeric_series(
        observed.loc[valid], expected, rtol=rtol, atol=atol
    )
    return {
        "path": str(path),
        "exists": True,
        **comparison,
        "n_checked": int(valid.sum()),
    }


def _expected_final_demand_environmental_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw F_Y provenance into the exact target-side passthrough view."""
    required = {
        "source_label",
        "extension",
        "kind",
        "stressor",
        "unit",
        "value",
        "monetary_denominator",
        "intensity",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            "Raw environmental sidecar is missing F_Y passthrough columns: "
            f"{sorted(missing)!r}."
        )
    fy = raw.loc[raw["kind"].astype(str).eq("F_Y")].copy()
    if fy.empty:
        return pd.DataFrame(columns=_ENVIRONMENTAL_GRID_COLUMNS)
    out = fy.rename(
        columns={"source_label": "sector", "monetary_denominator": "monetary_output"}
    ).copy()
    out.insert(2, "boundary", "scope_final_demand_direct")
    out.insert(3, "denominator_role", "scope_final_demand")
    group_cols = ["boundary", "denominator_role", "extension", "stressor", "unit"]
    out.insert(
        0,
        "sector_position",
        out.groupby(group_cols, sort=False, dropna=False).cumcount().add(1).astype("int64"),
    )
    return out.loc[:, _ENVIRONMENTAL_GRID_COLUMNS].reset_index(drop=True)


def _compare_final_demand_environmental_passthrough(
    observed: pd.DataFrame | None,
    expected: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual = (
        pd.DataFrame(columns=_ENVIRONMENTAL_GRID_COLUMNS)
        if observed is None
        else observed.loc[:, _ENVIRONMENTAL_GRID_COLUMNS].copy()
    )
    key_cols = [
        "sector_position",
        "sector",
        "boundary",
        "denominator_role",
        "extension",
        "stressor",
        "unit",
    ]
    value_cols = ["value", "monetary_output", "intensity"]
    return _compare_labelled_numeric_frames(
        actual.set_index(key_cols)[value_cols],
        expected.set_index(key_cols)[value_cols],
        rtol=rtol,
        atol=atol,
    )


def _validate_environmental_grid(
    *,
    environmental_path: Path,
    raw_path: Path | None,
    output: pd.Series,
    rtol: float,
    atol: float,
    production_groups: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate the product grid plus unchanged direct-final-demand passthrough."""
    errors: list[str] = []
    if not environmental_path.is_file():
        return {"ok": False, "errors": [f"Missing {environmental_path}"]}

    observed = pd.read_csv(environmental_path)
    raw: pd.DataFrame | None = None
    if raw_path is not None and raw_path.is_file():
        raw = pd.read_csv(raw_path)
        raw_columns = [
            "source_label",
            "extension",
            "kind",
            "stressor",
            "unit",
            "value",
            "monetary_denominator",
            "intensity",
        ]
        if list(raw.columns) != raw_columns:
            errors.append(f"raw schema={list(raw.columns)!r}")
    elif production_groups is None:
        return {
            "ok": False,
            "errors": [f"Missing raw sidecar and explicit production-group domain: {raw_path}"],
        }
    if list(observed.columns) != _ENVIRONMENTAL_GRID_COLUMNS:
        errors.append(f"environmental schema={list(observed.columns)!r}")
        return {"ok": False, "errors": errors}

    observed_all = observed.copy()
    observed_sector, observed_final_demand = (
        _partition_environmental_extensions_for_sector_reaggregation(observed)
    )
    observed = (
        observed_sector
        if observed_sector is not None
        else pd.DataFrame(columns=_ENVIRONMENTAL_GRID_COLUMNS)
    )

    if production_groups is None:
        assert raw is not None
        groups = _production_environmental_group_domain(
            raw, target_environmental_extensions=observed
        )
    else:
        group_cols = ["boundary", "denominator_role", "extension", "stressor", "unit"]
        missing_groups = set(group_cols) - set(production_groups.columns)
        if missing_groups:
            errors.append(
                f"production-group domain missing columns={sorted(missing_groups)!r}"
            )
            groups = pd.DataFrame(columns=group_cols)
        else:
            groups = (
                production_groups.loc[:, group_cols]
                .copy()
                .drop_duplicates(ignore_index=True)
            )
    group_cols = ["boundary", "denominator_role", "extension", "stressor", "unit"]
    key_cols = ["sector", *group_cols]
    canonical = output.copy().astype("float64")
    canonical.index = canonical.index.astype(str)
    sectors = canonical.index.tolist()
    if not canonical.index.is_unique:
        errors.append("final output sector index is not unique")
    if canonical.isna().any() or not np.isfinite(canonical.to_numpy()).all():
        errors.append("final output contains missing or non-finite values")

    expected_keys = pd.DataFrame(
        [
            {
                "sector_position": position,
                "sector": sector,
                "boundary": str(group.boundary),
                "denominator_role": str(group.denominator_role),
                "extension": str(group.extension),
                "stressor": str(group.stressor),
                "unit": str(group.unit),
            }
            for group in groups.itertuples(index=False)
            for position, sector in enumerate(sectors, start=1)
        ]
    )
    observed_keys = observed.loc[:, ["sector_position", *key_cols]].copy()
    if len(observed) != len(expected_keys):
        errors.append(
            f"row count={len(observed)}, expected={len(expected_keys)}"
        )
    if bool(observed.duplicated(key_cols).any()):
        errors.append("duplicate sector/extension/stressor/unit rows")
    if not observed_keys.equals(expected_keys):
        errors.append("group/sector order or sector_position differs from canonical grid")

    monetary = pd.to_numeric(observed["monetary_output"], errors="coerce")
    invalid_monetary = observed["monetary_output"].notna() & monetary.isna()
    if (
        bool(invalid_monetary.any())
        or bool(monetary.isna().any())
        or not np.isfinite(monetary.to_numpy(dtype="float64")).all()
    ):
        errors.append("monetary_output contains invalid or missing values")
    else:
        expected_output = observed["sector"].astype(str).map(canonical)
        if not np.allclose(
            monetary.to_numpy(dtype="float64"),
            expected_output.to_numpy(dtype="float64"),
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ):
            errors.append("monetary_output differs from canonical final x")

    value = pd.to_numeric(observed["value"], errors="coerce")
    intensity = pd.to_numeric(observed["intensity"], errors="coerce")
    invalid_value = observed["value"].notna() & value.isna()
    invalid_intensity = observed["intensity"].notna() & intensity.isna()
    if bool(invalid_value.any()) or not np.isfinite(value.dropna()).all():
        errors.append("value contains invalid or infinite observations")
    if bool(invalid_intensity.any()) or not np.isfinite(intensity.dropna()).all():
        errors.append("intensity contains invalid or infinite observations")

    missing_value = value.isna()
    if bool(intensity.loc[missing_value].notna().any()):
        errors.append("value=NA requires intensity=NA")

    # DOM is the strict target-side production boundary: positive activity must
    # have environmental evidence. IMP/TOTAL may propagate unavailable origin
    # evidence under the documented no-imputation contract.
    positive_domestic_unavailable = (
        observed["denominator_role"].astype(str).eq("target_domestic_output")
        & monetary.gt(0.0)
        & missing_value
    )
    if bool(positive_domestic_unavailable.any()):
        bad = (
            observed.loc[positive_domestic_unavailable, "sector"]
            .astype(str).drop_duplicates().head(20).tolist()
        )
        errors.append(
            "positive target domestic output requires available environmental "
            f"evidence; sectors={bad!r}"
        )

    finite_value = value.notna()
    nonzero_denominator = monetary.ne(0.0)
    ratio_rows = finite_value & nonzero_denominator
    if bool(ratio_rows.any()):
        expected_ratio = value.loc[ratio_rows] / monetary.loc[ratio_rows]
        observed_ratio = intensity.loc[ratio_rows]
        if bool(observed_ratio.isna().any()) or not np.allclose(
            observed_ratio.to_numpy(dtype="float64"),
            expected_ratio.to_numpy(dtype="float64"),
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ):
            errors.append("nonzero-denominator intensity does not equal value/output")
    zero_output_estimates = finite_value & monetary.eq(0.0)
    if bool(zero_output_estimates.any()):
        signed_zero_denominator = zero_output_estimates & observed[
            "denominator_role"
        ].isin({"target_import_use", "target_total_use"})
        ordinary_zero_denominator = zero_output_estimates & ~signed_zero_denominator
        zero_values = value.loc[ordinary_zero_denominator]
        zero_intensities = intensity.loc[zero_output_estimates]
        if not np.allclose(
            zero_values.to_numpy(dtype="float64"),
            0.0,
            rtol=rtol,
            atol=atol,
        ):
            errors.append("ordinary zero-output estimates must have value=0")
        if bool(zero_intensities.notna().any()):
            errors.append("zero-output supported estimates must have intensity=NA")

    nonzero_denominator_unavailable = (
        cast(pd.Series, missing_value)
        & cast(pd.Series, monetary.notna())
        & cast(pd.Series, monetary.ne(0.0))
    )
    unavailable_sectors = sorted(
        observed.loc[nonzero_denominator_unavailable, "sector"]
        .astype(str).drop_duplicates().tolist()
    )
    total_abs_monetary_mass = float(canonical.abs().sum())
    unavailable_abs_monetary_mass = float(
        canonical.reindex(unavailable_sectors).fillna(0.0).abs().sum()
    )
    unavailable_abs_monetary_share = (
        unavailable_abs_monetary_mass / total_abs_monetary_mass
        if total_abs_monetary_mass > 0.0 else 0.0
    )

    if raw is None:
        if observed_final_demand is not None and not observed_final_demand.empty:
            errors.append(
                "scope_final_demand_direct rows require raw F_Y provenance for passthrough validation"
            )
    else:
        try:
            expected_final_demand = _expected_final_demand_environmental_grid(raw)
            final_demand_comparison = _compare_final_demand_environmental_passthrough(
                observed_final_demand,
                expected_final_demand,
                rtol=rtol,
                atol=atol,
            )
            if not bool(final_demand_comparison["ok"]):
                errors.append(
                    "scope_final_demand_direct rows are not an unchanged F_Y passthrough: "
                    f"{final_demand_comparison.get('reason', final_demand_comparison)!r}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"F_Y passthrough validation failed: {exc}")

    return {
        "ok": not errors,
        "errors": errors,
        "n_groups": int(len(groups)),
        "n_sectors": int(len(canonical)),
        "n_rows": int(len(observed_all)),
        "n_final_demand_rows": int(
            0 if observed_final_demand is None else len(observed_final_demand)
        ),
        "n_nonzero_denominator_unavailable_rows": int(
            nonzero_denominator_unavailable.sum()
        ),
        "n_nonzero_denominator_unavailable_sectors": len(unavailable_sectors),
        "nonzero_denominator_unavailable_sector_sample": unavailable_sectors[:50],
        "nonzero_denominator_unavailable_abs_monetary_mass": unavailable_abs_monetary_mass,
        "total_abs_monetary_mass": total_abs_monetary_mass,
        "nonzero_denominator_unavailable_abs_monetary_share": unavailable_abs_monetary_share,
    }


def _load_augmented_mapping(
    *,
    mapping_csv: Path,
    split_dir: Path,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    mapping, mapping_df = read_aggregation_mapping(mapping_csv)
    edge_path = split_dir / "split_child_edges.csv"
    if not edge_path.exists():
        return mapping, mapping_df, pd.DataFrame()
    edge_metadata = pd.read_csv(edge_path)
    return _augment_mapping_from_split_child_edges(
        mapping=dict(mapping),
        mapping_df=mapping_df.copy(),
        edge_metadata=edge_metadata,
    )


def _validate_transformation_coherence_report(path: Path) -> dict[str, Any]:
    """Validate persisted evidence for the EXIO producer-conditioning operator."""
    if not path.is_file():
        return {"ok": False, "path": str(path), "reason": "missing report"}
    frame = pd.read_csv(path)
    required_columns = {
        "validation_class", "check_id", "flow", "status", "max_abs_gap", "rtol", "atol"
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        return {
            "ok": False,
            "path": str(path),
            "reason": "missing columns",
            "missing_columns": missing_columns,
        }
    if set(frame["validation_class"].astype(str)) != {"transformation_coherence"}:
        return {"ok": False, "path": str(path), "reason": "wrong validation_class"}
    keys = list(zip(frame["flow"].astype(str), frame["check_id"].astype(str)))
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    key_set = set(keys)
    missing_checks = sorted(_REQUIRED_TRANSFORMATION_COHERENCE_CHECKS - key_set)
    extra_checks = sorted(key_set - _REQUIRED_TRANSFORMATION_COHERENCE_CHECKS)
    failing = frame.loc[~frame["status"].astype(str).eq("pass"), ["flow", "check_id", "status"]]
    numeric = frame[["max_abs_gap", "rtol", "atol"]].apply(pd.to_numeric, errors="coerce")
    finite_numeric = bool(np.isfinite(numeric.to_numpy(dtype="float64")).all())
    ok = (
        not duplicate_keys
        and not missing_checks
        and not extra_checks
        and failing.empty
        and finite_numeric
    )
    return {
        "ok": ok,
        "path": str(path),
        "n_checks": int(len(frame)),
        "missing_checks": [list(item) for item in missing_checks],
        "extra_checks": [list(item) for item in extra_checks],
        "duplicate_checks": [list(item) for item in duplicate_keys],
        "failing_checks": failing.to_dict(orient="records"),
        "finite_numeric_diagnostics": finite_numeric,
    }


def validate_pipeline_outputs(
    *,
    split_dir: Path,
    agg_dir: Path | None = None,
    mapping_csv: Path | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Validate generated split/reaggregation outputs for the many-many pipeline."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    z_path = split_dir / "Z_split.csv"
    y_path = split_dir / "Y_split.csv"
    if not z_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Missing Z/Y split outputs under {split_dir}")

    Z = _numeric_frame(_read_frame(z_path))
    Y = _numeric_frame(_read_frame(y_path))
    report: dict[str, Any] = {
        "split_dir": str(split_dir),
        "agg_dir": str(agg_dir) if agg_dir is not None else "",
        "checks": [],
    }
    flow_role = str(split_dir.name).split("=", 1)[-1].upper()

    def add_check(name: str, ok: bool, **extra: Any) -> None:
        row = {"name": name, "ok": bool(ok)}
        row.update(extra)
        report["checks"].append(row)

    add_check("split_Z_square", Z.index.equals(Z.columns), n_rows=int(len(Z.index)))
    add_check("split_Y_index_matches_Z", Y.index.equals(Z.index), n_rows=int(len(Y.index)))

    x_path = split_dir / "x_split.csv"
    if not x_path.is_file():
        raise FileNotFoundError(f"Missing authoritative split output vector under {split_dir}")
    x_df = pd.read_csv(x_path, index_col=0)
    if x_df.shape[1] != 1:
        raise ValueError(f"Expected one output column in {x_path}, found {x_df.shape[1]}")
    x_observed = x_df.iloc[:, 0].astype("float64")
    x_observed.index = x_observed.index.astype(str)
    add_check("split_x_present", True, path=str(x_path))
    if flow_role == "TOTAL":
        split_info_path = split_dir / "split_info.json"
        if split_info_path.is_file():
            split_info = json.loads(split_info_path.read_text(encoding="utf-8"))
            if split_info.get("producer_column_conditioning") == "exio_technology_at_austrian_child_output":
                coherence = _validate_transformation_coherence_report(
                    split_dir.parent / "transformation_coherence_validation.csv"
                )
                coherence_extra = dict(coherence)
                coherence_ok = bool(coherence_extra.pop("ok", False))
                add_check(
                    "producer_conditioning_transformation_coherence",
                    coherence_ok,
                    **coherence_extra,
                )
    # Environmental direct accounts use the current DOM/IMP/TOTAL flow use/supply,
    # whereas aggregate_iot still uses the shared domestic activity x.
    split_flow_denominator = flow_use_total(Z, Y)

    for ratio in (
        _check_ratio_file(
            split_dir / "environmental_extensions.csv",
            numerator_col="value",
            denominator_col="monetary_output",
            ratio_col="intensity",
            rtol=rtol,
            atol=atol,
        ),
        _check_ratio_file(
            split_dir / "disaggregated_sector_value_per_physical_input.csv",
            numerator_col="monetary_output",
            denominator_col="quantity",
            ratio_col="value_per_physical_input",
            rtol=rtol,
            atol=atol,
        ),
    ):
        ratio_extra = dict(ratio)
        ratio_ok = bool(ratio_extra.pop("ok", False))
        add_check(f"ratio_file::{Path(str(ratio_extra.get('path', ''))).name}", ratio_ok, **ratio_extra)

    if agg_dir is not None or mapping_csv is not None:
        if agg_dir is None or mapping_csv is None:
            raise ValueError("agg_dir and mapping_csv must be supplied together for reaggregation validation.")
        agg_dir = Path(agg_dir)
        mapping_csv = Path(mapping_csv)
        mapping, mapping_df, edge_audit = _load_augmented_mapping(
            mapping_csv=mapping_csv, split_dir=split_dir
        )
        selected_path = split_dir / "exio_selected_sectors.json"
        selected_children = (
            _load_split_selected_children(selected_path)
            if selected_path.is_file()
            else set()
        )
        mapping_counts = mapping_df["from_sector"].astype(str).value_counts()
        missing_selected = sorted(
            child for child in selected_children if int(mapping_counts.get(child, 0)) == 0
        )
        duplicate_selected = sorted(
            child for child in selected_children if int(mapping_counts.get(child, 0)) > 1
        )
        add_check(
            "aggregation_mapping_covers_selected_split_children",
            not missing_selected and not duplicate_selected,
            n_selected=len(selected_children),
            n_missing=len(missing_selected),
            n_duplicate=len(duplicate_selected),
            missing_sample=missing_selected[:20],
            duplicate_sample=duplicate_selected[:20],
        )

        v_path = split_dir / "V_split.csv"
        m_path = split_dir / "M_split.csv"
        V = _read_frame(v_path).astype("float64") if v_path.is_file() else None
        M = _read_frame(m_path).astype("float64") if m_path.is_file() else None
        # flow_role is resolved once before split-level validation.
        recomputed = aggregate_iot(
            Z,
            Y,
            mapping,
            V=V,
            M=M,
            output=x_observed,
            flow_role=flow_role if flow_role in {"DOM", "IMP", "TOTAL"} else "TOTAL",
            keep_unmapped=True,
            create_io_system=False,
        )

        expected_outputs: list[tuple[str, pd.DataFrame | pd.Series | None, Path]] = [
            ("Z_agg", recomputed.Z, agg_dir / "Z_agg.csv"),
            ("Y_agg", recomputed.Y, agg_dir / "Y_agg.csv"),
            ("x_agg", recomputed.x, agg_dir / "x_agg.csv"),
            ("V_agg", recomputed.V, agg_dir / "V_agg.csv"),
            ("M_agg", recomputed.M, agg_dir / "M_agg.csv"),
        ]
        for name, expected, path in expected_outputs:
            if expected is None:
                continue
            if not path.exists():
                add_check(f"{name}_exists", False, path=str(path))
                continue
            observed_df = _read_frame(path)
            if isinstance(expected, pd.Series):
                observed_series = observed_df.iloc[:, 0].astype("float64")
                comparison = _compare_numeric_series(
                    observed_series, expected.astype("float64"), rtol=rtol, atol=atol
                )
            else:
                comparison = _compare_numeric_frames(
                    _numeric_frame(observed_df), expected.astype("float64"), rtol=rtol, atol=atol
                )
            add_check(f"{name}_matches_recomputed_reaggregation", **comparison)

        emitted_flow_output = flow_use_total(recomputed.Z, recomputed.Y)
        emitted_z_path = agg_dir / "Z_agg.csv"
        emitted_y_path = agg_dir / "Y_agg.csv"
        if emitted_z_path.is_file() and emitted_y_path.is_file():
            emitted_flow_output = flow_use_total(
                _numeric_frame(_read_frame(emitted_z_path)),
                _numeric_frame(_read_frame(emitted_y_path)),
            )

        split_environmental_path = split_dir / "environmental_extensions.csv"
        split_raw_environmental_path = split_dir / "exio_environmental_extensions.csv"
        if split_environmental_path.exists() or split_raw_environmental_path.exists():
            target_exists = split_environmental_path.exists()
            raw_exists = split_raw_environmental_path.exists()
            raw_required = flow_role == "DOM"
            add_check(
                "environmental_raw_provenance_paired",
                raw_exists or not raw_required,
                path=str(split_raw_environmental_path),
                required=raw_required,
                flow_role=flow_role,
            )
            expected_environmental = None
            group_cols = [
                "boundary", "denominator_role", "extension", "stressor", "unit"
            ]
            if target_exists:
                split_environmental = pd.read_csv(split_environmental_path)
                split_sector_environmental, split_final_demand_environmental = (
                    _partition_environmental_extensions_for_sector_reaggregation(
                        split_environmental
                    )
                )
                expected_sector_environmental = aggregate_environmental_extensions(
                    split_sector_environmental,
                    mapping,
                    output=split_flow_denominator,
                    keep_unmapped=True,
                    rtol=rtol,
                    atol=atol,
                )
                expected_parts = [
                    part
                    for part in (
                        expected_sector_environmental,
                        split_final_demand_environmental,
                    )
                    if part is not None and not part.empty
                ]
                expected_environmental = (
                    pd.concat(expected_parts, ignore_index=True)
                    if expected_parts
                    else None
                )
                if raw_exists:
                    if split_sector_environmental is None:
                        raise ValueError(
                            "Raw EXIO production provenance requires product-axis target environmental rows."
                        )
                    production_groups = _production_environmental_group_domain(
                        pd.read_csv(split_raw_environmental_path),
                        target_environmental_extensions=split_sector_environmental,
                    )
                else:
                    if split_sector_environmental is None:
                        production_groups = pd.DataFrame(columns=group_cols)
                    else:
                        missing_group_cols = set(group_cols) - set(
                            split_sector_environmental.columns
                        )
                        if missing_group_cols:
                            raise ValueError(
                                "Target environmental sidecar is missing production-group columns: "
                                f"{sorted(missing_group_cols)!r}"
                            )
                        production_groups = (
                            split_sector_environmental.loc[:, group_cols]
                            .copy()
                            .drop_duplicates(ignore_index=True)
                        )
            else:
                add_check(
                    "environmental_target_sidecar_paired",
                    False,
                    path=str(split_environmental_path),
                )
                production_groups = pd.DataFrame(columns=group_cols)

            observed_environmental_path = agg_dir / "environmental_extensions.csv"
            if production_groups.empty:
                add_check(
                    "environmental_production_grid_empty",
                    not observed_environmental_path.exists(),
                    path=str(observed_environmental_path),
                )
            elif not observed_environmental_path.exists():
                add_check(
                    "environmental_reaggregation_exists",
                    False,
                    path=str(observed_environmental_path),
                )
            else:
                observed_environmental = pd.read_csv(observed_environmental_path)
                scientific_rows = cast(
                    pd.DataFrame,
                    observed_environmental.loc[
                        pd.to_numeric(
                            observed_environmental["value"], errors="coerce"
                        ).notna()
                    ].copy(),
                )
                if expected_environmental is None:
                    comparison = (
                        _compare_labelled_numeric_frames(
                            pd.DataFrame(), pd.DataFrame(), rtol=rtol, atol=atol
                        )
                        if scientific_rows.empty
                        else _comparison_failure(reason="environmental expected rows are missing")
                    )
                else:
                    # Despite its historical name, this compares every keyed
                    # environmental extensive value and monetary denominator.
                    comparison = _compare_environmental_extensions(
                        scientific_rows, expected_environmental, rtol=rtol, atol=atol
                    )
                add_check(
                    "environmental_totals_match_reaggregation",
                    **comparison,
                )
                grid = _validate_environmental_grid(
                    environmental_path=observed_environmental_path,
                    raw_path=(
                        agg_dir / "exio_environmental_extensions.csv"
                        if (agg_dir / "exio_environmental_extensions.csv").is_file()
                        else None
                    ),
                    output=emitted_flow_output,
                    rtol=rtol,
                    atol=atol,
                    production_groups=production_groups,
                )
                grid_extra = dict(grid)
                grid_ok = bool(grid_extra.pop("ok", False))
                add_check(
                    "environmental_final_grid_contract",
                    grid_ok,
                    **grid_extra,
                )

        if not edge_audit.empty:
            audit_path = agg_dir / "aggregation_edge_mapping_audit.csv"
            add_check("aggregation_edge_mapping_audit_written", audit_path.exists(), path=str(audit_path), n_expected=int(len(edge_audit)))

    report["ok"] = all(bool(row.get("ok", False)) for row in report["checks"])
    return report


def _variant_dirs(base: Path, variant_dim: str) -> list[Path]:
    prefix = f"{variant_dim}="
    return sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith(prefix))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate many-many split/reaggregation pipeline outputs.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--agg-dir", type=Path, default=None)
    parser.add_argument("--mapping-csv", type=Path, default=None)
    parser.add_argument("--variant-dim", default="stk_flow")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    split_variants = _variant_dirs(args.split_dir, args.variant_dim)
    reports: list[dict[str, Any]] = []
    if split_variants:
        for split_variant in split_variants:
            agg_variant = args.agg_dir / split_variant.name if args.agg_dir is not None else None
            reports.append(
                validate_pipeline_outputs(
                    split_dir=split_variant,
                    agg_dir=agg_variant,
                    mapping_csv=args.mapping_csv,
                    rtol=float(args.rtol),
                    atol=float(args.atol),
                )
            )
    else:
        reports.append(
            validate_pipeline_outputs(
                split_dir=args.split_dir,
                agg_dir=args.agg_dir,
                mapping_csv=args.mapping_csv,
                rtol=float(args.rtol),
                atol=float(args.atol),
            )
        )

    output = {"ok": all(report["ok"] for report in reports), "reports": reports}
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote: {args.out_json.resolve()}", flush=True)
    else:
        print(text)
    return 0 if output["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

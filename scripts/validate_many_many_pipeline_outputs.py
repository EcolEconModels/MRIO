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


def _max_abs_frame_gap(actual: pd.DataFrame, expected: pd.DataFrame) -> float:
    if not actual.index.astype(str).equals(expected.index.astype(str)):
        return float("inf")
    if not actual.columns.astype(str).equals(expected.columns.astype(str)):
        return float("inf")
    actual_values = actual.astype("float64").to_numpy(dtype="float64")
    expected_values = expected.astype("float64").to_numpy(dtype="float64")
    if np.any(np.isnan(actual_values) != np.isnan(expected_values)):
        return float("inf")
    diff = actual.astype("float64") - expected.astype("float64")
    if diff.empty:
        return 0.0
    values = np.nan_to_num(np.abs(diff.to_numpy(dtype="float64")), nan=0.0)
    return float(np.max(values))


def _max_abs_series_gap(actual: pd.Series, expected: pd.Series) -> float:
    if not actual.index.astype(str).equals(expected.index.astype(str)):
        return float("inf")
    actual_values = actual.astype("float64").to_numpy(dtype="float64")
    expected_values = expected.astype("float64").to_numpy(dtype="float64")
    if np.any(np.isnan(actual_values) != np.isnan(expected_values)):
        return float("inf")
    diff = actual.astype("float64") - expected.astype("float64")
    if diff.empty:
        return 0.0
    values = np.nan_to_num(np.abs(diff.to_numpy(dtype="float64")), nan=0.0)
    return float(np.max(values))


def _max_abs_environmental_gap(actual: pd.DataFrame, expected: pd.DataFrame) -> float:
    key_cols = ["sector", "extension", "stressor", "unit"]
    value_cols = ["value", "monetary_output"]
    required = set(key_cols + value_cols)
    if not required.issubset(actual.columns) or not required.issubset(expected.columns):
        return float("inf")
    sort_keys = list(map(str, key_cols))
    actual_keyed = cast(
        pd.DataFrame, actual.loc[:, key_cols + value_cols].copy()
    ).sort_values(sort_keys).reset_index(drop=True)
    expected_keyed = cast(
        pd.DataFrame, expected.loc[:, key_cols + value_cols].copy()
    ).sort_values(sort_keys).reset_index(drop=True)
    if not actual_keyed[key_cols].equals(expected_keyed[key_cols]):
        return float("inf")
    if actual_keyed.empty:
        return 0.0
    delta = actual_keyed[value_cols].astype("float64") - expected_keyed[value_cols].astype("float64")
    return float(np.nanmax(np.abs(delta.to_numpy(dtype="float64"))))

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
    valid = den.replace(0.0, np.nan).notna() & observed.notna()
    if not bool(valid.any()):
        return {"path": str(path), "exists": True, "ok": True, "max_abs_gap": 0.0, "n_checked": 0}
    expected = num.loc[valid] / den.loc[valid]
    gap = float(np.nanmax(np.abs((observed.loc[valid] - expected).to_numpy(dtype="float64"))))
    return {
        "path": str(path),
        "exists": True,
        "ok": bool(np.isclose(gap, 0.0, rtol=rtol, atol=atol)),
        "max_abs_gap": gap,
        "n_checked": int(valid.sum()),
    }


def _validate_environmental_grid(
    *,
    environmental_path: Path,
    raw_path: Path,
    output: pd.Series,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Validate the public production-group by final-sector storage contract."""
    errors: list[str] = []
    if not environmental_path.is_file():
        return {"ok": False, "errors": [f"Missing {environmental_path}"]}
    if not raw_path.is_file():
        return {"ok": False, "errors": [f"Missing paired raw sidecar {raw_path}"]}

    observed = pd.read_csv(environmental_path)
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
    if list(observed.columns) != _ENVIRONMENTAL_GRID_COLUMNS:
        errors.append(f"environmental schema={list(observed.columns)!r}")
        return {"ok": False, "errors": errors}

    groups = _production_environmental_group_domain(
        raw, target_environmental_extensions=observed
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
    finite_value = value.notna()
    positive_output = monetary.gt(0.0)
    ratio_rows = finite_value & positive_output
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
            errors.append("positive-output intensity does not equal value/output")
    zero_output_estimates = finite_value & monetary.eq(0.0)
    if bool(zero_output_estimates.any()):
        zero_values = value.loc[zero_output_estimates]
        zero_intensities = intensity.loc[zero_output_estimates]
        if not np.allclose(
            zero_values.to_numpy(dtype="float64"),
            0.0,
            rtol=rtol,
            atol=atol,
        ):
            errors.append("zero-output supported estimates must have value=0")
        if bool(zero_intensities.notna().any()):
            errors.append("zero-output supported estimates must have intensity=NA")

    return {
        "ok": not errors,
        "errors": errors,
        "n_groups": int(len(groups)),
        "n_sectors": int(len(canonical)),
        "n_rows": int(len(observed)),
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
        flow_role = str(split_dir.name).split("=", 1)[-1].upper()
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
                gap = _max_abs_series_gap(observed_series, expected.astype("float64"))
            else:
                gap = _max_abs_frame_gap(_numeric_frame(observed_df), expected.astype("float64"))
            add_check(f"{name}_matches_recomputed_reaggregation", np.isclose(gap, 0.0, rtol=rtol, atol=atol), max_abs_gap=gap)

        split_environmental_path = split_dir / "environmental_extensions.csv"
        split_raw_environmental_path = split_dir / "exio_environmental_extensions.csv"
        if split_environmental_path.exists() or split_raw_environmental_path.exists():
            add_check(
                "environmental_raw_provenance_paired",
                split_raw_environmental_path.exists(),
                path=str(split_raw_environmental_path),
            )
            expected_environmental = None
            if split_environmental_path.exists():
                split_environmental = pd.read_csv(split_environmental_path)
                expected_environmental = aggregate_environmental_extensions(
                    split_environmental,
                    mapping,
                    output=split_flow_denominator,
                    keep_unmapped=True,
                    rtol=rtol,
                    atol=atol,
                )
            if split_raw_environmental_path.exists() and split_environmental_path.exists():
                production_groups = _production_environmental_group_domain(
                    pd.read_csv(split_raw_environmental_path),
                    target_environmental_extensions=pd.read_csv(split_environmental_path),
                )
            elif split_raw_environmental_path.exists():
                add_check(
                    "environmental_target_sidecar_paired",
                    False,
                    path=str(split_environmental_path),
                )
                production_groups = pd.DataFrame(
                    columns=["boundary", "denominator_role", "extension", "stressor", "unit"]
                )
            else:
                production_groups = pd.DataFrame(
                    columns=["boundary", "denominator_role", "extension", "stressor", "unit"]
                )
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
                        pd.to_numeric(observed_environmental["value"], errors="coerce").notna()
                    ].copy(),
                )
                if expected_environmental is None:
                    gap = 0.0 if scientific_rows.empty else float("inf")
                else:
                    gap = _max_abs_environmental_gap(
                        scientific_rows, expected_environmental
                    )
                add_check(
                    "environmental_totals_match_reaggregation",
                    np.isclose(gap, 0.0, rtol=rtol, atol=atol),
                    max_abs_gap=gap,
                )
                grid = _validate_environmental_grid(
                    environmental_path=observed_environmental_path,
                    raw_path=agg_dir / "exio_environmental_extensions.csv",
                    output=flow_use_total(recomputed.Z, recomputed.Y),
                    rtol=rtol,
                    atol=atol,
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

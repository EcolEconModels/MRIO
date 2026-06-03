#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_PLAN_FILES = [
    "component_values.csv",
    "split_shares.csv",
    "selector_residuals.csv",
    "parent_residuals.csv",
    "price_resolution.csv",
    "diagnostics_metadata.json",
]

REQUIRED_SPLIT_FILES = [
    "Z_split.csv",
    "Y_split.csv",
    "x_split.csv",
    "A_split.csv",
]

REQUIRED_REAGGREGATED_FILES = [
    "Z_reaggregated.csv",
    "Y_reaggregated.csv",
    "x_reaggregated.csv",
    "A_reaggregated.csv",
]


def _read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.astype("float64")


def _read_vector(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one-column vector at {path}, got shape={frame.shape}.")
    series = frame.iloc[:, 0].astype("float64")
    series.index = series.index.astype(str)
    return series


def _check_no_duplicates(index: pd.Index, *, what: str, errors: list[str]) -> None:
    dupes = index[index.duplicated()].astype(str).unique().tolist()
    if dupes:
        errors.append(f"{what} has duplicate labels: {dupes[:20]!r}")


def _check_finite(frame_or_series: pd.DataFrame | pd.Series, *, what: str, errors: list[str]) -> None:
    arr = frame_or_series.to_numpy(dtype="float64")
    if not np.isfinite(arr).all():
        errors.append(f"{what} contains non-finite values.")


def _check_matrix_system(prefix: Path, *, suffix: str, require_reaggregated: bool, errors: list[str]) -> None:
    z_name = f"Z_{suffix}.csv"
    y_name = f"Y_{suffix}.csv"
    x_name = f"x_{suffix}.csv"
    a_name = f"A_{suffix}.csv"

    Z = _read_matrix(prefix / z_name)
    Y = _read_matrix(prefix / y_name)
    x = _read_vector(prefix / x_name)
    A = _read_matrix(prefix / a_name)

    _check_no_duplicates(Z.index, what=f"{prefix}/{z_name} rows", errors=errors)
    _check_no_duplicates(Z.columns, what=f"{prefix}/{z_name} cols", errors=errors)
    _check_no_duplicates(Y.index, what=f"{prefix}/{y_name} rows", errors=errors)
    _check_no_duplicates(Y.columns, what=f"{prefix}/{y_name} cols", errors=errors)

    if not Z.index.equals(Z.columns):
        errors.append(f"{prefix}/{z_name} is not square with identical row/column labels.")
    if not Y.index.equals(Z.index):
        errors.append(f"{prefix}/{y_name} rows do not match {z_name} rows.")
    if not x.index.equals(Z.index):
        errors.append(f"{prefix}/{x_name} index does not match {z_name} rows.")
    if not A.index.equals(Z.index) or not A.columns.equals(Z.columns):
        errors.append(f"{prefix}/{a_name} labels do not match {z_name}.")

    _check_finite(Z, what=f"{prefix}/{z_name}", errors=errors)
    _check_finite(Y, what=f"{prefix}/{y_name}", errors=errors)
    _check_finite(x, what=f"{prefix}/{x_name}", errors=errors)
    _check_finite(A, what=f"{prefix}/{a_name}", errors=errors)

    x_calc = Z.sum(axis=1) + Y.sum(axis=1)
    max_x_err = float((x_calc - x).abs().max()) if len(x_calc) else 0.0
    if max_x_err > 1e-6:
        errors.append(f"{prefix}/{x_name} does not equal row-sum Z+Y; max abs error={max_x_err:.6g}.")

    x_safe = x.replace(0.0, np.nan)
    A_calc = Z.div(x_safe, axis=1).fillna(0.0)
    max_a_err = float((A_calc - A).abs().to_numpy().max()) if A.size else 0.0
    if max_a_err > 1e-8:
        errors.append(f"{prefix}/{a_name} does not equal Z divided by x columns; max abs error={max_a_err:.6g}.")

    if require_reaggregated and suffix == "reaggregated":
        if any(label.startswith("EB_") for label in Z.index.astype(str)):
            errors.append(f"{prefix}/{z_name} still contains EB_* split child labels after reaggregation.")


def validate_output_tree(outdir: Path, *, expected_flows: Iterable[str], require_reaggregated: bool) -> pd.DataFrame:
    errors: list[str] = []
    rows: list[dict[str, object]] = []

    plan_dir = outdir / "plan_diagnostics"
    if not plan_dir.exists():
        errors.append(f"Missing plan diagnostics directory: {plan_dir}")
    else:
        for name in REQUIRED_PLAN_FILES:
            if not (plan_dir / name).exists():
                errors.append(f"Missing plan diagnostics file: {plan_dir / name}")

        split_shares_path = plan_dir / "split_shares.csv"
        if split_shares_path.exists():
            shares = pd.read_csv(split_shares_path)
            required = {"parent_code", "child_code", "child_share"}
            if not required.issubset(shares.columns):
                errors.append(f"split_shares.csv missing columns {sorted(required - set(shares.columns))!r}")
            else:
                shares["child_share"] = pd.to_numeric(shares["child_share"], errors="coerce")
                bad = shares["child_share"].isna() | (shares["child_share"] < -1e-12)
                if bad.any():
                    errors.append("split_shares.csv has non-numeric or negative child shares.")
                sums = shares.groupby("parent_code")["child_share"].sum()
                bad_sums = sums[(sums - 1.0).abs() > 1e-8]
                if not bad_sums.empty:
                    errors.append(f"split_shares do not sum to one by parent: {bad_sums.to_dict()!r}")

        component_values_path = plan_dir / "component_values.csv"
        if component_values_path.exists():
            comp = pd.read_csv(component_values_path)
            for required_child in ["EB_C19_RESIDUAL", "EB_C20_OTHERS", "EB_D_OTHERS"]:
                if required_child not in set(comp.get("child_code", pd.Series(dtype=str)).astype(str)):
                    errors.append(f"component_values.csv missing required residual child {required_child!r}.")
            if "value_MEUR" in comp.columns:
                values = pd.to_numeric(comp["value_MEUR"], errors="coerce")
                if values.isna().any() or (values < -1e-8).any():
                    errors.append("component_values.csv has non-numeric or negative value_MEUR entries.")

        metadata_path = plan_dir / "diagnostics_metadata.json"
        if metadata_path.exists():
            try:
                json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"diagnostics_metadata.json is invalid JSON: {exc}")

    for flow in expected_flows:
        flow_dir = outdir / f"stk_flow={flow}"
        if not flow_dir.exists():
            errors.append(f"Missing flow output directory: {flow_dir}")
            continue
        for name in REQUIRED_SPLIT_FILES:
            if not (flow_dir / name).exists():
                errors.append(f"Missing split output file: {flow_dir / name}")
        if require_reaggregated:
            for name in REQUIRED_REAGGREGATED_FILES:
                if not (flow_dir / name).exists():
                    errors.append(f"Missing reaggregated output file: {flow_dir / name}")

        try:
            _check_matrix_system(flow_dir, suffix="split", require_reaggregated=False, errors=errors)
            split_z = _read_matrix(flow_dir / "Z_split.csv")
            rows.append(
                {
                    "flow": flow,
                    "kind": "split",
                    "n_rows": int(split_z.shape[0]),
                    "n_cols": int(split_z.shape[1]),
                    "total_Z": float(split_z.to_numpy().sum()),
                }
            )
        except Exception as exc:
            errors.append(f"Could not validate split system in {flow_dir}: {exc}")

        if require_reaggregated:
            try:
                _check_matrix_system(flow_dir, suffix="reaggregated", require_reaggregated=True, errors=errors)
                re_z = _read_matrix(flow_dir / "Z_reaggregated.csv")
                rows.append(
                    {
                        "flow": flow,
                        "kind": "reaggregated",
                        "n_rows": int(re_z.shape[0]),
                        "n_cols": int(re_z.shape[1]),
                        "total_Z": float(re_z.to_numpy().sum()),
                    }
                )
            except Exception as exc:
                errors.append(f"Could not validate reaggregated system in {flow_dir}: {exc}")

        # Plan diagnostics should not be duplicated in flow folders.
        duplicated_plan_file = flow_dir / "component_values.csv"
        if duplicated_plan_file.exists():
            errors.append(f"Plan diagnostics appear duplicated in flow directory: {duplicated_plan_file}")

    if not (outdir / "run_summary.csv").exists():
        errors.append(f"Missing run summary: {outdir / 'run_summary.csv'}")

    summary = pd.DataFrame(rows)
    if errors:
        raise SystemExit("Energy Balance pxp output validation failed:\n" + "\n".join(f"- {err}" for err in errors))
    return summary


def _parse_flows(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Energy Balance pxp pipeline output directory.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--flows", default="DOM,IMP,TOTAL")
    parser.add_argument("--require-reaggregated", action="store_true")
    parser.add_argument("--write-summary", default=None)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    summary = validate_output_tree(
        outdir,
        expected_flows=_parse_flows(args.flows),
        require_reaggregated=bool(args.require_reaggregated),
    )
    if args.write_summary:
        path = Path(args.write_summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
    print(f"Validated Energy Balance pxp outputs in {outdir}")
    if not summary.empty:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

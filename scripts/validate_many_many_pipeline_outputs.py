#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportGeneralTypeIssues=false
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_REPO_ROOT, _REPO_ROOT / "src"):
    _extra_text = str(_extra_path)
    if _extra_text not in sys.path:
        sys.path.insert(0, _extra_text)

from ecolecon_mrio.iot_aggregation import aggregate_iot
from ecolecon_mrio.mapping_csv import read_aggregation_mapping
from scripts.aggregate_iot_by_mapping import _augment_mapping_from_split_child_edges


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


def _find_first(base: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def _load_optional_frame(base: Path, names: Sequence[str]) -> pd.DataFrame | None:
    path = _find_first(base, names)
    if path is None:
        return None
    return _read_frame(path).astype("float64")


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
    diff = actual.astype("float64") - expected.astype("float64")
    if diff.empty:
        return 0.0
    return float(np.nanmax(np.abs(diff.to_numpy(dtype="float64"))))


def _max_abs_series_gap(actual: pd.Series, expected: pd.Series) -> float:
    if not actual.index.astype(str).equals(expected.index.astype(str)):
        return float("inf")
    diff = actual.astype("float64") - expected.astype("float64")
    if diff.empty:
        return 0.0
    return float(np.nanmax(np.abs(diff.to_numpy(dtype="float64"))))


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

    z_path = _find_first(split_dir, ("Z_split_balanced.csv", "Z_split.csv"))
    y_path = _find_first(split_dir, ("Y_split_balanced.csv", "Y_split.csv"))
    if z_path is None or y_path is None:
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

    x_expected = (Z.sum(axis=1) + Y.sum(axis=1)).astype("float64")
    x_path = _find_first(split_dir, ("x_split_balanced.csv", "x_split.csv"))
    if x_path is not None:
        x_df = pd.read_csv(x_path, index_col=0)
        x_observed = x_df.iloc[:, 0].astype("float64") if isinstance(x_df, pd.DataFrame) else pd.Series(dtype="float64")
        gap = _max_abs_series_gap(x_observed, x_expected)
        add_check("split_x_matches_Z_plus_Y", np.isclose(gap, 0.0, rtol=rtol, atol=atol), max_abs_gap=gap)

    for ratio in (
        _check_ratio_file(
            split_dir / "disaggregated_sector_co2_intensity.csv",
            numerator_col="emission",
            denominator_col="monetary_output",
            ratio_col="emission_intensity",
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
        mapping, _, edge_audit = _load_augmented_mapping(mapping_csv=mapping_csv, split_dir=split_dir)
        missing = sorted(set(map(str, Z.index)) - set(map(str, mapping.keys())))
        add_check("aggregation_mapping_covers_split_Z_index", len(missing) == 0, n_missing=len(missing), sample=missing[:20])

        V = _load_optional_frame(split_dir, ("V_split_balanced.csv", "V_split.csv"))
        M = _load_optional_frame(split_dir, ("M_split_balanced.csv", "M_split.csv"))
        recomputed = aggregate_iot(Z, Y, mapping, V=V, M=M, keep_unmapped=False, create_io_system=False)

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

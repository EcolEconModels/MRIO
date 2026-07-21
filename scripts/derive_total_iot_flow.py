#!/usr/bin/env python3
"""Construct the canonical TOTAL IOT only as DOM + IMP and validate it."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import load_eurostat_iot

_FRAME_FILES = ("Z", "Y", "V", "M")
_SPLIT_NAMES = {name: f"{name}_split.csv" for name in _FRAME_FILES}
_AGG_NAMES = {name: f"{name}_agg.csv" for name in _FRAME_FILES}
_ALIASES = {
    "Z_split.csv": "Z_split_balanced.csv",
    "Y_split.csv": "Y_split_balanced.csv",
    "x_split.csv": "x_split_balanced.csv",
    "A_split.csv": "A_split_balanced.csv",
    "V_split.csv": "V_split_balanced.csv",
    "M_split.csv": "M_split_balanced.csv",
}


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.astype("float64")


def _read_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one data column in {path}, found {frame.shape[1]}")
    series = frame.iloc[:, 0].astype("float64")
    series.index = series.index.astype(str)
    return series


def _add_frames(dom: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    rows = dom.index.union(imp.index, sort=False)
    cols = dom.columns.union(imp.columns, sort=False)
    return dom.reindex(index=rows, columns=cols, fill_value=0.0).add(
        imp.reindex(index=rows, columns=cols, fill_value=0.0), fill_value=0.0
    )


def _max_gap(left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series) -> float:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        rows = left.index.union(right.index, sort=False)
        cols = left.columns.union(right.columns, sort=False)
        diff = left.reindex(index=rows, columns=cols, fill_value=0.0) - right.reindex(
            index=rows, columns=cols, fill_value=0.0
        )
    elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
        idx = left.index.union(right.index, sort=False)
        diff = left.reindex(idx, fill_value=0.0) - right.reindex(idx, fill_value=0.0)
    else:
        raise TypeError("Both objects must be DataFrames or both must be Series")
    return float(np.nanmax(np.abs(diff.to_numpy()))) if diff.size else 0.0


def _assert_close(
    label: str,
    left: pd.DataFrame | pd.Series,
    right: pd.DataFrame | pd.Series,
    *,
    rtol: float,
    atol: float,
) -> float:
    gap = _max_gap(left, right)
    scale = max(
        float(np.nanmax(np.abs(left.to_numpy()))) if left.size else 0.0,
        float(np.nanmax(np.abs(right.to_numpy()))) if right.size else 0.0,
    )
    tolerance = max(atol, rtol * scale)
    if not np.isfinite(gap) or gap > tolerance:
        raise ValueError(f"{label} failed: max_abs_gap={gap:.12g}, tolerance={tolerance:.12g}")
    return gap


def _manifest_children(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    raise TypeError(f"Unexpected exio_selected_sectors value: {type(value).__name__}")


def _load_equal_manifests(dom_dir: Path, imp_dir: Path) -> dict[str, object]:
    paths = [dom_dir / "exio_selected_sectors.json", imp_dir / "exio_selected_sectors.json"]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("DOM and IMP must both contain exio_selected_sectors.json")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    topology = {str(parent): set(_manifest_children(children)) for parent, children in manifests[0].items()}
    other = {str(parent): set(_manifest_children(children)) for parent, children in manifests[1].items()}
    if topology != other:
        raise ValueError("DOM and IMP split-child topology differs; additive TOTAL is not well-defined")
    return manifests[0]


def _child_to_parent(manifest: Mapping[str, object], source_codes: Iterable[str]) -> dict[str, str]:
    mapping = {str(code): str(code) for code in source_codes}
    for parent, children in manifest.items():
        for child in _manifest_children(children):
            previous = mapping.get(child)
            if previous is not None and previous != str(parent) and child in set(map(str, source_codes)):
                raise ValueError(f"Split child {child} collides with unsplit source code {previous}")
            mapping[child] = str(parent)
    return mapping


def _aggregate_square(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    row_groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.index])
    col_groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.columns])
    row_agg = frame.groupby(row_groups, sort=False).sum()
    col_agg = row_agg.T.groupby(col_groups, sort=False).sum().T
    # Grouping a DataFrame on explicit row and column labels preserves a DataFrame at runtime.
    return cast(pd.DataFrame, col_agg.reindex(index=source_codes, columns=source_codes, fill_value=0.0))


def _aggregate_rows(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.index])
    # Grouping a DataFrame on explicit row labels preserves a DataFrame at runtime.
    return cast(pd.DataFrame, frame.groupby(groups, sort=False).sum().reindex(index=source_codes, fill_value=0.0))


def _aggregate_columns(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.columns])
    # Grouping a DataFrame on explicit column labels preserves a DataFrame at runtime.
    return cast(pd.DataFrame, frame.T.groupby(groups, sort=False).sum().T.reindex(columns=source_codes, fill_value=0.0))


def _write_frame(frame: pd.DataFrame, total_dir: Path, filename: str) -> None:
    frame.to_csv(total_dir / filename)
    alias = _ALIASES.get(filename)
    if alias:
        frame.to_csv(total_dir / alias)


def derive_total_flow(
    outdir: Path,
    *,
    eurostat_dataset: str,
    eurostat_filters: Mapping[str, object],
    eurostat_row_dim: str,
    eurostat_col_dim: str,
    eurostat_stk_flow_dim: str = "stk_flow",
    atol: float = 1e-6,
    rtol: float = 1e-9,
) -> None:
    dom_dir = outdir / f"{eurostat_stk_flow_dim}=DOM"
    imp_dir = outdir / f"{eurostat_stk_flow_dim}=IMP"
    total_dir = outdir / f"{eurostat_stk_flow_dim}=TOTAL"
    for directory in (dom_dir, imp_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required primitive flow directory not found: {directory}")

    manifest = _load_equal_manifests(dom_dir, imp_dir)
    if total_dir.exists():
        shutil.rmtree(total_dir)
    total_dir.mkdir(parents=True)

    derived: dict[str, pd.DataFrame] = {}
    for obj, filename in _SPLIT_NAMES.items():
        dom_path, imp_path = dom_dir / filename, imp_dir / filename
        if dom_path.is_file() != imp_path.is_file():
            raise ValueError(f"{filename} exists for only one of DOM/IMP")
        if dom_path.is_file():
            derived[obj] = _add_frames(_read_frame(dom_path), _read_frame(imp_path))
            _write_frame(derived[obj], total_dir, filename)
    if "Z" not in derived or "Y" not in derived:
        raise FileNotFoundError("DOM and IMP must both contain Z_split.csv and Y_split.csv")

    Z, Y = derived["Z"], derived["Y"]
    sectors = Z.index.union(Z.columns, sort=False).union(Y.index, sort=False)
    Z = Z.reindex(index=sectors, columns=sectors, fill_value=0.0)
    Y = Y.reindex(index=sectors, fill_value=0.0)
    _write_frame(Z, total_dir, "Z_split.csv")
    _write_frame(Y, total_dir, "Y_split.csv")
    x = Z.sum(axis=1).add(Y.sum(axis=1), fill_value=0.0).astype("float64")
    x.name = "x"
    x.to_csv(total_dir / "x_split.csv", header=True)
    x.to_csv(total_dir / _ALIASES["x_split.csv"], header=True)
    A = Z.divide(x.replace(0.0, np.nan), axis="columns").fillna(0.0)
    _write_frame(A, total_dir, "A_split.csv")

    total_filters = dict(eurostat_filters)
    total_filters[eurostat_stk_flow_dim] = "TOTAL"
    reference = load_eurostat_iot(
        dataset=eurostat_dataset,
        filters=total_filters,
        row_dim=eurostat_row_dim,
        col_dim=eurostat_col_dim,
        storage_folder=default_data_root() / "eurostat",
    )
    source_codes = reference.Z.index.astype(str)
    child_parent = _child_to_parent(manifest, source_codes)
    diagnostics: list[dict[str, object]] = []
    z_gap = _assert_close("TOTAL Z reconstruction", _aggregate_square(Z, child_parent, source_codes), reference.Z, rtol=rtol, atol=atol)
    y_gap = _assert_close("TOTAL Y reconstruction", _aggregate_rows(Y, child_parent, source_codes), reference.Y, rtol=rtol, atol=atol)
    diagnostics.extend([
        {"stage": "split_to_eurostat", "object": "Z", "max_abs_gap": z_gap, "ok": True},
        {"stage": "split_to_eurostat", "object": "Y", "max_abs_gap": y_gap, "ok": True},
    ])
    if "V" in derived and reference.V is not None:
        v_gap = _assert_close("TOTAL V reconstruction", _aggregate_columns(derived["V"], child_parent, source_codes), reference.V, rtol=rtol, atol=atol)
        diagnostics.append({"stage": "split_to_eurostat", "object": "V", "max_abs_gap": v_gap, "ok": True})
    if "M" in derived and reference.M is not None:
        m_gap = _assert_close("TOTAL M reconstruction", derived["M"], reference.M, rtol=rtol, atol=atol)
        diagnostics.append({"stage": "split_to_eurostat", "object": "M", "max_abs_gap": m_gap, "ok": True})

    (total_dir / "exio_selected_sectors.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    info = {
        "flow_construction": "derived_dom_plus_imp",
        "source_flows": ["DOM", "IMP"],
        "flow_identity_rtol": rtol,
        "flow_identity_atol": atol,
        "is_reconstruction_validated": True,
        "canonical_total_sidecars": "core monetary matrices, split topology, metadata, and measured reconstruction diagnostics only",
    }
    (total_dir / "split_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(diagnostics).to_csv(outdir / "derived_total_flow_identity.csv", index=False)


def validate_additive_outputs(outdir: Path, *, suffix: str, atol: float, rtol: float) -> None:
    dirs = {flow: outdir / f"stk_flow={flow}" for flow in ("DOM", "IMP", "TOTAL")}
    for directory in dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(f"Required flow directory not found: {directory}")
    diagnostics: list[dict[str, object]] = []
    names = _SPLIT_NAMES if suffix == "split" else _AGG_NAMES
    for obj in _FRAME_FILES:
        filename = names[obj]
        paths = {flow: directory / filename for flow, directory in dirs.items()}
        if not any(path.is_file() for path in paths.values()):
            continue
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"{filename} must exist for DOM, IMP, and TOTAL")
        expected = _add_frames(_read_frame(paths["DOM"]), _read_frame(paths["IMP"]))
        actual = _read_frame(paths["TOTAL"])
        gap = _assert_close(f"{suffix} {obj} DOM+IMP=TOTAL", expected, actual, rtol=rtol, atol=atol)
        diagnostics.append({"stage": suffix, "object": obj, "max_abs_gap": gap, "ok": True})
    x_name = "x_split.csv" if suffix == "split" else "x_agg.csv"
    x_paths = {flow: directory / x_name for flow, directory in dirs.items()}
    if all(path.is_file() for path in x_paths.values()):
        expected_x = _read_series(x_paths["DOM"]).add(_read_series(x_paths["IMP"]), fill_value=0.0)
        gap = _assert_close(f"{suffix} x DOM+IMP=TOTAL", expected_x, _read_series(x_paths["TOTAL"]), rtol=rtol, atol=atol)
        diagnostics.append({"stage": suffix, "object": "x", "max_abs_gap": gap, "ok": True})
    pd.DataFrame(diagnostics).to_csv(outdir / f"{suffix}_flow_identity.csv", index=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--validate-only", choices=("split", "agg"))
    parser.add_argument("--eurostat-dataset")
    parser.add_argument("--eurostat-filters")
    parser.add_argument("--eurostat-row-dim")
    parser.add_argument("--eurostat-col-dim")
    parser.add_argument("--eurostat-stk-flow-dim", default="stk_flow")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.validate_only:
        validate_additive_outputs(args.outdir, suffix=args.validate_only, rtol=args.rtol, atol=args.atol)
        return 0
    required = (args.eurostat_dataset, args.eurostat_filters, args.eurostat_row_dim, args.eurostat_col_dim)
    if not all(required):
        parser.error("derivation requires Eurostat dataset, filters, row dimension, and column dimension")
    derive_total_flow(
        args.outdir,
        eurostat_dataset=str(args.eurostat_dataset),
        eurostat_filters=json.loads(str(args.eurostat_filters)),
        eurostat_row_dim=str(args.eurostat_row_dim),
        eurostat_col_dim=str(args.eurostat_col_dim),
        eurostat_stk_flow_dim=str(args.eurostat_stk_flow_dim),
        rtol=args.rtol,
        atol=args.atol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

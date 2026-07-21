#!/usr/bin/env python3
"""Construct canonical additive flows and validate the single IO contract."""
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
from ecolecon_mrio.iot_flow_semantics import (
    flow_use_total,
    refresh_supplementary_rows,
    technical_coefficients,
)

_FRAME_FILES = ("Z", "Y", "V", "M")
_SPLIT_NAMES = {name: f"{name}_split.csv" for name in _FRAME_FILES}
_AGG_NAMES = {name: f"{name}_agg.csv" for name in _FRAME_FILES}
_ALIASES = {name: name.replace("_split.csv", "_split_balanced.csv") for name in (
    "Z_split.csv", "Y_split.csv", "x_split.csv", "A_split.csv", "V_split.csv", "M_split.csv"
)}


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


def _write_series(series: pd.Series, path: Path) -> None:
    out = series.astype("float64").copy()
    out.name = out.name or "value"
    out.to_csv(path, header=True)


def _write_frame(frame: pd.DataFrame, directory: Path, filename: str) -> None:
    frame.to_csv(directory / filename)
    alias = _ALIASES.get(filename)
    if alias:
        frame.to_csv(directory / alias)


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
        diff = left.reindex(index=rows, columns=cols, fill_value=0.0) - right.reindex(index=rows, columns=cols, fill_value=0.0)
    elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
        idx = left.index.union(right.index, sort=False)
        diff = left.reindex(idx, fill_value=0.0) - right.reindex(idx, fill_value=0.0)
    else:
        raise TypeError("Both objects must be DataFrames or both must be Series")
    return float(np.nanmax(np.abs(diff.to_numpy()))) if diff.size else 0.0


def _assert_close(label: str, left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series, *, rtol: float, atol: float) -> float:
    gap = _max_gap(left, right)
    scale = max(float(np.nanmax(np.abs(left.to_numpy()))) if left.size else 0.0, float(np.nanmax(np.abs(right.to_numpy()))) if right.size else 0.0)
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
    source_set = set(map(str, source_codes))
    mapping = {code: code for code in source_set}
    for parent, children in manifest.items():
        for child in _manifest_children(children):
            previous = mapping.get(child)
            if previous is not None and previous != str(parent) and child in source_set:
                raise ValueError(f"Split child {child} collides with unsplit source code {previous}")
            mapping[child] = str(parent)
    return mapping


def _aggregate_square(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    rows = pd.Index([child_parent.get(str(code), str(code)) for code in frame.index])
    cols = pd.Index([child_parent.get(str(code), str(code)) for code in frame.columns])
    row_agg = cast(pd.DataFrame, frame.groupby(rows, sort=False).sum())
    col_agg = cast(pd.DataFrame, row_agg.T.groupby(cols, sort=False).sum().T)
    return cast(pd.DataFrame, col_agg.reindex(index=source_codes, columns=source_codes, fill_value=0.0))


def _aggregate_rows(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.index])
    return cast(pd.DataFrame, frame.groupby(groups, sort=False).sum().reindex(index=source_codes, fill_value=0.0))


def _aggregate_columns(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.columns])
    return cast(pd.DataFrame, frame.T.groupby(groups, sort=False).sum().T.reindex(columns=source_codes, fill_value=0.0))


def _apply_contract(dirs: Mapping[str, Path], *, suffix: str, write: bool) -> list[dict[str, object]]:
    z_name = "Z_split.csv" if suffix == "split" else "Z_agg.csv"
    y_name = "Y_split.csv" if suffix == "split" else "Y_agg.csv"
    x_name = "x_split.csv" if suffix == "split" else "x_agg.csv"
    a_name = "A_split.csv" if suffix == "split" else "A_agg.csv"
    use_name = "use_total_split.csv" if suffix == "split" else "use_total_agg.csv"
    v_name = "V_split.csv" if suffix == "split" else "V_agg.csv"
    m_name = "M_split.csv" if suffix == "split" else "M_agg.csv"
    Z = {flow: _read_frame(path / z_name) for flow, path in dirs.items()}
    Y = {flow: _read_frame(path / y_name) for flow, path in dirs.items()}
    use = {flow: flow_use_total(Z[flow], Y[flow]) for flow in dirs}
    output = use["DOM"].copy().rename("x")
    diagnostics: list[dict[str, object]] = []
    for flow, directory in dirs.items():
        expected_A = technical_coefficients(Z[flow], output)
        if write:
            _write_series(use[flow], directory / use_name)
            _write_series(output, directory / x_name)
            if suffix == "split":
                _write_series(output, directory / _ALIASES[x_name])
            _write_frame(expected_A, directory, a_name)
            V = _read_frame(directory / v_name) if (directory / v_name).is_file() else None
            M = _read_frame(directory / m_name) if (directory / m_name).is_file() else None
            V, M = refresh_supplementary_rows(V, M, Z=Z[flow], Y=Y[flow], output=output, flow_role=flow)  # type: ignore[arg-type]
            if V is not None:
                _write_frame(V, directory, v_name)
            if M is not None:
                _write_frame(M, directory, m_name)
        else:
            _assert_close(f"{suffix} {flow} use_total", _read_series(directory / use_name), use[flow], rtol=1e-9, atol=1e-6)
            _assert_close(f"{suffix} {flow} output", _read_series(directory / x_name), output, rtol=1e-9, atol=1e-6)
            _assert_close(f"{suffix} {flow} A", _read_frame(directory / a_name), expected_A, rtol=1e-9, atol=1e-6)
    diagnostics.extend([
        {"stage": suffix, "object": "use_total", "contract": "atomic final demand; TOTAL=DOM+IMP"},
        {"stage": suffix, "object": "x", "contract": "shared domestic product output"},
        {"stage": suffix, "object": "A", "contract": "A_flow=Z_flow/domestic output"},
    ])
    return diagnostics


def derive_total_flow(outdir: Path, *, eurostat_dataset: str, eurostat_filters: Mapping[str, object], eurostat_row_dim: str, eurostat_col_dim: str, eurostat_stk_flow_dim: str = "stk_flow", atol: float = 1e-6, rtol: float = 1e-9) -> None:
    dirs = {flow: outdir / f"{eurostat_stk_flow_dim}={flow}" for flow in ("DOM", "IMP", "TOTAL")}
    for flow in ("DOM", "IMP"):
        if not dirs[flow].is_dir():
            raise FileNotFoundError(f"Required primitive flow directory not found: {dirs[flow]}")
    manifest = _load_equal_manifests(dirs["DOM"], dirs["IMP"])
    if dirs["TOTAL"].exists():
        shutil.rmtree(dirs["TOTAL"])
    dirs["TOTAL"].mkdir(parents=True)
    derived: dict[str, pd.DataFrame] = {}
    for obj, filename in _SPLIT_NAMES.items():
        dom_path, imp_path = dirs["DOM"] / filename, dirs["IMP"] / filename
        if dom_path.is_file() != imp_path.is_file():
            raise ValueError(f"{filename} exists for only one of DOM/IMP")
        if dom_path.is_file():
            derived[obj] = _add_frames(_read_frame(dom_path), _read_frame(imp_path))
            _write_frame(derived[obj], dirs["TOTAL"], filename)
    if "Z" not in derived or "Y" not in derived:
        raise FileNotFoundError("DOM and IMP must both contain Z_split.csv and Y_split.csv")
    sectors = derived["Z"].index.union(derived["Z"].columns, sort=False).union(derived["Y"].index, sort=False)
    derived["Z"] = derived["Z"].reindex(index=sectors, columns=sectors, fill_value=0.0)
    derived["Y"] = derived["Y"].reindex(index=sectors, fill_value=0.0)
    _write_frame(derived["Z"], dirs["TOTAL"], "Z_split.csv")
    _write_frame(derived["Y"], dirs["TOTAL"], "Y_split.csv")
    semantic_diagnostics = _apply_contract(dirs, suffix="split", write=True)

    total_filters = dict(eurostat_filters)
    total_filters[eurostat_stk_flow_dim] = "TOTAL"
    reference = load_eurostat_iot(dataset=eurostat_dataset, filters=total_filters, row_dim=eurostat_row_dim, col_dim=eurostat_col_dim, storage_folder=default_data_root() / "eurostat")
    source_codes = reference.Z.index.astype(str)
    child_parent = _child_to_parent(manifest, source_codes)
    diagnostics: list[dict[str, object]] = [
        {"stage": "split_to_eurostat", "object": "Z", "max_abs_gap": _assert_close("TOTAL Z reconstruction", _aggregate_square(derived["Z"], child_parent, source_codes), reference.Z, rtol=rtol, atol=atol), "ok": True},
        {"stage": "split_to_eurostat", "object": "Y", "max_abs_gap": _assert_close("TOTAL Y reconstruction", _aggregate_rows(derived["Y"], child_parent, source_codes), reference.Y, rtol=rtol, atol=atol), "ok": True},
    ]
    if "V" in derived and reference.V is not None:
        diagnostics.append({"stage": "split_to_eurostat", "object": "V", "max_abs_gap": _assert_close("TOTAL V reconstruction", _aggregate_columns(_read_frame(dirs["TOTAL"] / "V_split.csv"), child_parent, source_codes), reference.V, rtol=rtol, atol=atol), "ok": True})
    if "M" in derived and reference.M is not None:
        diagnostics.append({"stage": "split_to_eurostat", "object": "M", "max_abs_gap": _assert_close("TOTAL M reconstruction", _read_frame(dirs["TOTAL"] / "M_split.csv"), reference.M, rtol=rtol, atol=atol), "ok": True})
    (dirs["TOTAL"] / "exio_selected_sectors.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dirs["TOTAL"] / "split_info.json").write_text(json.dumps({
        "flow_construction": "derived_dom_plus_imp",
        "source_flows": ["DOM", "IMP"],
        "x_semantics": "domestic product output from Z plus atomic final demand",
        "use_total_semantics": "flow-specific Z plus atomic final demand",
        "A_semantics": "flow-specific Z divided by shared domestic output",
        "supplementary_semantics": "TOTAL complete identities; DOM/IMP partial-table identities explicitly blanked where not meaningful",
        "is_reconstruction_validated": True,
    }, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(diagnostics + semantic_diagnostics).to_csv(outdir / "derived_total_flow_identity.csv", index=False)


def validate_additive_outputs(outdir: Path, *, suffix: str, atol: float, rtol: float) -> None:
    dirs = {flow: outdir / f"stk_flow={flow}" for flow in ("DOM", "IMP", "TOTAL")}
    for directory in dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(f"Required flow directory not found: {directory}")
    names = _SPLIT_NAMES if suffix == "split" else _AGG_NAMES
    diagnostics: list[dict[str, object]] = []
    for obj in _FRAME_FILES:
        filename = names[obj]
        paths = {flow: directory / filename for flow, directory in dirs.items()}
        if not any(path.is_file() for path in paths.values()):
            continue
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"{filename} must exist for DOM, IMP, and TOTAL")
        expected = _add_frames(_read_frame(paths["DOM"]), _read_frame(paths["IMP"]))
        diagnostics.append({"stage": suffix, "object": obj, "max_abs_gap": _assert_close(f"{suffix} {obj} DOM+IMP=TOTAL", expected, _read_frame(paths["TOTAL"]), rtol=rtol, atol=atol), "ok": True})
    diagnostics.extend(_apply_contract(dirs, suffix=suffix, write=False))
    use_name = "use_total_split.csv" if suffix == "split" else "use_total_agg.csv"
    use = {flow: _read_series(directory / use_name) for flow, directory in dirs.items()}
    diagnostics.append({"stage": suffix, "object": "use_total", "max_abs_gap": _assert_close(f"{suffix} use TOTAL=DOM+IMP", use["DOM"].add(use["IMP"], fill_value=0.0), use["TOTAL"], rtol=rtol, atol=atol), "ok": True})
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
    derive_total_flow(args.outdir, eurostat_dataset=str(args.eurostat_dataset), eurostat_filters=json.loads(str(args.eurostat_filters)), eurostat_row_dim=str(args.eurostat_row_dim), eurostat_col_dim=str(args.eurostat_col_dim), eurostat_stk_flow_dim=str(args.eurostat_stk_flow_dim), rtol=args.rtol, atol=args.atol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

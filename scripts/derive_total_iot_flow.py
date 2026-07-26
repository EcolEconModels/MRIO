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
from ecolecon_mrio.eurostat_iot import ParsedEurostatIOT, load_eurostat_iot
from ecolecon_mrio.iot_flow_semantics import (
    derived_final_demand_control_columns,
    enforce_final_demand_display_identities,
    flow_use_total,
    refresh_supplementary_rows,
    rounded_source_control_diagnostic,
    source_atomic_final_demand,
    technical_coefficients,
)
from ecolecon_mrio.iot_identities import find_imports_vector, supplementary_accounting_identity_kind

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
    if type(left) is not type(right):
        raise TypeError("Both objects must be DataFrames or both must be Series")
    if isinstance(left, pd.DataFrame):
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            return float("inf")
        if not left.isna().equals(right.isna()):
            return float("inf")
        values = (left - right).to_numpy(dtype="float64")
    else:
        if not left.index.equals(right.index):
            return float("inf")
        if not left.isna().equals(right.isna()):
            return float("inf")
        values = (left - right).to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def _assert_close(label: str, left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series, *, rtol: float, atol: float) -> float:
    gap = _max_gap(left, right)
    scale = max(float(np.nanmax(np.abs(left.to_numpy()))) if left.size else 0.0, float(np.nanmax(np.abs(right.to_numpy()))) if right.size else 0.0)
    tolerance = max(atol, rtol * scale)
    if not np.isfinite(gap) or gap > tolerance:
        raise ValueError(f"{label} failed: max_abs_gap={gap:.12g}, tolerance={tolerance:.12g}")
    return gap


_ROUNDED_Y_CONTROLS = ("P3", "P5M", "P5", "TFU", "TU")


def _resolve_source_rounding_quantum(
    references: Mapping[str, ParsedEurostatIOT],
    override: float | None,
) -> float:
    if override is not None:
        if not np.isfinite(override) or override <= 0.0:
            raise ValueError("--source-rounding-quantum must be finite and positive")
        return float(override)
    inferred = {
        float(reference.meta["source_rounding_quantum"])
        for reference in references.values()
        if reference.meta.get("source_rounding_quantum") is not None
    }
    if len(inferred) != 1:
        raise ValueError(
            "Could not establish one source rounding quantum across DOM/IMP/TOTAL; "
            f"inferred={sorted(inferred)!r}. Pass --source-rounding-quantum explicitly."
        )
    quantum = inferred.pop()
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError(f"Invalid inferred source rounding quantum: {quantum!r}")
    return quantum


def _derived_y_control(
    reference: ParsedEurostatIOT,
    identity: str,
) -> tuple[pd.Series, pd.Series]:
    Y = reference.Y.astype("float64")
    if identity == "P3":
        detail = ["P3_S13", "P3_S14", "P3_S15"]
        if not set(detail).issubset(Y.columns):
            raise ValueError("P3 published-control validation requires its complete detail partition")
        values = Y.loc[:, detail].sum(axis=1).astype("float64")
        counts = Y.loc[:, detail].notna().sum(axis=1).astype("int64")
        return values, counts
    if identity == "P5":
        detail = ["P51G", "P52", "P53"]
        if not set(detail).issubset(Y.columns):
            raise ValueError("P5 published-control validation requires its complete detail partition")
        values = Y.loc[:, detail].sum(axis=1).astype("float64")
        counts = Y.loc[:, detail].notna().sum(axis=1).astype("int64")
        return values, counts
    if identity == "P5M":
        detail = ["P52", "P53"]
        if not set(detail).issubset(Y.columns):
            raise ValueError("P5M published-control validation requires P52 and P53")
        values = Y.loc[:, detail].sum(axis=1).astype("float64")
        counts = Y.loc[:, detail].notna().sum(axis=1).astype("int64")
        return values, counts

    atomic = source_atomic_final_demand(Y)
    values = atomic.sum(axis=1).astype("float64")
    counts = atomic.notna().sum(axis=1).astype("int64")
    if identity == "TFU":
        return values, counts
    if identity == "TU":
        values = reference.Z.sum(axis=1).add(values, fill_value=0.0).astype("float64")
        counts = reference.Z.notna().sum(axis=1).add(counts, fill_value=0).astype("int64")
        return values, counts
    raise ValueError(f"Unsupported rounded Y control: {identity}")


def _derived_source_output(reference: ParsedEurostatIOT) -> tuple[pd.Series, pd.Series]:
    """Derive source output and term counts without exact control validation."""
    atomic = source_atomic_final_demand(reference.Y)
    values = reference.Z.sum(axis=1).add(atomic.sum(axis=1), fill_value=0.0).astype("float64")
    counts = reference.Z.notna().sum(axis=1).add(
        atomic.notna().sum(axis=1), fill_value=0
    ).astype("int64")
    return values, counts


def _validate_rounded_series(
    *,
    flow: str,
    identity: str,
    published: pd.Series,
    derived: pd.Series,
    component_counts: pd.Series,
    source_rounding_quantum: float,
    rtol: float,
    atol: float,
) -> list[dict[str, object]]:
    labels = published.index
    if not derived.index.equals(labels) or not component_counts.index.equals(labels):
        raise ValueError(f"{flow} {identity}: rounded-control index mismatch")
    diagnostics: list[dict[str, object]] = []
    for code in labels.astype(str):
        diagnostic = rounded_source_control_diagnostic(
            identity=identity,
            published=float(published.loc[code]),
            derived=float(derived.loc[code]),
            component_term_count=int(component_counts.loc[code]),
            source_rounding_quantum=source_rounding_quantum,
            rtol=rtol,
            atol=atol,
        )
        diagnostic.update({"stage": "source_control", "flow": flow, "product_code": code})
        diagnostics.append(diagnostic)
    return diagnostics


def _validate_y_source_reconstruction(
    *,
    flow: str,
    reconstructed: pd.DataFrame,
    reference: ParsedEurostatIOT,
    source_rounding_quantum: float,
    rtol: float,
    atol: float,
) -> list[dict[str, object]]:
    common_controls = [
        code for code in _ROUNDED_Y_CONTROLS
        if code in reference.Y.columns and code in reconstructed.columns
    ]
    ordinary_columns = [code for code in reference.Y.columns if str(code) not in set(common_controls)]
    _assert_close(
        f"{flow} ordinary Y reconstruction",
        reconstructed.loc[:, ordinary_columns],
        reference.Y.loc[:, ordinary_columns],
        rtol=rtol,
        atol=atol,
    )
    diagnostics: list[dict[str, object]] = []
    for identity in common_controls:
        derived, counts = _derived_y_control(reference, identity)
        _assert_close(
            f"{flow} generated {identity} reconstruction",
            reconstructed.loc[:, identity],
            derived,
            rtol=rtol,
            atol=atol,
        )
        diagnostics.extend(
            _validate_rounded_series(
                flow=flow,
                identity=identity,
                published=reference.Y.loc[:, identity].astype("float64"),
                derived=derived,
                component_counts=counts,
                source_rounding_quantum=source_rounding_quantum,
                rtol=rtol,
                atol=atol,
            )
        )
    return diagnostics


def _published_v_row(reference: ParsedEurostatIOT, identity_kind: str) -> tuple[str, pd.Series] | None:
    if reference.V is None:
        return None
    rows = [
        str(code) for code in reference.V.index
        if supplementary_accounting_identity_kind(str(code)) == identity_kind
    ]
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError(f"Expected at most one {identity_kind} source row, found {rows!r}")
    return rows[0], reference.V.loc[rows[0]].astype("float64")

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
    expected = set(source_codes.astype(str))
    if set(col_agg.index.astype(str)) != expected or set(col_agg.columns.astype(str)) != expected:
        raise ValueError(
            "Split reconstruction produced missing/orphan parent sectors: "
            f"rows_extra={sorted(set(col_agg.index.astype(str)) - expected)!r}, "
            f"rows_missing={sorted(expected - set(col_agg.index.astype(str)))!r}, "
            f"cols_extra={sorted(set(col_agg.columns.astype(str)) - expected)!r}, "
            f"cols_missing={sorted(expected - set(col_agg.columns.astype(str)))!r}."
        )
    return cast(pd.DataFrame, col_agg.reindex(index=source_codes, columns=source_codes))


def _aggregate_rows(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.index])
    grouped = cast(pd.DataFrame, frame.groupby(groups, sort=False).sum())
    expected = set(source_codes.astype(str))
    if set(grouped.index.astype(str)) != expected:
        raise ValueError(
            "Row reconstruction produced missing/orphan parent sectors: "
            f"extra={sorted(set(grouped.index.astype(str)) - expected)!r}, "
            f"missing={sorted(expected - set(grouped.index.astype(str)))!r}."
        )
    return cast(pd.DataFrame, grouped.reindex(index=source_codes))


def _aggregate_columns(frame: pd.DataFrame, child_parent: Mapping[str, str], source_codes: pd.Index) -> pd.DataFrame:
    groups = pd.Index([child_parent.get(str(code), str(code)) for code in frame.columns])
    grouped = cast(pd.DataFrame, frame.T.groupby(groups, sort=False).sum().T)
    expected = set(source_codes.astype(str))
    if set(grouped.columns.astype(str)) != expected:
        raise ValueError(
            "Column reconstruction produced missing/orphan parent sectors: "
            f"extra={sorted(set(grouped.columns.astype(str)) - expected)!r}, "
            f"missing={sorted(expected - set(grouped.columns.astype(str)))!r}."
        )
    return cast(pd.DataFrame, grouped.reindex(columns=source_codes))


_V_OUTPUT_PROXY_ROWS = {"B1G", "D1", "D11", "D21X31", "D29X39", "P51C", "B2A3G", "B2A3N"}
_V_TOTAL_INPUT_PROXY_ROWS = {"P2_ADJ"}
_V_IMP_INPUT_PROXY_ROWS = {"P7", "P7_B0", "P7_D0", "P7_U2", "P7_U3"}


def _rounded_v_formula(
    *, flow: str, row: str, published: pd.Series, formula: pd.Series,
    term_counts: pd.Series, quantum: float, rtol: float, atol: float,
) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for parent in published.index.astype(str):
        diag = rounded_source_control_diagnostic(
            identity=f"V:{row}", published=float(published.loc[parent]),
            derived=float(formula.loc[parent]), component_term_count=int(term_counts.loc[parent]),
            source_rounding_quantum=quantum, rtol=rtol, atol=atol,
        )
        diag.update({"stage": "V_source_control", "flow": flow, "row": row, "parent": parent})
        diagnostics.append(diag)
    return diagnostics


def _construct_canonical_v(
    *, flow: str, reference: ParsedEurostatIOT, child_parent: Mapping[str, str],
    Z: pd.DataFrame, x_domestic: pd.Series, Z_total: pd.DataFrame, Z_imp: pd.DataFrame,
    source_total_Z: pd.DataFrame, imp_product_proxy: pd.Series,
    quantum: float, rtol: float, atol: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[dict[str, object]]]:
    if reference.V is None:
        return None, None, []
    source = reference.V.copy().astype("float64")
    source.index = source.index.astype(str); source.columns = source.columns.astype(str)
    children = pd.Index(Z.columns.astype(str))
    V = pd.DataFrame(0.0, index=source.index, columns=children, dtype="float64")
    diagnostics: list[dict[str, object]] = []
    parent_children: dict[str, list[str]] = {str(parent): [] for parent in source.columns}
    for child in children:
        parent = child_parent.get(str(child), str(child))
        if parent in parent_children:
            parent_children[parent].append(str(child))
    dom_output = x_domestic.reindex(children).astype("float64")
    total_inputs = Z_total.sum(axis=0).reindex(children).astype("float64")
    imp_inputs = Z_imp.sum(axis=0).reindex(children).astype("float64")
    imp_products = imp_product_proxy.reindex(children).astype("float64")
    flow_inputs = Z.sum(axis=0).reindex(children).astype("float64")
    source_z_inputs = reference.Z.sum(axis=0).astype("float64")
    source_z_counts = reference.Z.notna().sum(axis=0).astype("int64")
    for row in source.index.astype(str):
        published = source.loc[row].astype("float64")
        if row == "CPA_TOTAL":
            values, formula, counts = flow_inputs, source_z_inputs, source_z_counts
        elif row == "P1":
            # P1 is source-controlled; canonical x was constructed from DOM P1.
            values, formula, counts = dom_output, None, None
        elif row in _V_OUTPUT_PROXY_ROWS:
            values = dom_output; formula = None; counts = None
        elif row in _V_TOTAL_INPUT_PROXY_ROWS:
            values = total_inputs; formula = None; counts = None
        elif row in _V_IMP_INPUT_PROXY_ROWS:
            values = imp_products; formula = None; counts = None
        elif row == "IMP":
            values = imp_inputs; formula = None; counts = None
        elif row == "TS_BP":
            if flow != "TOTAL" or "P7" not in V.index:
                raise ValueError(f"{flow} TS_BP requires TOTAL P7 before derivation")
            formula = source.loc["P1"].astype("float64") + source.loc["P7"].astype("float64")
            counts = pd.Series(2, index=source.columns, dtype="int64")
            diagnostics.extend(_rounded_v_formula(flow=flow, row=row, published=published, formula=formula, term_counts=counts, quantum=quantum, rtol=rtol, atol=atol))
            V.loc[row, children] = x_domestic.reindex(children).to_numpy() + V.loc["P7", children].to_numpy()
            reconstructed = _aggregate_columns(V.loc[[row]], child_parent, source.columns).loc[row]
            _assert_close(f"{flow} TS_BP parent reconstruction", reconstructed, formula, rtol=0.0, atol=1e-9)
            diagnostics.append({"flow": flow, "row_code": row, "policy": "derived_P1_plus_P7", "source_value": float(published.sum()), "child_sum": float(V.loc[row].sum()), "absolute_gap": float((reconstructed - published).abs().max()), "proxy_denominator": float("nan")})
            continue
        else:
            if float(np.nan_to_num(published.to_numpy(dtype="float64"), nan=0.0).sum()) != 0.0:
                raise ValueError(f"Unknown nonzero V row: flow={flow!r}, row={row!r}")
            continue
        if formula is not None:
            diagnostics.extend(_rounded_v_formula(flow=flow, row=row, published=published, formula=formula.reindex(source.columns), term_counts=counts.reindex(source.columns), quantum=quantum, rtol=rtol, atol=atol))
        for parent in source.columns.astype(str):
            child_list = parent_children.get(parent, [])
            value = float(published.loc[parent])
            if len(child_list) == 1:
                V.loc[row, child_list[0]] = value
                diagnostics.append({"flow": flow, "row_code": row, "parent": parent, "children": "|".join(child_list), "policy": "direct_copy", "source_value": value, "child_sum": value, "absolute_gap": 0.0, "proxy_denominator": float("nan")})
                continue
            if value == 0.0:
                continue
            proxy = values.reindex(child_list).astype("float64")
            denominator = float(proxy.sum())
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(
                    f"V proxy has no usable mass: flow={flow!r}, row={row!r}, parent={parent!r}, "
                    f"children={child_list!r}, published value={value!r}, proxy={proxy.to_dict()!r}, denominator={denominator!r}."
                )
            V.loc[row, child_list] = proxy.to_numpy() * (value / denominator)
            diagnostics.append({
                "flow": flow, "row_code": row, "parent": parent,
                "children": "|".join(child_list), "policy": (
                    "derived" if row in {"CPA_TOTAL", "P1"}
                    else "domestic_output_share" if row in _V_OUTPUT_PROXY_ROWS
                    else "total_input_share" if row in _V_TOTAL_INPUT_PROXY_ROWS
                    else "IMP_product_row_use_share" if row in _V_IMP_INPUT_PROXY_ROWS
                    else "imported_input_share"
                ), "source_value": value, "child_sum": float(V.loc[row, child_list].sum()),
                "absolute_gap": abs(float(V.loc[row, child_list].sum()) - value),
                "proxy_denominator": denominator,
            })
        reconstructed = _aggregate_columns(V.loc[[row]], child_parent, source.columns).loc[row]
        _assert_close(f"{flow} V parent reconstruction {row}", reconstructed, published, rtol=0.0, atol=1e-9)
        diagnostics.append({"stage": "V_parent_reconstruction", "flow": flow, "row": row, "max_abs_gap": 0.0, "ok": True})
    if {"P1", "P2_ADJ", "B1G"}.issubset(source.index):
        base = source.loc["P1"] - source_total_Z.sum(axis=0).reindex(source.columns) - source.loc["B1G"]
        plus = base - source.loc["P2_ADJ"]
        minus = base + source.loc["P2_ADJ"]
        production = plus if float(plus.abs().sum()) <= float(minus.abs().sum()) else minus
        diagnostics.append({"stage": "B1G_diagnostic", "flow": flow, "row_code": "B1G", "policy": "published_source_controlled", "production_residual_max_abs": float(production.abs().max()), "p2_adj_sign": -1 if production is plus else 1})
    if {"B1G", "D1", "D29X39", "B2A3G"}.issubset(source.index):
        income = source.loc["B1G"] - (source.loc["D1"] + source.loc["D29X39"] + source.loc["B2A3G"])
        diagnostics.append({"stage": "B1G_diagnostic", "flow": flow, "row_code": "B1G", "policy": "published_source_controlled", "income_residual_max_abs": float(income.abs().max())})
    diagnostic_frame = pd.DataFrame(diagnostics)
    return V, diagnostic_frame, diagnostics


def _allocate_source_parent_vector(
    *, published: pd.Series, child_parent: Mapping[str, str], children: pd.Index,
    proxy: pd.Series, flow: str, row_code: str,
) -> pd.Series:
    out = pd.Series(0.0, index=children, dtype="float64")
    for parent in published.index.astype(str):
        child_list = [str(child) for child in children if child_parent.get(str(child), str(child)) == parent]
        value = float(published.loc[parent])
        if value == 0.0:
            continue
        weights = proxy.reindex(child_list).astype("float64")
        denominator = float(weights.sum())
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                f"No usable P1 allocation proxy: flow={flow!r}, row_code={row_code!r}, "
                f"parent={parent!r}, children={child_list!r}, published_value={value!r}, "
                f"proxy_denominator={denominator!r}."
            )
        out.loc[child_list] = weights.to_numpy() * (value / denominator)
    return out


def _v_policy_preflight(*, references: Mapping[str, ParsedEurostatIOT], child_parent: Mapping[str, str], children: pd.Index, x: pd.Series, total_inputs: pd.Series, imp_inputs: pd.Series, imp_products: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policy_rows = {"P1", "CPA_TOTAL", "TS_BP"} | _V_OUTPUT_PROXY_ROWS | _V_TOTAL_INPUT_PROXY_ROWS | _V_IMP_INPUT_PROXY_ROWS | {"IMP"}
    for flow, reference in references.items():
        if reference.V is None:
            continue
        for code in reference.V.index.astype(str):
            for parent in reference.V.columns.astype(str):
                value = float(reference.V.loc[code, parent])
                child_list = [str(c) for c in children if child_parent.get(str(c), str(c)) == parent]
                if code in {"P1", "CPA_TOTAL", "TS_BP"}:
                    proxy, name, status = pd.Series(dtype="float64"), "derived", "derived_identity_ok"
                elif code in _V_OUTPUT_PROXY_ROWS:
                    proxy, name, status = x.reindex(child_list), "domestic_output", "allocatable"
                elif code in _V_TOTAL_INPUT_PROXY_ROWS:
                    proxy, name, status = total_inputs.reindex(child_list), "TOTAL_input", "allocatable"
                elif code in _V_IMP_INPUT_PROXY_ROWS:
                    proxy, name, status = imp_products.reindex(child_list), "IMP_product_row_use", "allocatable"
                elif code == "IMP":
                    proxy, name, status = imp_inputs.reindex(child_list), "IMP_input", "allocatable"
                else:
                    proxy, name, status = pd.Series(dtype="float64"), "none", "unknown_row"
                denom = float(proxy.sum()) if len(proxy) else float("nan")
                if len(child_list) == 1:
                    status = "direct_copy"
                elif value == 0.0:
                    status = "zero_source"
                elif code not in policy_rows:
                    status = "unknown_row"
                elif status == "allocatable" and (not np.isfinite(denom) or denom <= 0.0):
                    status = "zero_proxy_nonzero_source"
                rows.append({"flow": flow, "row_code": code, "parent": parent, "children": "|".join(child_list), "n_children": len(child_list), "source_value": value, "policy": code, "proxy_name": name, "proxy_denominator": denom, "derived_source_value": float("nan"), "identity_gap": float("nan"), "status": status})
    return pd.DataFrame(rows)


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



def _validate_v_flow_contract(
    *,
    dirs: Mapping[str, Path],
    suffix: str,
    atol: float,
    rtol: float,
) -> list[dict[str, object]]:
    name = "V_split.csv" if suffix == "split" else "V_agg.csv"
    paths = {flow: directory / name for flow, directory in dirs.items()}
    if not any(path.is_file() for path in paths.values()):
        return []
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"{name} must exist for DOM, IMP, and TOTAL")
    frames = {flow: _read_frame(path) for flow, path in paths.items()}
    rows = frames["TOTAL"].index
    if any(not frame.index.equals(rows) or not frame.columns.equals(frames["TOTAL"].columns) for frame in frames.values()):
        raise ValueError(f"{suffix} V row/column schema differs across flows")
    diagnostics: list[dict[str, object]] = []
    x = {flow: _read_series(dirs[flow] / ("x_split.csv" if suffix == "split" else "x_agg.csv")) for flow in dirs}
    Z = {flow: _read_frame(dirs[flow] / ("Z_split.csv" if suffix == "split" else "Z_agg.csv")) for flow in dirs}
    for row in rows.astype(str):
        kind = supplementary_accounting_identity_kind(row)
        if kind == "output":
            for flow in ("DOM", "IMP", "TOTAL"):
                gap = _assert_close(f"{suffix} V output {flow}", frames[flow].loc[row], x[flow], rtol=rtol, atol=atol)
                diagnostics.append({"stage": suffix, "object": f"V:{row}:{flow}", "contract": "shared_output", "max_abs_gap": gap, "ok": True})
        elif kind == "intermediate_use":
            for flow in ("DOM", "IMP", "TOTAL"):
                expected = Z[flow].sum(axis=0).astype("float64")
                gap = _assert_close(f"{suffix} V intermediate use {flow}", frames[flow].loc[row], expected, rtol=rtol, atol=atol)
                diagnostics.append({"stage": suffix, "object": f"V:{row}:{flow}", "contract": "flow_specific_Z_column_sum", "max_abs_gap": gap, "ok": True})
        elif kind == "value_added":
            expected = x["TOTAL"] - Z["TOTAL"].sum(axis=0).astype("float64")
            gap = _assert_close(f"{suffix} V TOTAL value added", frames["TOTAL"].loc[row], expected, rtol=rtol, atol=atol)
            diagnostics.append({"stage": suffix, "object": f"V:{row}:TOTAL", "contract": "total_only_identity", "max_abs_gap": gap, "ok": True})
        elif kind == "total_supply":
            diagnostics.append({"stage": suffix, "object": f"V:{row}:TOTAL", "contract": "validated_against_source_flow_reconstruction", "ok": True})
        else:
            expected = _add_frames(frames["DOM"].loc[[row]], frames["IMP"].loc[[row]]).iloc[0]
            gap = _assert_close(f"{suffix} V additive row {row}", expected, frames["TOTAL"].loc[row], rtol=rtol, atol=atol)
            diagnostics.append({"stage": suffix, "object": f"V:{row}", "contract": "DOM+IMP=TOTAL", "max_abs_gap": gap, "ok": True})
    return diagnostics


def _validate_m_flow_contract(
    *,
    dirs: Mapping[str, Path],
    suffix: str,
    atol: float,
    rtol: float,
) -> list[dict[str, object]]:
    name = "M_split.csv" if suffix == "split" else "M_agg.csv"
    paths = {flow: directory / name for flow, directory in dirs.items()}
    if not any(path.is_file() for path in paths.values()):
        return []
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"{name} must exist for DOM, IMP, and TOTAL")
    frames = {flow: _read_frame(path) for flow, path in paths.items()}
    expected = _add_frames(frames["DOM"], frames["IMP"])
    gap = _assert_close(f"{suffix} M DOM+IMP=TOTAL", expected, frames["TOTAL"], rtol=rtol, atol=atol)
    return [{"stage": suffix, "object": "M", "contract": "DOM+IMP=TOTAL including applicability mask", "max_abs_gap": gap, "ok": True}]


def derive_total_flow(outdir: Path, *, eurostat_dataset: str, eurostat_filters: Mapping[str, object], eurostat_row_dim: str, eurostat_col_dim: str, eurostat_stk_flow_dim: str = "stk_flow", source_rounding_quantum: float | None = None, atol: float = 1e-6, rtol: float = 1e-9) -> None:
    """Canonicalise primitive DOM/IMP Z/Y into TOTAL, shared x/A, and source M."""
    dirs = {flow: outdir / f"{eurostat_stk_flow_dim}={flow}" for flow in ("DOM", "IMP", "TOTAL")}
    for flow in ("DOM", "IMP"):
        if not dirs[flow].is_dir():
            raise FileNotFoundError(f"Required primitive flow directory not found: {dirs[flow]}")
    manifest = _load_equal_manifests(dirs["DOM"], dirs["IMP"])
    Z_dom, Z_imp = (_read_frame(dirs[flow] / "Z_split.csv") for flow in ("DOM", "IMP"))
    Y_dom, Y_imp = (_read_frame(dirs[flow] / "Y_split.csv") for flow in ("DOM", "IMP"))
    if not Z_dom.index.equals(Z_imp.index) or not Z_dom.columns.equals(Z_imp.columns):
        raise ValueError("DOM and IMP child topology/order differs for Z; cannot construct TOTAL.")
    if not Y_dom.index.equals(Y_imp.index) or not Y_dom.columns.equals(Y_imp.columns):
        raise ValueError("DOM and IMP child topology/order differs for Y; cannot construct TOTAL.")
    if dirs["TOTAL"].exists():
        shutil.rmtree(dirs["TOTAL"])
    dirs["TOTAL"].mkdir(parents=True)
    Z_total = Z_dom + Z_imp
    controls = set(derived_final_demand_control_columns(Y_dom.columns))
    Y_total = Y_dom.loc[:, [c for c in Y_dom.columns if c not in controls]] + Y_imp.loc[:, [c for c in Y_imp.columns if c not in controls]]
    Y_total = enforce_final_demand_display_identities(Y_total, Z=Z_total)
    _write_frame(Z_total, dirs["TOTAL"], "Z_split.csv")
    _write_frame(Y_total, dirs["TOTAL"], "Y_split.csv")
    Z = {"DOM": Z_dom, "IMP": Z_imp, "TOTAL": Z_total}
    Y = {"DOM": Y_dom, "IMP": Y_imp, "TOTAL": Y_total}
    use = {flow: flow_use_total(Z[flow], Y[flow]) for flow in Z}
    references: dict[str, ParsedEurostatIOT] = {}
    diagnostics: list[dict[str, object]] = []
    for flow in ("DOM", "IMP", "TOTAL"):
        filters = dict(eurostat_filters)
        filters[eurostat_stk_flow_dim] = flow
        references[flow] = load_eurostat_iot(dataset=eurostat_dataset, filters=filters, row_dim=eurostat_row_dim, col_dim=eurostat_col_dim, storage_folder=default_data_root() / "eurostat")
        if references[flow].M is not None:
            source_m = references[flow].M.copy().astype("float64")
            _write_frame(source_m, dirs[flow], "M_split.csv")
            _assert_close(f"{flow} published M", _read_frame(dirs[flow] / "M_split.csv"), source_m, rtol=0.0, atol=0.0)
    quantum = _resolve_source_rounding_quantum(references, source_rounding_quantum)
    source_codes = references["TOTAL"].Z.index.astype(str)
    child_parent = _child_to_parent(manifest, source_codes)
    if references["DOM"].V is None or "P1" not in references["DOM"].V.index:
        raise ValueError("DOM source V must contain the source-controlled P1 row.")
    p1_dom = references["DOM"].V.loc["P1"].astype("float64")
    x_domestic = _allocate_source_parent_vector(
        published=p1_dom, child_parent=child_parent, children=Z_dom.index,
        proxy=use["DOM"], flow="DOM", row_code="P1",
    ).rename("x")
    p1_reconstructed = _aggregate_columns(x_domestic.to_frame("P1").T, child_parent, p1_dom.index).loc["P1"]
    _assert_close("DOM P1 parent reconstruction", p1_reconstructed, p1_dom, rtol=0.0, atol=1e-9)
    if references["TOTAL"].V is not None and "P1" in references["TOTAL"].V.index:
        p1_total = references["TOTAL"].V.loc["P1"].astype("float64")
        _rounded_v_formula(flow="TOTAL", row="P1_DOM_vs_TOTAL", published=p1_total, formula=p1_dom.reindex(p1_total.index), term_counts=pd.Series(1, index=p1_total.index), quantum=quantum, rtol=rtol, atol=atol)
    for flow in Z:
        _write_series(use[flow], dirs[flow] / "use_total_split.csv")
        _write_series(x_domestic, dirs[flow] / "x_split.csv")
        _write_frame(technical_coefficients(Z[flow], x_domestic), dirs[flow], "A_split.csv")
    diagnostics.append({"stage": "row_use_diagnostic", "flow": "DOM", "object": "row_use_minus_p1", "max_abs_gap": float((use["DOM"] - x_domestic).abs().max())})
    for flow in ("DOM", "IMP"):
        reference = references[flow]
        reconstructed_z = _aggregate_square(Z[flow], child_parent, reference.Z.index.astype(str))
        diagnostics.append({"stage": "source_reconstruction", "flow": flow, "object": "Z", "max_abs_gap": _assert_close(f"{flow} Z reconstruction", reconstructed_z, reference.Z, rtol=rtol, atol=atol), "ok": True})
        diagnostics.extend(_validate_y_source_reconstruction(flow=flow, reconstructed=_aggregate_rows(Y[flow], child_parent, reference.Z.index.astype(str)), reference=reference, source_rounding_quantum=quantum, rtol=rtol, atol=atol))
    # Published TOTAL can differ from DOM+IMP by independently rounded source cells.
    total_z = _aggregate_square(Z_total, child_parent, references["TOTAL"].Z.index.astype(str))
    total_atomic = _aggregate_rows(source_atomic_final_demand(Y_total), child_parent, references["TOTAL"].Z.index.astype(str))
    source_atomic = source_atomic_final_demand(references["TOTAL"].Y).reindex(index=total_atomic.index, columns=total_atomic.columns)
    for object_name, generated, published in (("Z", total_z, references["TOTAL"].Z), ("atomic_Y", total_atomic, source_atomic)):
        for row in generated.index.astype(str):
            for col in generated.columns.astype(str):
                diag = rounded_source_control_diagnostic(identity=f"TOTAL:{object_name}", published=float(published.loc[row, col]), derived=float(generated.loc[row, col]), component_term_count=2, source_rounding_quantum=quantum, rtol=rtol, atol=atol)
                diag.update({"stage": "TOTAL_source_reconstruction", "flow": "TOTAL", "object": object_name, "row": row, "column": col})
                diagnostics.append(diag)
    total_reconstructed_y = _aggregate_rows(Y_total, child_parent, references["TOTAL"].Z.index.astype(str))
    for identity in _ROUNDED_Y_CONTROLS:
        if identity in total_reconstructed_y.columns:
            expected = _derived_y_control(ParsedEurostatIOT(Z=references["TOTAL"].Z, Y=total_reconstructed_y, V=None, M=None, meta={}), identity)[0]
            _assert_close(f"TOTAL generated {identity}", total_reconstructed_y[identity], expected, rtol=rtol, atol=atol)
    preflight = _v_policy_preflight(
        references=references, child_parent=child_parent, children=Z_dom.index,
        x=x_domestic, total_inputs=Z_total.sum(axis=0), imp_inputs=Z_imp.sum(axis=0),
        imp_products=use["IMP"],
    )
    preflight.to_csv(outdir / "v_policy_preflight.csv", index=False)
    problems = preflight.loc[
        preflight["status"].isin(["unknown_row", "zero_proxy_nonzero_source"])
        & (preflight["source_value"] != 0.0)
    ]
    if not problems.empty:
        raise ValueError("V policy preflight found genuine gaps:\n" + problems.to_string(index=False))
    for flow in ("DOM", "IMP", "TOTAL"):
        V, v_diagnostics, rows = _construct_canonical_v(flow=flow, reference=references[flow], child_parent=child_parent, Z=Z[flow], x_domestic=x_domestic, Z_total=Z_total, Z_imp=Z_imp, source_total_Z=references["TOTAL"].Z, imp_product_proxy=use["IMP"], quantum=quantum, rtol=rtol, atol=atol)
        if V is not None:
            _write_frame(V, dirs[flow], "V_split.csv")
            v_diagnostics.to_csv(dirs[flow] / "supplementary_row_split_diagnostics.csv", index=False)
            diagnostics.extend(rows)
    (dirs["TOTAL"] / "exio_selected_sectors.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dirs["TOTAL"] / "split_info.json").write_text(json.dumps({"flow_construction": "DOM_plus_IMP", "x_semantics": "shared_DOM_use_total", "A_semantics": "Z_flow_divided_by_shared_DOM_x", "V_status": "explicit_row_policy"}, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(diagnostics).to_csv(outdir / "derived_total_flow_identity.csv", index=False)


def validate_additive_outputs(outdir: Path, *, suffix: str, atol: float, rtol: float) -> None:
    dirs = {flow: outdir / f"stk_flow={flow}" for flow in ("DOM", "IMP", "TOTAL")}
    for directory in dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(f"Required flow directory not found: {directory}")
    names = _SPLIT_NAMES if suffix == "split" else _AGG_NAMES
    diagnostics: list[dict[str, object]] = []
    for obj in ("Z", "Y"):
        filename = names[obj]
        paths = {flow: directory / filename for flow, directory in dirs.items()}
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"{filename} must exist for DOM, IMP, and TOTAL")
        expected = _add_frames(_read_frame(paths["DOM"]), _read_frame(paths["IMP"]))
        diagnostics.append({"stage": suffix, "object": obj, "max_abs_gap": _assert_close(f"{suffix} {obj} DOM+IMP=TOTAL", expected, _read_frame(paths["TOTAL"]), rtol=rtol, atol=atol), "ok": True})
    diagnostics.extend(_apply_contract(dirs, suffix=suffix, write=False))
    diagnostics.extend(_validate_v_flow_contract(dirs=dirs, suffix=suffix, atol=atol, rtol=rtol))
    # M is a published, flow-specific supplementary block.  It is validated
    # against its corresponding source during canonical construction, not by
    # a false DOM+IMP additivity relation.
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
    parser.add_argument(
        "--source-rounding-quantum",
        type=float,
        help="Override the inferred source reporting quantum for rounded published controls.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.validate_only:
        validate_additive_outputs(args.outdir, suffix=args.validate_only, rtol=args.rtol, atol=args.atol)
        return 0
    required = (args.eurostat_dataset, args.eurostat_filters, args.eurostat_row_dim, args.eurostat_col_dim)
    if not all(required):
        parser.error("derivation requires Eurostat dataset, filters, row dimension, and column dimension")
    derive_total_flow(args.outdir, eurostat_dataset=str(args.eurostat_dataset), eurostat_filters=json.loads(str(args.eurostat_filters)), eurostat_row_dim=str(args.eurostat_row_dim), eurostat_col_dim=str(args.eurostat_col_dim), eurostat_stk_flow_dim=str(args.eurostat_stk_flow_dim), source_rounding_quantum=args.source_rounding_quantum, rtol=args.rtol, atol=args.atol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from ecolecon_mrio.eurostat_iot_source_policy import NAIO_10_CP1700_ACCOUNTING_POLICY
from ecolecon_mrio.iot_full_table_validation import validate_eurostat_iot_full_table
from ecolecon_mrio.iot_flow_semantics import (
    derived_final_demand_control_columns,
    enforce_final_demand_display_identities,
    flow_use_total,
    rounded_source_control_diagnostic,
    source_atomic_final_demand,
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
    if type(left) is not type(right):
        raise TypeError("Both objects must be DataFrames or both must be Series")
    if isinstance(left, pd.DataFrame):
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            return float("inf")
        if not left.isna().equals(right.isna()):
            return float("inf")
    else:
        if not left.index.equals(right.index):
            return float("inf")
        if not left.isna().equals(right.isna()):
            return float("inf")
    left_values = left.to_numpy(dtype="float64")
    right_values = right.to_numpy(dtype="float64")
    if np.isinf(left_values).any() or np.isinf(right_values).any():
        return float("inf")
    values = left_values - right_values
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


def _assert_close(label: str, left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series, *, rtol: float, atol: float) -> float:
    gap = _max_gap(left, right)
    finite_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for values in (
                left.to_numpy(dtype="float64"),
                right.to_numpy(dtype="float64"),
            )
        ]
    )
    scale = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
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
    grouped = cast(
        pd.DataFrame,
        frame.T.groupby(groups, sort=False).sum(min_count=1).T,
    )
    expected = set(source_codes.astype(str))
    if set(grouped.columns.astype(str)) != expected:
        raise ValueError(
            "Column reconstruction produced missing/orphan parent sectors: "
            f"extra={sorted(set(grouped.columns.astype(str)) - expected)!r}, "
            f"missing={sorted(expected - set(grouped.columns.astype(str)))!r}."
        )
    return cast(pd.DataFrame, grouped.reindex(columns=source_codes))


_V_OUTPUT_PRIMITIVES = {"D1", "D11", "D29X39", "P51C", "B3G"}


def _allocate_source_parent_vector(
    *, published: pd.Series, child_parent: Mapping[str, str], children: pd.Index,
    proxy: pd.Series, flow: str, row_code: str,
) -> pd.Series:
    """Allocate one signed primitive parent vector with non-negative child weights."""
    out = pd.Series(0.0, index=children, dtype="float64")
    for parent in published.index.astype(str):
        child_list = [str(child) for child in children if child_parent.get(str(child), str(child)) == parent]
        value = float(published.loc[parent])
        if not child_list:
            raise ValueError(f"No emitted children for V parent {parent!r}")
        if np.isnan(value):
            out.loc[child_list] = np.nan
            continue
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite primitive V value: flow={flow!r}, "
                f"row_code={row_code!r}, parent={parent!r}, value={value!r}."
            )
        if len(child_list) == 1:
            out.loc[child_list[0]] = value
            continue
        if value == 0.0:
            continue
        weights = proxy.reindex(child_list).astype("float64")
        if not np.isfinite(weights.to_numpy(dtype="float64")).all() or bool((weights < 0.0).any()):
            raise ValueError(
                f"Invalid V allocation proxy: flow={flow!r}, row_code={row_code!r}, "
                f"parent={parent!r}, proxy={weights.to_dict()!r}."
            )
        denominator = float(weights.sum())
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                f"No usable V allocation proxy: flow={flow!r}, row_code={row_code!r}, "
                f"parent={parent!r}, children={child_list!r}, published_value={value!r}, "
                f"proxy_denominator={denominator!r}."
            )
        out.loc[child_list] = weights.to_numpy() * (value / denominator)
    return out


def _source_row(reference: ParsedEurostatIOT, row: str) -> pd.Series | None:
    """Return one finite SDMX primitive row, or None when the row is unavailable."""
    if reference.V is None or row not in reference.V.index:
        return None
    values = reference.V.loc[row].astype("float64")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError(f"Source V row {row!r} contains non-finite product values")
    return values


def _allocate_import_partition(
    *, aggregate: pd.Series, published_part: pd.Series, published_total: pd.Series,
    child_parent: Mapping[str, str], children: pd.Index, row_code: str,
) -> pd.Series:
    """Apply source geographic proportions to transaction-derived imported supply."""
    out = pd.Series(0.0, index=children, dtype="float64")
    for parent in published_total.index.astype(str):
        child_list = [str(c) for c in children if child_parent.get(str(c), str(c)) == parent]
        total = float(published_total.loc[parent])
        part = float(published_part.loc[parent])
        derived = aggregate.reindex(child_list).astype("float64")
        if total == 0.0:
            if abs(part) > 0.0:
                raise ValueError(f"{row_code} has a nonzero component under zero source P7 for {parent}")
            continue
        out.loc[child_list] = derived.to_numpy() * (part / total)
    return out


def _construct_transaction_v(
    *, flow: str, reference: ParsedEurostatIOT, child_parent: Mapping[str, str],
    Z: pd.DataFrame, Z_total: pd.DataFrame, Z_imp: pd.DataFrame,
    x_domestic: pd.Series, imported_supply: pd.Series,
    primitive_reference: ParsedEurostatIOT | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Construct V from primitive observations and exact transaction identities."""
    if reference.V is None:
        return None, pd.DataFrame()
    source = reference.V.copy().astype("float64")
    source.index = source.index.astype(str)
    source.columns = source.columns.astype(str)
    primitive = reference if primitive_reference is None else primitive_reference
    primitive_v = None if primitive.V is None else primitive.V.astype("float64")
    children = pd.Index(Z.columns.astype(str))
    V = pd.DataFrame(np.nan, index=source.index, columns=children, dtype="float64")
    diagnostics: list[dict[str, object]] = []
    output_proxy = x_domestic.reindex(children).astype("float64")
    input_proxy = Z_total.sum(axis=0).reindex(children).astype("float64")

    # Primitive production rows retain their SDMX parent totals while using the
    # transaction-derived domestic output distribution inside each parent.
    for row in sorted(_V_OUTPUT_PRIMITIVES.intersection(source.index)):
        if primitive_v is None or row not in primitive_v.index:
            raise ValueError(f"Authoritative primitive V row {row!r} is unavailable")
        allocated = _allocate_source_parent_vector(
            published=primitive_v.loc[row], child_parent=child_parent, children=children,
            proxy=output_proxy, flow=flow, row_code=row,
        )
        V.loc[row] = allocated
        reconstructed = _aggregate_columns(
            allocated.rename(row).to_frame().T, child_parent, primitive_v.columns
        ).loc[row]
        gap = _assert_close(
            f"{flow} primitive V reconstruction {row}",
            reconstructed,
            primitive_v.loc[row],
            rtol=0.0,
            atol=1e-9,
        )
        diagnostics.append({
            "flow": flow, "row": row, "role": "primitive_output_weighted",
            "max_abs_gap": gap,
        })

    # Product taxes on intermediate use follow purchaser-column input mass, not
    # producer output, because they bridge basic-price inputs to P2_ADJ.
    if "D21X31" in source.index:
        if primitive_v is None or "D21X31" not in primitive_v.index:
            raise ValueError("Authoritative primitive V row 'D21X31' is unavailable")
        allocated_d21 = _allocate_source_parent_vector(
            published=primitive_v.loc["D21X31"], child_parent=child_parent, children=children,
            proxy=input_proxy, flow=flow, row_code="D21X31",
        )
        V.loc["D21X31"] = allocated_d21
        reconstructed_d21 = _aggregate_columns(
            allocated_d21.rename("D21X31").to_frame().T,
            child_parent,
            primitive_v.columns,
        ).loc["D21X31"]
        d21_gap = _assert_close(
            f"{flow} primitive V reconstruction D21X31",
            reconstructed_d21,
            primitive_v.loc["D21X31"],
            rtol=0.0,
            atol=1e-9,
        )
        diagnostics.append({
            "flow": flow, "row": "D21X31", "role": "primitive_input_weighted",
            "max_abs_gap": d21_gap,
        })

    # Exact transaction identities are populated only where the source flow
    # exposes the corresponding row; IMP never receives a production account.
    if "CPA_TOTAL" in source.index:
        V.loc["CPA_TOTAL"] = Z.sum(axis=0).reindex(children)
    if "P1" in source.index:
        V.loc["P1"] = x_domestic.reindex(children) if flow in {"DOM", "TOTAL"} else 0.0
    if "IMP" in source.index:
        V.loc["IMP"] = Z_imp.sum(axis=0).reindex(children)
    if "P7" in source.index:
        V.loc["P7"] = imported_supply.reindex(children)

    # Import geography is a source composition applied to derived P7. The final
    # component in each exhaustive partition is a residual so identities remain exact.
    source_p7 = _source_row(reference, "P7")
    if source_p7 is not None and "P7_B0" in source.index and "P7_D0" in source.index:
        b0 = _allocate_import_partition(
            aggregate=imported_supply, published_part=source.loc["P7_B0"],
            published_total=source_p7, child_parent=child_parent, children=children,
            row_code="P7_B0",
        )
        V.loc["P7_B0"] = b0
        V.loc["P7_D0"] = imported_supply.reindex(children) - b0
        if "P7_U2" in source.index and "P7_U3" in source.index:
            u2 = _allocate_import_partition(
                aggregate=b0, published_part=source.loc["P7_U2"],
                published_total=source.loc["P7_B0"], child_parent=child_parent,
                children=children, row_code="P7_U2",
            )
            V.loc["P7_U2"] = u2
            V.loc["P7_U3"] = b0 - u2

    # The production column closes after taxes on products; operating surplus is
    # the declared residual rather than an independently proxied child margin.
    if {"P2_ADJ", "B1G", "B2A3N", "B2A3G"}.intersection(source.index):
        if flow not in {"DOM", "TOTAL"}:
            raise ValueError(f"{flow} exposes production-account rows unexpectedly")
        if "D21X31" not in V.index or V.loc["D21X31"].isna().any():
            raise ValueError(f"{flow} production account requires D21X31")
        p2 = input_proxy + V.loc["D21X31"]
        b1g = output_proxy - p2
        if "P2_ADJ" in source.index:
            V.loc["P2_ADJ"] = p2
        if "B1G" in source.index:
            V.loc["B1G"] = b1g
        required = {"D1", "D29X39", "P51C"}
        if {"B2A3N", "B2A3G"}.intersection(source.index) and not required.issubset(V.index):
            raise ValueError(f"{flow} surplus closure requires {sorted(required)!r}")
        if "B2A3N" in source.index or "B2A3G" in source.index:
            b2n = b1g - V.loc["D1"] - V.loc["D29X39"] - V.loc["P51C"]
            if "B2A3N" in source.index:
                V.loc["B2A3N"] = b2n
            if "B2A3G" in source.index:
                V.loc["B2A3G"] = b2n + V.loc["P51C"]
    if "TS_BP" in source.index:
        if flow != "TOTAL" or "P7" not in V.index:
            raise ValueError(f"{flow} TS_BP requires TOTAL P7")
        V.loc["TS_BP"] = output_proxy + imported_supply.reindex(children)

    # A numeric source row without a declared primitive or identity role is a
    # genuine accounting gap; all-blank template-only rows are handled at export.
    for row in source.index.astype(str):
        if V.loc[row].isna().all():
            values = source.loc[row].to_numpy(dtype="float64")
            if np.any(np.abs(np.nan_to_num(values, nan=0.0)) > 0.0):
                raise ValueError(f"Unknown nonzero V row: flow={flow!r}, row={row!r}")
    return V, pd.DataFrame(diagnostics)


def _construct_transaction_m(
    *, flow: str, reference: ParsedEurostatIOT, Y: pd.DataFrame,
    Z: pd.DataFrame, Y_imp: pd.DataFrame, Z_imp: pd.DataFrame,
    V: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Refresh M identities while preserving primitive SDMX values and masks."""
    if reference.M is None:
        return None
    M = reference.M.copy().astype("float64")
    M.index = M.index.astype(str)
    M.columns = M.columns.astype(str)
    if V is None:
        return M

    # TOTAL imports by final use come from the transformed IMP transactions.
    # Source geography is a composition prior only; residual components make
    # both declared partitions exact without inventing equal shares.
    if flow == "TOTAL" and "P7" in M.index:
        for column in M.columns.intersection(Y_imp.columns):
            M.loc["P7", column] = float(Y_imp[column].sum())
        for aggregate_row, first_row, residual_row in (
            ("P7", "P7_B0", "P7_D0"),
            ("P7_B0", "P7_U2", "P7_U3"),
        ):
            if not {aggregate_row, first_row, residual_row}.issubset(M.index):
                continue
            source_m = reference.M
            assert source_m is not None
            for column in M.columns:
                if column == "TOTAL" or column not in source_m.columns:
                    continue
                aggregate_value = float(M.loc[aggregate_row, column])
                source_total = float(source_m.loc[aggregate_row, column])
                source_part = float(source_m.loc[first_row, column])
                if np.isnan(source_total) and np.isnan(source_part):
                    continue
                if not np.isfinite(source_total) or not np.isfinite(source_part):
                    raise ValueError(
                        f"Incomplete import-partition proxy in M: {aggregate_row}/{first_row}, "
                        f"column={column!r}"
                    )
                if source_total == 0.0:
                    if aggregate_value != 0.0 or source_part != 0.0:
                        raise ValueError(
                            f"No usable import-partition proxy in M: {aggregate_row}, "
                            f"column={column!r}"
                        )
                    first_value = 0.0
                else:
                    first_value = aggregate_value * source_part / source_total
                M.loc[first_row, column] = first_value
                M.loc[residual_row, column] = aggregate_value - first_value

    # The TOTAL display column under each supplementary row is the matching V
    # grand total; it is not an independently allocated final-demand cell.
    if "TOTAL" in M.columns:
        shared_rows = cast(pd.Index, M.index).intersection(cast(pd.Index, V.index))
        for row in shared_rows:
            M.loc[row, "TOTAL"] = float(V.loc[row].sum(min_count=1))
    if "CPA_TOTAL" in M.index:
        for column in M.columns:
            if column == "TOTAL":
                continue
            if column in Y.columns:
                M.loc["CPA_TOTAL", column] = float(Y[column].sum())
    if "IMP" in M.index and flow == "DOM":
        for column in M.columns:
            if column == "TOTAL":
                M.loc["IMP", column] = float(Z_imp.to_numpy(dtype="float64").sum())
            elif column in Y_imp.columns:
                M.loc["IMP", column] = float(Y_imp[column].sum())
    if "P2_ADJ" in M.index and "D21X31" in M.index:
        for column in M.columns:
            if column == "TOTAL":
                continue
            if column in Y.columns and not pd.isna(M.loc["D21X31", column]):
                M.loc["P2_ADJ", column] = float(Y[column].sum()) + float(M.loc["D21X31", column])

    # Rebuild non-overlapping display controls only where the source schema says
    # the supplementary observation is applicable.
    m_rows = cast(pd.Index, M.index).astype(str)
    for row in m_rows:
        if {"P3_S13", "P3_S14", "P3_S15"}.issubset(M.columns) and not pd.isna(M.loc[row, "P3"]):
            M.loc[row, "P3"] = float(M.loc[row, ["P3_S13", "P3_S14", "P3_S15"]].sum())
        if {"P52", "P53"}.issubset(M.columns) and not pd.isna(M.loc[row, "P5M"]):
            M.loc[row, "P5M"] = float(M.loc[row, ["P52", "P53"]].sum())
        if {"P51G", "P52", "P53"}.issubset(M.columns) and not pd.isna(M.loc[row, "P5"]):
            M.loc[row, "P5"] = float(M.loc[row, ["P51G", "P52", "P53"]].sum())
        atomic_cols = [c for c in source_atomic_final_demand(Y).columns if c in M.columns]
        if "TFU" in M.columns and not pd.isna(M.loc[row, "TFU"]):
            M.loc[row, "TFU"] = float(M.loc[row, atomic_cols].sum())
        if "TU" in M.columns and not pd.isna(M.loc[row, "TU"]):
            intermediate = float(M.loc[row, "TOTAL"]) if "TOTAL" in M.columns else 0.0
            final = float(M.loc[row, "TFU"]) if "TFU" in M.columns else float(M.loc[row, atomic_cols].sum())
            M.loc[row, "TU"] = intermediate + final
    return M


def _apply_contract(dirs: Mapping[str, Path], *, suffix: str, write: bool) -> list[dict[str, object]]:
    z_name = "Z_split.csv" if suffix == "split" else "Z_agg.csv"
    y_name = "Y_split.csv" if suffix == "split" else "Y_agg.csv"
    x_name = "x_split.csv" if suffix == "split" else "x_agg.csv"
    a_name = "A_split.csv" if suffix == "split" else "A_agg.csv"
    use_name = "use_total_split.csv" if suffix == "split" else "use_total_agg.csv"
    Z = {flow: _read_frame(path / z_name) for flow, path in dirs.items()}
    Y = {flow: _read_frame(path / y_name) for flow, path in dirs.items()}
    use = {flow: flow_use_total(Z[flow], Y[flow]) for flow in dirs}
    output = _read_series(dirs["DOM"] / x_name)
    diagnostics: list[dict[str, object]] = []
    for flow, directory in dirs.items():
        expected_A = technical_coefficients(Z[flow], output)
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
    frames = {flow: _read_frame(path) for flow, path in paths.items() if path.is_file()}
    diagnostics: list[dict[str, object]] = []
    x_name = "x_split.csv" if suffix == "split" else "x_agg.csv"
    z_name = "Z_split.csv" if suffix == "split" else "Z_agg.csv"
    stored_x = {flow: _read_series(dirs[flow] / x_name) for flow in dirs}
    Z = {flow: _read_frame(dirs[flow] / z_name) for flow in dirs}
    for flow, frame in frames.items():
        if "P1" in frame.index:
            if flow == "IMP":
                gap = _assert_close(f"{suffix} IMP V P1", frame.loc["P1"], pd.Series(0.0, index=frame.columns), rtol=rtol, atol=atol)
                diagnostics.append({"stage": suffix, "object": "V:P1:IMP", "contract": "absent_or_zero", "max_abs_gap": gap, "ok": True})
            else:
                expected = stored_x[flow].reindex(frame.columns)
                gap = _assert_close(f"{suffix} V P1 {flow}", frame.loc["P1"], expected, rtol=rtol, atol=atol)
                diagnostics.append({"stage": suffix, "object": f"V:P1:{flow}", "contract": "canonical_x", "max_abs_gap": gap, "ok": True})
        if "CPA_TOTAL" in frame.index:
            expected = Z[flow].sum(axis=0).reindex(frame.columns).astype("float64")
            gap = _assert_close(f"{suffix} V CPA_TOTAL {flow}", frame.loc["CPA_TOTAL"], expected, rtol=rtol, atol=atol)
            diagnostics.append({"stage": suffix, "object": f"V:CPA_TOTAL:{flow}", "contract": "Z_column_sum", "max_abs_gap": gap, "ok": True})
    if "TOTAL" in frames and "TS_BP" in frames["TOTAL"].index:
        total = frames["TOTAL"]
        if not {"P1", "P7"}.issubset(total.index):
            raise ValueError(f"{suffix} TOTAL TS_BP requires P1 and P7")
        gap = _assert_close(f"{suffix} TOTAL V TS_BP", total.loc["TS_BP"], total.loc["P1"] + total.loc["P7"], rtol=rtol, atol=atol)
        diagnostics.append({"stage": suffix, "object": "V:TS_BP:TOTAL", "contract": "P1_plus_P7", "max_abs_gap": gap, "ok": True})
    return diagnostics


def _validate_generated_full_table(
    *, variants: Mapping[str, ParsedEurostatIOT], report_path: Path
) -> None:
    """Run the single generated-table accounting gate and persist its evidence."""
    report = validate_eurostat_iot_full_table(
        variants,
        policy=NAIO_10_CP1700_ACCOUNTING_POLICY,
        mode="generated",
    )
    report.to_frame().to_csv(report_path, index=False)
    report.require_ok()


def _read_generated_variants(
    dirs: Mapping[str, Path], *, suffix: str
) -> dict[str, ParsedEurostatIOT]:
    names = _SPLIT_NAMES if suffix == "split" else _AGG_NAMES
    return {
        flow: ParsedEurostatIOT(
            Z=_read_frame(directory / names["Z"]),
            Y=_read_frame(directory / names["Y"]),
            V=_read_frame(directory / names["V"]),
            M=_read_frame(directory / names["M"]),
            meta={},
        )
        for flow, directory in dirs.items()
    }


def derive_total_flow(outdir: Path, *, eurostat_dataset: str, eurostat_filters: Mapping[str, object], eurostat_row_dim: str, eurostat_col_dim: str, eurostat_stk_flow_dim: str = "stk_flow", source_rounding_quantum: float | None = None, atol: float = 1e-6, rtol: float = 1e-9) -> None:
    """Canonicalise DOM/IMP transactions into a cleared production account.

    Domestic product output is the complete domestic-origin row use. Imported
    supply is the complete import-origin row use. Primitive supplementary rows
    retain source parent totals; generated rows are recomputed from transactions.
    """
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
    quantum = _resolve_source_rounding_quantum(references, source_rounding_quantum)
    source_codes = references["TOTAL"].Z.index.astype(str)
    child_parent = _child_to_parent(manifest, source_codes)
    # Row conservation supplies the only child-output margin. EXIO producer
    # output shares remain donor diagnostics and are not a second hard target.
    x_domestic = use["DOM"].rename("x")
    imported_supply = use["IMP"].rename("imports")
    published_p1 = _source_row(references["DOM"], "P1")
    if published_p1 is not None:
        derived_p1 = cast(
            pd.Series,
            _aggregate_rows(
                x_domestic.to_frame("P1"), child_parent, published_p1.index
            )["P1"],
        )
        counts = references["DOM"].Z.notna().sum(axis=1).add(
            source_atomic_final_demand(references["DOM"].Y).notna().sum(axis=1),
            fill_value=0,
        ).astype("int64")
        diagnostics.extend(_validate_rounded_series(
            flow="DOM", identity="P1", published=published_p1,
            derived=derived_p1, component_counts=counts,
            source_rounding_quantum=quantum, rtol=rtol, atol=atol,
        ))
    for flow in Z:
        _write_series(use[flow], dirs[flow] / "use_total_split.csv")
        _write_series(x_domestic, dirs[flow] / "x_split.csv")
        _write_frame(technical_coefficients(Z[flow], x_domestic), dirs[flow], "A_split.csv")
    _assert_close("DOM Leontief output identity", x_domestic, use["DOM"], rtol=0.0, atol=0.0)
    diagnostics.append({
        "stage": "analytical_contract", "flow": "DOM", "object": "P1",
        "contract": "transaction_derived_DOM_row_use", "max_abs_gap": 0.0, "ok": True,
    })
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
    canonical_v: dict[str, pd.DataFrame | None] = {}
    canonical_m: dict[str, pd.DataFrame | None] = {}
    for flow in ("DOM", "IMP", "TOTAL"):
        V, v_diagnostics = _construct_transaction_v(
            flow=flow, reference=references[flow], child_parent=child_parent,
            Z=Z[flow], Z_total=Z_total, Z_imp=Z_imp,
            x_domestic=x_domestic, imported_supply=imported_supply,
            primitive_reference=references["TOTAL"] if flow in {"DOM", "TOTAL"} else None,
        )
        canonical_v[flow] = V
        if V is not None:
            _write_frame(V, dirs[flow], "V_split.csv")
            v_diagnostics.to_csv(
                dirs[flow] / "supplementary_row_split_diagnostics.csv",
                index=False,
            )
        M = _construct_transaction_m(
            flow=flow, reference=references[flow], Y=Y[flow], Z=Z[flow],
            Y_imp=Y_imp, Z_imp=Z_imp, V=V,
        )
        canonical_m[flow] = M
        if M is not None:
            _write_frame(M, dirs[flow], "M_split.csv")
    if eurostat_dataset == NAIO_10_CP1700_ACCOUNTING_POLICY.dataset:
        _validate_generated_full_table(
            variants={
                flow: ParsedEurostatIOT(
                    Z=Z[flow],
                    Y=Y[flow],
                    V=canonical_v[flow],
                    M=canonical_m[flow],
                    meta={},
                )
                for flow in ("DOM", "IMP", "TOTAL")
            },
            report_path=outdir / "split_full_table_validation.csv",
        )
    (dirs["TOTAL"] / "exio_selected_sectors.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dirs["TOTAL"] / "split_info.json").write_text(json.dumps({
        "flow_construction": "DOM_plus_IMP",
        "x_semantics": "transaction_derived_DOM_P1",
        "A_semantics": {
            "DOM": "cleared_domestic_leontief_coefficients",
            "IMP": "imported_inputs_per_domestic_output",
            "TOTAL": "total_inputs_per_domestic_output",
        },
        "V_status": "transaction_identities_plus_primitive_rows",
        "balancing_applied": False,
    }, indent=2) + "\n", encoding="utf-8")
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
        observed = _read_frame(paths["TOTAL"])
        # TOTAL follows the published source display schema.  DOM/IMP can carry
        # display-only TU/TFU controls which are absent from that source schema;
        # validate every emitted TOTAL column without inventing those rows.
        if obj == "Y":
            expected = expected.reindex(index=observed.index, columns=observed.columns)
        diagnostics.append({"stage": suffix, "object": obj, "max_abs_gap": _assert_close(f"{suffix} {obj} DOM+IMP=TOTAL", expected, observed, rtol=rtol, atol=atol), "ok": True})
    diagnostics.extend(_apply_contract(dirs, suffix=suffix, write=False))
    diagnostics.extend(_validate_v_flow_contract(dirs=dirs, suffix=suffix, atol=atol, rtol=rtol))
    _validate_generated_full_table(
        variants=_read_generated_variants(dirs, suffix=suffix),
        report_path=outdir / f"{suffix}_full_table_validation.csv",
    )
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

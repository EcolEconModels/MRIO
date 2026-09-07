#!/usr/bin/env python3
"""Read-only census of environmental donor-support states for split MRIO outputs.

The census is deliberately diagnostic. It does not construct model-facing
environmental estimates, alter source data, or relax any production validator.
It reconstructs the exact emitted-sector projection from successful split
artifacts, streams the selected EXIOBASE extension corpus, and enumerates the
support states that would otherwise be discovered serially by a fail-fast
all-extension production run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_REQUIRED_EXTENSION_FILES = ("F.txt", "F_Y.txt", "unit.txt")

_SOURCE_COMPLETENESS_COLUMNS = (
    "extension", "has_F", "has_F_Y", "has_unit", "complete", "missing_files"
)
_PRODUCTION_SUPPORT_COLUMNS = (
    "extension", "stressor", "unit", "sector", "mapped_F", "mapped_x",
    "target_output", "projection_support", "support_state", "target_state",
    "method_blocking_positive_target", "reaggregation_parent",
)
_PRODUCTION_LINEAGE_COLUMNS = (
    "extension", "stressor", "unit", "sector", "reaggregation_parent",
    "source_code", "projection_weight", "raw_scope_F", "raw_scope_x",
    "weighted_F_contribution", "weighted_x_contribution", "mapped_F",
    "mapped_x", "target_output", "support_state", "target_state",
    "method_blocking_positive_target",
)
_IMPORT_EXCEPTION_COLUMNS = (
    "origin", "target_sector", "mapped_import_use", "mapped_production_output",
    "method_blocking",
)
_FY_SUPPORT_COLUMNS = (
    "extension", "stressor", "unit", "final_demand_category", "F_Y",
    "monetary_final_demand", "support_state", "method_blocking",
)
_SOURCE_ERROR_COLUMNS = ("extension", "kind", "error")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return {str(key): value for key, value in data.items()}


def validate_split_provenance(
    *,
    split_root: Path,
    source_zip: Path,
    scope_code: str,
    scope_members: Sequence[str],
    scope_mapping_csv: Path | None,
) -> dict[str, object]:
    """Prove that census source/scope bytes match the persisted monetary split."""
    source_sha = _sha256_file(source_zip)
    expected_members = list(map(str, scope_members))
    mapping_sha = _sha256_file(scope_mapping_csv) if scope_mapping_csv is not None else None
    flow_records: dict[str, dict[str, object]] = {}
    for flow in ("DOM", "IMP"):
        info_path = Path(split_root) / f"stk_flow={flow}" / "split_info.json"
        info = _load_json_object(info_path)
        if info.get("is_reconstruction_validated") is not True:
            raise ValueError(f"{flow} split_info is not marked reconstruction-validated.")
        variant = str(info.get("variant_value", "")).upper()
        if variant != flow:
            raise ValueError(
                f"{flow} split_info flow identity mismatch: variant_value={variant!r}."
            )
        recorded_sha = str(info.get("exio_source_zip_sha256", "")).strip()
        if recorded_sha != source_sha:
            raise ValueError(
                f"{flow} split/source EXIO SHA-256 mismatch: split={recorded_sha!r}, "
                f"census={source_sha!r}."
            )
        recorded_scope = str(info.get("exio_scope_code", "")).strip()
        if recorded_scope != str(scope_code):
            raise ValueError(
                f"{flow} split scope-code mismatch: split={recorded_scope!r}, "
                f"census={scope_code!r}."
            )
        raw_members = info.get("exio_scope_members", [])
        if not isinstance(raw_members, list):
            raise ValueError(f"{flow} split scope members must be a JSON list.")
        recorded_members = [str(value) for value in raw_members]
        if recorded_members != expected_members:
            raise ValueError(
                f"{flow} split scope-member mismatch: split={recorded_members!r}, "
                f"census={expected_members!r}."
            )
        recorded_mapping_sha = info.get("region_scope_mapping_sha256")
        normalized_recorded = None if recorded_mapping_sha in (None, "") else str(recorded_mapping_sha)
        if normalized_recorded != mapping_sha:
            raise ValueError(
                f"{flow} split region-scope mapping SHA mismatch: "
                f"split={normalized_recorded!r}, census={mapping_sha!r}."
            )
        flow_records[flow] = {
            "split_info": str(info_path),
            "exio_source_zip_sha256": recorded_sha,
            "exio_scope_code": recorded_scope,
            "exio_scope_members": recorded_members,
            "region_scope_mapping_sha256": normalized_recorded,
        }
    return {
        "verified": True,
        "exio_source_zip_sha256": source_sha,
        "scope_code": str(scope_code),
        "scope_members": expected_members,
        "region_scope_mapping_sha256": mapping_sha,
        "flows": flow_records,
    }


def _numeric_vector_csv(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one numeric data column in {path}, got {frame.shape[1]}.")
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    if not isinstance(series, pd.Series):
        raise TypeError(f"Could not read one-dimensional vector from {path}.")
    series.index = pd.Index([str(value).strip() for value in series.index])
    if not series.index.is_unique or any(not value for value in series.index):
        raise ValueError(f"Vector index must be unique and non-blank: {path}")
    values = series.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError(f"Vector contains missing or non-finite values: {path}")
    return series.astype("float64")


def _flatten_selected_children(value: object) -> list[str]:
    out: list[str] = []

    def visit(obj: object) -> None:
        if obj is None:
            return
        if isinstance(obj, Mapping):
            for nested in obj.values():
                visit(nested)
            return
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            for nested in obj:
                visit(nested)
            return
        text = str(obj).strip()
        if text:
            out.append(text)

    visit(value)
    return list(dict.fromkeys(out))


def _read_emitted_axis(flow_dir: Path) -> pd.Index:
    for name in ("x_split.csv", "use_total_split.csv"):
        path = flow_dir / name
        if path.is_file():
            return pd.Index(_numeric_vector_csv(path).index.astype(str), name="sector")
    z_path = flow_dir / "Z_split.csv"
    if z_path.is_file():
        frame = pd.read_csv(z_path, index_col=0)
        labels = pd.Index([str(value).strip() for value in frame.index], name="sector")
        if not labels.is_unique:
            raise ValueError(f"Split sector axis is not unique: {z_path}")
        return labels
    raise FileNotFoundError(
        f"Could not determine emitted sector axis in {flow_dir}; expected x_split.csv, "
        "use_total_split.csv, or Z_split.csv."
    )


def _mapping_from_weights(flow_dir: Path) -> dict[str, dict[str, float]]:
    path = flow_dir / "flow_specific_exio_to_eurostat_weights.csv"
    frame = pd.read_csv(path)
    required = {"source_code", "target_code", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)!r}")
    mapping: dict[str, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        source = str(getattr(row, "source_code")).strip()
        target = str(getattr(row, "target_code")).strip()
        weight = float(getattr(row, "weight"))
        if not source or not target or not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Malformed resolved mapping row in {path}: {row!r}")
        target_map = mapping.setdefault(target, {})
        if source in target_map:
            raise ValueError(
                "Duplicate persisted projection edge in "
                f"{path}: target={target!r}, source={source!r}."
            )
        target_map[source] = weight
    return mapping


def _selected_child_parents(flow_dir: Path) -> dict[str, str]:
    path = flow_dir / "exio_selected_sectors.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping object in {path}.")
    parents: dict[str, str] = {}
    for raw_parent, children in data.items():
        parent = str(raw_parent).strip()
        for child in _flatten_selected_children(children):
            owner = parents.get(child)
            if owner is not None and owner != parent:
                raise ValueError(
                    f"Persisted split child {child!r} has multiple parents: {owner!r}, {parent!r}."
                )
            parents[child] = parent
    return parents


def _edge_metadata(flow_dir: Path) -> dict[str, dict[str, object]]:
    path = flow_dir / "split_child_edges.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "parent_code",
        "current_child_code",
        "source_code",
        "concordance_target_code",
        "edge_weight_current_reference",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)!r}")
    out: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        child = str(row["current_child_code"]).strip()
        if not child:
            continue
        if child in out:
            raise ValueError(f"Duplicate split-child edge metadata for {child!r} in {path}.")
        out[child] = row
    return out


def build_projection_from_split_artifacts(
    flow_dir: Path,
    *,
    raw_exio_sectors: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Reconstruct the emitted-label-by-raw-EXIO projection from persisted provenance.

    This is intentionally an audit reconstruction. Every selected child must be
    reconstructible from the persisted child/edge provenance; otherwise the
    census fails rather than guessing a donor relation.
    """
    flow_dir = Path(flow_dir)
    raw_order = list(dict.fromkeys(str(value).strip() for value in raw_exio_sectors))
    if not raw_order or any(not value for value in raw_order):
        raise ValueError("raw_exio_sectors must be a non-empty unique code sequence.")
    emitted = list(map(str, _read_emitted_axis(flow_dir)))
    weights = _mapping_from_weights(flow_dir)
    selected_parents = _selected_child_parents(flow_dir)
    edges = _edge_metadata(flow_dir)
    available_raw = set(raw_order)

    P = pd.DataFrame(
        0.0,
        index=pd.Index(emitted, name="sector"),
        columns=pd.Index(raw_order, name="source_code"),
        dtype="float64",
    )
    parent_by_label = {label: label for label in emitted}

    for label in emitted:
        if label in edges:
            row = edges[label]
            source = str(row["source_code"]).strip()
            target = str(row["concordance_target_code"]).strip()
            if source not in available_raw:
                raise ValueError(
                    f"Persisted edge child {label!r} refers to absent EXIO source {source!r}."
                )
            try:
                weight = float(weights[target][source])
            except KeyError as exc:
                raise ValueError(
                    f"No authoritative persisted concordance weight for edge child {label!r}: "
                    f"source={source!r}, target={target!r}."
                ) from exc
            raw_weight = str(row["edge_weight_current_reference"]).strip()
            if raw_weight:
                edge_weight = float(raw_weight)
                if not math.isclose(edge_weight, weight, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        "split_child_edges weight disagrees with authoritative "
                        "flow-specific mapping: "
                        f"child={label!r}, source={source!r}, target={target!r}, "
                        f"edge_weight={edge_weight!r}, mapping_weight={weight!r}."
                    )
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"Invalid authoritative edge weight for {label!r}: {weight!r}")
            P.loc[label, source] = weight
            parent_by_label[label] = str(row["parent_code"]).strip() or selected_parents.get(label, label)
            continue

        if label in selected_parents:
            if label not in available_raw:
                raise ValueError(
                    "Persisted selected child cannot be reconstructed exactly from the "
                    f"available provenance: child={label!r}. Expected a raw EXIO code or "
                    "materialized source@target edge. Re-run the monetary split with richer "
                    "projection provenance before using the census."
                )
            P.loc[label, label] = 1.0
            parent_by_label[label] = selected_parents[label]
            continue

        if label in weights:
            for source, weight in weights[label].items():
                if source not in available_raw:
                    raise ValueError(
                        f"Resolved mapping for retained target {label!r} refers to absent "
                        f"EXIO source {source!r}."
                    )
                P.loc[label, source] = float(weight)
            continue

        if label in available_raw:
            P.loc[label, label] = 1.0
            continue

    values = P.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("Reconstructed environmental projection must be finite and non-negative.")
    column_sums = P.sum(axis=0).astype("float64")
    bad = column_sums.loc[~np.isclose(column_sums, 1.0, rtol=0.0, atol=1e-9)]
    if not bad.empty:
        sample = {str(k): float(v) for k, v in bad.iloc[:10].items()}
        raise ValueError(
            "Persisted split provenance does not reconstruct a complete source-product "
            f"partition; column-sum mismatches={sample!r}."
        )
    return P, parent_by_label


def classify_production_support(
    *,
    mapped_value: pd.Series,
    mapped_output: pd.Series,
    target_output: pd.Series,
    projection_support: pd.Series,
) -> pd.DataFrame:
    labels = pd.Index(mapped_value.index.astype(str), name="sector")
    for name, series in (
        ("mapped_output", mapped_output),
        ("target_output", target_output),
        ("projection_support", projection_support),
    ):
        if not pd.Index(series.index.astype(str)).equals(labels):
            raise ValueError(f"{name} must use the exact mapped-value sector axis.")
    rows: list[dict[str, object]] = []
    for label in labels:
        f_value = float(mapped_value.loc[label])
        x_value = float(mapped_output.loc[label])
        target = float(target_output.loc[label])
        supported = bool(projection_support.loc[label])
        if not all(math.isfinite(value) for value in (f_value, x_value, target)):
            state = "NONFINITE"
        elif not supported:
            state = "NO_PROJECTION_SUPPORT"
        elif x_value < 0.0:
            state = "NEGATIVE_MAPPED_OUTPUT"
        elif x_value > 0.0:
            state = "SUPPORTED"
        elif f_value == 0.0:
            state = "ZERO_ZERO"
        else:
            state = "NONZERO_F_ZERO_X"
        target_state = "POSITIVE" if target > 0.0 else ("ZERO" if target == 0.0 else "NEGATIVE")
        method_blocking = target > 0.0 and (
            (not supported) or (not math.isfinite(x_value)) or x_value <= 0.0
        )
        rows.append(
            {
                "sector": str(label),
                "mapped_F": f_value,
                "mapped_x": x_value,
                "target_output": target,
                "projection_support": supported,
                "support_state": state,
                "target_state": target_state,
                "method_blocking_positive_target": bool(method_blocking),
            }
        )
    return pd.DataFrame(rows)


def classify_final_demand_support(
    *,
    physical: pd.Series,
    monetary: pd.Series,
) -> pd.DataFrame:
    if not pd.Index(physical.index.astype(str)).equals(pd.Index(monetary.index.astype(str))):
        raise ValueError("F_Y physical and monetary category axes must match exactly.")
    rows: list[dict[str, object]] = []
    for label in physical.index:
        value = float(physical.loc[label])
        denominator = float(monetary.loc[label])
        if not math.isfinite(value) or not math.isfinite(denominator):
            state = "NONFINITE"
        elif denominator == 0.0 and value != 0.0:
            state = "NONZERO_FY_ZERO_DENOMINATOR"
        elif denominator == 0.0:
            state = "ZERO_ZERO"
        else:
            state = "SUPPORTED"
        rows.append(
            {
                "final_demand_category": str(label),
                "F_Y": value,
                "monetary_final_demand": denominator,
                "support_state": state,
                "method_blocking": state in {"NONZERO_FY_ZERO_DENOMINATOR", "NONFINITE"},
            }
        )
    return pd.DataFrame(rows)


def _source_completeness(source_zip: Path) -> tuple[pd.DataFrame, list[str]]:
    with zipfile.ZipFile(source_zip, "r") as archive:
        present: dict[str, set[str]] = {}
        for member in archive.namelist():
            parts = Path(member).parts
            if len(parts) == 2 and parts[1] in _REQUIRED_EXTENSION_FILES:
                present.setdefault(str(parts[0]), set()).add(str(parts[1]))
    candidates = sorted(
        extension
        for extension, files in present.items()
        if {"F.txt", "F_Y.txt"}.intersection(files)
    )
    rows: list[dict[str, object]] = []
    complete: list[str] = []
    for extension in candidates:
        files = present[extension]
        missing = [name for name in _REQUIRED_EXTENSION_FILES if name not in files]
        if not missing:
            complete.append(extension)
        rows.append(
            {
                "extension": extension,
                "has_F": "F.txt" in files,
                "has_F_Y": "F_Y.txt" in files,
                "has_unit": "unit.txt" in files,
                "complete": not missing,
                "missing_files": ";".join(missing),
            }
        )
    return pd.DataFrame(rows), complete


def _coded_basis_and_output(
    source_zip: Path,
    *,
    system: str,
) -> tuple[pd.Series, pd.MultiIndex, tuple[str, ...], tuple[str, ...]]:
    from ecolecon_mrio.data import _read_exio_x_from_zip
    from ecolecon_mrio.exiobase3 import _code_index, _exio_label_to_code_lookup

    output, basis = _read_exio_x_from_zip(source_zip)
    regions = basis.get_level_values("region").astype(str)
    sectors = pd.Index(basis.get_level_values("sector").astype(str))
    coded = _code_index(sectors, label_to_code=_exio_label_to_code_lookup(str(system)))
    coded_basis = pd.MultiIndex.from_arrays([regions, coded], names=["region", "sector"])
    coded_output = pd.Series(
        output.to_numpy(dtype="float64"),
        index=coded_basis,
        dtype="float64",
        name="x",
    )
    region_order = tuple(dict.fromkeys(map(str, regions)))
    sector_order = tuple(dict.fromkeys(map(str, coded)))
    return coded_output, basis, region_order, sector_order


def _resolve_scope(
    *,
    scope_code: str,
    scope_mapping_csv: Path | None,
    source_regions: Sequence[str],
):
    from ecolecon_mrio.region_aggregation import RegionScope, compile_region_aggregation

    if scope_mapping_csv is None:
        if scope_code not in set(source_regions):
            raise ValueError(
                f"Singleton scope {scope_code!r} is not an EXIO source region; provide --scope-mapping-csv."
            )
        return RegionScope(code=scope_code, members=(scope_code,))
    plan = compile_region_aggregation(
        source_regions=source_regions,
        mapping=scope_mapping_csv,
        scopes=(scope_code,),
        include_complement=False,
    )
    if len(plan.scopes) != 1:
        raise RuntimeError("Scope mapping did not compile exactly one requested scope.")
    return plan.scopes[0]


def _scope_sector_sum(
    values: pd.Series,
    *,
    coded_basis: pd.MultiIndex,
    members: Sequence[str],
    sector_order: Sequence[str],
) -> pd.Series:
    coded_values = pd.Series(
        values.to_numpy(dtype="float64"), index=coded_basis, dtype="float64"
    )
    regions = coded_basis.get_level_values("region").astype(str)
    mask = np.asarray(regions.isin(set(map(str, members))), dtype=bool)
    subset = coded_values.loc[mask]
    grouped = subset.groupby(level="sector", sort=False).sum().reindex(list(sector_order))
    if grouped.isna().any() or not np.isfinite(grouped.to_numpy(dtype="float64")).all():
        raise ValueError("Scope aggregation lost or corrupted the canonical EXIO sector basis.")
    return grouped.astype("float64")


def _target_output_from_dom(dom_dir: Path, axis: pd.Index) -> pd.Series:
    path = dom_dir / "x_split.csv"
    if not path.is_file():
        raise FileNotFoundError(f"DOM census requires canonical target output: {path}")
    output = _numeric_vector_csv(path)
    output = output.reindex(axis)
    if output.isna().any():
        raise ValueError("DOM x_split does not cover the reconstructed environmental projection axis.")
    if bool((output < 0.0).any()):
        raise ValueError("DOM target output must be non-negative for the production-support census.")
    return output.astype("float64")


def _write_csv(
    frame: pd.DataFrame, path: Path, *, columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = list(map(str, columns))
    if frame.empty:
        out = pd.DataFrame(columns=pd.Index(expected))
    else:
        missing = set(expected) - set(frame.columns)
        extra = set(frame.columns) - set(expected)
        if missing or extra:
            raise ValueError(
                f"Audit CSV schema mismatch for {path}: missing={sorted(missing)!r}, "
                f"extra={sorted(extra)!r}."
            )
        out = frame.loc[:, expected].copy()
    out.to_csv(path, index=False)


def _production_census(
    *,
    source_zip: Path,
    extensions: Sequence[str],
    source_basis: pd.MultiIndex,
    coded_basis: pd.MultiIndex,
    coded_output: pd.Series,
    scope: Any,
    P: pd.DataFrame,
    target_output: pd.Series,
    parent_by_label: Mapping[str, str],
    system: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    from ecolecon_mrio.exiobase3_extensions import load_exiobase3_extension_rows_with_units

    members = tuple(map(str, scope.members))
    sector_order = tuple(map(str, P.columns))
    x_scope = _scope_sector_sum(
        coded_output, coded_basis=coded_basis, members=members, sector_order=sector_order
    )
    mapped_x = pd.Series(
        P.to_numpy(dtype="float64") @ x_scope.to_numpy(dtype="float64"),
        index=P.index,
        dtype="float64",
    )
    projection_support = P.abs().sum(axis=1).gt(0.0)
    target_output = target_output.reindex(P.index).astype("float64")

    support_frames: list[pd.DataFrame] = []
    lineage_rows: list[dict[str, object]] = []
    source_errors: list[dict[str, object]] = []
    for extension in extensions:
        try:
            rows = load_exiobase3_extension_rows_with_units(
                source_zip,
                extension=str(extension),
                kind="F",
                keywords=(),
                region=None,
                system=str(system),
                basis_index=source_basis,
            )
        except Exception as exc:  # diagnostic census records source-level failures and continues
            source_errors.append(
                {"extension": extension, "kind": "F", "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for row in rows:
            f_scope = _scope_sector_sum(
                row.values,
                coded_basis=coded_basis,
                members=members,
                sector_order=sector_order,
            )
            mapped_f = pd.Series(
                P.to_numpy(dtype="float64") @ f_scope.to_numpy(dtype="float64"),
                index=P.index,
                dtype="float64",
            )
            classified = classify_production_support(
                mapped_value=mapped_f,
                mapped_output=mapped_x,
                target_output=target_output,
                projection_support=projection_support,
            )
            classified.insert(0, "unit", row.unit)
            classified.insert(0, "stressor", row.stressor)
            classified.insert(0, "extension", row.extension)
            classified["reaggregation_parent"] = classified["sector"].map(
                lambda value: parent_by_label.get(str(value), str(value))
            )
            support_frames.append(classified)

            exceptional = classified.loc[
                classified["support_state"].ne("SUPPORTED")
                | classified["method_blocking_positive_target"].astype(bool)
            ]
            for exception in exceptional.itertuples(index=False):
                sector = str(exception.sector)
                weights = P.loc[sector]
                active_sources = weights.index[weights.ne(0.0)]
                for source in active_sources:
                    weight = float(weights.loc[source])
                    lineage_rows.append(
                        {
                            "extension": row.extension,
                            "stressor": row.stressor,
                            "unit": row.unit,
                            "sector": sector,
                            "reaggregation_parent": parent_by_label.get(sector, sector),
                            "source_code": str(source),
                            "projection_weight": weight,
                            "raw_scope_F": float(f_scope.loc[source]),
                            "raw_scope_x": float(x_scope.loc[source]),
                            "weighted_F_contribution": weight * float(f_scope.loc[source]),
                            "weighted_x_contribution": weight * float(x_scope.loc[source]),
                            "mapped_F": float(exception.mapped_F),
                            "mapped_x": float(exception.mapped_x),
                            "target_output": float(exception.target_output),
                            "support_state": str(exception.support_state),
                            "target_state": str(exception.target_state),
                            "method_blocking_positive_target": bool(
                                exception.method_blocking_positive_target
                            ),
                        }
                    )
    support = pd.concat(support_frames, ignore_index=True) if support_frames else pd.DataFrame()
    lineage = pd.DataFrame(lineage_rows)
    return support, lineage, source_errors


def _external_import_census(
    *,
    source_zip: Path,
    coded_basis: pd.MultiIndex,
    coded_output: pd.Series,
    scope: Any,
    P: pd.DataFrame,
    system: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    from ecolecon_mrio.exiobase3 import (
        _code_index,
        _exio_label_to_code_lookup,
        read_exiobase3_scope_external_import_use_by_origin,
    )

    raw_use = read_exiobase3_scope_external_import_use_by_origin(source_zip=source_zip, scope=scope)
    use_regions = raw_use.index.get_level_values("region").astype(str)
    use_sectors = pd.Index(raw_use.index.get_level_values("sector").astype(str))
    use_codes = _code_index(use_sectors, label_to_code=_exio_label_to_code_lookup(str(system)))
    coded_use = pd.Series(
        raw_use.to_numpy(dtype="float64"),
        index=pd.MultiIndex.from_arrays([use_regions, use_codes], names=["region", "sector"]),
        dtype="float64",
    )
    coded_use_basis = coded_use.index
    if not isinstance(coded_use_basis, pd.MultiIndex):
        raise TypeError("External-import use must retain a region-sector MultiIndex.")
    members = set(map(str, scope.members))
    all_regions = tuple(dict.fromkeys(map(str, coded_basis.get_level_values("region"))))
    origins = tuple(region for region in all_regions if region not in members)
    rows: list[dict[str, object]] = []
    total_pairs = 0
    active_pairs = 0
    for origin in origins:
        x_origin = _scope_sector_sum(
            coded_output,
            coded_basis=coded_basis,
            members=(origin,),
            sector_order=tuple(map(str, P.columns)),
        )
        u_origin = _scope_sector_sum(
            coded_use,
            coded_basis=coded_use_basis,
            members=(origin,),
            sector_order=tuple(map(str, P.columns)),
        )
        mapped_x = P.to_numpy(dtype="float64") @ x_origin.to_numpy(dtype="float64")
        mapped_u = P.to_numpy(dtype="float64") @ u_origin.to_numpy(dtype="float64")
        for pos, target in enumerate(P.index.astype(str)):
            total_pairs += 1
            x_value = float(mapped_x[pos])
            use_value = float(mapped_u[pos])
            if use_value != 0.0:
                active_pairs += 1
            if use_value != 0.0 and x_value <= 0.0:
                rows.append(
                    {
                        "origin": origin,
                        "target_sector": str(target),
                        "mapped_import_use": use_value,
                        "mapped_production_output": x_value,
                        "method_blocking": True,
                    }
                )
    return pd.DataFrame(rows), {"total_origin_target_pairs": total_pairs, "nonzero_use_pairs": active_pairs}


def _fy_census(
    *,
    source_zip: Path,
    extensions: Sequence[str],
    scope: Any,
    system: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    from ecolecon_mrio.exiobase3 import read_exiobase3_scope_final_demand_totals
    from ecolecon_mrio.exiobase3_extensions import load_exiobase3_extension_rows_with_units

    denominator = read_exiobase3_scope_final_demand_totals(source_zip=source_zip, scope=scope)
    denominator.index = pd.Index([str(value) for value in denominator.index])
    rows_out: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    members = set(map(str, scope.members))
    for extension in extensions:
        try:
            rows = load_exiobase3_extension_rows_with_units(
                source_zip,
                extension=str(extension),
                kind="F_Y",
                keywords=(),
                region=None,
                system=str(system),
            )
        except Exception as exc:
            errors.append(
                {"extension": extension, "kind": "F_Y", "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for row in rows:
            idx = row.values.index
            regions = idx.get_level_values("region").astype(str)
            categories = idx.get_level_values("final_demand_category").astype(str)
            mask = np.asarray(regions.isin(members), dtype=bool)
            physical = pd.Series(
                row.values.to_numpy(dtype="float64")[mask],
                index=pd.Index(categories[mask], name="final_demand_category"),
                dtype="float64",
            ).groupby(level=0, sort=False).sum()
            if not isinstance(physical, pd.Series):
                raise TypeError("F_Y aggregation must produce a one-dimensional category vector.")
            physical = physical.reindex(denominator.index)
            if bool(np.asarray(physical.isna(), dtype=bool).any()):
                raise ValueError(
                    f"F_Y scope categories do not match monetary denominator for {extension!r}, {row.stressor!r}."
                )
            classified = classify_final_demand_support(physical=physical, monetary=denominator)
            classified.insert(0, "unit", row.unit)
            classified.insert(0, "stressor", row.stressor)
            classified.insert(0, "extension", row.extension)
            rows_out.append(classified)
    return (pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()), errors


def _summary_counts(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, object]]:
    if frame.empty:
        return []
    grouped = frame.groupby(list(columns), dropna=False, sort=True).size()
    if not isinstance(grouped, pd.Series):
        raise TypeError("Summary grouping must produce a one-dimensional count vector.")
    summary_frame = grouped.rename("count").reset_index()
    return cast(list[dict[str, object]], summary_frame.to_dict(orient="records"))


def _source_discovery_errors(complete_extensions: Sequence[str]) -> list[dict[str, object]]:
    if complete_extensions:
        return []
    return [
        {
            "extension": "",
            "kind": "DISCOVERY",
            "error": "No complete EXIOBASE extension directories with F.txt, F_Y.txt, and unit.txt were discovered.",
        }
    ]


def run_census(args: argparse.Namespace) -> dict[str, object]:
    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_year_dir

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.source_zip:
        source_zip = Path(args.source_zip).resolve()
        if not source_zip.is_file():
            raise FileNotFoundError(source_zip)
    else:
        year_dir = resolve_exiobase_year_dir(int(args.year), data_root=args.data_root)
        source_zip = ensure_exiobase_year_zip(
            int(args.year),
            data_dir=year_dir,
            system=str(args.system),
            force_download=False,
            allow_download_missing=False,
            resume_downloads=True,
            show_download_progress=False,
            doi=str(args.doi),
            verbose=False,
        )

    completeness, complete_extensions = _source_completeness(source_zip)
    _write_csv(
        completeness, outdir / "source_extension_completeness.csv",
        columns=_SOURCE_COMPLETENESS_COLUMNS,
    )
    incomplete_count = int((~completeness["complete"].astype(bool)).sum()) if not completeness.empty else 0

    coded_output, source_basis, source_regions, raw_sector_order = _coded_basis_and_output(
        source_zip, system=str(args.system)
    )
    coded_basis = cast(pd.MultiIndex, coded_output.index)
    scope = _resolve_scope(
        scope_code=str(args.scope_code),
        scope_mapping_csv=None if args.scope_mapping_csv is None else Path(args.scope_mapping_csv),
        source_regions=source_regions,
    )

    split_root = Path(args.split_root).resolve()
    dom_dir = split_root / "stk_flow=DOM"
    imp_dir = split_root / "stk_flow=IMP"
    if not dom_dir.is_dir() or not imp_dir.is_dir():
        raise FileNotFoundError(
            f"split_root must contain stk_flow=DOM and stk_flow=IMP directories: {split_root}"
        )

    scope_mapping_path = (
        None if args.scope_mapping_csv is None else Path(args.scope_mapping_csv).resolve()
    )
    provenance = validate_split_provenance(
        split_root=split_root,
        source_zip=source_zip,
        scope_code=str(scope.code),
        scope_members=tuple(map(str, scope.members)),
        scope_mapping_csv=scope_mapping_path,
    )

    P_dom, parent_by_label = build_projection_from_split_artifacts(
        dom_dir, raw_exio_sectors=raw_sector_order
    )
    target_output = _target_output_from_dom(dom_dir, P_dom.index)
    production, lineage, prod_errors = _production_census(
        source_zip=source_zip,
        extensions=complete_extensions,
        source_basis=source_basis,
        coded_basis=coded_basis,
        coded_output=coded_output,
        scope=scope,
        P=P_dom,
        target_output=target_output,
        parent_by_label=parent_by_label,
        system=str(args.system),
    )
    _write_csv(production, outdir / "production_F_support.csv", columns=_PRODUCTION_SUPPORT_COLUMNS)
    _write_csv(lineage, outdir / "production_F_exception_lineage.csv", columns=_PRODUCTION_LINEAGE_COLUMNS)

    P_imp, _imp_parent = build_projection_from_split_artifacts(
        imp_dir, raw_exio_sectors=raw_sector_order
    )
    if not P_imp.index.equals(P_dom.index) or not P_imp.columns.equals(P_dom.columns):
        raise ValueError("DOM and IMP persisted projections must use the same emitted/raw axes.")
    import_exceptions, import_counts = _external_import_census(
        source_zip=source_zip,
        coded_basis=coded_basis,
        coded_output=coded_output,
        scope=scope,
        P=P_imp,
        system=str(args.system),
    )
    _write_csv(import_exceptions, outdir / "external_import_support_exceptions.csv", columns=_IMPORT_EXCEPTION_COLUMNS)

    fy_support, fy_errors = _fy_census(
        source_zip=source_zip,
        extensions=complete_extensions,
        scope=scope,
        system=str(args.system),
    )
    _write_csv(fy_support, outdir / "final_demand_FY_support.csv", columns=_FY_SUPPORT_COLUMNS)
    source_errors = pd.DataFrame([*_source_discovery_errors(complete_extensions), *prod_errors, *fy_errors])
    _write_csv(source_errors, outdir / "source_row_errors.csv", columns=_SOURCE_ERROR_COLUMNS)

    production_summary = _summary_counts(
        production, ["support_state", "target_state", "method_blocking_positive_target"]
    )
    fy_summary = _summary_counts(fy_support, ["support_state", "method_blocking"])
    blocking_production = (
        int(production["method_blocking_positive_target"].astype(bool).sum())
        if not production.empty
        else 0
    )
    nonzero_zero_target_zero = (
        int(
            (
                production["support_state"].eq("NONZERO_F_ZERO_X")
                & production["target_state"].eq("ZERO")
            ).sum()
        )
        if not production.empty
        else 0
    )
    nonzero_zero_target_positive = (
        int(
            (
                production["support_state"].eq("NONZERO_F_ZERO_X")
                & production["target_state"].eq("POSITIVE")
            ).sum()
        )
        if not production.empty
        else 0
    )
    blocking_fy = int(fy_support["method_blocking"].astype(bool).sum()) if not fy_support.empty else 0
    blocking_import = int(len(import_exceptions))
    source_error_count = int(len(source_errors))
    summary: dict[str, object] = {
        "source_zip": str(source_zip),
        "provenance": provenance,
        "scope_code": str(scope.code),
        "scope_members": list(map(str, scope.members)),
        "split_root": str(split_root),
        "extension_count": len(complete_extensions),
        "incomplete_extension_count": incomplete_count,
        "production_rows": int(len(production)),
        "production_support_counts": production_summary,
        "nonzero_F_zero_x_target_zero": nonzero_zero_target_zero,
        "nonzero_F_zero_x_target_positive": nonzero_zero_target_positive,
        "production_positive_target_blockers": blocking_production,
        "external_import_nonzero_use_nonpositive_x": blocking_import,
        "external_import_counts": import_counts,
        "final_demand_support_counts": fy_summary,
        "final_demand_nonzero_FY_zero_denominator": blocking_fy,
        "source_row_error_count": source_error_count,
        "strict_method_blocker_count": (
            incomplete_count
            + source_error_count
            + blocking_production
            + blocking_import
            + blocking_fy
        ),
    }
    (outdir / "census_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate EXIO production-F, external-import, and F_Y denominator-support "
            "states against an already-successful monetary split. The census is read-only "
            "and does not construct or validate model-facing environmental estimates."
        )
    )
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--source-zip", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--doi", default="10.5281/zenodo.20051562")
    parser.add_argument("--system", default="pxp")
    parser.add_argument("--scope-code", required=True)
    parser.add_argument("--scope-mapping-csv", default=None)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with status 2 after writing the complete census when any source-integrity "
            "or current-method support blocker is present. Without --strict the census always "
            "finishes successfully after recording support states."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_census(args)
    blocker_count = summary["strict_method_blocker_count"]
    if not isinstance(blocker_count, (int, float)):
        raise TypeError("Census summary has a non-numeric strict blocker count.")
    if args.strict and int(blocker_count) != 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import time
import re
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np
import pandas as pd

from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI, default_data_root, load_mrio_iot_matrices
from ecolecon_mrio.exio_share_split import (
    load_local_iot,
    split_iot_using_exiobase_split_profiles,
    write_split_report_xlsx,
    map_exio_value_per_physical_input,
    emitted_sector_to_exio_members_from_split_profiles,
)
from ecolecon_mrio.region_aggregation import (
    MrioScopeSlice,
    RegionScope,
    ScopeFlow,
    compile_region_aggregation,
    load_mrio_iot_scope_slice,
)
from ecolecon_mrio.exiobase3_extensions import discover_exiobase3_extensions
from ecolecon_mrio.eurostat_iot import (
    ParsedEurostatIOT,
    build_eurostat_atomic_final_demand,
    build_eurostat_product_use_reference,
    load_eurostat_iot,
    load_eurostat_iot_variants,
)
from ecolecon_mrio.canonical_mapping_csv import read_canonical_split_mapping
from ecolecon_mrio.mapping_csv import (
    load_weighted_mapping_frame,
    read_split_mapping,
    read_weighted_mapping,
    validate_mapping_weights,
)


DEFAULT_PATTERNS_19: tuple[str, ...] = ("coke", "petroleum", "refined")
DEFAULT_PATTERNS_35: tuple[str, ...] = (
    "electricity",
    "gas",
    "steam",
    "hot water",
    "air conditioning",
)
DEFAULT_ELECTRICITY_BY_SOURCE_PATTERNS: tuple[str, ...] = (
    "electricity from",
    "electricity by",
    "nuclear",
    "hydro",
    "wind",
    "solar",
    "biomass",
    "geothermal",
)


def _load_exio_scope_slice(
    *,
    year: int,
    scope: RegionScope,
    flow: str,
    data_root: Path,
    doi: str,
    force_download: bool,
    verbose: bool,
) -> MrioScopeSlice:
    """Load one already-resolved singleton or grouped EXIO donor scope."""
    scope_flow = {"domestic": "internal", "imports": "external_imports", "total": "total"}.get(
        str(flow), str(flow)
    )
    return load_mrio_iot_scope_slice(
        "exiobase",
        int(year),
        scope=scope,
        exio_system="pxp",
        flow=cast(ScopeFlow, scope_flow),
        data_root=data_root,
        doi=doi,
        force_download=force_download,
        verbose=verbose,
    )


def _resolve_pipeline_scope(
    *,
    scope_code: str,
    mapping_csv: str | Path | None,
    source_regions: Sequence[str] | None,
) -> RegionScope:
    code = str(scope_code).strip()
    if not code:
        raise ValueError("EXIO donor scope code must not be blank.")
    if mapping_csv is None:
        return RegionScope(code=code, members=(code,))
    if source_regions is None:
        raise ValueError(
            "Mapping-backed donor scopes require the actual EXIO source-region basis."
        )
    plan = compile_region_aggregation(
        source_regions=source_regions,
        mapping=Path(mapping_csv),
        scopes=(code,),
        include_complement=False,
    )
    if len(plan.scopes) != 1 or plan.scopes[0].code != code:
        raise RuntimeError("Region mapping did not compile exactly the requested donor scope.")
    return plan.scopes[0]



def _sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact local input bytes used by a run."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _vprint(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


class _Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


def _is_canonical_mapping_csv(path: str | Path) -> bool:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    required = {
        "source_code",
        "source_label",
        "target_code",
        "target_label",
        "weight",
        "enabled",
        "notes",
    }
    return required.issubset(set(header.columns))


def _canonical_source_dictionary_path(
    mapping_csv: str | Path, classification: str
) -> Path:
    base = Path(mapping_csv).resolve().parent
    if classification == "industry":
        return base / "exiobase_industries_dictionary.csv"
    return base / "exiobase_products_dictionary.csv"


def _canonical_target_dictionary_path(
    mapping_csv: str | Path, classification: str
) -> Path:
    base = Path(mapping_csv).resolve().parent
    if classification == "industry":
        return base / "eurostat_nace_rev_2_1_industries_dictionary.csv"
    return base / "eurostat_cpa65_products_dictionary.csv"


def _write_text(path: Path, content: str, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_df_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_series_csv(
    series: pd.Series, path: Path, *, verbose: bool, header: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, header=header)
    _vprint(verbose, f"Wrote: {path.resolve()}")


def _write_rows_csv(df: pd.DataFrame, path: Path, *, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _vprint(verbose, f"Wrote: {path.resolve()}")


SPLIT_MANAGED_OUTPUT_NAMES: tuple[str, ...] = (
    "Z_split.csv",
    "Z_split_balanced.csv",
    "Y_split.csv",
    "Y_split_balanced.csv",
    "x_split.csv",
    "x_split_balanced.csv",
    "A_split.csv",
    "A_split_balanced.csv",
    "V_split.csv",
    "V_split_balanced.csv",
    "M_split.csv",
    "M_split_balanced.csv",
    "use_total_split.csv",
    "supplementary_row_split_diagnostics.csv",
    "split_output_shares.csv",
    "fallback_diagnostics.csv",
    "exio_selected_sectors.json",
    "split_info.json",
    "balance_info.json",
    "ras_info.json",
    "flow_specific_exio_to_eurostat_weights.csv",
    "mapped_output_consistency.csv",
    "split_child_edges.csv",
    "exio_quantity_and_price.csv",
    "exio_quantity_and_price_proxies.csv",
    "disaggregated_sector_value_per_physical_input.csv",
    "exio_environmental_extensions.csv",
    "environmental_extensions.csv",
    # Removed CO2-only sidecars remain managed only so reruns clear stale files.
    "exio_co2_intensity.csv",
    "disaggregated_sector_co2_intensity.csv",
    "exio_final_demand_co2_intensity.csv",
    "eurostat_sector_co2_intensity.csv",
    "energy_split_report.xlsx",
)


def _clear_managed_split_outputs(outdir: Path) -> None:
    for name in SPLIT_MANAGED_OUTPUT_NAMES:
        (outdir / name).unlink(missing_ok=True)


def _split_info_payload(
    result,
    *,
    variant_dim: str | None = None,
    variant_value: str | None = None,
    **extra_info: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "method": result.balancing_method,
        "is_reconstruction_validated": True,
        "balancing_applied": False,
        "fallback_count": result.fallback_count,
        "has_fallback_diagnostics": result.fallback_diagnostics is not None
        and not result.fallback_diagnostics.empty,
    }
    if variant_dim is not None:
        payload["variant_dim"] = variant_dim
    if variant_value is not None:
        payload["variant_value"] = variant_value
    payload.update(
        {str(key): value for key, value in extra_info.items() if value is not None}
    )
    return payload


def write_split_core_outputs(
    res, outdir: Path, *, verbose: bool, extra_info: dict | None = None
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name in (
        "x_split.csv",
        "A_split.csv",
        "V_split.csv",
        "M_split.csv",
        "use_total_split.csv",
        "supplementary_row_split_diagnostics.csv",
        "split_output_shares.csv",
        "fallback_diagnostics.csv",
    ):
        (outdir / name).unlink(missing_ok=True)
    share_rows = []
    for parent, profile in res.split_profiles.items():
        children = list(map(str, profile.subsectors))
        if len(children) <= 1:
            continue
        shares = profile.output_share.reindex(children).astype("float64")
        if not np.isfinite(shares.to_numpy()).all() or (shares < 0).any() or abs(float(shares.sum()) - 1.0) > 1e-12:
            raise ValueError(f"Invalid split output shares for {parent!r}.")
        share_rows.extend({"parent": str(parent), "child": child, "output_share": float(shares.loc[child])} for child in children)
    if share_rows:
        _write_rows_csv(pd.DataFrame(share_rows), outdir / "split_output_shares.csv", verbose=verbose)
    _write_df_csv(res.Z, outdir / "Z_split.csv", verbose=verbose)
    _write_df_csv(res.Y, outdir / "Y_split.csv", verbose=verbose)
    if res.use_total is not None:
        _write_series_csv(
            res.use_total, outdir / "use_total_split.csv", verbose=verbose, header=True
        )

    _write_text(
        outdir / "exio_selected_sectors.json",
        json.dumps(res.exio_selected_sectors, indent=2),
        verbose=verbose,
    )
    split_info = _split_info_payload(res, **(extra_info or {}))
    _write_text(
        outdir / "split_info.json", json.dumps(split_info, indent=2), verbose=verbose
    )
    if res.fallback_diagnostics is not None and not res.fallback_diagnostics.empty:
        _write_df_csv(
            res.fallback_diagnostics,
            outdir / "fallback_diagnostics.csv",
            verbose=verbose,
        )
    if getattr(res, "V", None) is not None and not res.V.empty:
        _write_df_csv(res.V, outdir / "V_split.csv", verbose=verbose)
    if getattr(res, "M", None) is not None and not res.M.empty:
        _write_df_csv(res.M, outdir / "M_split.csv", verbose=verbose)
    if (
        getattr(res, "supplementary_diagnostics", None) is not None
        and not res.supplementary_diagnostics.empty
    ):
        _write_rows_csv(
            res.supplementary_diagnostics,
            outdir / "supplementary_row_split_diagnostics.csv",
            verbose=verbose,
        )
    if (
        getattr(res, "exio_environmental_extensions", None) is not None
        and not res.exio_environmental_extensions.empty
    ):
        _write_rows_csv(
            res.exio_environmental_extensions,
            outdir / "exio_environmental_extensions.csv",
            verbose=verbose,
        )
    if (
        getattr(res, "environmental_extensions", None) is not None
        and not res.environmental_extensions.empty
    ):
        _write_rows_csv(
            res.environmental_extensions,
            outdir / "environmental_extensions.csv",
            verbose=verbose,
        )


def _read_exio_to_eurostat_mapping_with_notes(
    path: str | Path,
    notes_col: str = "notes",
    *,
    target_output_by_code: object | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    df = load_weighted_mapping_frame(path)
    mapping = read_weighted_mapping(path, target_output_by_code=target_output_by_code)
    validate_mapping_weights(mapping)

    notes_by_sector: dict[str, list[str]] = {}
    if notes_col in df.columns:
        for _, row in df.iterrows():
            agg = str(row["aggregate_sector"])
            note_val = str(row.get(notes_col, "")).strip()
            if note_val:
                notes_by_sector.setdefault(agg, []).append(note_val)

    condensed_notes: dict[str, str] = {}
    for agg, notes in notes_by_sector.items():
        uniq = []
        seen = set()
        for n in notes:
            if n not in seen:
                uniq.append(n)
                seen.add(n)
        condensed_notes[agg] = " | ".join(uniq)

    return mapping, condensed_notes


def _eurostat_export_final_demand_codes(parsed: ParsedEurostatIOT) -> tuple[str, ...]:
    """Return Eurostat export final-demand columns present in a parsed IOT."""
    return tuple(
        str(column)
        for column in parsed.Y.columns
        if str(column) == "P6" or str(column).startswith("P6_")
    )


def _eurostat_non_export_use_reference_from_parsed(
    parsed: ParsedEurostatIOT,
) -> pd.Series:
    """Build a destination-use reference for flow-specific concordance weights.

    This excludes export final-demand columns so DOM/IMP/TOTAL weights are based
    on products used by the destination country rather than producer-output or
    export-inclusive final demand.
    """
    reference_raw = build_eurostat_product_use_reference(
        parsed,
        exclude_final_demand_codes=_eurostat_export_final_demand_codes(parsed),
    )
    reference = (
        cast(pd.Series, pd.to_numeric(reference_raw, errors="coerce"))
        .fillna(0.0)
        .astype("float64")
    )
    reference.index = reference.index.astype(str)
    return reference


def _resolve_exio_to_eurostat_mapping_for_parsed(
    path: str | Path,
    parsed: ParsedEurostatIOT,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Resolve EXIO-to-Eurostat weights from this flow's Eurostat use reference."""
    return _read_exio_to_eurostat_mapping_with_notes(
        path,
        target_output_by_code=_eurostat_non_export_use_reference_from_parsed(parsed),
    )


def _mapping_weights_frame(
    sector_mapping: Mapping[str, Mapping[str, float]],
    *,
    variant_dim: str | None = None,
    variant_value: str | None = None,
    exio_flow: str | None = None,
    reference_basis: str = "eurostat_Z_plus_non_export_Y",
) -> pd.DataFrame:
    """Flatten resolved weights for reproducibility diagnostics."""
    rows: list[dict[str, object]] = []
    for target_code, source_weights in sector_mapping.items():
        for source_code, weight in source_weights.items():
            rows.append(
                {
                    "variant_dim": variant_dim or "",
                    "variant_value": variant_value or "",
                    "exio_flow": exio_flow or "",
                    "source_code": str(source_code),
                    "target_code": str(target_code),
                    "weight": float(weight),
                    "reference_basis": reference_basis,
                }
            )
    return pd.DataFrame(
        rows,
        columns=pd.Index([
            "variant_dim",
            "variant_value",
            "exio_flow",
            "source_code",
            "target_code",
            "weight",
            "reference_basis",
        ]),
    )


def _flatten_selected_child_codes(value: object) -> list[str]:
    """Flatten split-output selected-child structures while preserving order."""
    out: list[str] = []

    def _visit(obj: object) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            for nested in obj.values():
                _visit(nested)
            return
        if isinstance(obj, IterableABC) and not isinstance(obj, (str, bytes)):
            for nested in obj:
                _visit(nested)
            return
        text_value = str(obj).strip()
        if text_value:
            out.append(text_value)

    _visit(value)
    return list(dict.fromkeys(out))


def _source_targets_from_sector_mapping(
    sector_mapping: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    source_targets: dict[str, dict[str, float]] = {}
    if sector_mapping is None:
        return source_targets
    for target_code, source_weights in sector_mapping.items():
        target = str(target_code)
        for source_code, weight in source_weights.items():
            source = str(source_code)
            weight_float = float(weight)
            if weight_float == 0.0:
                continue
            source_targets.setdefault(source, {})[target] = weight_float
    return source_targets


def _parse_edge_child_code(child_code: object) -> tuple[str, str] | None:
    text = str(child_code).strip()
    if "@" not in text:
        return None
    source, target = text.split("@", 1)
    source = source.strip()
    target = target.strip()
    if not source or not target:
        return None
    return source, target


def _build_split_child_edge_metadata(
    *,
    selected_children: Mapping[str, object],
    sector_mapping: Mapping[str, Mapping[str, float]] | None,
    variant_dim: str | None = None,
    variant_value: str | None = None,
    exio_flow: str | None = None,
) -> pd.DataFrame:
    """Build metadata for Phase-4 edge-aware split children."""
    source_targets = _source_targets_from_sector_mapping(sector_mapping)
    rows: list[dict[str, object]] = []
    for parent_code, child_values in selected_children.items():
        parent = str(parent_code)
        for child_code in _flatten_selected_child_codes(child_values):
            edge = _parse_edge_child_code(child_code)
            if edge is not None:
                source_code, concordance_target_code = edge
                target_weights = source_targets.get(source_code, {})
                source_n_targets = len(target_weights) or 1
                proposed_child_code = child_code
                edge_weight = target_weights.get(concordance_target_code)
                phase = "phase4_edge_output"
                notes = (
                    "Current child code is a materialized source-target edge child. "
                    "source_code is the raw EXIO source and concordance_target_code is the mapped Eurostat target."
                )
                selected_parent = (
                    concordance_target_code in target_weights
                    if target_weights
                    else False
                )
            else:
                source_code = child_code
                concordance_target_code = parent
                target_weights = source_targets.get(source_code, {})
                source_n_targets = len(target_weights) or 1
                edge_weight = target_weights.get(parent)
                selected_parent = parent in target_weights if target_weights else False
                proposed_child_code = (
                    f"{source_code}@{parent}"
                    if source_n_targets > 1 and selected_parent
                    else source_code
                )
                if source_n_targets > 1:
                    phase = "phase4_incomplete_edge_materialization"
                    notes = (
                        "Raw child code still represents a many-target EXIO source; "
                        "edge-aware output should materialize proposed_edge_child_code."
                    )
                else:
                    phase = "phase4_raw_single_target_output"
                    notes = "Raw child code has a single mapped target or no weighted concordance split."

            is_many_target_source = source_n_targets > 1
            rows.append(
                {
                    "variant_dim": variant_dim or "",
                    "variant_value": variant_value or "",
                    "exio_flow": exio_flow or "",
                    "parent_code": parent,
                    "current_child_code": child_code,
                    "proposed_edge_child_code": proposed_child_code,
                    "source_code": source_code,
                    "concordance_target_code": concordance_target_code,
                    "source_n_targets": int(source_n_targets),
                    "is_many_target_source": bool(is_many_target_source),
                    "is_edge_child_candidate": bool(is_many_target_source),
                    "edge_weight_current_reference": ""
                    if edge_weight is None
                    else edge_weight,
                    "selected_parent_in_concordance": bool(selected_parent),
                    "phase": phase,
                    "notes": notes,
                }
            )
    columns = [
        "variant_dim",
        "variant_value",
        "exio_flow",
        "parent_code",
        "current_child_code",
        "proposed_edge_child_code",
        "source_code",
        "concordance_target_code",
        "source_n_targets",
        "is_many_target_source",
        "is_edge_child_candidate",
        "edge_weight_current_reference",
        "selected_parent_in_concordance",
        "phase",
        "notes",
    ]
    return pd.DataFrame(rows, columns=pd.Index(columns))


def _eurostat_output_reference_from_frames(
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
) -> pd.Series:
    """Build the Eurostat destination-use reference used for output checks.

    This intentionally matches the boundary used to resolve flow-specific
    EXIO->Eurostat weights: product rows plus non-export final demand. Exports
    are excluded for DOM, IMP, and TOTAL so mapped EXIO domestic/import/total
    use slices are checked against the same destination-use object used for the
    split mapping weights.
    """
    parsed = ParsedEurostatIOT(Z=eurostat_Z, Y=eurostat_Y, meta={})
    reference_raw = build_eurostat_product_use_reference(
        parsed,
        exclude_final_demand_codes=_eurostat_export_final_demand_codes(parsed),
    )
    reference = (
        cast(pd.Series, pd.to_numeric(reference_raw, errors="coerce"))
        .fillna(0.0)
        .astype("float64")
    )
    reference.index = reference.index.astype(str)
    return reference


def _eurostat_output_rounding_bound(parsed: ParsedEurostatIOT) -> float:
    quantum_value = parsed.meta.get("source_rounding_quantum")
    if quantum_value is None:
        return 0.0
    quantum = float(quantum_value)
    if not np.isfinite(quantum) or quantum <= 0.0:
        raise ValueError(f"Invalid Eurostat source rounding quantum: {quantum_value!r}")
    atomic_non_export = build_eurostat_atomic_final_demand(
        parsed,
        include_exports=False,
    )
    component_terms = int(parsed.Z.size + atomic_non_export.size)
    return 0.5 * quantum * component_terms


def _assert_mapping_integrity(
    *,
    sector_mapping: dict[str, dict[str, float]],
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    required_target_codes: Sequence[str] | None = None,
    label: str,
) -> None:
    euro_x = _eurostat_output_reference_from_frames(eurostat_Z, eurostat_Y)
    if required_target_codes is None:
        required_target_codes = tuple(map(str, euro_x.index))
    required_codes = [
        str(code) for code in required_target_codes if str(code).strip() != ""
    ]
    missing = sorted([code for code in required_codes if code not in sector_mapping])
    if missing:
        show = ", ".join(missing[:10])
        raise SystemExit(
            "Mapping integrity check failed: missing required target codes in mapping "
            f"({label}). Examples: {show}"
        )
    try:
        validate_mapping_weights(sector_mapping)
    except ValueError as exc:
        raise SystemExit(f"Mapping integrity check failed ({label}): {exc}") from exc


def _build_output_consistency_frame(
    *,
    sector_mapping: dict[str, dict[str, float]],
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_slice: MrioScopeSlice,
) -> tuple[pd.DataFrame, float, float]:
    euro_x = _eurostat_output_reference_from_frames(eurostat_Z, eurostat_Y)

    # Match the EXIO side to the same destination-use boundary as the Eurostat
    # reference: rows from the selected EXIO flow into the destination region plus
    # that destination region's final demand. Scope production x is deliberately
    # not used for IMP/TOTAL destination-use checks.
    exio_Z = exio_slice.Z.copy().astype("float64")
    exio_Y = exio_slice.Y.copy().astype("float64")
    exio_Z.index = exio_Z.index.astype(str)
    exio_Y.index = exio_Y.index.astype(str)
    exio_out = (
        exio_Z.sum(axis=1) + exio_Y.sum(axis=1).reindex(exio_Z.index).fillna(0.0)
    ).astype("float64")
    exio_out.index = exio_out.index.astype(str)

    mapped_rows: list[dict[str, float | str | bool]] = []
    for cpa, euro_val in euro_x.items():
        total = 0.0
        for exio, weight in sector_mapping.get(str(cpa), {}).items():
            if exio in exio_out.index:
                total += float(exio_out.get(exio, 0.0)) * float(weight)  # pyright: ignore[reportArgumentType]  # pandas Series.get stubs allow None; membership guard/default keep this numeric
        mapped_rows.append(
            {
                "cpa": str(cpa),
                "eurostat_output": float(euro_val),
                "exio_mapped_output": float(total),
            }
        )

    frame = pd.DataFrame(mapped_rows)
    exio_total = float(frame["exio_mapped_output"].sum())
    euro_total = float(frame["eurostat_output"].sum())
    if (
        np.isfinite(exio_total)
        and exio_total > 0
        and np.isfinite(euro_total)
        and euro_total > 0
    ):
        scale = euro_total / exio_total
    else:
        scale = np.nan

    frame["exio_scale_to_eurostat_total"] = scale
    frame["exio_scaled_output"] = frame["exio_mapped_output"] * scale
    frame["abs_diff_scaled"] = (
        frame["exio_scaled_output"] - frame["eurostat_output"]
    ).abs()
    frame["rel_diff_scaled"] = np.where(
        frame["eurostat_output"].abs() > 0,
        frame["abs_diff_scaled"] / frame["eurostat_output"].abs(),
        np.nan,
    )
    return frame, exio_total, euro_total


def _assert_output_consistency(
    *,
    sector_mapping: dict[str, dict[str, float]],
    eurostat_Z: pd.DataFrame,
    eurostat_Y: pd.DataFrame,
    exio_slice: MrioScopeSlice,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    _, exio_total, euro_total = _build_output_consistency_frame(
        sector_mapping=sector_mapping,
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        exio_slice=exio_slice,
    )
    if (
        not np.isfinite(exio_total)
        or not np.isfinite(euro_total)
        or exio_total <= 0
        or euro_total <= 0
    ):
        raise SystemExit(
            "Mapping consistency check failed: invalid total outputs for comparison "
            f"({label}). EXIO destination-use total={exio_total}, Eurostat destination-use total={euro_total}."
        )

    diff_total = abs(exio_total - euro_total)
    tol_total = max(atol, rtol * abs(euro_total))
    if diff_total > tol_total:
        raise SystemExit(
            "Mapping consistency check failed: EXIO-mapped total output does not match the Eurostat "
            f"total output reference within tolerances (rtol={rtol}, atol={atol}) for {label}. "
            f"EXIO destination-use total={exio_total:.6g}, Eurostat destination-use total={euro_total:.6g}, diff={diff_total:.6g}, tol={tol_total:.6g}."
        )


def _extract_year(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if s.isdigit() and len(s) == 4:
        return int(s)
    m = re.search(r"(\\d{4})", s)
    if m:
        return int(m.group(1))
    return None


def _should_check_output(
    flow_value: object, eurostat_year: int | None, exio_year: int
) -> bool:
    """Return whether a stk_flow variant should run mapped-output checks.

    The check is deliberately flow-based, not year-gated. The fuels pipeline can
    compare Eurostat 2021 with EXIOBASE 2021 by default, but users may still run
    sensitivity checks with another EXIO year; the diagnostic should run and
    report the boundary-consistent gap instead of silently skipping.
    """
    flow = str(flow_value).strip().upper()
    return flow in {"DOM", "IMP", "TOTAL", "TOT", "DOMESTIC", "IMPORTS"}


def _flow_identity_key(flow_value: object) -> str:
    flow = str(flow_value).strip().upper()
    if flow in {"TOTAL", "TOT"}:
        return "TOTAL"
    if flow in {"DOM", "DOMESTIC"}:
        return "DOM"
    if flow in {"IMP", "IMPORT", "IMPORTS"}:
        return "IMP"
    return flow


def _build_mapped_output_flow_identity_frame(
    flow_totals: Mapping[str, Mapping[str, float]],
    *,
    rtol: float,
    atol: float,
) -> pd.DataFrame:
    """Compare DOM + IMP against TOTAL for boundary-correct mapped-output totals."""
    required = ("DOM", "IMP", "TOTAL")
    columns = [
        "quantity",
        "dom_plus_imp",
        "total",
        "abs_diff",
        "source_rounding_allowance",
        "tolerance",
        "ok",
    ]
    if any(flow not in flow_totals for flow in required):
        return pd.DataFrame(columns=pd.Index(columns))

    rows: list[dict[str, object]] = []
    for quantity in ("eurostat_output_total", "exio_mapped_output_total"):
        dom = float(flow_totals["DOM"].get(quantity, 0.0))
        imp = float(flow_totals["IMP"].get(quantity, 0.0))
        total = float(flow_totals["TOTAL"].get(quantity, 0.0))
        lhs = dom + imp
        diff = abs(lhs - total)
        source_rounding_allowance = (
            sum(
                float(flow_totals[flow].get("eurostat_rounding_bound", 0.0))
                for flow in required
            )
            if quantity == "eurostat_output_total"
            else 0.0
        )
        tol = (
            max(float(atol), float(rtol) * abs(total))
            + source_rounding_allowance
        )
        rows.append(
            {
                "quantity": quantity,
                "dom_plus_imp": lhs,
                "total": total,
                "abs_diff": diff,
                "source_rounding_allowance": source_rounding_allowance,
                "tolerance": tol,
                "ok": bool(diff <= tol),
            }
        )
    return pd.DataFrame(rows, columns=pd.Index(columns))


_ENVIRONMENTAL_EXTENSION_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "none": {},
    "combustion-co2": {"air_emissions": ("CO2 - combustion - air",)},
}


def _parse_environmental_extension_requests(
    raw: str,
) -> dict[str, tuple[str, ...]]:
    """Parse the advanced JSON environmental-selection form."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --environmental-extensions-json JSON: {exc}") from exc
    requests: dict[str, tuple[str, ...]] = {}
    if isinstance(value, list):
        for item in value:
            name = str(item).strip()
            if not name:
                raise SystemExit("Environmental extension names must not be blank.")
            requests[name] = ()
        return requests
    if not isinstance(value, dict):
        raise SystemExit(
            "--environmental-extensions-json must be a JSON list of extension names or "
            "an object mapping extension names to stressor keyword lists."
        )
    for extension, keywords in value.items():
        name = str(extension).strip()
        if not name:
            raise SystemExit("Environmental extension names must not be blank.")
        if not isinstance(keywords, list):
            raise SystemExit(
                f"Environmental extension {name!r} must map to a JSON list of "
                "stressor keywords; use [] to select all stressors."
            )
        requests[name] = tuple(str(keyword) for keyword in keywords)
    return requests


def _parse_environmental_extension_specs(
    specs: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Parse repeatable ``EXTENSION[=PATTERN]`` selectors.

    Extension names are exact directory identifiers from the selected EXIOBASE
    archive (for example ``air_emissions``, ``material`` or ``water`` in the
    currently tested beta archive). A bare extension requests all
    indicator/stressor rows. Each ``=PATTERN`` is a substring filter that may match
    multiple rows; repeated patterns are combined for that extension.
    """
    requests: dict[str, tuple[str, ...]] = {}
    for raw in specs:
        extension_raw, separator, stressor_raw = str(raw).partition("=")
        extension = extension_raw.strip()
        if not extension:
            raise SystemExit("Environmental extension names must not be blank.")
        if not separator:
            requests[extension] = ()
            continue
        stressor = stressor_raw.strip()
        if not stressor:
            raise SystemExit(
                f"Environmental extension selector {raw!r} has an empty stressor; "
                "omit '=' to select all stressors."
            )
        if extension in requests and requests[extension] == ():
            continue
        existing = requests.get(extension, ())
        requests[extension] = tuple(dict.fromkeys((*existing, stressor)))
    return requests


def _resolve_environmental_extension_requests(
    *,
    preset: str | None,
    specs: Sequence[str] | None,
    raw_json: str | None,
) -> dict[str, tuple[str, ...]]:
    """Resolve exactly one environmental-selection interface to the core mapping."""
    if raw_json is not None:
        return _parse_environmental_extension_requests(raw_json)
    if specs:
        return _parse_environmental_extension_specs(specs)
    preset_name = str(preset or "combustion-co2")
    if preset_name not in _ENVIRONMENTAL_EXTENSION_PRESETS:
        raise SystemExit(
            f"Unknown environmental preset {preset_name!r}; "
            f"choose from {sorted(_ENVIRONMENTAL_EXTENSION_PRESETS)!r}."
        )
    return {
        extension: tuple(keywords)
        for extension, keywords in _ENVIRONMENTAL_EXTENSION_PRESETS[preset_name].items()
    }

def _environmental_extension_requests_from_archive(
    source_zip: str | Path,
) -> dict[str, tuple[str, ...]]:
    """Select every complete extension directory present in the source archive."""
    names = discover_exiobase3_extensions(source_zip)
    if not names:
        raise SystemExit(
            f"No complete EXIOBASE extension directories found in {source_zip}."
        )
    return {name: () for name in names}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    def _normalize_stk_flow(v: str) -> str:
        # Eurostat uses "TOTAL" (not "TOT") for stk_flow.
        if str(v).upper() == "TOT":
            return "TOTAL"
        return str(v)

    def _normalize_eurostat_filters(filters: dict) -> dict:
        if args.eurostat_stk_flow_dim == "stk_flow" and "stk_flow" in filters:
            cur = filters["stk_flow"]
            if isinstance(cur, (list, tuple)):
                filters["stk_flow"] = [_normalize_stk_flow(x) for x in cur]
            else:
                filters["stk_flow"] = _normalize_stk_flow(cur)
        return filters

    def _exio_flow_for_eurostat_flow_value(flow_value: str) -> str:
        if args.exio_share_flow != "auto":
            return str(args.exio_share_flow)
        if args.eurostat_stk_flow_dim == "stk_flow":
            f = str(flow_value).upper()
            if f == "DOM":
                return "domestic"
            if f == "IMP":
                return "imports"
            if f in {"TOTAL", "TOT"}:
                return "total"
        return "domestic"

    p = argparse.ArgumentParser(
        description="Split Eurostat IOT sectors using EXIOBASE donor blocks and conservative source-parent reconstruction."
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Base folder for cached datasets (defaults to repo-root `data/`, overridable via env var MRIO_DATA_ROOT).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and full paths for outputs.",
    )
    p.add_argument(
        "--classification",
        choices=["product", "industry"],
        default="product",
        help="Choose the source/target dictionary pair for the canonical mapping CSV.",
    )

    euro = p.add_argument_group("Eurostat input")
    euro.add_argument("--eurostat-z", help="Path to Eurostat Z (square).")
    euro.add_argument("--eurostat-y", help="Path to Eurostat Y (same index as Z).")
    euro.add_argument(
        "--eurostat-dataset", help="Eurostat dataset code (e.g. naio_10_cp1700)."
    )
    euro.add_argument(
        "--eurostat-filters", help="JSON dict of Eurostat dimension filters."
    )
    euro.add_argument(
        "--eurostat-row-dim", help="Row dimension id in the Eurostat payload."
    )
    euro.add_argument(
        "--eurostat-col-dim", help="Column dimension id in the Eurostat payload."
    )
    euro.add_argument("--eurostat-force-download", action="store_true")
    euro.add_argument("--eurostat-force-reparse", action="store_true")
    euro.add_argument(
        "--eurostat-stk-flows",
        help='JSON list of stk_flow values to run in one go (e.g. \'["DOM","IMP"]\'). Requires --eurostat-dataset mode.',
    )
    euro.add_argument("--eurostat-stk-flow-dim", default="stk_flow")

    p.add_argument("--exio-year", type=int, default=2021)
    p.add_argument(
        "--exio-region",
        default="AT",
        help="EXIO donor scope code. Without --region-scope-mapping-csv this is a singleton EXIO region.",
    )
    p.add_argument(
        "--region-scope-mapping-csv",
        default=None,
        help="Optional many-member region-scope mapping CSV; required for grouped scopes such as EU27_2020.",
    )
    p.add_argument(
        "--exio-doi",
        default=DEFAULT_EXIOBASE3_DOI,
        help="EXIOBASE Zenodo DOI or doi.org URL used for source zip discovery/download.",
    )
    p.add_argument(
        "--exio-share-flow",
        default="auto",
        choices=["auto", "domestic", "imports", "total"],
        help="Which EXIO use-slice to derive split shares from. "
        "auto maps Eurostat stk_flow DOM/IMP/TOTAL -> EXIO domestic/imports/total.",
    )
    p.add_argument(
        "--source-rounding-quantum",
        type=float,
        default=None,
        help="Optional reporting quantum for independently rounded published Eurostat controls; overrides dataset metadata and is required for rounded local-file controls.",
    )
    p.add_argument(
        "--no-mapping-consistency-check",
        action="store_true",
        help="Disable consistency diagnostics between mapped EXIO use totals and Eurostat use totals.",
    )
    p.add_argument(
        "--strict-mapping-consistency-check",
        action="store_true",
        help=(
            "Fail if mapped EXIO destination-use totals differ from Eurostat destination-use totals. "
            "By default these cross-dataset level differences are written as diagnostics only; "
            "DOM+IMP=TOTAL flow identity remains a hard internal check."
        ),
    )
    p.add_argument(
        "--mapping-consistency-rtol",
        type=float,
        default=0.2,
        help="Relative tolerance for mapping consistency checks (scaled EXIO vs Eurostat output).",
    )
    p.add_argument(
        "--mapping-consistency-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for mapping consistency checks.",
    )
    p.add_argument(
        "--flow-identity-rtol",
        type=float,
        default=1e-9,
        help=(
            "Relative tolerance for the internal DOM+IMP=TOTAL flow-identity check. "
            "This is intentionally strict and independent of the looser cross-dataset mapping-consistency tolerance."
        ),
    )
    p.add_argument(
        "--flow-identity-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the internal DOM+IMP=TOTAL flow-identity check.",
    )

    p.add_argument(
        "--dump-exio-sectors-csv",
        default=None,
        help="Write the available EXIO sector labels for the selected region slice to CSV and exit.",
    )
    p.add_argument(
        "--exio-mapping-csv",
        default=None,
        help="Canonical annotated split mapping CSV. When provided, pattern matching is disabled (recommended).",
    )
    p.add_argument(
        "--require-mapping-coverage",
        action="store_true",
        help=(
            "Require a constituent coverage check before splitting. When enabled, the mapping CSV must list "
            "the full constituent set for each aggregate (including any disabled rows). The split will error "
            "if the enabled subset covers less than (1 - --coverage-tol) of the constituent total, as measured "
            "in the selected EXIO flow."
        ),
    )
    p.add_argument(
        "--coverage-tol",
        type=float,
        default=0.0,
        help=(
            "Tolerance for the coverage check (default 0.0). Example: 1e-6 allows up to 0.0001%% of EXIO Z+Y row totals "
            "to be omitted/disabled."
        ),
    )
    p.add_argument(
        "--allow-normalized-exio-match",
        action="store_true",
        help="Allow whitespace/case normalized matching of EXIO sector labels (default: strict exact match).",
    )
    p.add_argument(
        "--write-mapping-template-csv",
        default=None,
        help="Write a mapping CSV template generated by the current pattern selection and exit.",
    )

    # Energy-specific defaults belong to the CLI (module code is generic).
    p.add_argument(
        "--sector-19", default="C19", help="Aggregate sector code for refined products."
    )
    p.add_argument(
        "--sector-35",
        default="D35",
        help="Aggregate sector code for electricity/gas/steam.",
    )
    p.add_argument(
        "--patterns-19",
        default=None,
        help="JSON list of EXIO patterns for sector-19 subsectors.",
    )
    p.add_argument(
        "--patterns-35",
        default=None,
        help="JSON list of EXIO patterns for sector-35 subsectors.",
    )
    p.add_argument(
        "--electricity-by-source",
        action="store_true",
        help="Add more granular electricity patterns.",
    )

    p.add_argument(
        "--no-exio-quantity-price",
        action="store_true",
        help="Skip extracting physical quantities and value-per-physical-input proxies from EXIO extensions.",
    )
    p.add_argument(
        "--qp-extensions",
        default='["energy"]',
        help='JSON list of EXIO extension names to read for quantities (default: ["energy"]).',
    )
    p.add_argument(
        "--qp-stressor-keywords",
        default='["Energy use -"]',
        help='JSON list of stressor substring keywords to select quantity rows (default: ["Energy use -"]).',
    )

    environmental_selection = p.add_mutually_exclusive_group()
    environmental_selection.add_argument(
        "--environmental-preset",
        choices=tuple(_ENVIRONMENTAL_EXTENSION_PRESETS),
        default="combustion-co2",
        help=(
            "Named environmental selection (default: combustion-co2). "
            "Use 'none' to disable environmental extensions."
        ),
    )
    environmental_selection.add_argument(
        "--environmental-extension",
        action="append",
        default=None,
        metavar="EXTENSION[=PATTERN]",
        help=(
            "Repeatable EXIOBASE extension selector using official extension directory "
            "names. A bare name selects all indicator/stressor rows; '=PATTERN' "
            "selects every row whose label contains that substring. Repeat to combine "
            "patterns or extensions."
        ),
    )
    environmental_selection.add_argument(
        "--environmental-extensions-json",
        default=None,
        help=(
            "Advanced JSON selection: a list of EXIO extension names or an object "
            "mapping extension names to indicator/stressor substring lists."
        ),
    )
    environmental_selection.add_argument(
        "--environmental-all-extensions",
        action="store_true",
        help=(
            "Select every extension directory discovered in the actual EXIO source "
            "archive that provides F.txt, F_Y.txt, and unit.txt. No hard-coded "
            "release-specific extension-name list is used."
        ),
    )
    p.add_argument(
        "--exio-to-eurostat-mapping-csv",
        default=None,
        help=(
            "Optional canonical annotated weighted EXIO-to-Eurostat mapping CSV. "
            "It is also used to project environmental donor intensities to retained unsplit sectors."
        ),
    )

    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory (required unless using a dump/template mode).",
    )
    p.add_argument(
        "--no-xlsx-report", action="store_true", help="Skip writing the XLSX report."
    )
    args = p.parse_args(argv)

    data_root = (
        Path(args.data_root) if args.data_root is not None else default_data_root()
    )
    eurostat_storage = data_root / "eurostat"
    _vprint(args.verbose, f"Using data root: {data_root.resolve()}")

    if (args.eurostat_z or args.eurostat_y) and args.eurostat_dataset:
        raise SystemExit(
            "Provide either --eurostat-z/--eurostat-y OR --eurostat-dataset (not both)."
        )

    patterns_19 = (
        DEFAULT_PATTERNS_19
        if args.patterns_19 is None
        else tuple(json.loads(args.patterns_19))
    )
    patterns_35 = (
        DEFAULT_PATTERNS_35
        if args.patterns_35 is None
        else tuple(json.loads(args.patterns_35))
    )
    if args.electricity_by_source:
        patterns_35 = tuple(patterns_35) + DEFAULT_ELECTRICITY_BY_SOURCE_PATTERNS
    sector_patterns = {
        str(args.sector_19): patterns_19,
        str(args.sector_35): patterns_35,
    }

    if args.outdir is None and not (
        args.dump_exio_sectors_csv is not None
        or args.write_mapping_template_csv is not None
    ):
        raise SystemExit(
            "Provide --outdir (or use --dump-exio-sectors-csv / --write-mapping-template-csv)."
        )

    outdir = Path(args.outdir) if args.outdir is not None else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        _clear_managed_split_outputs(outdir)
        for diagnostic_name in (
            "mapped_output_flow_identity.csv",
            "mapped_output_consistency_total_reference.csv",
        ):
            (outdir / diagnostic_name).unlink(missing_ok=True)

    extract_qp = not args.no_exio_quantity_price
    qp_extensions = tuple(json.loads(args.qp_extensions))
    qp_keywords = tuple(json.loads(args.qp_stressor_keywords))
    # Archive-driven all-extension selection can only be resolved after the
    # DOI-checked/local EXIO source zip has been located. Other selectors resolve now.
    environmental_requests = (
        {}
        if args.environmental_all_extensions
        else _resolve_environmental_extension_requests(
            preset=args.environmental_preset,
            specs=args.environmental_extension,
            raw_json=args.environmental_extensions_json,
        )
    )
    full_sector_mapping: dict[str, dict[str, float]] | None = None
    full_sector_mapping_csv: str | None = args.exio_to_eurostat_mapping_csv

    environmental_source_output: pd.Series | None = None
    environmental_source_basis: pd.MultiIndex | None = None
    source_regions: tuple[str, ...] | None = None
    need_global_x = (
        bool(args.region_scope_mapping_csv)
        or bool(environmental_requests)
        or bool(args.environmental_all_extensions)
    )
    if need_global_x:
        source_matrices, source_basis, _source_meta = load_mrio_iot_matrices(
            "exiobase",
            int(args.exio_year),
            matrices=("x",),
            data_root=data_root,
            doi=args.exio_doi,
            exio_system="pxp",
            force_download=False,
            verbose=args.verbose,
        )
        environmental_source_output = cast(pd.Series, source_matrices["x"])
        environmental_source_basis = source_basis
        source_regions = tuple(dict.fromkeys(
            map(str, source_basis.get_level_values("region"))
        ))
    exio_scope = _resolve_pipeline_scope(
        scope_code=args.exio_region,
        mapping_csv=args.region_scope_mapping_csv,
        source_regions=source_regions,
    )
    if len(exio_scope.members) > 1 and extract_qp:
        raise SystemExit(
            "Grouped EXIO donor scopes do not yet define the optional quantity/value proxy sidecar. "
            "Pass --no-exio-quantity-price; this does not affect monetary IOT splitting or environmental extensions."
        )

    region_scope_mapping_sha256 = (
        _sha256_file(args.region_scope_mapping_csv)
        if args.region_scope_mapping_csv is not None
        else None
    )
    exio_split_mapping_sha256 = (
        _sha256_file(args.exio_mapping_csv)
        if args.exio_mapping_csv is not None
        else None
    )
    exio_to_eurostat_mapping_sha256 = (
        _sha256_file(args.exio_to_eurostat_mapping_csv)
        if args.exio_to_eurostat_mapping_csv is not None
        else None
    )

    qp_cache_dir = (
        data_root
        / f"exiobase3_{args.exio_year}"
        / "slices"
        / args.exio_region
        / "pxp"
        / "extensions"
    )

    if (
        args.dump_exio_sectors_csv is not None
        or args.write_mapping_template_csv is not None
    ):
        _vprint(args.verbose, "Calling: load_mrio_iot_scope_slice(...)")
        t = _Timer()
        exio_slice = _load_exio_scope_slice(
            year=args.exio_year,
            scope=exio_scope,
            flow="domestic",
            data_root=data_root,
            doi=args.exio_doi,
            force_download=False,
            verbose=args.verbose,
        )
        _vprint(
            args.verbose,
            f"Returned: load_mrio_iot_scope_slice ({t.elapsed_s():.2f}s)",
        )

        if args.dump_exio_sectors_csv is not None:
            out = Path(args.dump_exio_sectors_csv)
            _write_rows_csv(
                pd.DataFrame({"exio_sector": list(map(str, exio_slice.Z.index))}),
                out,
                verbose=args.verbose,
            )
            return 0

        # Template from pattern selection (not robust; intended to be edited into an explicit mapping).
        assert args.write_mapping_template_csv is not None
        rows: list[dict[str, str]] = []
        for agg, pats in sector_patterns.items():
            matched = []
            for s in map(str, exio_slice.Z.index):
                s_l = s.casefold()
                if any(p.casefold() in s_l for p in pats):
                    matched.append(s)
            for s in matched:
                rows.append(
                    {
                        "aggregate_sector": str(agg),
                        "subsector": str(s),
                        "exio_sector": str(s),
                        "enabled": "1",
                    }
                )
        out = Path(args.write_mapping_template_csv)
        _write_rows_csv(pd.DataFrame(rows), out, verbose=args.verbose)
        return 0

    sector_to_exio_sectors = None
    coverage_expected_exio = None
    if args.exio_mapping_csv is not None:
        if _is_canonical_mapping_csv(args.exio_mapping_csv):
            _vprint(args.verbose, "Calling: read_canonical_split_mapping(...)")
            t = _Timer()
            sector_to_exio_sectors = read_canonical_split_mapping(
                args.exio_mapping_csv,
                _canonical_source_dictionary_path(
                    args.exio_mapping_csv, args.classification
                ),
                _canonical_target_dictionary_path(
                    args.exio_mapping_csv, args.classification
                ),
            )
            _vprint(
                args.verbose,
                f"Returned: read_canonical_split_mapping ({t.elapsed_s():.2f}s)",
            )
        else:
            _vprint(args.verbose, "Calling: read_split_mapping(...)")
            t = _Timer()
            sector_to_exio_sectors = read_split_mapping(
                args.exio_mapping_csv,
                exio_to_eurostat_mapping_csv=args.exio_to_eurostat_mapping_csv,
            )
            _vprint(
                args.verbose, f"Returned: read_split_mapping ({t.elapsed_s():.2f}s)"
            )
        if args.verbose and (
            args.patterns_19 is not None
            or args.patterns_35 is not None
            or args.electricity_by_source
        ):
            _vprint(
                args.verbose,
                "NOTE: --exio-mapping-csv provided; ignoring pattern selection flags.",
            )
        if args.require_mapping_coverage:

            def _flatten(values: object) -> list[str]:
                if isinstance(values, dict):
                    out: list[str] = []
                    for inner in values.values():
                        if isinstance(inner, dict):
                            out.extend(list(map(str, inner.keys())))
                        else:
                            out.extend(list(map(str, inner)))
                    return list(dict.fromkeys(out))
                if isinstance(values, IterableABC) and not isinstance(
                    values, (str, bytes)
                ):
                    return list(map(str, values))
                raise TypeError(
                    f"Expected iterable EXIO members, got {type(values).__name__}"
                )

            coverage_expected_exio = {
                str(aggregate): _flatten(members)
                for aggregate, members in (sector_to_exio_sectors or {}).items()
            }
    report_aggregates = (
        tuple(sector_to_exio_sectors.keys())
        if sector_to_exio_sectors is not None
        else (args.sector_19, args.sector_35)
    )

    base_filters: dict[str, object] = {}
    eurostat_year = None
    parsed: ParsedEurostatIOT | None = None
    if args.eurostat_dataset:
        if not (
            args.eurostat_filters and args.eurostat_row_dim and args.eurostat_col_dim
        ):
            raise SystemExit(
                "When using --eurostat-dataset, also provide --eurostat-filters, --eurostat-row-dim, --eurostat-col-dim."
            )
        base_filters = _normalize_eurostat_filters(json.loads(args.eurostat_filters))
        time_val = base_filters.get("time")
        if isinstance(time_val, (list, tuple)) and time_val:
            time_val = time_val[0]
        eurostat_year = _extract_year(time_val)

        if args.eurostat_stk_flows is not None:
            flows = list(map(str, json.loads(args.eurostat_stk_flows)))
            if args.eurostat_stk_flow_dim == "stk_flow":
                flows = [_normalize_stk_flow(f) for f in flows]
            _vprint(args.verbose, "Calling: load_eurostat_iot_variants(...)")
            t = _Timer()
            variants = load_eurostat_iot_variants(
                dataset=args.eurostat_dataset,
                base_filters=base_filters,
                variant_dim=args.eurostat_stk_flow_dim,
                variant_values=flows,
                row_dim=args.eurostat_row_dim,
                col_dim=args.eurostat_col_dim,
                storage_folder=eurostat_storage,
                force_download=args.eurostat_force_download,
                force_reparse=args.eurostat_force_reparse,
            )
            _vprint(
                args.verbose,
                f"Returned: load_eurostat_iot_variants ({t.elapsed_s():.2f}s)",
            )

            if full_sector_mapping_csv:
                _vprint(
                    args.verbose,
                    "Phase 3: resolving EXIO-to-Eurostat weights separately for each stk_flow "
                    "from that flow's Eurostat Z + non-export Y reference.",
                )

            exio_slices: dict[str, MrioScopeSlice] = {}
            for flow in flows:
                exio_flow = _exio_flow_for_eurostat_flow_value(flow)
                if exio_flow in exio_slices:
                    continue
                _vprint(
                    args.verbose,
                    f"Calling: load_mrio_iot_scope_slice(..., flow={exio_flow})",
                )
                t = _Timer()
                exio_slices[exio_flow] = _load_exio_scope_slice(
            year=args.exio_year,
            scope=exio_scope,
            flow=exio_flow,
            data_root=data_root,
            doi=args.exio_doi,
            force_download=False,
            verbose=args.verbose,
        )
                _vprint(
                    args.verbose,
                    f"Returned: load_mrio_iot_scope_slice ({t.elapsed_s():.2f}s)",
                )

            if args.environmental_all_extensions:
                domestic_slice = exio_slices.get("domestic")
                if domestic_slice is None:
                    environmental_requests = {}
                    _vprint(
                        args.verbose,
                        "Archive-driven environmental selection requested, but no DOM "
                        "flow is present; production extensions remain omitted.",
                    )
                else:
                    environmental_requests = _environmental_extension_requests_from_archive(
                        cast(str | Path, domestic_slice.metadata["source_zip"])
                    )
                    _vprint(
                        args.verbose,
                        "Discovered EXIO environmental extensions: "
                        + ", ".join(environmental_requests),
                    )
                    if environmental_source_output is None or environmental_source_basis is None:
                        raise RuntimeError(
                            "Archive-driven environmental selection requires the already-loaded "
                            "global EXIO x/basis context."
                        )

            source_zip_paths = {
                Path(str(scope_slice.metadata["source_zip"]))
                for scope_slice in exio_slices.values()
            }
            if len(source_zip_paths) != 1:
                raise RuntimeError(
                    "All EXIO donor-flow slices must originate from one source archive; "
                    f"got {sorted(map(str, source_zip_paths))!r}."
                )
            exio_source_zip = next(iter(source_zip_paths))
            exio_source_zip_sha256 = _sha256_file(exio_source_zip)

            assert outdir is not None
            flow_identity_totals: dict[str, dict[str, float]] = {}
            dom_column_profiles = None
            dom_column_mapping = None
            ordered_flows = sorted(
                variants,
                key=lambda value: (0 if str(value) == "DOM" else 1, str(value)),
            )
            for flow in ordered_flows:
                parsed_variant = variants[flow]
                flow_dir = outdir / f"{args.eurostat_stk_flow_dim}={flow}"
                flow_dir.mkdir(parents=True, exist_ok=True)
                _clear_managed_split_outputs(flow_dir)

                _vprint(
                    args.verbose,
                    f"Calling: split_iot_using_exiobase_split_profiles(...) [{flow}]",
                )
                t = _Timer()
                exio_flow = _exio_flow_for_eurostat_flow_value(flow)
                exio_slice = exio_slices[exio_flow]
                flow_has_quantity_price_proxies = exio_flow != "imports"
                if exio_flow == "imports" and extract_qp:
                    _vprint(
                        args.verbose,
                        "Skipping EXIO quantity/value proxy outputs for IMP flow: "
                        "origin-weighted import proxies are not implemented.",
                    )
                if exio_flow == "imports" and dom_column_profiles is None:
                    raise SystemExit(
                        "IMP primitive splitting requires DOM in --eurostat-stk-flows "
                        "so production-column profiles are domestic."
                    )
                flow_sector_mapping: dict[str, dict[str, float]] | None = None
                if full_sector_mapping_csv is not None:
                    _vprint(
                        args.verbose,
                        f"Calling: resolve flow-specific EXIO-to-Eurostat mapping [{flow}]",
                    )
                    t_map = _Timer()
                    flow_sector_mapping, _ = (
                        _resolve_exio_to_eurostat_mapping_for_parsed(
                            full_sector_mapping_csv,
                            parsed_variant,
                        )
                    )
                    _vprint(
                        args.verbose,
                        f"Returned: resolve flow-specific EXIO-to-Eurostat mapping ({t_map.elapsed_s():.2f}s)",
                    )
                    out = flow_dir / "flow_specific_exio_to_eurostat_weights.csv"
                    _write_rows_csv(
                        _mapping_weights_frame(
                            flow_sector_mapping,
                            variant_dim=args.eurostat_stk_flow_dim,
                            variant_value=str(flow),
                            exio_flow=exio_flow,
                        ),
                        out,
                        verbose=args.verbose,
                    )
                if (
                    flow_sector_mapping is not None
                    and not args.no_mapping_consistency_check
                ):
                    label = f"{args.eurostat_stk_flow_dim}={flow}"
                    _assert_mapping_integrity(
                        sector_mapping=flow_sector_mapping,
                        eurostat_Z=parsed_variant.Z,
                        eurostat_Y=parsed_variant.Y,
                        required_target_codes=report_aggregates,
                        label=label,
                    )
                    if _should_check_output(flow, eurostat_year, args.exio_year):
                        output_consistency, _, _ = _build_output_consistency_frame(
                            sector_mapping=flow_sector_mapping,
                            eurostat_Z=parsed_variant.Z,
                            eurostat_Y=parsed_variant.Y,
                            exio_slice=exio_slice,
                        )
                        out = flow_dir / "mapped_output_consistency.csv"
                        output_consistency.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")
                        flow_identity_totals[_flow_identity_key(flow)] = {
                            "eurostat_output_total": float(
                                output_consistency["eurostat_output"].sum()
                            ),
                            "exio_mapped_output_total": float(
                                output_consistency["exio_mapped_output"].sum()
                            ),
                            "eurostat_rounding_bound": _eurostat_output_rounding_bound(
                                parsed_variant
                            ),
                        }
                        if args.strict_mapping_consistency_check:
                            _assert_output_consistency(
                                sector_mapping=flow_sector_mapping,
                                eurostat_Z=parsed_variant.Z,
                                eurostat_Y=parsed_variant.Y,
                                exio_slice=exio_slice,
                                rtol=float(args.mapping_consistency_rtol),
                                atol=float(args.mapping_consistency_atol),
                                label=label,
                            )
                    else:
                        _vprint(
                            args.verbose,
                            f"Skipping output consistency check for {label} "
                            "because the stk_flow value is not one of DOM, IMP, or TOTAL.",
                        )
                res = split_iot_using_exiobase_split_profiles(
                    eurostat_Z=parsed_variant.Z,
                    eurostat_Y=parsed_variant.Y,
                    eurostat_V=parsed_variant.V,
                    eurostat_M=parsed_variant.M,
                    exio_slice=exio_slice,
                    sector_to_exio_sectors=sector_to_exio_sectors,
                    sector_patterns=None
                    if sector_to_exio_sectors is not None
                    else sector_patterns,
                    eurostat_sector_mapping=flow_sector_mapping,
                    column_split_profiles=(
                        dom_column_profiles if exio_flow == "imports" else None
                    ),
                    column_eurostat_sector_mapping=(
                        dom_column_mapping if exio_flow == "imports" else None
                    ),
                    coverage_expected_exio_sectors=coverage_expected_exio,
                    coverage_tol=args.coverage_tol,
                    source_rounding_quantum=args.source_rounding_quantum
                    if args.source_rounding_quantum is not None
                    else parsed_variant.meta.get("source_rounding_quantum"),
                    extract_quantity_and_price=extract_qp
                    and flow_has_quantity_price_proxies,
                    quantity_price_extensions=qp_extensions
                    if extract_qp and flow_has_quantity_price_proxies
                    else None,
                    quantity_price_stressor_keywords=qp_keywords
                    if extract_qp and flow_has_quantity_price_proxies
                    else None,
                    quantity_price_cache_dir=qp_cache_dir
                    if extract_qp and flow_has_quantity_price_proxies
                    else None,
                    environmental_extension_requests=(
                        environmental_requests
                    ),
                    environmental_source_output=environmental_source_output,
                    environmental_source_basis=environmental_source_basis,
                )
                _vprint(
                    args.verbose,
                    f"Returned: split_iot_using_exiobase_split_profiles ({t.elapsed_s():.2f}s)",
                )

                if str(flow) == "DOM":
                    dom_column_profiles = res.split_profiles
                    dom_column_mapping = flow_sector_mapping

                write_split_core_outputs(
                    res,
                    flow_dir,
                    verbose=args.verbose,
                    extra_info={
                        "variant_dim": args.eurostat_stk_flow_dim,
                        "variant_value": flow,
                        "exio_doi": args.exio_doi,
                        "exio_source_zip": str(exio_source_zip),
                        "exio_source_zip_sha256": exio_source_zip_sha256,
                        "output_share_basis": "EXIO_scope_producer_output",
                        "exio_scope_code": exio_scope.code,
                        "exio_scope_members": list(exio_scope.members),
                        "region_scope_mapping_csv": args.region_scope_mapping_csv,
                        "region_scope_mapping_sha256": region_scope_mapping_sha256,
                        "exio_split_mapping_csv": args.exio_mapping_csv,
                        "exio_split_mapping_sha256": exio_split_mapping_sha256,
                        "exio_to_eurostat_mapping_csv": args.exio_to_eurostat_mapping_csv,
                        "exio_to_eurostat_mapping_sha256": exio_to_eurostat_mapping_sha256,
                    },
                )

                if flow_sector_mapping is not None:
                    edge_metadata = _build_split_child_edge_metadata(
                        selected_children=res.exio_selected_sectors,
                        sector_mapping=flow_sector_mapping,
                        variant_dim=args.eurostat_stk_flow_dim,
                        variant_value=str(flow),
                        exio_flow=exio_flow,
                    )
                    if not edge_metadata.empty:
                        out = flow_dir / "split_child_edges.csv"
                        edge_metadata.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")
                if res.exio_quantity_and_price is not None:
                    out = flow_dir / "exio_quantity_and_price.csv"
                    res.exio_quantity_and_price.to_csv(out, index=False)
                    _vprint(args.verbose, f"Wrote: {out.resolve()}")

                    # Compute and output disaggregated sector values per physical input
                    mapping_for_values = (
                        emitted_sector_to_exio_members_from_split_profiles(
                            res.split_profiles
                        )
                    )
                    disagg_values = map_exio_value_per_physical_input(
                        exio_quantity_and_price=res.exio_quantity_and_price,
                        sector_to_exio_sectors=mapping_for_values,
                        split_profiles=res.split_profiles,
                        eurostat_sector_mapping=flow_sector_mapping,
                    )
                    if disagg_values is not None:
                        out = (
                            flow_dir
                            / "disaggregated_sector_value_per_physical_input.csv"
                        )
                        disagg_values.to_csv(out, index=False)
                        _vprint(args.verbose, f"Wrote: {out.resolve()}")

                # Environmental extensions are owned by the reusable split API.
                if not args.no_xlsx_report:
                    out = flow_dir / "energy_split_report.xlsx"
                    write_split_report_xlsx(
                        out,
                        result=res,
                        aggregates=report_aggregates,
                    )
                    _vprint(args.verbose, f"Wrote: {out.resolve()}")

            identity_keys = {_flow_identity_key(flow) for flow in flows}
            if (
                full_sector_mapping_csv is not None
                and not args.no_mapping_consistency_check
                and args.eurostat_stk_flow_dim == "stk_flow"
                and {"DOM", "IMP"}.issubset(identity_keys)
                and "TOTAL" not in identity_keys
            ):
                _vprint(
                    args.verbose,
                    "Loading diagnostic-only TOTAL references for the mapped-output "
                    "DOM + IMP = TOTAL identity; no TOTAL split is generated.",
                )
                total_filters = dict(base_filters)
                total_filters[args.eurostat_stk_flow_dim] = "TOTAL"
                parsed_total = load_eurostat_iot(
                    dataset=args.eurostat_dataset,
                    filters=total_filters,
                    row_dim=args.eurostat_row_dim,
                    col_dim=args.eurostat_col_dim,
                    storage_folder=eurostat_storage,
                    force_download=args.eurostat_force_download,
                    force_reparse=args.eurostat_force_reparse,
                )
                total_mapping, _ = _resolve_exio_to_eurostat_mapping_for_parsed(
                    full_sector_mapping_csv,
                    parsed_total,
                )
                _assert_mapping_integrity(
                    sector_mapping=total_mapping,
                    eurostat_Z=parsed_total.Z,
                    eurostat_Y=parsed_total.Y,
                    required_target_codes=report_aggregates,
                    label=f"{args.eurostat_stk_flow_dim}=TOTAL diagnostic reference",
                )
                total_slice = _load_exio_scope_slice(
            year=args.exio_year,
            scope=exio_scope,
            flow="total",
            data_root=data_root,
            doi=args.exio_doi,
            force_download=False,
            verbose=args.verbose,
        )
                total_consistency, _, _ = _build_output_consistency_frame(
                    sector_mapping=total_mapping,
                    eurostat_Z=parsed_total.Z,
                    eurostat_Y=parsed_total.Y,
                    exio_slice=total_slice,
                )
                total_out = outdir / "mapped_output_consistency_total_reference.csv"
                total_consistency.to_csv(total_out, index=False)
                _vprint(args.verbose, f"Wrote: {total_out.resolve()}")
                flow_identity_totals["TOTAL"] = {
                    "eurostat_output_total": float(
                        total_consistency["eurostat_output"].sum()
                    ),
                    "exio_mapped_output_total": float(
                        total_consistency["exio_mapped_output"].sum()
                    ),
                    "eurostat_rounding_bound": _eurostat_output_rounding_bound(
                        parsed_total
                    ),
                }
                if args.strict_mapping_consistency_check:
                    _assert_output_consistency(
                        sector_mapping=total_mapping,
                        eurostat_Z=parsed_total.Z,
                        eurostat_Y=parsed_total.Y,
                        exio_slice=total_slice,
                        rtol=float(args.mapping_consistency_rtol),
                        atol=float(args.mapping_consistency_atol),
                        label=f"{args.eurostat_stk_flow_dim}=TOTAL diagnostic reference",
                    )

            flow_identity = _build_mapped_output_flow_identity_frame(
                flow_identity_totals,
                rtol=float(args.flow_identity_rtol),
                atol=float(args.flow_identity_atol),
            )
            if not flow_identity.empty:
                out = outdir / "mapped_output_flow_identity.csv"
                flow_identity.to_csv(out, index=False)
                _vprint(args.verbose, f"Wrote: {out.resolve()}")
                identity_ok = all(bool(value) for value in flow_identity["ok"].tolist())
                if not identity_ok:
                    raise SystemExit(
                        "Mapping consistency check failed: DOM + IMP does not match TOTAL "
                        "for boundary-correct mapped-output totals. See "
                        f"{out.resolve()}"
                    )
            return 0

        _vprint(args.verbose, "Calling: load_eurostat_iot(...)")
        t = _Timer()
        parsed = load_eurostat_iot(
            dataset=args.eurostat_dataset,
            filters=base_filters,
            row_dim=args.eurostat_row_dim,
            col_dim=args.eurostat_col_dim,
            storage_folder=eurostat_storage,
            force_download=args.eurostat_force_download,
            force_reparse=args.eurostat_force_reparse,
        )
        _vprint(args.verbose, f"Returned: load_eurostat_iot ({t.elapsed_s():.2f}s)")
        Z, Y = parsed.Z, parsed.Y
        exio_flow_single = _exio_flow_for_eurostat_flow_value(
            str(base_filters.get(args.eurostat_stk_flow_dim, "DOM"))
        )
    else:
        if not (args.eurostat_z and args.eurostat_y):
            raise SystemExit(
                "Provide --eurostat-z and --eurostat-y, or use --eurostat-dataset mode."
            )
        _vprint(args.verbose, "Calling: load_local_iot(...)")
        t = _Timer()
        Z, Y = load_local_iot(args.eurostat_z, args.eurostat_y)
        _vprint(args.verbose, f"Returned: load_local_iot ({t.elapsed_s():.2f}s)")
        exio_flow_single = (
            "domestic" if args.exio_share_flow == "auto" else str(args.exio_share_flow)
        )

    if full_sector_mapping_csv is not None and full_sector_mapping is None:
        parsed_for_mapping = (
            parsed if parsed is not None else ParsedEurostatIOT(Z=Z, Y=Y, meta={})
        )
        _vprint(
            args.verbose,
            "Calling: resolve flow-specific EXIO-to-Eurostat mapping [single]",
        )
        t_map = _Timer()
        full_sector_mapping, _ = (
            _resolve_exio_to_eurostat_mapping_for_parsed(
                full_sector_mapping_csv,
                parsed_for_mapping,
            )
        )
        _vprint(
            args.verbose,
            f"Returned: resolve flow-specific EXIO-to-Eurostat mapping ({t_map.elapsed_s():.2f}s)",
        )
        if outdir is not None:
            out = outdir / "flow_specific_exio_to_eurostat_weights.csv"
            _write_rows_csv(
                _mapping_weights_frame(full_sector_mapping, exio_flow=exio_flow_single),
                out,
                verbose=args.verbose,
            )

    single_has_quantity_price_proxies = exio_flow_single != "imports"
    if exio_flow_single == "imports" and extract_qp:
        _vprint(
            args.verbose,
            "Skipping EXIO quantity/value proxy outputs for single-flow imports: "
            "origin-weighted import proxies are not implemented.",
        )

    # Load the single-flow EXIO slice once. It supplies both donor matrices and,
    # when requested, the actual archive manifest used for extension discovery.
    exio_slice_single = _load_exio_scope_slice(
            year=args.exio_year,
            scope=exio_scope,
            flow=exio_flow_single,
            data_root=data_root,
            doi=args.exio_doi,
            force_download=False,
            verbose=args.verbose,
        )
    exio_source_zip_single = Path(str(exio_slice_single.metadata["source_zip"]))
    exio_source_zip_single_sha256 = _sha256_file(exio_source_zip_single)

    if args.environmental_all_extensions:
        if exio_flow_single in {"domestic", "imports"}:
            environmental_requests = _environmental_extension_requests_from_archive(
                cast(str | Path, exio_slice_single.metadata["source_zip"])
            )
            _vprint(
                args.verbose,
                "Discovered EXIO environmental extensions: "
                + ", ".join(environmental_requests),
            )
        else:
            environmental_requests = {}

    _vprint(args.verbose, "Calling: split_iot_using_exiobase_split_profiles(...)")
    t = _Timer()
    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=Z,
        eurostat_Y=Y,
        eurostat_V=parsed.V if parsed is not None else None,
        eurostat_M=parsed.M if parsed is not None else None,
        exio_slice=exio_slice_single,
        sector_to_exio_sectors=sector_to_exio_sectors,
        sector_patterns=None if sector_to_exio_sectors is not None else sector_patterns,
        eurostat_sector_mapping=full_sector_mapping,
        coverage_expected_exio_sectors=coverage_expected_exio,
        coverage_tol=args.coverage_tol,
        source_rounding_quantum=args.source_rounding_quantum
        if args.source_rounding_quantum is not None
        else (
            parsed.meta.get("source_rounding_quantum") if parsed is not None else None
        ),
        extract_quantity_and_price=extract_qp
        and single_has_quantity_price_proxies,
        quantity_price_extensions=qp_extensions
        if extract_qp and single_has_quantity_price_proxies
        else None,
        quantity_price_stressor_keywords=qp_keywords
        if extract_qp and single_has_quantity_price_proxies
        else None,
        quantity_price_cache_dir=qp_cache_dir
        if extract_qp and single_has_quantity_price_proxies
        else None,
        environmental_extension_requests=environmental_requests,
        environmental_source_output=environmental_source_output,
        environmental_source_basis=environmental_source_basis,
    )
    _vprint(
        args.verbose,
        f"Returned: split_iot_using_exiobase_shares ({t.elapsed_s():.2f}s)",
    )

    assert outdir is not None
    write_split_core_outputs(
        res,
        outdir,
        verbose=args.verbose,
        extra_info={
            "exio_doi": args.exio_doi,
            "exio_source_zip": str(exio_source_zip_single),
            "exio_source_zip_sha256": exio_source_zip_single_sha256,
            "output_share_basis": "EXIO_scope_producer_output",
            "exio_scope_code": exio_scope.code,
            "exio_scope_members": list(exio_scope.members),
            "region_scope_mapping_csv": args.region_scope_mapping_csv,
            "region_scope_mapping_sha256": region_scope_mapping_sha256,
            "exio_split_mapping_csv": args.exio_mapping_csv,
            "exio_split_mapping_sha256": exio_split_mapping_sha256,
            "exio_to_eurostat_mapping_csv": args.exio_to_eurostat_mapping_csv,
            "exio_to_eurostat_mapping_sha256": exio_to_eurostat_mapping_sha256,
        },
    )

    if full_sector_mapping is not None:
        edge_metadata = _build_split_child_edge_metadata(
            selected_children=res.exio_selected_sectors,
            sector_mapping=full_sector_mapping,
            exio_flow=exio_flow_single,
        )
        if not edge_metadata.empty:
            out = outdir / "split_child_edges.csv"
            edge_metadata.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")
    if res.exio_quantity_and_price is not None:
        out = outdir / "exio_quantity_and_price_proxies.csv"
        res.exio_quantity_and_price.to_csv(out, index=False)
        _vprint(args.verbose, f"Wrote: {out.resolve()}")

        # Compute and output disaggregated sector values per physical input
        mapping_for_values = emitted_sector_to_exio_members_from_split_profiles(
            res.split_profiles
        )
        disagg_values = map_exio_value_per_physical_input(
            exio_quantity_and_price=res.exio_quantity_and_price,
            sector_to_exio_sectors=mapping_for_values,
            split_profiles=res.split_profiles,
            eurostat_sector_mapping=full_sector_mapping,
        )
        if disagg_values is not None:
            out = outdir / "disaggregated_sector_value_per_physical_input.csv"
            disagg_values.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")

    # Reuse the already-loaded EXIO slice for mapping diagnostics.
    if full_sector_mapping is not None and not args.no_mapping_consistency_check:
        _assert_mapping_integrity(
            sector_mapping=full_sector_mapping,
            eurostat_Z=Z,
            eurostat_Y=Y,
            required_target_codes=report_aggregates,
            label="single",
        )
        flow_value_single = None
        if args.eurostat_dataset:
            flow_value_single = base_filters.get(args.eurostat_stk_flow_dim, "DOM")
        if _should_check_output(flow_value_single, eurostat_year, args.exio_year):
            output_consistency, _, _ = _build_output_consistency_frame(
                sector_mapping=full_sector_mapping,
                eurostat_Z=Z,
                eurostat_Y=Y,
                exio_slice=exio_slice_single,
            )
            out = outdir / "mapped_output_consistency.csv"
            output_consistency.to_csv(out, index=False)
            _vprint(args.verbose, f"Wrote: {out.resolve()}")
            if args.strict_mapping_consistency_check:
                _assert_output_consistency(
                    sector_mapping=full_sector_mapping,
                    eurostat_Z=Z,
                    eurostat_Y=Y,
                    exio_slice=exio_slice_single,
                    rtol=float(args.mapping_consistency_rtol),
                    atol=float(args.mapping_consistency_atol),
                    label="single",
                )
        else:
            _vprint(
                args.verbose,
                "Skipping output consistency check for single flow "
                f"(flow != TOTAL or year mismatch: Eurostat={eurostat_year}, EXIO={args.exio_year}).",
            )
    # Environmental extensions are owned by the reusable split API.
    if not args.no_xlsx_report:
        out = outdir / "energy_split_report.xlsx"
        write_split_report_xlsx(out, result=res, aggregates=report_aggregates)
        _vprint(args.verbose, f"Wrote: {out.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# pyright: reportMissingImports=false
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterable, Sequence, cast
import zipfile

import pandas as pd

from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors
from ecolecon_mrio.data import (
    DEFAULT_EXIOBASE3_DOI,
    _normalize_exiobase_doi,
    _numeric_tail_from_exio_line,
    _open_exio_zip_text,
)

from ecolecon_mrio.region_aggregation import MrioScopeSlice, RegionScope, ScopeFlow

try:
    import pymrio
except Exception:  # pragma: no cover
    pymrio = None  # type: ignore[assignment]


def list_local_exiobase3_years(*, data_root: str | Path | None = None) -> list[int]:
    if data_root is None:
        from ecolecon_mrio.data import default_data_root

        root = default_data_root()
    else:
        root = Path(data_root)
    years: list[int] = []
    for p in root.glob("exiobase3_*"):
        try:
            years.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(years))


def _exio_label_to_code_lookup(system: str) -> dict[str, str]:
    concordance = load_pymrio_exio3_sectors(system=system).sectors
    required = {"ExioCode", "ExioName"}
    missing = required.difference(set(concordance.columns))
    if missing:
        raise ValueError(f"EXIO sector concordance missing expected columns: {sorted(missing)}")
    codes = concordance["ExioCode"].astype(str)
    names = concordance["ExioName"].astype(str)
    return {
        name: code
        for code, name in zip(codes, names, strict=False)
        if code.strip() != "" and name.strip() != ""
    }


def _code_index(index: pd.Index, *, label_to_code: dict[str, str]) -> pd.Index:
    labels = [str(value) for value in index.astype(str)]
    known_codes = set(label_to_code.values())
    if all(value in known_codes for value in labels):
        return pd.Index(labels)
    missing = [label for label in labels if label not in label_to_code]
    if missing:
        raise ValueError(f"Could not map EXIO sector labels to codes: {missing[:10]!r}")
    return pd.Index([label_to_code[label] for label in labels])



def _normalize_scope_flow(flow: str) -> str:
    flow_norm = str(flow).strip().lower()
    if flow_norm in {"internal", "external_imports", "total"}:
        return flow_norm
    raise ValueError("scope flow must be one of: internal, external_imports, total")


def _scope_members(scope: RegionScope) -> tuple[str, ...]:
    members = tuple(str(member).strip() for member in scope.members)
    if not str(scope.code).strip():
        raise ValueError("RegionScope.code must not be blank.")
    if not members or any(not member for member in members):
        raise ValueError(f"RegionScope {scope.code!r} must contain non-blank members.")
    if len(set(members)) != len(members):
        raise ValueError(f"RegionScope {scope.code!r} contains duplicate members.")
    return members


def _read_scope_x_from_exio_zip(*, zip_path: Path, scope: RegionScope) -> pd.Series:
    """Sum producer-side x over scope members without fabricating missing output."""
    members = _scope_members(scope)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, "x.txt") as fh:
            frame = pd.read_csv(fh, sep="\t")
    required = {"region", "sector"}
    if not required.issubset(frame.columns):
        raise ValueError("EXIOBASE x.txt must contain region and sector columns.")
    value_col = "indout" if "indout" in frame.columns else frame.columns[-1]
    frame_regions = frame["region"].astype(str)
    available_regions = set(frame_regions)
    missing_regions = [member for member in members if member not in available_regions]
    if missing_regions:
        raise ValueError(
            f"EXIOBASE x.txt has no rows for scope members: {missing_regions!r}."
        )

    sector_order: pd.Index | None = None
    total: pd.Series | None = None
    # Sum each member's producer output on a common sector basis. Missing member
    # sectors are malformed source evidence, not zeros to be synthesized.
    for member in members:
        member_frame = frame.loc[frame_regions == member, ["sector", value_col]].copy()
        sectors = pd.Index(member_frame["sector"].astype(str), name="sector")
        if not sectors.is_unique:
            raise ValueError(
                f"EXIOBASE x.txt contains duplicate sectors for region={member!r}."
            )
        raw_values = member_frame[value_col]
        if not isinstance(raw_values, pd.Series):
            raise ValueError(
                f"EXIOBASE x.txt output column {value_col!r} is not one-dimensional."
            )
        values = pd.to_numeric(raw_values, errors="coerce")
        if not isinstance(values, pd.Series):
            raise TypeError("EXIOBASE x.txt numeric output conversion did not produce a Series.")
        nonfinite = ~np.isfinite(values.to_numpy(dtype="float64"))
        if bool(nonfinite.any()):
            bad = [str(sectors[i]) for i in np.flatnonzero(nonfinite)[:10]]
            raise ValueError(
                "EXIOBASE x.txt contains blank, malformed, or non-finite output values "
                f"for region={member!r}; sample={bad!r}."
            )
        member_values = pd.Series(
            values.to_numpy(dtype=float),
            index=sectors,
            dtype="float64",
        )
        if sector_order is None:
            sector_order = sectors
            total = member_values.copy()
            continue
        if set(sectors.astype(str)) != set(sector_order.astype(str)):
            missing = sorted(set(sector_order.astype(str)) - set(sectors.astype(str)))
            extra = sorted(set(sectors.astype(str)) - set(sector_order.astype(str)))
            raise ValueError(
                "EXIOBASE x.txt sector basis differs across scope members: "
                f"region={member!r}, missing={missing[:10]!r}, extra={extra[:10]!r}."
            )
        assert total is not None
        total = total.add(member_values.reindex(sector_order), fill_value=0.0)

    if sector_order is None or total is None:
        raise ValueError(f"EXIOBASE x.txt produced an empty scope {scope.code!r}.")
    total.name = "x_scope"
    total.index = pd.Index(total.index.astype(str), name="sector")
    return total.astype("float64")


def _exio_table_header_for_scope(
    fh,
    *,
    table_name: str,
    scope: RegionScope,
) -> tuple[list[int], list[str], list[int]]:
    """Select scope destination columns and map duplicates to one collapsed label axis."""
    members = _scope_members(scope)
    regions = next(fh).rstrip("\n\r").split("\t")[2:]
    labels = next(fh).rstrip("\n\r").split("\t")[2:]
    next(fh)
    if len(regions) != len(labels):
        raise ValueError(
            f"EXIOBASE {table_name} column-header length mismatch: "
            f"regions={len(regions)}, labels={len(labels)}."
        )
    missing_destinations = [member for member in members if member not in set(map(str, regions))]
    if missing_destinations:
        raise ValueError(
            f"EXIOBASE {table_name} has no destination columns for scope members: "
            f"{missing_destinations!r}."
        )
    keep_positions: list[int] = []
    collapse_positions: list[int] = []
    reference_labels: tuple[str, ...] | None = None
    for member in members:
        member_positions = [
            idx for idx, col_region in enumerate(regions) if str(col_region) == member
        ]
        member_labels = tuple(str(labels[idx]) for idx in member_positions)
        if len(set(member_labels)) != len(member_labels):
            raise ValueError(
                f"EXIOBASE {table_name} contains duplicate destination labels for "
                f"scope member {member!r}: {member_labels!r}."
            )
        if reference_labels is None:
            reference_labels = member_labels
        elif member_labels != reference_labels:
            missing = [label for label in reference_labels if label not in set(member_labels)]
            extra = [label for label in member_labels if label not in set(reference_labels)]
            raise ValueError(
                f"EXIOBASE {table_name} destination label basis differs across scope members: "
                f"member={member!r}, missing={missing[:10]!r}, extra={extra[:10]!r}, "
                f"order_matches={set(member_labels) == set(reference_labels)}."
            )
        keep_positions.extend(member_positions)
        collapse_positions.extend(range(len(member_labels)))
    if reference_labels is None or not reference_labels:
        raise ValueError(f"EXIOBASE {table_name} has an empty destination basis for scope {scope.code!r}.")
    return keep_positions, list(reference_labels), collapse_positions


def _row_belongs_to_scope_flow(
    *, row_region: str, members: set[str], flow: str
) -> bool:
    if flow == "internal":
        return str(row_region) in members
    if flow == "external_imports":
        return str(row_region) not in members
    if flow == "total":
        return True
    raise ValueError(f"Unsupported normalized scope flow {flow!r}")


def _read_scope_use_table_from_exio_zip(
    *,
    zip_path: Path,
    member_name: str,
    scope: RegionScope,
    flow: str,
) -> pd.DataFrame:
    """Stream and collapse a Z.txt or Y.txt destination-scope use block."""
    flow_norm = _normalize_scope_flow(flow)
    members = set(_scope_members(scope))
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, member_name) as fh:
            keep_positions, output_labels, collapse_positions = _exio_table_header_for_scope(
                fh,
                table_name=member_name,
                scope=scope,
            )
            collapse_index = np.asarray(collapse_positions, dtype=np.int64)
            acc: dict[str, np.ndarray] = {}
            max_keep = max(keep_positions)
            origin_rows: dict[str, list[str]] = {}
            origin_seen: dict[str, set[str]] = {}
            reference_row_order: tuple[str, ...] | None = None
            # Stream producer rows once. Origin membership determines internal versus
            # external-import use; destination membership was fixed by the header slice.
            for row_idx, line in enumerate(fh):
                try:
                    row_region, row_sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(
                        f"EXIOBASE {member_name} malformed row at position {row_idx}"
                    ) from None
                origin = str(row_region)
                sector = str(row_sector)
                seen = origin_seen.setdefault(origin, set())
                if sector in seen:
                    raise ValueError(
                        f"EXIOBASE {member_name} contains duplicate producer sector {sector!r} "
                        f"for origin {origin!r}."
                    )
                seen.add(sector)
                origin_rows.setdefault(origin, []).append(sector)
                if not _row_belongs_to_scope_flow(
                    row_region=row_region, members=members, flow=flow_norm
                ):
                    continue
                values = _numeric_tail_from_exio_line(line)
                if values.size <= max_keep:
                    raise ValueError(
                        f"EXIOBASE {member_name} row width mismatch at row {row_idx}: "
                        f"got {values.size}, need column {max_keep + 1}."
                    )
                selected = values[keep_positions]
                if not np.isfinite(selected).all():
                    raise ValueError(
                        f"EXIOBASE {member_name} contains non-finite monetary values "
                        f"at row {row_idx} for destination scope={scope.code!r}."
                    )
                # Sum the selected destination-country columns onto one scope-level
                # sector/category axis before accumulating across producer origins.
                collapsed = np.zeros(len(output_labels), dtype="float64")
                np.add.at(collapsed, collapse_index, selected)
                if sector not in acc:
                    acc[sector] = np.zeros(len(output_labels), dtype="float64")
                acc[sector] += collapsed
    if not origin_rows:
        raise ValueError(f"EXIOBASE {member_name} contains no source rows.")
    for origin, rows in origin_rows.items():
        ordered = tuple(rows)
        if reference_row_order is None:
            reference_row_order = ordered
        elif ordered != reference_row_order:
            missing = [sector for sector in reference_row_order if sector not in set(ordered)]
            extra = [sector for sector in ordered if sector not in set(reference_row_order)]
            raise ValueError(
                f"EXIOBASE {member_name} producer sector basis differs across origins: "
                f"origin={origin!r}, missing={missing[:10]!r}, extra={extra[:10]!r}, "
                f"order_matches={set(ordered) == set(reference_row_order)}."
            )
    assert reference_row_order is not None
    data = np.vstack(
        [acc.get(sector, np.zeros(len(output_labels), dtype="float64")) for sector in reference_row_order]
    )
    return pd.DataFrame(
        data,
        index=pd.Index(reference_row_order, name="sector"),
        columns=pd.Index(output_labels),
        dtype="float64",
    )


def _read_region_exports_member_from_exio_zip(
    *,
    zip_path: Path,
    member_name: str,
    region: str,
) -> pd.Series:
    """Sum one region's product exports to all foreign destinations.

    For ``Z.txt`` this is intermediate exports; for ``Y.txt`` it is final-demand
    exports. Rows are restricted to products produced in ``region`` and columns
    to destination regions other than ``region``.
    """
    with zipfile.ZipFile(zip_path) as zf:
        with _open_exio_zip_text(zf, member_name) as fh:
            destination_regions = next(fh).rstrip("\n\r").split("\t")[2:]
            next(fh)  # sector/final-demand labels
            next(fh)  # units
            keep_positions = [
                idx for idx, destination in enumerate(destination_regions)
                if str(destination) != str(region)
            ]
            if not keep_positions:
                raise ValueError(f"EXIOBASE {member_name} has no foreign destination columns.")
            max_keep = max(keep_positions)
            values_by_sector: dict[str, float] = {}
            order: list[str] = []
            for row_idx, line in enumerate(fh):
                try:
                    row_region, row_sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(
                        f"EXIOBASE {member_name} malformed row at position {row_idx}"
                    ) from None
                if str(row_region) != str(region):
                    continue
                numeric = _numeric_tail_from_exio_line(line)
                if numeric.size <= max_keep:
                    raise ValueError(
                        f"EXIOBASE {member_name} row width mismatch at row {row_idx}: "
                        f"got {numeric.size}, need column {max_keep + 1}."
                    )
                selected = numeric[keep_positions]
                if not np.isfinite(selected).all():
                    raise ValueError(
                        f"EXIOBASE {member_name} contains non-finite export-use values "
                        f"at row {row_idx}."
                    )
                sector = str(row_sector)
                if sector not in values_by_sector:
                    values_by_sector[sector] = 0.0
                    order.append(sector)
                values_by_sector[sector] += float(selected.sum())
    return pd.Series(
        [values_by_sector[sector] for sector in order],
        index=pd.Index(order, name="sector"),
        dtype="float64",
        name=f"{member_name}_exports",
    )


def read_exiobase3_region_export_vector(*, source_zip: str | Path, region: str) -> pd.Series:
    """Return product exports from ``region`` to all foreign intermediate/final users."""
    zip_path = Path(source_zip)
    z_exports = _read_region_exports_member_from_exio_zip(
        zip_path=zip_path, member_name="Z.txt", region=region
    )
    y_exports = _read_region_exports_member_from_exio_zip(
        zip_path=zip_path, member_name="Y.txt", region=region
    )
    sectors = z_exports.index.union(y_exports.index, sort=False)
    return z_exports.reindex(sectors, fill_value=0.0).add(
        y_exports.reindex(sectors, fill_value=0.0), fill_value=0.0
    ).rename("exports")

def read_exiobase3_destination_final_demand_totals(
    *, source_zip: str | Path, region: str
) -> pd.Series:
    """Return one destination region's final demand by category across all origins.

    ``F_Y`` is indexed by the destination final-demand category, not by producer
    product.  Its monetary denominator must therefore sum ``Y.txt`` over every
    producer origin supplying the requested destination region.  This helper
    streams only that destination's Y columns from the source zip.
    """
    total_use = _read_scope_use_table_from_exio_zip(
        zip_path=Path(source_zip),
        member_name="Y.txt",
        scope=RegionScope(code=str(region), members=(str(region),)),
        flow="total",
    )
    totals = total_use.sum(axis=0).astype("float64")
    if not isinstance(totals, pd.Series):
        raise TypeError("EXIO final-demand total reduction did not return a Series.")
    totals.index = pd.Index(
        [str(label) for label in totals.index], name="final_demand_category"
    )
    if not totals.index.is_unique:
        raise ValueError("EXIO destination final-demand categories must be unique.")
    if totals.isna().any() or not np.isfinite(totals.to_numpy(dtype="float64")).all():
        raise ValueError("EXIO destination final-demand totals must be finite.")
    return totals.rename("final_demand_total")

def _load_exio_scope_slice_direct_from_zip(
    *,
    year: int,
    system: str,
    scope: RegionScope,
    source_zip: Path,
    flow: str,
    doi: str | None = None,
) -> MrioScopeSlice:
    """Build one EXIOBASE destination-scope use slice from x.txt/Z.txt/Y.txt."""
    flow_norm = _normalize_scope_flow(flow)
    # Z and Y are extensive destination-use accounts. Collapse geography first; no
    # coefficient or intensity is formed in this work package.
    Z_use = _read_scope_use_table_from_exio_zip(
        zip_path=source_zip,
        member_name="Z.txt",
        scope=scope,
        flow=flow_norm,
    )
    Y_use = _read_scope_use_table_from_exio_zip(
        zip_path=source_zip,
        member_name="Y.txt",
        scope=scope,
        flow=flow_norm,
    )
    # A missing Y product row is unavailable/malformed source evidence, not a valid
    # all-zero final-demand row. Preserve the existing fail-closed singleton rule.
    z_rows = pd.Index(Z_use.index.astype(str))
    y_rows = pd.Index(Y_use.index.astype(str))
    missing_y = z_rows.difference(y_rows)
    extra_y = y_rows.difference(z_rows)
    if len(missing_y) or len(extra_y):
        raise ValueError(
            "EXIOBASE Y.txt row basis does not match Z.txt for the scope slice: "
            f"missing={list(map(str, missing_y[:10]))!r}, "
            f"extra={list(map(str, extra_y[:10]))!r}."
        )
    Y_use = Y_use.reindex(z_rows)
    x_output = _read_scope_x_from_exio_zip(zip_path=source_zip, scope=scope)

    # Translate EXIO source labels to the canonical product codes used by existing
    # split/concordance consumers only after the monetary aggregation is complete.
    label_to_code = _exio_label_to_code_lookup(system)
    Z_use = Z_use.copy()
    Y_use = Y_use.copy()
    Z_use.index = _code_index(Z_use.index, label_to_code=label_to_code).rename("sector")
    Z_use.columns = _code_index(Z_use.columns, label_to_code=label_to_code)
    Y_use.index = _code_index(Y_use.index, label_to_code=label_to_code).rename("sector")
    x_output = x_output.copy()
    try:
        x_output.index = _code_index(x_output.index, label_to_code=label_to_code)
    except ValueError as exc:
        raise ValueError(
            "EXIOBASE x.txt sector basis does not exactly match the scope Z basis."
        ) from exc

    sectors = pd.Index(Z_use.index.astype(str), name="sector")
    x_sectors = pd.Index(x_output.index.astype(str), name="sector")
    missing_x = sectors.difference(x_sectors)
    extra_x = x_sectors.difference(sectors)
    if len(missing_x) or len(extra_x):
        raise ValueError(
            "EXIOBASE x.txt sector basis does not exactly match the scope Z basis: "
            f"missing={list(map(str, missing_x[:10]))!r}, "
            f"extra={list(map(str, extra_x[:10]))!r}."
        )
    x_scope = x_output.reindex(sectors)
    metadata: dict[str, object] = {
        "dataset": "exiobase",
        "year": int(year),
        "system": str(system),
        "scope_code": str(scope.code),
        "scope_members": list(map(str, scope.members)),
        "flow": flow_norm,
        "source_zip": str(source_zip),
        "cache_kind": "exiobase_zip_stream_scope_slice",
        "zip_only_cache": True,
        "source_zip_retained": True,
    }
    if doi is not None:
        metadata["exio_doi"] = _normalize_exiobase_doi(str(doi))
    return MrioScopeSlice(
        scope=scope,
        Z=Z_use.astype("float64"),
        Y=Y_use.astype("float64"),
        x_scope=x_scope.astype("float64").rename("x_scope"),
        flow=cast(ScopeFlow, flow_norm),
        metadata=metadata,
    )


def _load_exiobase3_iot_scope_slice(
    *,
    year: int,
    scope: RegionScope,
    system: str = "pxp",
    flow: str = "internal",
    force_download: bool = False,
    data_root: str | Path | None = None,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    verbose: bool = False,
) -> MrioScopeSlice:
    """EXIOBASE provider for the public low-memory IOT scope-slice boundary."""
    if data_root is None:
        from ecolecon_mrio.data import default_data_root

        data_root = default_data_root()
    root = Path(data_root)
    flow_norm = _normalize_scope_flow(flow)
    _scope_members(scope)

    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_cache_paths

    exio_paths = resolve_exiobase_cache_paths(year, system=system, data_root=root)
    data_dir = exio_paths["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    source_zip = ensure_exiobase_year_zip(
        year,
        data_dir=data_dir,
        system=system,
        force_download=force_download,
        allow_download_missing=allow_download_missing,
        resume_downloads=resume_downloads,
        show_download_progress=show_download_progress,
        download_chunk_mb=download_chunk_mb,
        download_timeout_seconds=download_timeout_seconds,
        download_retries=download_retries,
        doi=str(doi),
        verbose=verbose,
    )
    if verbose:
        print(
            f"Streaming EXIOBASE scope slice directly from {source_zip.name} "
            f"for scope={scope.code} members={list(scope.members)!r} year={year} "
            f"system={system} flow={flow_norm} (source-zip only) ..."
        )
    return _load_exio_scope_slice_direct_from_zip(
        year=year,
        system=system,
        scope=scope,
        source_zip=source_zip,
        flow=flow_norm,
        doi=doi,
    )

def select_by_patterns(values: Iterable[str], patterns: Sequence[str]) -> list[str]:
    patterns_l = [p.lower() for p in patterns]
    hits: list[str] = []
    for v in values:
        v_l = str(v).lower()
        if any(p in v_l for p in patterns_l):
            hits.append(str(v))
    return hits

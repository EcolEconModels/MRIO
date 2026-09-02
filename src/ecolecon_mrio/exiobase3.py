# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass
import json
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


@dataclass(frozen=True)
class ExioRegionSlice:
    year: int
    system: str
    region: str
    sectors: pd.Index
    Z_domestic: pd.DataFrame
    Y_region: pd.DataFrame
    x_region: pd.Series
    source_zip: Path
    flow: str = "domestic"

    def __post_init__(self) -> None:
        """Keep single-region sector axes consistently named.

        Direct zip streaming and legacy pickle slices may construct identical values
        with different pandas Index.name metadata.  Downstream code treats these
        axes as sector axes, so normalize only the row/index names here without
        changing values or column labels.
        """

        if not isinstance(self.sectors, pd.MultiIndex) and self.sectors.name != "sector":
            object.__setattr__(self, "sectors", pd.Index(self.sectors, name="sector"))
        for frame in (self.Z_domestic, self.Y_region):
            if not isinstance(frame.index, pd.MultiIndex) and frame.index.name != "sector":
                frame.index = pd.Index(frame.index, name="sector")
        if not isinstance(self.x_region.index, pd.MultiIndex) and self.x_region.index.name != "sector":
            self.x_region.index = pd.Index(self.x_region.index, name="sector")


def _coerce_single_column_df_to_series(x: pd.Series | pd.DataFrame, *, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas Series or DataFrame, got {type(x)}")
    if x.shape[1] != 1:
        raise ValueError(f"{name} must have exactly 1 column when provided as DataFrame, got shape={x.shape}")
    s = x.iloc[:, 0]
    # Keep the original index, but ensure a stable name for downstream code.
    s.name = s.name or name
    return s


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


def _normalize_flow(flow: str) -> str:
    flow_norm = str(flow).strip().lower()
    if flow_norm in {"dom", "domestic"}:
        return "domestic"
    if flow_norm in {"imp", "import", "imports"}:
        return "imports"
    if flow_norm in {"tot", "total"}:
        return "total"
    raise ValueError("flow must be one of: domestic, imports, total")


def _compute_region_output_from_mrio(*, Z: pd.DataFrame, Y: pd.DataFrame, region: str) -> pd.Series:
    """Return producer-side output for one region across all destinations/final-demand columns."""
    Z_from_r = Z.xs(region, level=0, axis=0)
    Y_from_r = Y.xs(region, level=0, axis=0)
    x_output = (Z_from_r.sum(axis=1) + Y_from_r.sum(axis=1)).astype("float64")
    x_output.index = x_output.index.astype(str)
    return x_output


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
    member_set = set(members)
    keep_positions = [
        idx for idx, col_region in enumerate(regions) if str(col_region) in member_set
    ]
    keep_labels = [str(labels[idx]) for idx in keep_positions]
    # Every destination country repeats the same sector/final-demand labels. Collapse
    # those duplicate destination labels while retaining first-occurrence order.
    collapsed_labels = list(dict.fromkeys(keep_labels))
    label_position = {label: idx for idx, label in enumerate(collapsed_labels)}
    collapse_positions = [label_position[label] for label in keep_labels]
    return keep_positions, collapsed_labels, collapse_positions


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
            row_order: list[str] = []
            all_row_order: list[str] = []
            seen_all: set[str] = set()
            max_keep = max(keep_positions)
            # Stream producer rows once. Origin membership determines internal versus
            # external-import use; destination membership was fixed by the header slice.
            for row_idx, line in enumerate(fh):
                try:
                    row_region, row_sector, _rest = line.rstrip("\n\r").split("\t", 2)
                except ValueError:
                    raise ValueError(
                        f"EXIOBASE {member_name} malformed row at position {row_idx}"
                    ) from None
                sector = str(row_sector)
                if sector not in seen_all:
                    seen_all.add(sector)
                    all_row_order.append(sector)
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
                    row_order.append(sector)
                acc[sector] += collapsed
    if not all_row_order:
        raise ValueError(f"EXIOBASE {member_name} contains no source rows.")
    if not row_order:
        # The all-regions external-import limiting case is a valid zero account.
        row_order = all_row_order
    data = np.vstack(
        [acc.get(sector, np.zeros(len(output_labels), dtype="float64")) for sector in row_order]
    )
    return pd.DataFrame(
        data,
        index=pd.Index(row_order, name="sector"),
        columns=pd.Index(output_labels),
        dtype="float64",
    )


def _read_region_use_table_from_exio_zip(
    *,
    zip_path: Path,
    member_name: str,
    region: str,
    flow: str,
) -> pd.DataFrame:
    """Backward-compatible singleton adapter to the scope-streaming implementation."""
    flow_norm = _normalize_flow(flow)
    scope_flow = {
        "domestic": "internal",
        "imports": "external_imports",
        "total": "total",
    }[flow_norm]
    return _read_scope_use_table_from_exio_zip(
        zip_path=zip_path,
        member_name=member_name,
        scope=RegionScope(code=str(region), members=(str(region),)),
        flow=scope_flow,
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
    total_use = _read_region_use_table_from_exio_zip(
        zip_path=Path(source_zip),
        member_name="Y.txt",
        region=str(region),
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
    x_output.index = _code_index(x_output.index, label_to_code=label_to_code)

    sectors = pd.Index(Z_use.index.astype(str), name="sector")
    x_scope = x_output.reindex(sectors)
    missing_x = [str(sector) for sector, value in x_scope.items() if pd.isna(value)]
    if missing_x:
        raise ValueError(
            f"EXIOBASE x.txt is missing output values for scope sectors: {missing_x[:10]!r}"
        )
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
        x_scope=x_scope.astype("float64"),
        flow=cast(ScopeFlow, flow_norm),
        metadata=metadata,
    )


def _load_exio_region_slice_direct_from_zip(
    *,
    year: int,
    system: str,
    region: str,
    source_zip: Path,
    flow: str,
) -> ExioRegionSlice:
    """Backward-compatible singleton adapter over the scope streaming core."""
    flow_norm = _normalize_flow(flow)
    scope_flow = {
        "domestic": "internal",
        "imports": "external_imports",
        "total": "total",
    }[flow_norm]
    scope_slice = _load_exio_scope_slice_direct_from_zip(
        year=year,
        system=system,
        scope=RegionScope(code=str(region), members=(str(region),)),
        source_zip=source_zip,
        flow=scope_flow,
    )
    sectors = pd.Index(scope_slice.Z.index.astype(str), name="sector")
    return ExioRegionSlice(
        year=int(year),
        system=str(system),
        region=str(region),
        sectors=sectors,
        Z_domestic=scope_slice.Z,
        Y_region=scope_slice.Y,
        x_region=scope_slice.x_scope.rename("x_region"),
        source_zip=source_zip,
        flow=flow_norm,
    )

def _region_slice_cache_paths(
    *,
    cache_root: Path,
    year: int,
    region: str,
    system: str,
    flow: str,
) -> dict[str, Path]:
    flow_norm = _normalize_flow(flow)
    slice_dir = cache_root / f"exiobase3_{int(year)}" / "slices" / str(region) / str(system)
    if flow_norm == "domestic":
        z_name, y_name = "Z_domestic.pkl", "Y_region.pkl"
    elif flow_norm == "imports":
        z_name, y_name = "Z_imports.pkl", "Y_imports.pkl"
    else:
        z_name, y_name = "Z_total.pkl", "Y_total.pkl"
    return {
        "dir": slice_dir,
        "z": slice_dir / z_name,
        "y": slice_dir / y_name,
        "x": slice_dir / "x_region.pkl",
        "meta": slice_dir / "meta.json",
    }


def _read_cached_region_slice_if_valid(
    *,
    cache_root: Path,
    year: int,
    region: str,
    system: str,
    flow: str,
    doi: str,
    source_zip: Path,
) -> ExioRegionSlice | None:
    paths = _region_slice_cache_paths(cache_root=cache_root, year=year, region=region, system=system, flow=flow)
    required_paths = [paths["z"], paths["y"], paths["x"], paths["meta"]]
    if not any(path.exists() for path in required_paths):
        return None
    if not all(path.exists() for path in required_paths):
        return None
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cached EXIOBASE region slice metadata is not readable: {paths['meta']}") from exc
    flow_norm = _normalize_flow(flow)
    expected = {
        "year": int(year),
        "system": str(system),
        "region": str(region),
        "flow": flow_norm,
        "exio_doi": _normalize_exiobase_doi(doi),
    }
    actual = {
        "year": int(meta.get("year", -1)),
        "system": str(meta.get("system", "")),
        "region": str(meta.get("region", "")),
        "flow": _normalize_flow(str(meta.get("flow", "domestic"))),
        "exio_doi": _normalize_exiobase_doi(str(meta.get("exio_doi", ""))),
    }
    if actual != expected:
        raise ValueError(
            "Cached EXIOBASE region slice metadata does not match the requested source. "
            f"Expected {expected}; found {actual}. Use force_rebuild_slice=True or cache_region_slice=False."
        )
    label_to_code = _exio_label_to_code_lookup(system)
    Z_use = cast(pd.DataFrame, pd.read_pickle(paths["z"])).copy()
    Y_use = cast(pd.DataFrame, pd.read_pickle(paths["y"])).copy()
    x_output = _coerce_single_column_df_to_series(pd.read_pickle(paths["x"]), name="x_region").copy()
    Z_use.index = _code_index(Z_use.index, label_to_code=label_to_code)
    Z_use.columns = _code_index(Z_use.columns, label_to_code=label_to_code)
    Y_use.index = _code_index(Y_use.index, label_to_code=label_to_code)
    x_output.index = _code_index(x_output.index, label_to_code=label_to_code)
    sectors = pd.Index(Z_use.index.astype(str), name="sector")
    return ExioRegionSlice(
        year=int(year),
        system=str(system),
        region=str(region),
        sectors=sectors,
        Z_domestic=Z_use.astype("float64"),
        Y_region=Y_use.astype("float64"),
        x_region=x_output.reindex(sectors).astype("float64"),
        source_zip=source_zip,
        flow=flow_norm,
    )


def _write_region_slice_cache(
    *,
    cache_root: Path,
    region_slice: ExioRegionSlice,
    doi: str,
) -> None:
    paths = _region_slice_cache_paths(
        cache_root=cache_root,
        year=region_slice.year,
        region=region_slice.region,
        system=region_slice.system,
        flow=region_slice.flow,
    )
    paths["dir"].mkdir(parents=True, exist_ok=True)
    region_slice.Z_domestic.to_pickle(paths["z"])
    region_slice.Y_region.to_pickle(paths["y"])
    region_slice.x_region.to_pickle(paths["x"])
    paths["meta"].write_text(
        json.dumps(
            {
                "year": int(region_slice.year),
                "system": str(region_slice.system),
                "region": str(region_slice.region),
                "flow": str(region_slice.flow),
                "exio_doi": _normalize_exiobase_doi(doi),
                "source_zip": str(region_slice.source_zip),
                "cache_kind": "exiobase_region_slice_optional_pickle",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

def build_exio_region_use_slice_from_mrio(
    *,
    year: int,
    system: str,
    region: str,
    Z: pd.DataFrame,
    Y: pd.DataFrame,
    source_zip: Path,
    flow: str = "domestic",
) -> ExioRegionSlice:
    """
    Build an EXIO-like single-region *use* slice from a full MRIO (Z,Y).

    Definitions (destination = `region`):
    - flow='domestic': origin == region
    - flow='imports' : sum over origins != region
    - flow='total'   : sum over all origins

    Returns an `ExioRegionSlice` whose `Z_domestic`/`Y_region` contain the chosen use matrices,
    while `x_region` stores producer-side regional output across all destinations.
    """
    flow_norm = _normalize_flow(flow)

    if not isinstance(Z.index, pd.MultiIndex) or not isinstance(Z.columns, pd.MultiIndex):
        raise ValueError("Expected MRIO Z to have MultiIndex rows/cols (region, sector).")
    if not isinstance(Y.index, pd.MultiIndex) or not isinstance(Y.columns, pd.MultiIndex):
        raise ValueError("Expected MRIO Y to have MultiIndex rows/cols.")

    # Slice use side (columns) to destination region (drops region level in columns).
    Z_to_r = Z.xs(region, level=0, axis=1)
    Y_to_r = Y.xs(region, level=0, axis=1)

    # Domestic use (origin == region).
    Z_dom = Z_to_r.xs(region, level=0, axis=0)
    Y_dom = Y_to_r.xs(region, level=0, axis=0)

    # Import use (origin != region), summed over origins -> by-sector.
    Z_foreign = Z_to_r.loc[Z_to_r.index.get_level_values(0) != region]
    Y_foreign = Y_to_r.loc[Y_to_r.index.get_level_values(0) != region]
    Z_imp = Z_foreign.groupby(level=1).sum()
    Y_imp = Y_foreign.groupby(level=1).sum()

    # Total use, summed over all origins -> by-sector.
    Z_tot = Z_to_r.groupby(level=1).sum()
    Y_tot = Y_to_r.groupby(level=1).sum()

    if flow_norm == "domestic":
        Z_use, Y_use = Z_dom, Y_dom
    elif flow_norm == "imports":
        Z_use, Y_use = Z_imp, Y_imp
    else:
        Z_use, Y_use = Z_tot, Y_tot

    Z_use = cast(pd.DataFrame, Z_use)
    Y_use = cast(pd.DataFrame, Y_use)
    sectors = pd.Index(Z_use.index.astype(str))
    x_output = cast(pd.Series, _compute_region_output_from_mrio(Z=Z, Y=Y, region=region).reindex(sectors)).astype("float64")

    return ExioRegionSlice(
        year=year,
        system=system,
        region=region,
        sectors=sectors,
        Z_domestic=Z_use,
        Y_region=Y_use,
        x_region=x_output,
        source_zip=source_zip,
        flow=flow_norm,
    )


def load_exiobase3_scope_slice(
    *,
    year: int,
    scope: RegionScope,
    system: str = "pxp",
    flow: str = "internal",
    force_download: bool = False,
    cache_root: str | Path | None = None,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    verbose: bool = False,
) -> MrioScopeSlice:
    """Stream one EXIOBASE destination-scope use block without a derived cache."""
    if cache_root is None:
        from ecolecon_mrio.data import default_data_root

        cache_root = default_data_root()
    cache_root = Path(cache_root)
    flow_norm = _normalize_scope_flow(flow)
    _scope_members(scope)

    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_cache_paths

    exio_paths = resolve_exiobase_cache_paths(year, system=system, data_root=cache_root)
    data_dir = exio_paths["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the repository low-RAM contract: ensure/reuse only the DOI-checked
    # source ZIP and do not create a persistent derived scope cache.
    source_zip = ensure_exiobase_year_zip(
        year,
        data_dir=data_dir,
        system=system,
        force_download=force_download,
        allow_download_missing=True,
        doi=str(doi),
        verbose=verbose,
    )
    if verbose:
        print(
            f"Streaming EXIOBASE scope slice directly from {source_zip.name} "
            f"for scope={scope.code} members={list(scope.members)!r} year={year} "
            f"system={system} flow={flow_norm} (without derived scope cache) ..."
        )
    return _load_exio_scope_slice_direct_from_zip(
        year=year,
        system=system,
        scope=scope,
        source_zip=source_zip,
        flow=flow_norm,
        doi=doi,
    )

def load_exiobase3_region_slice(
    *,
    year: int,
    region: str = "AT",
    system: str = "pxp",
    flow: str = "domestic",
    force_download: bool = False,
    force_rebuild_slice: bool = False,
    cache_region_slice: bool = False,
    cache_root: str | Path | None = None,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    verbose: bool = False,
) -> ExioRegionSlice:
    """
    Load a single-region EXIOBASE use slice.

    By default this streams x.txt, Z.txt, and Y.txt directly from the DOI-checked
    source zip and writes no derived slice cache. Set cache_region_slice=True only
    when a caller explicitly wants optional pickle caching of the derived slice.
    """
    if cache_root is None:
        from ecolecon_mrio.data import default_data_root

        cache_root = default_data_root()
    cache_root = Path(cache_root)
    flow_norm = _normalize_flow(flow)

    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_cache_paths

    exio_paths = resolve_exiobase_cache_paths(year, system=system, data_root=cache_root)
    data_dir = exio_paths["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    source_zip = ensure_exiobase_year_zip(
        year,
        data_dir=data_dir,
        system=system,
        force_download=force_download,
        allow_download_missing=True,
        doi=str(doi),
        verbose=verbose,
    )

    if cache_region_slice and not force_rebuild_slice:
        cached = _read_cached_region_slice_if_valid(
            cache_root=cache_root,
            year=year,
            region=region,
            system=system,
            flow=flow_norm,
            doi=doi,
            source_zip=source_zip,
        )
        if cached is not None:
            if verbose:
                print(
                    f"Loading cached EXIOBASE region slice for region={region} year={year} "
                    f"system={system} flow={flow_norm} ..."
                )
            return cached

    if verbose:
        cache_msg = "with optional slice cache write" if cache_region_slice else "without slice cache write"
        print(
            f"Streaming EXIOBASE region slice directly from {source_zip.name} "
            f"for region={region} year={year} system={system} flow={flow_norm} ({cache_msg}) ..."
        )
    region_slice = _load_exio_region_slice_direct_from_zip(
        year=year,
        system=system,
        region=region,
        source_zip=source_zip,
        flow=flow_norm,
    )
    if cache_region_slice:
        _write_region_slice_cache(cache_root=cache_root, region_slice=region_slice, doi=doi)
    return region_slice


def load_exiobase3_region_slices(
    *,
    year: int,
    regions: Sequence[str],
    system: str = "pxp",
    flow: str = "domestic",
    force_download: bool = False,
    force_rebuild_slice: bool = False,
    cache_region_slice: bool = False,
    cache_root: str | Path | None = None,
    doi: str = DEFAULT_EXIOBASE3_DOI,
) -> dict[str, ExioRegionSlice]:
    """Load multiple region slices through the direct-from-zip region-slice boundary."""
    if cache_root is None:
        from ecolecon_mrio.data import default_data_root

        cache_root = default_data_root()
    cache_root = Path(cache_root)
    regions = list(dict.fromkeys(map(str, regions)))
    if not regions:
        return {}
    return {
        region: load_exiobase3_region_slice(
            year=year,
            region=region,
            system=system,
            flow=flow,
            force_download=force_download,
            force_rebuild_slice=force_rebuild_slice,
            cache_region_slice=cache_region_slice,
            cache_root=cache_root,
            doi=doi,
        )
        for region in regions
    }

def select_by_patterns(values: Iterable[str], patterns: Sequence[str]) -> list[str]:
    patterns_l = [p.lower() for p in patterns]
    hits: list[str] = []
    for v in values:
        v_l = str(v).lower()
        if any(p in v_l for p in patterns_l):
            hits.append(str(v))
    return hits


def build_split_profiles_from_exio_domestic_block(
    exio_slice: ExioRegionSlice,
    *,
    aggregate_sector: str,
    exio_sector_patterns: Sequence[str],
    prefix_labels_with: str | None = None,
    selector: str = "patterns",
):
    """
    Build 1D split profiles for disaggregation using EXIOBASE domestic block totals.

    Returns (SplitProfiles, matched_exio_sectors).
    """
    from ecolecon_mrio.disaggregation import SplitProfiles, normalize_share

    Z_rr = exio_slice.Z_domestic
    Y_r = exio_slice.Y_region
    # Use-side "output" for the slice (matches Eurostat-style Z+Y accounting).
    x_use = (Y_r.sum(axis=1) + Z_rr.sum(axis=1)).astype("float64")

    exio_sector_index = pd.Index(Z_rr.index.astype(str))
    if selector == "patterns":
        matched = select_by_patterns(exio_sector_index, exio_sector_patterns)
    elif selector == "exact":
        available = set(map(str, exio_sector_index))
        matched = []
        missing = []
        for s in exio_sector_patterns:
            s_str = str(s)
            if s_str not in available:
                missing.append(s_str)
            else:
                matched.append(s_str)
        if missing:
            raise ValueError(
                f"Missing EXIO sector labels for aggregate_sector={aggregate_sector!r} in region={exio_slice.region}: "
                f"{missing}"
            )
    else:
        raise ValueError("selector must be one of: 'patterns', 'exact'")
    if not matched:
        raise ValueError(
            f"No EXIO sectors matched patterns={list(exio_sector_patterns)} in region={exio_slice.region}."
        )

    subsectors = [
        f"{prefix_labels_with}{s}" if prefix_labels_with is not None else s for s in matched
    ]
    subsector_to_exio = {sub: [exio] for exio, sub in zip(matched, subsectors)}

    matched_idx = pd.Index(matched)
    x_sub = x_use.loc[matched_idx].astype("float64")
    domestic_sales_intermediate = Z_rr.loc[matched_idx, :].sum(axis=1).astype("float64")
    domestic_inputs = Z_rr.loc[:, matched_idx].sum(axis=0).astype("float64")
    domestic_final_demand = Y_r.loc[matched_idx, :].sum(axis=1).astype("float64")

    output_share = normalize_share(x_sub, name=f"{aggregate_sector}.output_share")
    sales_share = normalize_share(
        domestic_sales_intermediate, name=f"{aggregate_sector}.intermediate_sales_share"
    )
    input_share = normalize_share(
        domestic_inputs, name=f"{aggregate_sector}.intermediate_input_share"
    )

    try:
        final_demand_share = normalize_share(
            domestic_final_demand, name=f"{aggregate_sector}.final_demand_share"
        )
    except ValueError:
        final_demand_share = sales_share.copy()

    rename = dict(zip(matched, subsectors))
    profiles = SplitProfiles(
        aggregate_sector=aggregate_sector,
        subsectors=tuple(subsectors),
        output_share=output_share.rename(index=rename),
        intermediate_sales_share=sales_share.rename(index=rename),
        intermediate_input_share=input_share.rename(index=rename),
        final_demand_share=final_demand_share.rename(index=rename),
        subsector_to_exio_sectors=subsector_to_exio,
    )
    return profiles, matched

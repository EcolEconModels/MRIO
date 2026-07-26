"""
Eurostat input-output table (IOT) downloader + parser utilities.

Design goals
------------
- Pymrio-like API: `download_*`, `parse_*`, `load_*`.
- Cache-first: never re-download unless `force_download=True`.
- General filters: caller specifies Eurostat dataset + dimension filters explicitly.
- Helpful errors: when a requested dimension name is wrong, print available dimensions
  and example category codes/labels observed in the downloaded payload.

Notes
-----
Eurostat's Dissemination API returns SDMX-JSON. This module implements a small,
dependency-light SDMX-JSON -> pandas converter that works for typical Eurostat
responses (sparse observation mapping via `value`).

Network access is required at runtime for download functions.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root


EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


class EurostatError(RuntimeError):
    pass


class EurostatHTTPError(EurostatError):
    pass


class EurostatParseError(EurostatError):
    pass


class EurostatFilterError(EurostatError):
    pass


@dataclass(frozen=True)
class EurostatDataset:
    dataset: str
    filters: dict[str, tuple[str, ...]]
    request_url: str
    raw_path: Path


def default_eurostat_storage_folder() -> Path:
    return default_data_root() / "eurostat"


def _canonicalize_filters(filters: Mapping[str, Any] | None) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    if not filters:
        return out
    for k, v in filters.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            vals = tuple(str(x) for x in v)
        else:
            vals = (str(v),)
        out[str(k)] = tuple(vals)
    return out


def _filters_to_query(filters: Mapping[str, Sequence[str]]) -> str:
    params: list[tuple[str, str]] = []
    for k, vals in filters.items():
        for v in vals:
            params.append((k, v))
    return urllib.parse.urlencode(params, doseq=True)


def _request_hash(dataset: str, filters: Mapping[str, Sequence[str]]) -> str:
    payload = json.dumps({"dataset": dataset, "filters": dict(filters)}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_eurostat_dataset(
    dataset: str,
    *,
    filters: Mapping[str, Any] | None = None,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> EurostatDataset:
    """
    Download an arbitrary Eurostat dataset slice as SDMX-JSON and cache it.

    Parameters
    ----------
    dataset:
        Eurostat dataset code (e.g. "naio_10_cp1700").
    filters:
        Mapping of Eurostat dimension -> selected category code(s).
        Values may be str or list/tuple/set of str.
    storage_folder:
        Base folder for caching raw payloads.
    force_download:
        If True, overwrite cached raw payload.
    timeout_s:
        Request timeout in seconds.

    Returns
    -------
    EurostatDataset
        Object describing the cached raw download.
    """
    filt = _canonicalize_filters(filters)
    h = _request_hash(dataset, filt)
    root = default_eurostat_storage_folder() if storage_folder is None else Path(storage_folder)
    base = root / dataset / h
    _ensure_dir(base)
    raw_path = base / "raw.json"

    query = _filters_to_query(filt)
    url = f"{EUROSTAT_API_BASE}/{urllib.parse.quote(dataset)}"
    if query:
        url = f"{url}?{query}"

    if raw_path.exists() and not force_download:
        return EurostatDataset(dataset=dataset, filters=filt, request_url=url, raw_path=raw_path)

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except Exception as e:  # pragma: no cover (network)
        raise EurostatHTTPError(
            f"Eurostat download failed for dataset={dataset} url={url}: {e}"
        ) from e

    raw_path.write_bytes(data)
    (base / "request.json").write_text(
        json.dumps({"dataset": dataset, "filters": filt, "url": url}, indent=2),
        encoding="utf-8",
    )
    return EurostatDataset(dataset=dataset, filters=filt, request_url=url, raw_path=raw_path)


def read_eurostat_raw(raw_path: str | Path) -> dict[str, Any]:
    raw = Path(raw_path).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise EurostatParseError(f"Invalid JSON at {raw_path}: {e}") from e


def eurostat_dimensions(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """
    Extract dimension category code -> label mappings from a Eurostat SDMX-JSON payload.

    Returns
    -------
    dict[dim_id, dict[code, label]]
    """
    dim = payload.get("dimension", {})
    ids = payload.get("id", []) or dim.get("id", [])
    out: dict[str, dict[str, str]] = {}
    for d in ids:
        entry = dim.get(d, {})
        cat = entry.get("category", {})
        labels = cat.get("label", {}) or {}
        index = cat.get("index", {}) or {}
        if labels:
            out[str(d)] = {str(code): str(labels.get(code, code)) for code in labels.keys()}
        elif index:
            out[str(d)] = {str(code): str(code) for code in index.keys()}
        else:
            out[str(d)] = {}
    return out


def sdmx_json_to_series(payload: Mapping[str, Any], *, value_key: str = "value") -> pd.Series:
    """
    Convert Eurostat SDMX-JSON payload into a pandas Series with MultiIndex.

    The Series index levels are the dataset dimensions in payload["id"] order.
    """
    ids = payload.get("id")
    sizes = payload.get("size")
    dim = payload.get("dimension")
    values = payload.get(value_key)

    if not isinstance(ids, list) or not isinstance(sizes, list) or not isinstance(dim, dict):
        raise EurostatParseError("Payload missing expected SDMX keys: id/size/dimension.")
    if not isinstance(values, dict):
        raise EurostatParseError("Payload missing expected SDMX observation mapping at key 'value'.")

    dims = [str(d) for d in ids]
    sizes_i = [int(s) for s in sizes]
    if len(dims) != len(sizes_i):
        raise EurostatParseError("SDMX id/size length mismatch.")

    code_by_pos: list[list[str]] = []
    for d, sz in zip(dims, sizes_i, strict=True):
        cat = dim.get(d, {}).get("category", {})
        idx = cat.get("index", {}) or {}
        if not isinstance(idx, dict) or not idx:
            if int(sz) == 0:
                raise EurostatParseError(
                    f"Empty category index for dimension '{d}' (size=0). "
                    "This usually indicates an invalid filter value or no data for the selected slice."
                )
            raise EurostatParseError(f"Missing category index for dimension '{d}'.")
        rev: list[str | None] = [None] * sz
        for code, pos in idx.items():
            rev[int(pos)] = str(code)
        if any(x is None for x in rev):
            raise EurostatParseError(f"Incomplete category index for dimension '{d}'.")
        code_by_pos.append([str(x) for x in rev])

    strides: list[int] = []
    prod = 1
    for sz in reversed(sizes_i):
        strides.append(prod)
        prod *= sz
    strides = list(reversed(strides))

    obs_idx = np.fromiter((int(k) for k in values.keys()), dtype=np.int64)
    obs_val = np.fromiter((float(v) for v in values.values()), dtype=np.float64)

    coords: list[np.ndarray] = []
    for dim_i, (sz, stride) in enumerate(zip(sizes_i, strides, strict=True)):
        pos = (obs_idx // stride) % sz
        mapping = np.array(code_by_pos[dim_i], dtype=object)
        coords.append(mapping[pos])

    tuples = list(zip(*coords, strict=True))
    mi = pd.MultiIndex.from_tuples(tuples, names=dims)
    return pd.Series(obs_val, index=mi)


def infer_source_rounding_quantum(
    payload: Mapping[str, Any],
    *,
    value_key: str = "value",
) -> float | None:
    """Infer the smallest decimal reporting quantum present in an SDMX payload.

    Eurostat controls are independently rounded source observations.  The
    inference uses the decimal representation retained by ``json.loads`` and
    therefore avoids imposing a dataset-wide hard-coded quantum.
    """
    values = payload.get(value_key)
    if not isinstance(values, dict) or not values:
        return None

    max_decimal_places = 0
    found_numeric = False
    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        found_numeric = True
        text = format(value, ".15g").lower()
        mantissa, _, exponent_text = text.partition("e")
        exponent = int(exponent_text) if exponent_text else 0
        decimals = len(mantissa.partition(".")[2])
        max_decimal_places = max(max_decimal_places, max(0, decimals - exponent))

    if not found_numeric:
        return None
    return 10.0 ** (-max_decimal_places)

def describe_eurostat_dataset(
    dataset: str,
    *,
    sample_filters: Mapping[str, Any] | None = None,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    max_codes_per_dim: int = 50,
) -> dict[str, Any]:
    """
    Download a (typically filtered) Eurostat payload and return dimension metadata.

    This is intended for interactive exploration: it shows which dimensions exist and
    which category codes/labels appear under the current filtering.
    """
    ds = download_eurostat_dataset(
        dataset,
        filters=sample_filters,
        storage_folder=storage_folder,
        force_download=force_download,
    )
    payload = read_eurostat_raw(ds.raw_path)
    dims = eurostat_dimensions(payload)

    summary: dict[str, Any] = {
        "dataset": dataset,
        "request_url": ds.request_url,
        "dimensions": {},
    }

    for dim_id, codes in dims.items():
        items = list(codes.items())
        summary["dimensions"][dim_id] = {
            "num_codes": len(items),
            "example_codes": items[:max_codes_per_dim],
        }
    return summary


@dataclass(frozen=True)
class ParsedEurostatIOT:
    Z: pd.DataFrame
    Y: pd.DataFrame
    meta: dict[str, Any]
    V: pd.DataFrame | None = None
    M: pd.DataFrame | None = None


def select_eurostat_atomic_final_demand_columns(
    parsed: ParsedEurostatIOT,
    *,
    include_exports: bool = True,
) -> list[str]:
    cols = [str(col) for col in parsed.Y.columns]
    selected: list[str] = []

    p3_detail = [code for code in ("P3_S13", "P3_S14", "P3_S15") if code in cols]
    if p3_detail:
        selected.extend(p3_detail)
    elif "P3" in cols:
        selected.append("P3")

    p5_detail = [code for code in ("P51G", "P52", "P53") if code in cols]
    if p5_detail:
        selected.extend(p5_detail)
    elif "P5" in cols:
        selected.append("P5")

    if include_exports:
        if "P6" in cols:
            selected.append("P6")
        else:
            selected.extend(sorted(code for code in cols if code.startswith("P6_")))

    return selected


def build_eurostat_atomic_final_demand(
    parsed: ParsedEurostatIOT,
    *,
    include_exports: bool = True,
) -> pd.DataFrame:
    cols = select_eurostat_atomic_final_demand_columns(parsed, include_exports=include_exports)
    if not cols:
        return parsed.Y.iloc[:, 0:0].copy()
    return parsed.Y.loc[:, cols].astype("float64").copy()


def build_eurostat_product_use_reference(
    parsed: ParsedEurostatIOT,
    *,
    exclude_final_demand_codes: Sequence[str] = (),
) -> pd.Series:
    y_block = build_eurostat_atomic_final_demand(parsed, include_exports=True)
    if exclude_final_demand_codes:
        exclude = {str(code) for code in exclude_final_demand_codes}
        drop_cols = [col for col in y_block.columns if str(col) in exclude]
        if drop_cols:
            y_block = y_block.drop(columns=drop_cols)
    reference = (parsed.Z.sum(axis=1) + y_block.sum(axis=1)).astype("float64")
    reference.index = reference.index.astype(str)
    return reference


def build_eurostat_intermediate_use_reference_from_rows(
    parsed: ParsedEurostatIOT,
    *,
    source_row_codes: Sequence[str],
) -> pd.Series:
    """
    Build a target-sector weighting reference from a subset of PRD_USE rows.

    For each target product/sector column in ``parsed.Z``, this sums the selected
    source-product rows and returns an intermediate-use-only reference vector.
    """
    selected = [str(code) for code in source_row_codes]
    if not selected:
        raise ValueError("source_row_codes must not be empty.")

    available = set(parsed.Z.index.astype(str))
    missing = [code for code in selected if code not in available]
    if missing:
        raise KeyError(f"Requested PRD_USE source rows not found: {missing}")

    reference = parsed.Z.loc[selected, :].sum(axis=0).astype("float64")
    reference.index = reference.index.astype(str)
    return reference


def parse_eurostat_iot(
    payload: Mapping[str, Any],
    *,
    row_dim: str,
    col_dim: str,
    value_key: str = "value",
) -> ParsedEurostatIOT:
    """
    Parse a Eurostat SDMX-JSON slice into (Z, Y, V, M) using specified row/column dimensions.

    The common Eurostat IOT layout is:
      - Rows: industries
      - Columns: industries + final demand categories
    In that layout, Z uses columns whose codes are also present in row industries.
    Y uses all remaining columns.
    """
    series = sdmx_json_to_series(payload, value_key=value_key)
    dims = list(series.index.names)
    if row_dim not in dims or col_dim not in dims:
        dim_info = eurostat_dimensions(payload)
        raise EurostatFilterError(
            "row_dim/col_dim not present in payload. "
            f"row_dim={row_dim} col_dim={col_dim} available_dims={dims}. "
            f"Observed dimension codes (filtered): { {k: list(v.keys())[:20] for k, v in dim_info.items()} }"
        )

    df_obj = series.unstack(col_dim)
    if not isinstance(df_obj, pd.DataFrame):
        raise EurostatParseError("Expected unstacked Eurostat table to be a DataFrame.")
    df = df_obj
    for d in dims:
        if d in {row_dim, col_dim}:
            continue
        if d not in df.index.names:
            continue
        # Require other dims to be singletons in this payload slice.
        level_vals = df.index.get_level_values(d).unique()
        if len(level_vals) != 1:
            raise EurostatParseError(
                f"Non-singleton extra dimension '{d}' in payload: {list(map(str, level_vals[:10]))}..."
            )
        df = df.droplevel(d)

    if not isinstance(df.index, pd.Index):
        raise EurostatParseError("Unexpected index type after reshaping SDMX payload.")

    industries = pd.Index(df.index.astype(str))
    columns = pd.Index(df.columns.astype(str))
    industry_set = set(map(str, industries))
    row_codes = set(map(str, industries))

    ind_cols = [c for c in columns if c in industry_set]
    fd_cols = [c for c in columns if c not in industry_set]
    if not ind_cols:
        raise EurostatParseError(
            "Could not identify intermediate-demand columns as intersection of row/column codes. "
            "Provide a more specific payload slice or post-process columns."
        )

    # Keep a square intermediate block by restricting rows to the same codes
    # that appear as intermediate-demand columns.
    Z = df.loc[ind_cols, ind_cols].astype("float64").fillna(0.0)
    Y = df.loc[ind_cols, fd_cols].astype("float64").fillna(0.0)
    lower_rows = [r for r in industries if r not in set(ind_cols)]
    V = df.loc[lower_rows, ind_cols].astype("float64").fillna(0.0) if lower_rows else None
    M = df.loc[lower_rows, fd_cols].astype("float64") if lower_rows and fd_cols else None

    if not Z.index.equals(pd.Index(ind_cols)):
        # Rows are industries; columns are a subset of industries but may be missing/ordered differently.
        # Keep Z as-is; caller can reorder if desired.
        pass

    meta = {
        "dimensions": {k: v for k, v in eurostat_dimensions(payload).items()},
        "row_dim": row_dim,
        "col_dim": col_dim,
        "identified_industry_columns": ind_cols,
        "identified_final_demand_columns": fd_cols,
        "row_codes_not_in_intermediate_columns": sorted(row_codes - set(ind_cols)),
        "intermediate_columns_not_in_rows": sorted(set(ind_cols) - row_codes),
        "identified_supplementary_rows": lower_rows,
        "source_rounding_quantum": infer_source_rounding_quantum(payload, value_key=value_key),
    }
    return ParsedEurostatIOT(Z=Z, Y=Y, meta=meta, V=V, M=M)


def load_eurostat_iot(
    *,
    dataset: str,
    filters: Mapping[str, Any],
    row_dim: str,
    col_dim: str,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> ParsedEurostatIOT:
    """
    Download/cache the raw Eurostat source payload and parse an IOT slice into (Z, Y, V, M).

    Only the raw SDMX-JSON source response is persisted by this convenience wrapper.
    Derived parsed pickle caches are intentionally not used: parsing is cheap relative
    to downloading, and stale parsed caches can silently drop newer parsed components
    such as supplementary rows ``V`` and ``M``.

    ``force_reparse`` is retained for CLI/API compatibility. It has no effect because
    there is no derived parsed cache to bypass.
    """
    _ = force_reparse
    ds = download_eurostat_dataset(
        dataset,
        filters=filters,
        storage_folder=storage_folder,
        force_download=force_download,
    )
    payload = read_eurostat_raw(ds.raw_path)
    return parse_eurostat_iot(payload, row_dim=row_dim, col_dim=col_dim)


def load_eurostat_iot_variants(
    *,
    dataset: str,
    base_filters: Mapping[str, Any],
    variant_dim: str,
    variant_values: Sequence[str],
    row_dim: str,
    col_dim: str,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> dict[str, ParsedEurostatIOT]:
    """
    Load multiple variants (e.g., `stk_flow` = DOM/IMP/TOTAL) of the same dataset slice.

    Uses the same raw-source cache layout as `load_eurostat_iot`; each variant is cached under its own
    hashed filter set, so repeated calls don't re-download unless forced. Derived parsed caches are not used.
    """
    out: dict[str, ParsedEurostatIOT] = {}
    for v in variant_values:
        filters = dict(base_filters)
        filters[variant_dim] = v
        out[str(v)] = load_eurostat_iot(
            dataset=dataset,
            filters=filters,
            row_dim=row_dim,
            col_dim=col_dim,
            storage_folder=storage_folder,
            force_download=force_download,
            force_reparse=force_reparse,
        )
    return out


def parse_eurostat_table_2d(
    payload: Mapping[str, Any],
    *,
    row_dim: str,
    col_dim: str,
    value_key: str = "value",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Parse a Eurostat SDMX-JSON slice into a 2D DataFrame (row_dim x col_dim).

    All remaining dimensions must be fixed singletons (typically enforced by filters).
    """
    series = sdmx_json_to_series(payload, value_key=value_key)
    dims = list(series.index.names)
    if row_dim not in dims or col_dim not in dims:
        dim_info = eurostat_dimensions(payload)
        raise EurostatFilterError(
            "row_dim/col_dim not present in payload. "
            f"row_dim={row_dim} col_dim={col_dim} available_dims={dims}. "
            f"Observed dimension codes (filtered): { {k: list(v.keys())[:20] for k, v in dim_info.items()} }"
        )

    df_obj = series.unstack(col_dim)
    if not isinstance(df_obj, pd.DataFrame):
        raise EurostatParseError("Expected unstacked Eurostat table to be a DataFrame.")
    df = df_obj
    for d in dims:
        if d in {row_dim, col_dim}:
            continue
        if d not in df.index.names:
            continue
        level_vals = df.index.get_level_values(d).unique()
        if len(level_vals) != 1:
            raise EurostatParseError(
                f"Non-singleton extra dimension '{d}' in payload: {list(map(str, level_vals[:10]))}..."
            )
        df = df.droplevel(d)

    df = df.astype("float64").fillna(0.0)
    meta = {"dimensions": eurostat_dimensions(payload), "row_dim": row_dim, "col_dim": col_dim}
    return df, meta


def load_eurostat_table_2d(
    *,
    dataset: str,
    filters: Mapping[str, Any],
    row_dim: str,
    col_dim: str,
    value_key: str = "value",
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Download/cache the raw Eurostat source payload and parse a 2D table.

    Only the raw SDMX-JSON response is persisted. Legacy derived parsed
    pickle caches are intentionally ignored, matching `load_eurostat_iot()`.
    ``force_reparse`` is retained for CLI/API compatibility and has no
    effect because there is no derived parsed cache to bypass.
    """
    _ = force_reparse
    ds = download_eurostat_dataset(
        dataset,
        filters=filters,
        storage_folder=storage_folder,
        force_download=force_download,
    )
    payload = read_eurostat_raw(ds.raw_path)
    return parse_eurostat_table_2d(payload, row_dim=row_dim, col_dim=col_dim, value_key=value_key)

def parse_eurostat_series_1d(
    payload: Mapping[str, Any],
    *,
    dim: str,
    value_key: str = "value",
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Parse a Eurostat SDMX-JSON slice into a Series indexed by `dim`.

    All remaining dimensions must be fixed singletons.
    """
    series = sdmx_json_to_series(payload, value_key=value_key)
    dims = list(series.index.names)
    if dim not in dims:
        dim_info = eurostat_dimensions(payload)
        raise EurostatFilterError(
            f"Requested dim not present in payload: dim={dim} available_dims={dims}. "
            f"Observed dimension codes (filtered): {list(dim_info.keys())}"
        )

    for d in dims:
        if d == dim:
            continue
        level_vals = series.index.get_level_values(d).unique()
        if len(level_vals) != 1:
            raise EurostatParseError(
                f"Non-singleton extra dimension '{d}' in payload: {list(map(str, level_vals[:10]))}..."
            )
        series = series.droplevel(d)

    if not isinstance(series.index, pd.Index):
        raise EurostatParseError("Unexpected index type after dropping singleton dims.")

    series.index = series.index.astype(str)
    numeric_obj = pd.to_numeric(series, errors="coerce")
    if not isinstance(numeric_obj, pd.Series):
        raise EurostatParseError("Expected parsed Eurostat 1D object to remain a Series.")
    series = numeric_obj.fillna(0.0).astype("float64")
    meta = {"dimensions": eurostat_dimensions(payload), "dim": dim}
    return series, meta


def load_eurostat_series_1d(
    *,
    dataset: str,
    filters: Mapping[str, Any],
    dim: str,
    value_key: str = "value",
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Download/cache the raw Eurostat source payload and parse a 1D Series.

    Only the raw SDMX-JSON response is persisted. Legacy derived parsed
    pickle caches are intentionally ignored, matching `load_eurostat_iot()`.
    ``force_reparse`` is retained for CLI/API compatibility and has no
    effect because there is no derived parsed cache to bypass.
    """
    _ = force_reparse
    ds = download_eurostat_dataset(
        dataset,
        filters=filters,
        storage_folder=storage_folder,
        force_download=force_download,
    )
    payload = read_eurostat_raw(ds.raw_path)
    return parse_eurostat_series_1d(payload, dim=dim, value_key=value_key)

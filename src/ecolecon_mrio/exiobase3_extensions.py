from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, cast
import csv
import zipfile

import numpy as np
import pandas as pd

from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors
from ecolecon_mrio.data import DEFAULT_EXIOBASE3_DOI


MatrixKind = Literal["F", "F_Y"]






def _extension_column_label_level_name(kind: MatrixKind) -> str:
    """Return the second column-index level name for an EXIOBASE extension kind."""

    kind_clean = str(kind)
    if kind_clean == "F":
        return "sector"
    if kind_clean == "F_Y":
        return "final_demand_category"
    raise ValueError(f"Unsupported extension matrix kind: {kind!r}")


@dataclass(frozen=True)
class ExtensionRowResult:
    """One streamed EXIOBASE extension row plus unit metadata.

    The numeric row remains a pandas Series indexed by a two-level MultiIndex.
    For ``F`` rows the second level is ``sector``; for ``F_Y`` rows it is
    ``final_demand_category``.  Unit handling is deliberately limited to exposing
    the source unit label for reporting; no unit conversion is attempted here.
    """

    year: int | None
    values: pd.Series
    stressor: str
    unit: str
    unit_source: str
    extension: str
    kind: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class ExioExtension:
    name: str
    unit: pd.Series  # index=stressor, value=unit
    F: pd.DataFrame | None
    F_Y: pd.DataFrame | None



def _exio_code_to_name_lookup(system: str = "pxp") -> dict[str, str]:
    concordance = load_pymrio_exio3_sectors(system=system).sectors
    required = {"ExioCode", "ExioName"}
    missing = required.difference(set(concordance.columns))
    if missing:
        raise ValueError(f"EXIO sector concordance missing expected columns: {sorted(missing)}")
    codes = concordance["ExioCode"].astype(str)
    names = concordance["ExioName"].astype(str)
    return {
        code: name
        for code, name in zip(codes, names, strict=False)
        if code.strip() != "" and name.strip() != ""
    }



def _open_text_from_zip(zf: zipfile.ZipFile, member: str) -> TextIOWrapper:
    return TextIOWrapper(zf.open(member, "r"), encoding="utf-8", newline="")


def _resolve_container(path: str | Path) -> tuple[str, Path]:
    p = Path(path)
    if p.is_dir():
        return "dir", p
    if p.is_file() and p.suffix.lower() == ".zip":
        return "zip", p
    raise ValueError(f"Unsupported EXIOBASE container: {p}")


_REQUIRED_EXTENSION_FILES: frozenset[str] = frozenset(
    {"F.txt", "F_Y.txt", "unit.txt"}
)


def discover_exiobase3_extensions(container: str | Path) -> tuple[str, ...]:
    """Discover complete EXIO extension directories from the actual source archive.

    A directory is an extension candidate when it contains ``F.txt`` or
    ``F_Y.txt``. Every candidate must also contain the complementary matrix and
    ``unit.txt``. Exact archive directory names are returned; no release-specific
    alias table is maintained.
    """
    ctype, cpath = _resolve_container(container)
    present_by_extension: dict[str, set[str]] = {}
    if ctype == "zip":
        with zipfile.ZipFile(cpath, "r") as zf:
            for member in zf.namelist():
                parts = Path(member).parts
                if len(parts) != 2 or parts[1] not in _REQUIRED_EXTENSION_FILES:
                    continue
                present_by_extension.setdefault(str(parts[0]), set()).add(str(parts[1]))
    else:
        for child in cpath.iterdir():
            if not child.is_dir():
                continue
            present = {
                name for name in _REQUIRED_EXTENSION_FILES if (child / name).is_file()
            }
            if present:
                present_by_extension[str(child.name)] = present

    candidates = {
        name: present
        for name, present in present_by_extension.items()
        if {"F.txt", "F_Y.txt"}.intersection(present)
    }
    incomplete = {
        name: sorted(_REQUIRED_EXTENSION_FILES - present)
        for name, present in candidates.items()
        if present != _REQUIRED_EXTENSION_FILES
    }
    if incomplete:
        sample = dict(list(sorted(incomplete.items()))[:10])
        raise ValueError(
            "EXIOBASE extension directories are incomplete; selected/all-extension "
            f"processing requires F.txt, F_Y.txt, and unit.txt. Missing={sample!r}."
        )
    return tuple(
        sorted(
            name
            for name, present in candidates.items()
            if present == _REQUIRED_EXTENSION_FILES
        )
    )


def _read_header_lines(stream: TextIOWrapper) -> tuple[list[str], list[str], str]:
    line1 = stream.readline()
    line2 = stream.readline()
    line3 = stream.readline()
    if not line1 or not line2:
        raise ValueError("Extension file is missing header lines.")
    return line1.rstrip("\r\n").split("\t"), line2.rstrip("\r\n").split("\t"), line3.rstrip("\r\n")


def _read_extension_matrix(
    container: str | Path,
    *,
    extension: str,
    kind: MatrixKind,
    region: str | None = None,
) -> pd.DataFrame:
    """
    Read one EXIOBASE 3 extension matrix (F or F_Y) from a zip or folder.

    This is optimized for region subsetting: if `region` is given, it only reads
    the columns for that region.
    """
    ctype, cpath = _resolve_container(container)
    rel = f"{extension}/{kind}.txt"

    if ctype == "zip":
        with zipfile.ZipFile(cpath, "r") as zf:
            with _open_text_from_zip(zf, rel) as fh:
                h1, h2, h3 = _read_header_lines(fh)
            with _open_text_from_zip(zf, rel) as fh:
                keep = [i for i in range(1, len(h1)) if region is None or h1[i] == region]
                usecols = [0] + keep
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    header=None,
                    index_col=0,
                    usecols=usecols,  # pyright: ignore[reportCallIssue, reportArgumentType]
                    skiprows=3,
                    engine="c",
                )
    else:
        file_path = cpath / rel
        with file_path.open("r", encoding="utf-8") as fh:
            h1, h2, h3 = _read_header_lines(fh)
        keep = [i for i in range(1, len(h1)) if region is None or h1[i] == region]
        usecols = [0] + keep
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            index_col=0,
            usecols=usecols,  # pyright: ignore[reportCallIssue, reportArgumentType]
            skiprows=3,
            engine="c",
        )

    # Set consistent MultiIndex column names
    label_level_name = _extension_column_label_level_name(kind)
    region_codes = [h1[i] for i in keep]
    labels = [h2[i] for i in keep]
    df.columns = pd.MultiIndex.from_arrays(
        [region_codes, labels], names=["region", label_level_name]
    )
    df.index = df.index.astype(str)
    df.index.name = "stressor"
    return df.astype("float64")


def read_extension_units(container: str | Path, *, extension: str) -> pd.Series:
    ctype, cpath = _resolve_container(container)
    rel = f"{extension}/unit.txt"

    if ctype == "zip":
        with zipfile.ZipFile(cpath, "r") as zf:
            with _open_text_from_zip(zf, rel) as fh:
                df = pd.read_csv(fh, sep="\t")
    else:
        df = pd.read_csv(cpath / rel, sep="\t")

    if "stressor" not in df.columns or "unit" not in df.columns:
        raise ValueError(f"Unexpected unit file format for {extension}: columns={list(df.columns)}")
    out = pd.Series(df["unit"].to_numpy(), index=df["stressor"].astype(str))
    out.name = "unit"
    return out





def _select_extension_row_label(
    labels: Sequence[str],
    *,
    exact_priority: Sequence[str],
    keywords: Sequence[str],
) -> str:
    """Choose one EXIOBASE extension stressor row using explicit priority.

    The analysis_v2 direct-CO2 calculation needs a single stressor row from
    ``emissions/F.txt``.  Historical analysis code used the exact row
    ``CO2 - combustion - air`` when available, so exact-priority matches are
    tried before broader keyword matches.  This function returns the label only;
    the row reader then streams the matching values without loading the whole
    extension object or requiring ``unit.txt``.
    """
    available = [str(label) for label in labels]
    by_lower = {label.lower(): label for label in available}
    for candidate in exact_priority:
        hit = by_lower.get(str(candidate).lower())
        if hit is not None:
            return hit
    lowered_keywords = [str(keyword).lower() for keyword in keywords if str(keyword).strip()]
    for label in available:
        lowered_label = label.lower()
        if any(keyword in lowered_label for keyword in lowered_keywords):
            return label
    raise KeyError(
        "Could not find extension row matching "
        f"exact_priority={list(exact_priority)} or keywords={list(keywords)}; "
        f"first available labels={available[:10]}"
    )



def _align_production_extension_row_to_basis(
    row: pd.Series,
    basis_index: pd.MultiIndex,
    *,
    system: str,
) -> pd.Series:
    """Align one authoritative EXIO F row to the monetary x/Z/Y basis.

    Raw EXIO extension headers can use internal sector codes while x.txt exposes
    sector names.  Translate only official EXIO codes, then require a bijective
    label match before reordering.  This source-loader alignment is deliberately
    stricter than a generic ``reindex``: unknown, missing, extra, or duplicate
    labels remain scientific errors.
    """

    if not isinstance(basis_index, pd.MultiIndex) or list(basis_index.names) != [
        "region",
        "sector",
    ]:
        raise ValueError("basis_index must be the canonical MultiIndex(region, sector)")
    if not basis_index.is_unique:
        raise ValueError("basis_index must contain unique region-sector labels")
    if not isinstance(row.index, pd.MultiIndex) or list(row.index.names) != [
        "region",
        "sector",
    ]:
        raise ValueError("production extension row must use MultiIndex(region, sector)")
    if not row.index.is_unique:
        raise ValueError("production extension row contains duplicate region-sector labels")
    if row.index.equals(basis_index):
        return row
    if len(row.index) != len(basis_index):
        raise ValueError(
            "production extension row length differs from canonical MRIO basis: "
            f"{len(row.index)} != {len(basis_index)}"
        )

    code_to_name = _exio_code_to_name_lookup(system=system)
    raw_regions = row.index.get_level_values("region").astype(str)
    raw_sectors = row.index.get_level_values("sector").astype(str)
    canonical_sectors = [code_to_name.get(str(label), str(label)) for label in raw_sectors]
    translated_index = pd.MultiIndex.from_arrays(
        [raw_regions, canonical_sectors], names=["region", "sector"]
    )
    if not translated_index.is_unique:
        raise ValueError("EXIO sector-code translation created duplicate extension labels")

    missing = basis_index.difference(translated_index)
    extra = translated_index.difference(basis_index)
    if len(missing) or len(extra):
        first_raw = row.index[0] if len(row.index) else None
        first_translated = translated_index[0] if len(translated_index) else None
        first_expected = basis_index[0] if len(basis_index) else None
        raise ValueError(
            "production extension labels do not match the canonical MRIO basis after "
            "official EXIO code-to-name translation; "
            f"missing={len(missing)}, extra={len(extra)}, first raw={first_raw!r}, "
            f"translated={first_translated!r}, expected={first_expected!r}"
        )

    aligned = pd.Series(
        row.to_numpy(dtype="float64", copy=False),
        index=translated_index,
        dtype="float64",
        name=row.name,
    ).reindex(basis_index)
    if aligned.isna().any():
        raise RuntimeError("canonical EXIO extension alignment introduced missing values")
    return aligned

def load_exiobase3_extension_row(
    container: str | Path,
    *,
    extension: str,
    kind: MatrixKind = "F",
    exact_priority: Sequence[str] = (),
    keywords: Sequence[str] = (),
    region: str | None = None,
    system: str = "pxp",
    basis_index: pd.MultiIndex | None = None,
) -> pd.Series:
    """Stream one row from an EXIOBASE extension matrix.

    This is the low-memory API analysis scripts should use when they need one
    stressor, such as direct CO2 from ``emissions/F.txt``.  It deliberately does
    not call :func:`read_extension_units`, because several source zips contain
    ``F.txt`` but not ``unit.txt``.  Units are irrelevant for this API: callers
    request a named row and receive the numeric row aligned to the same
    ``(region, sector)`` MultiIndex as the corresponding extension matrix.
    """
    ctype, cpath = _resolve_container(container)
    rel = f"{extension}/{kind}.txt"

    def _open() -> TextIOWrapper:
        if ctype == "zip":
            zf = zipfile.ZipFile(cpath, "r")
            try:
                raw = zf.open(rel, "r")
            except Exception:
                zf.close()
                raise
            wrapper = TextIOWrapper(raw, encoding="utf-8", newline="")
            original_close = wrapper.close

            def _close_with_zip() -> None:
                try:
                    original_close()
                finally:
                    zf.close()

            wrapper.close = _close_with_zip  # type: ignore[method-assign]
            return wrapper
        return (cpath / rel).open("r", encoding="utf-8", newline="")

    with _open() as fh:
        h1, h2, _h3 = _read_header_lines(fh)
        keep = [i for i in range(1, len(h1)) if region is None or h1[i] == region]
        if not keep:
            raise ValueError(f"No columns matched region={region!r} in {rel}")
        labels_seen: list[str] = []
        stored_rows: dict[str, list[str]] = {}
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            label = str(row[0])
            labels_seen.append(label)
            lowered = label.lower()
            exact_hit = any(lowered == str(candidate).lower() for candidate in exact_priority)
            keyword_hit = any(str(keyword).lower() in lowered for keyword in keywords)
            if exact_hit or keyword_hit:
                stored_rows[label] = row
        chosen = _select_extension_row_label(labels_seen, exact_priority=exact_priority, keywords=keywords)
        selected_row = stored_rows.get(chosen)
        if selected_row is None:
            raise KeyError(f"Selected row {chosen!r} was not retained while streaming {rel}")
        values = [float(selected_row[i]) for i in keep]
        region_codes = [h1[i] for i in keep]
        sector_labels = [h2[i] for i in keep]
        label_level_name = _extension_column_label_level_name(kind)
        index = pd.MultiIndex.from_arrays([region_codes, sector_labels], names=["region", label_level_name])
        out = pd.Series(values, index=index, dtype="float64")
        out.name = chosen
        if basis_index is not None:
            if str(kind) != "F":
                raise ValueError("basis_index alignment is valid only for production-side kind='F'")
            out = _align_production_extension_row_to_basis(
                out, basis_index, system=str(system)
            )
        return out


def load_exiobase3_extension_row_for_year(
    year: int,
    *,
    extension: str,
    kind: MatrixKind = "F",
    system: str = "pxp",
    data_root: str | Path | None = None,
    exact_priority: Sequence[str] = (),
    keywords: Sequence[str] = (),
    region: str | None = None,
    basis_index: pd.MultiIndex | None = None,
    force_download: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    verbose: bool = False,
) -> pd.Series:
    """Load one EXIOBASE extension row from the standard yearly source zip.

    This is the MRIO-owned API boundary for analysis scripts that need one
    extension stressor.  It keeps source-zip discovery/download policy in the
    MRIO repo and returns only the requested row, avoiding full extension loads.
    """
    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_year_dir

    year_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
    year_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ensure_exiobase_year_zip(
        int(year),
        data_dir=year_dir,
        system=str(system),
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
    return load_exiobase3_extension_row(
        zip_path,
        extension=str(extension),
        kind=kind,
        exact_priority=exact_priority,
        keywords=keywords,
        region=region,
        system=str(system),
        basis_index=basis_index,
    )


def load_exiobase3_extension_row_with_unit(
    container: str | Path,
    *,
    extension: str,
    kind: MatrixKind = "F",
    exact_priority: Sequence[str] = (),
    keywords: Sequence[str] = (),
    region: str | None = None,
    system: str = "pxp",
    basis_index: pd.MultiIndex | None = None,
    unit_override: str | None = None,
    require_unit: bool = True,
) -> ExtensionRowResult:
    """Stream one extension row and attach its unit label when available.

    This combines the existing low-memory row reader with ``read_extension_units``.
    It is still row-level and zip-stream friendly: the full extension matrix is
    not materialized.  If a source archive lacks ``unit.txt``, callers may provide
    an explicit override; otherwise ``require_unit=True`` fails loudly so figure
    tables do not report dimensionless emissions by accident.
    """

    row = load_exiobase3_extension_row(
        container,
        extension=str(extension),
        kind=kind,
        exact_priority=exact_priority,
        keywords=keywords,
        region=region,
        system=str(system),
        basis_index=basis_index,
    )
    stressor = str(row.name)
    if unit_override is not None and str(unit_override).strip():
        unit = str(unit_override).strip()
        unit_source = "explicit_override"
    else:
        try:
            units = read_extension_units(container, extension=str(extension))
            unit_value = units.get(stressor)
        except Exception as exc:
            if require_unit:
                raise ValueError(
                    f"Could not read unit metadata for extension={extension!r}, stressor={stressor!r}"
                ) from exc
            unit_value = None
        if unit_value is None or str(unit_value).strip() == "":
            if require_unit:
                raise ValueError(
                    f"No unit metadata for extension={extension!r}, stressor={stressor!r}; "
                    "pass unit_override=... only if the unit has been verified externally."
                )
            unit = ""
            unit_source = "missing_unit_allowed"
        else:
            unit = str(unit_value).strip()
            unit_source = f"{extension}/unit.txt"
    return ExtensionRowResult(
        year=None,
        values=row,
        stressor=stressor,
        unit=unit,
        unit_source=unit_source,
        extension=str(extension),
        kind=str(kind),
        metadata={
            "container": str(container),
            "region": region,
            "row_index_names": list(row.index.names) if isinstance(row.index, pd.MultiIndex) else [row.index.name],
        },
    )


def load_exiobase3_extension_row_with_unit_for_year(
    year: int,
    *,
    extension: str,
    kind: MatrixKind = "F",
    system: str = "pxp",
    data_root: str | Path | None = None,
    exact_priority: Sequence[str] = (),
    keywords: Sequence[str] = (),
    region: str | None = None,
    basis_index: pd.MultiIndex | None = None,
    force_download: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    unit_override: str | None = None,
    require_unit: bool = True,
    verbose: bool = False,
) -> ExtensionRowResult:
    """Load one yearly EXIOBASE extension row together with unit metadata."""

    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_year_dir

    year_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
    year_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ensure_exiobase_year_zip(
        int(year),
        data_dir=year_dir,
        system=str(system),
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
    result = load_exiobase3_extension_row_with_unit(
        zip_path,
        extension=str(extension),
        kind=kind,
        exact_priority=exact_priority,
        keywords=keywords,
        region=region,
        system=str(system),
        basis_index=basis_index,
        unit_override=unit_override,
        require_unit=require_unit,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "year": int(year),
            "system": str(system),
            "source_zip": str(zip_path),
            "exio_doi": str(doi),
        }
    )
    return ExtensionRowResult(
        year=int(year),
        values=result.values,
        stressor=result.stressor,
        unit=result.unit,
        unit_source=result.unit_source,
        extension=result.extension,
        kind=result.kind,
        metadata=metadata,
    )


def load_exiobase3_extension_for_year(
    year: int,
    *,
    extension: str,
    system: str = "pxp",
    data_root: str | Path | None = None,
    region: str | None = None,
    force_download: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str = DEFAULT_EXIOBASE3_DOI,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
) -> ExioExtension:
    """Load one EXIOBASE 3 extension by year using the standard MRIO cache layout.

    This helper keeps the raw EXIO zip in the shared year directory and reads the
    extension directly from that zip unless an explicit extension cache directory
    is provided.
    """
    from ecolecon_mrio.data import ensure_exiobase_year_zip, resolve_exiobase_year_dir

    year_dir = resolve_exiobase_year_dir(int(year), data_root=data_root)
    year_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ensure_exiobase_year_zip(
        int(year),
        data_dir=year_dir,
        system=str(system),
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
    return load_exiobase3_extension(
        zip_path,
        extension=str(extension),
        region=region,
        cache_dir=cache_dir,
    )


@lru_cache(maxsize=32)
def load_exiobase3_extension(
    container: str | Path,
    *,
    extension: str,
    region: str | None = None,
    cache_dir: str | Path | None = None,
) -> ExioExtension:
    """
    Load one EXIOBASE 3 extension with optional region subsetting.

    Uses an in-process cache to avoid re-reading large files repeatedly.
    """
    cache_root = Path(cache_dir) if cache_dir is not None else None

    def _pkl(name: str) -> Path | None:
        if cache_root is None:
            return None
        tag = "ALL" if region is None else str(region)
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"exioext_{extension}_{name}_{tag}.pkl"

    def _read_cached_frame(path: Path, *, label: str) -> pd.DataFrame:
        obj = pd.read_pickle(path)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Cached EXIOBASE extension {label} is not a DataFrame: {path}")
        return obj

    unit = read_extension_units(container, extension=extension)

    F: pd.DataFrame | None = None
    f_path = _pkl("F")
    if f_path is not None and f_path.exists():
        F = _read_cached_frame(f_path, label="F")
    else:
        try:
            F = _read_extension_matrix(container, extension=extension, kind="F", region=region)
        except (KeyError, FileNotFoundError):
            F = None
        if f_path is not None and F is not None:
            F.to_pickle(f_path)

    F_Y: pd.DataFrame | None = None
    fy_path = _pkl("F_Y")
    if fy_path is not None and fy_path.exists():
        F_Y = _read_cached_frame(fy_path, label="F_Y")
    else:
        try:
            F_Y = _read_extension_matrix(container, extension=extension, kind="F_Y", region=region)
        except (KeyError, FileNotFoundError):
            F_Y = None
        if fy_path is not None and F_Y is not None:
            F_Y.to_pickle(fy_path)

    return ExioExtension(name=extension, unit=unit, F=F, F_Y=F_Y)



def _select_rows_by_keywords(index: Iterable[str], keywords: Sequence[str]) -> list[str]:
    """Select stressor rows; an empty keyword list explicitly means all rows."""
    keys = [str(keyword).lower() for keyword in keywords if str(keyword).strip()]
    available = [str(row) for row in index]
    if not keys:
        return available
    return [row for row in available if any(key in row.lower() for key in keys)]

def extract_extension_values(
    *,
    container: str | Path,
    region: str,
    source_labels: Sequence[str],
    monetary_denominator: pd.Series,
    extension_requests: Mapping[str, Sequence[str]],
    kind: MatrixKind = "F",
    required: bool = True,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Extract arbitrary EXIOBASE extension values in one long-form schema.

    ``F`` rows are production-side environmental totals by EXIO sector and
    ``F_Y`` rows are direct final-demand totals by EXIO final-demand category.
    The raw physical value and its monetary denominator are retained separately;
    intensity is derived only as ``value / monetary_denominator``.  Empty
    stressor keyword lists request every row in the named extension.

    ``required`` controls whether this low-level extractor may skip an unavailable
    matrix. The fuels environmental pipeline requires both ``F`` and ``F_Y`` for
    every selected extension and therefore calls this helper with ``required=True``
    for both accounting boundaries.
    """
    kind_clean = str(kind)
    if kind_clean not in {"F", "F_Y"}:
        raise ValueError(f"Unsupported extension matrix kind: {kind!r}")
    if not extension_requests:
        return None

    labels = list(dict.fromkeys(str(label) for label in source_labels))
    if not labels:
        return None
    denominator = monetary_denominator.copy()
    denominator.index = denominator.index.astype(str)
    if not denominator.index.is_unique:
        raise ValueError("monetary_denominator index must be unique")
    denominator = denominator.reindex(labels).astype("float64")
    if denominator.isna().any() or not np.isfinite(denominator.to_numpy()).all():
        missing_index = pd.Index(denominator.index[denominator.isna().to_numpy(dtype="bool")])
        missing = list(map(str, missing_index))
        raise ValueError(
            "Extension monetary denominator must be finite and complete; "
            f"missing={missing[:10]!r}."
        )
    if kind_clean == "F" and bool((denominator < 0.0).any()):
        raise ValueError("Production-side extension monetary output must be non-negative.")

    rows: list[pd.DataFrame] = []
    for extension, requested_keywords in extension_requests.items():
        extension_name = str(extension).strip()
        if not extension_name:
            raise ValueError("Extension name must not be blank.")
        keywords = tuple(str(value) for value in requested_keywords)
        ext_obj = load_exiobase3_extension(
            container,
            extension=extension_name,
            region=str(region),
            cache_dir=cache_dir,
        )
        matrix = ext_obj.F if kind_clean == "F" else ext_obj.F_Y
        if matrix is None:
            if required:
                raise ValueError(
                    f"Requested EXIO extension matrix is unavailable: "
                    f"extension={extension_name!r}, kind={kind_clean!r}."
                )
            continue
        if not isinstance(matrix.columns, pd.MultiIndex) or matrix.columns.nlevels != 2:
            raise ValueError(
                f"Unexpected EXIO extension columns for {extension_name}/{kind_clean}."
            )
        try:
            region_matrix = matrix.xs(str(region), level=0, axis=1)
        except KeyError as exc:
            if required:
                raise ValueError(
                    f"EXIO extension {extension_name}/{kind_clean} has no region {region!r}."
                ) from exc
            continue
        if not isinstance(region_matrix, pd.DataFrame):
            raise TypeError(
                f"Expected two-dimensional region slice for {extension_name}/{kind_clean}."
            )
        region_matrix.columns = pd.Index(map(str, region_matrix.columns))
        available = set(map(str, region_matrix.columns))

        matrix_labels = labels
        if kind_clean == "F" and not set(labels).issubset(available):
            code_to_name = _exio_code_to_name_lookup()
            matrix_labels = [code_to_name.get(label, label) for label in labels]
        missing = [
            source_label
            for source_label, matrix_label in zip(labels, matrix_labels, strict=True)
            if matrix_label not in available
        ]
        if missing:
            if required:
                raise ValueError(
                    f"EXIO extension {extension_name}/{kind_clean} is missing requested labels: "
                    f"{missing[:10]!r}."
                )
            continue

        stressors = _select_rows_by_keywords(region_matrix.index, keywords)
        if not stressors:
            if required:
                raise ValueError(
                    f"No stressors matched extension={extension_name!r}, "
                    f"keywords={list(keywords)!r}."
                )
            continue

        for stressor in stressors:
            raw_values = region_matrix.loc[stressor, matrix_labels]
            if not isinstance(raw_values, pd.Series):
                raise TypeError(
                    f"Expected one-dimensional EXIO extension row for {stressor!r}."
                )
            values = pd.to_numeric(raw_values, errors="coerce")
            if not isinstance(values, pd.Series):
                raise TypeError("Extension numeric conversion did not return a Series.")
            # Blank selected satellite cells remain unavailable observations;
            # they must never be converted into a physical zero before validation.
            value_series = pd.Series(
                values.to_numpy(dtype="float64"),
                index=pd.Index(labels),
                dtype="float64",
            )
            if not np.isfinite(value_series.to_numpy()).all():
                raise ValueError(
                    f"Non-finite values in EXIO extension {extension_name!r}, "
                    f"stressor={stressor!r}."
                )
            unit_value = ext_obj.unit.get(stressor, None)
            if unit_value is None or not str(unit_value).strip():
                raise ValueError(
                    f"Missing unit for EXIO extension={extension_name!r}, stressor={stressor!r}."
                )
            inconsistent_zero = denominator.eq(0.0) & value_series.ne(0.0)
            if bool(inconsistent_zero.any()):
                bad = list(map(str, denominator.index[inconsistent_zero.to_numpy()][:10]))
                raise ValueError(
                    "EXIO extension has a nonzero physical value with zero monetary "
                    f"denominator: extension={extension_name!r}, kind={kind_clean!r}, "
                    f"stressor={stressor!r}, labels={bad!r}."
                )
            intensity = value_series / denominator.replace(0.0, np.nan)
            rows.append(
                pd.DataFrame(
                    {
                        "source_label": labels,
                        "extension": extension_name,
                        "kind": kind_clean,
                        "stressor": str(stressor),
                        "unit": str(unit_value).strip(),
                        "value": value_series.to_numpy(dtype="float64"),
                        "monetary_denominator": denominator.to_numpy(dtype="float64"),
                        "intensity": intensity.to_numpy(dtype="float64"),
                    }
                )
            )

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)

def extract_quantity_and_price(
    *,
    container: str | Path,
    region: str,
    sectors: Sequence[str],
    monetary_output: pd.Series,
    extensions: Sequence[str] = ("energy",),
    stressor_keywords: Sequence[str] = ("energy use",),
    cache_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """
    Extract physical quantities from EXIOBASE 3 extensions and compute quantity shares + unit-value proxies.

    Notes
    - Physical data in EXIOBASE 3 is provided via extensions (e.g., `energy/F.txt` in TJ).
    - The value per physical input here is a unit-value proxy: monetary_output / physical_quantity.
    """
    if not sectors:
        return None

    sectors = list(map(str, sectors))
    monetary_output = monetary_output.reindex(sectors).astype("float64")

    code_to_name = _exio_code_to_name_lookup()

    rows: list[pd.DataFrame] = []
    for ext in extensions:
        ext_obj = load_exiobase3_extension(
            container, extension=ext, region=region, cache_dir=cache_dir
        )
        if ext_obj.F is None:
            continue

        # The extension columns are labeled by sector names; ensure we can align.
        F = ext_obj.F
        if not isinstance(F.columns, pd.MultiIndex) or F.columns.nlevels != 2:
            continue

        # Pull the one-region columns and drop the region level.
        try:
            F_r = F.xs(region, level=0, axis=1)
        except Exception:
            continue

        if not isinstance(F_r.columns, pd.Index):
            F_r.columns = pd.Index(map(str, F_r.columns))
        F_r = F_r.loc[:, F_r.columns.astype(str)]

        sector_columns = sectors
        available = set(map(str, F_r.columns))
        if not all(sector in available for sector in sectors):
            sector_columns = [code_to_name.get(sector, sector) for sector in sectors]
        missing = [sector for sector, column in zip(sectors, sector_columns, strict=False) if column not in available]
        if missing:
            continue

        stressors = _select_rows_by_keywords(F_r.index, stressor_keywords)
        if not stressors:
            continue

        for stressor in stressors:
            qty = pd.Series(F_r.loc[stressor, sector_columns].to_numpy(dtype="float64"), index=sectors)
            total_qty = float(qty.sum())
            if not np.isfinite(total_qty) or total_qty == 0:
                continue
            qty_share = qty / total_qty
            value_per_physical_input = monetary_output / qty.replace(0.0, np.nan)
            unit = ext_obj.unit.get(stressor, None)
            out = pd.DataFrame(
                {
                    "exio_sector": sectors,
                    "extension": ext,
                    "stressor": stressor,
                    "unit": unit,
                    "quantity": qty,
                    "quantity_share": qty_share,
                    "monetary_output": monetary_output,
                    "value_per_physical_input": value_per_physical_input,
                }
            )
            rows.append(out)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def extract_emissions_and_intensity(
    *,
    container: str | Path,
    region: str,
    sectors: Sequence[str],
    monetary_output: pd.Series,
    extension: str = "air_emissions",
    stressor_keywords: Sequence[str] = ("CO2 - combustion - air",),
    cache_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Compatibility view over :func:`extract_extension_values` for diagnostics."""
    generic = extract_extension_values(
        container=container,
        region=region,
        source_labels=sectors,
        monetary_denominator=monetary_output,
        extension_requests={str(extension): tuple(stressor_keywords)},
        kind="F",
        required=False,
        cache_dir=cache_dir,
    )
    if generic is None:
        return None
    legacy = generic.rename(
        columns={
            "source_label": "exio_sector",
            "value": "emission",
            "monetary_denominator": "monetary_output",
            "intensity": "emission_intensity",
        }
    )
    return cast(pd.DataFrame, legacy.loc[
        :,
        [
            "exio_sector",
            "extension",
            "stressor",
            "unit",
            "emission",
            "monetary_output",
            "emission_intensity",
        ],
    ])

def extract_final_demand_emissions_and_intensity(
    *,
    container: str | Path,
    region: str,
    monetary_final_demand: pd.Series,
    extension: str = "air_emissions",
    stressor_keywords: Sequence[str] = ("CO2 - combustion - air",),
    cache_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Compatibility view over generic ``F_Y`` extension extraction."""
    generic = extract_extension_values(
        container=container,
        region=region,
        source_labels=list(map(str, monetary_final_demand.index)),
        monetary_denominator=monetary_final_demand,
        extension_requests={str(extension): tuple(stressor_keywords)},
        kind="F_Y",
        required=False,
        cache_dir=cache_dir,
    )
    if generic is None:
        return None
    legacy = generic.rename(
        columns={
            "source_label": "category",
            "value": "emission",
            "monetary_denominator": "monetary_final_demand",
            "intensity": "final_demand_emission_intensity",
        }
    )
    return cast(pd.DataFrame, legacy.loc[
        :,
        [
            "extension",
            "stressor",
            "unit",
            "category",
            "emission",
            "monetary_final_demand",
            "final_demand_emission_intensity",
        ],
    ])

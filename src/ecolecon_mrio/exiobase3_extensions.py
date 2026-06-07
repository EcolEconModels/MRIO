from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import TextIOWrapper
from pathlib import Path
from typing import Iterable, Literal, Sequence
import csv
import zipfile

import numpy as np
import pandas as pd

from ecolecon_mrio.exio_concordance import load_pymrio_exio3_sectors


MatrixKind = Literal["F", "F_Y"]


@dataclass(frozen=True)
class ExioExtension:
    name: str
    unit: pd.Series  # index=stressor, value=unit
    F: pd.DataFrame | None
    F_Y: pd.DataFrame | None


def _exio_code_to_name_lookup(system: str = "pxp") -> dict[str, str]:
    concordance = load_pymrio_exio3_sectors(system=system).sectors
    return {
        str(row.ExioCode): str(row.ExioName)
        for row in concordance.itertuples(index=False)
        if str(row.ExioCode).strip() != "" and str(row.ExioName).strip() != ""
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


def _read_header_lines(stream: TextIOWrapper) -> tuple[list[str], list[str], str]:
    line1 = stream.readline()
    line2 = stream.readline()
    line3 = stream.readline()
    if not line1 or not line2:
        raise ValueError("Extension file is missing header lines.")
    return line1.rstrip("\n").split("\t"), line2.rstrip("\n").split("\t"), line3.rstrip("\n")


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
                    usecols=usecols,
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
            usecols=usecols,
            skiprows=3,
            engine="c",
        )

    # Set consistent MultiIndex column names
    label_level_name = str(h2[0]).strip() or "label"
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


def load_exiobase3_extension_row(
    container: str | Path,
    *,
    extension: str,
    kind: MatrixKind = "F",
    exact_priority: Sequence[str] = (),
    keywords: Sequence[str] = (),
    region: str | None = None,
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
        index = pd.MultiIndex.from_arrays([region_codes, sector_labels], names=["region", "sector"])
        out = pd.Series(values, index=index, dtype="float64")
        out.name = chosen
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
    force_download: bool = False,
    allow_download_missing: bool = True,
    resume_downloads: bool = True,
    show_download_progress: bool = True,
    download_chunk_mb: int = 1,
    download_timeout_seconds: float | None = None,
    download_retries: int = 3,
    doi: str = "10.5281/zenodo.3583070",
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
    doi: str = "10.5281/zenodo.3583070",
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

    unit = read_extension_units(container, extension=extension)

    F = None
    f_path = _pkl("F")
    if f_path is not None and f_path.exists():
        F = pd.read_pickle(f_path)
    else:
        try:
            F = _read_extension_matrix(container, extension=extension, kind="F", region=region)
        except Exception:
            F = None
        if f_path is not None and F is not None:
            F.to_pickle(f_path)

    F_Y = None
    fy_path = _pkl("F_Y")
    if fy_path is not None and fy_path.exists():
        F_Y = pd.read_pickle(fy_path)
    else:
        try:
            F_Y = _read_extension_matrix(container, extension=extension, kind="F_Y", region=region)
        except Exception:
            F_Y = None
        if fy_path is not None and F_Y is not None:
            F_Y.to_pickle(fy_path)

    return ExioExtension(name=extension, unit=unit, F=F, F_Y=F_Y)


def _select_rows_by_keywords(index: Iterable[str], keywords: Sequence[str]) -> list[str]:
    keys = [k.lower() for k in keywords]
    out: list[str] = []
    for r in index:
        rl = str(r).lower()
        if any(k in rl for k in keys):
            out.append(str(r))
    return out


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
    """Extract emissions from one EXIOBASE extension and compute emissions intensity.

    This is the direct (production-based) environmental extension intensity:
        emission_intensity = emissions / monetary_output

    Notes
    - `monetary_output` must be aligned to EXIO sector labels and be in the same
      monetary basis as the IO system you will apply it to.
    - Units come from the extension's `unit.txt` (e.g. kg for CO2 stressors).
    """
    if not sectors:
        return None

    sectors = list(map(str, sectors))
    monetary_output = monetary_output.reindex(sectors).astype("float64")

    code_to_name = _exio_code_to_name_lookup()

    ext_obj = load_exiobase3_extension(
        container, extension=str(extension), region=str(region), cache_dir=cache_dir
    )
    if ext_obj.F is None:
        return None

    F = ext_obj.F
    if not isinstance(F.columns, pd.MultiIndex) or F.columns.nlevels != 2:
        return None

    try:
        F_r = F.xs(region, level=0, axis=1)
    except Exception:
        return None

    if not isinstance(F_r.columns, pd.Index):
        F_r.columns = pd.Index(map(str, F_r.columns))
    F_r = F_r.loc[:, F_r.columns.astype(str)]

    available = set(map(str, F_r.columns))
    sector_columns = sectors
    if not all(sector in available for sector in sectors):
        sector_columns = [code_to_name.get(sector, sector) for sector in sectors]
    missing = [sector for sector, column in zip(sectors, sector_columns, strict=False) if column not in available]
    if missing:
        return None

    stressors = _select_rows_by_keywords(F_r.index, stressor_keywords)
    if not stressors:
        return None

    rows: list[pd.DataFrame] = []
    for stressor in stressors:
        emis = pd.Series(F_r.loc[stressor, sector_columns].to_numpy(dtype="float64"), index=sectors)
        unit = ext_obj.unit.get(stressor, None)
        emission_intensity = emis / monetary_output.replace(0.0, np.nan)
        out = pd.DataFrame(
            {
                "exio_sector": sectors,
                "extension": str(extension),
                "stressor": stressor,
                "unit": unit,
                "emission": emis,
                "monetary_output": monetary_output,
                "emission_intensity": emission_intensity,
            }
        )
        rows.append(out)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def extract_final_demand_emissions_and_intensity(
    *,
    container: str | Path,
    region: str,
    monetary_final_demand: pd.Series,
    extension: str = "air_emissions",
    stressor_keywords: Sequence[str] = ("CO2 - combustion - air",),
    cache_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Extract direct emissions tied to final demand categories (EXIO F_Y).

    This captures *direct* emissions that occur in final demand (e.g. households'
    combustion), represented in EXIOBASE as an extension matrix F_Y.

    Computes:
        final_demand_emission_intensity = emission / monetary_final_demand

    Args:
        monetary_final_demand: Series indexed by EXIO final demand category labels
            (matching the second column header line in Y.txt / F_Y.txt), values in EUR.

    Returns:
        DataFrame with columns: extension, stressor, unit, category, emission,
        monetary_final_demand, final_demand_emission_intensity
    """
    monetary_final_demand = monetary_final_demand.copy()
    monetary_final_demand.index = monetary_final_demand.index.astype(str)
    monetary_final_demand = pd.to_numeric(monetary_final_demand, errors="coerce")

    ext_obj = load_exiobase3_extension(
        container, extension=str(extension), region=str(region), cache_dir=cache_dir
    )
    if ext_obj.F_Y is None:
        return None

    F_Y = ext_obj.F_Y
    if not isinstance(F_Y.columns, pd.MultiIndex) or F_Y.columns.nlevels != 2:
        return None

    try:
        F_Y_r = F_Y.xs(region, level=0, axis=1)
    except Exception:
        return None

    if not isinstance(F_Y_r.columns, pd.Index):
        F_Y_r.columns = pd.Index(map(str, F_Y_r.columns))
    categories = pd.Index(map(str, F_Y_r.columns))

    stressors = _select_rows_by_keywords(F_Y_r.index, stressor_keywords)
    if not stressors:
        return None

    rows: list[pd.DataFrame] = []
    for stressor in stressors:
        emis = F_Y_r.loc[stressor, categories].astype("float64")
        unit = ext_obj.unit.get(stressor, None)
        denom = monetary_final_demand.reindex(categories).astype("float64")
        intensity = emis / denom.replace(0.0, np.nan)
        out = pd.DataFrame(
            {
                "extension": str(extension),
                "stressor": stressor,
                "unit": unit,
                "category": categories,
                "emission": emis.to_numpy(),
                "monetary_final_demand": denom.to_numpy(),
                "final_demand_emission_intensity": intensity.to_numpy(),
            }
        )
        rows.append(out)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)

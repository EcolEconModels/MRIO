from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import json
import zipfile

import pandas as pd


@dataclass(frozen=True)
class ExioSectorConcordance:
    """
    Sector concordance for EXIOBASE 3 as shipped with pymrio.

    This is not a Eurostat CPA/NACE concordance; it is the EXIOBASE sector list
    (names + internal codes) that can be used to generate robust split mappings
    without ad-hoc substring matching.
    """

    system: str  # "pxp" or "ixi"
    sectors: pd.DataFrame  # columns from pymrio's sectors.tsv


@dataclass(frozen=True)
class ExioMetadata:
    """
    EXIOBASE sector metadata from a source-facing metadata loader.

    EXIOBASE IOT zips expose sector names in ``x.txt``/``unit.txt`` and
    version/system metadata in ``metadata.json``.  They do not expose the short
    stable codes used by this repo, such as ``p01.a`` or ``i01.a``; those codes
    come from the EXIOBASE sector metadata packaged with ``pymrio``.
    """

    system: str
    sectors: pd.DataFrame
    source: str
    source_zip: Path | None = None
    version: str = ""
    zip_sector_names: tuple[str, ...] = ()


def _normalize_system(system: str) -> str:
    s = str(system).strip().lower()
    if s in {"pxp", "product", "products", "productxproduct"}:
        return "pxp"
    if s in {"ixi", "industry", "industries", "industryxindustry"}:
        return "ixi"
    raise ValueError("system must be one of: pxp, ixi")


def _default_exiobase_zip(*, year: int, system: str, data_root: str | Path) -> Path:
    """Return the conventional raw EXIOBASE zip path for a year/system."""

    return Path(data_root) / f"exiobase3_{year}" / f"IOT_{year}_{system}.zip"


def load_pymrio_exio3_sectors(system: str = "pxp") -> ExioSectorConcordance:
    """
    Load the EXIOBASE 3 sector list packaged inside pymrio.

    The returned table is typically used to:
    - validate mapping CSVs against a canonical EXIO sector list
    - generate mapping CSVs via EXIO internal code prefixes (e.g. p23.20.*)
    """
    system_norm = _normalize_system(system)
    pkg = f"pymrio.mrio_models.exio3_{system_norm}"
    path = resources.files(pkg).joinpath("sectors.tsv")
    with path.open("r", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", dtype=str).fillna("")

    # Normalise the expected core columns to always exist (even if future pymrio adds/removes).
    for col in ("ExioName", "ExioCode", "ExioLabel"):
        if col not in df.columns:
            df[col] = ""
    df["ExioName"] = df["ExioName"].astype(str)
    df["ExioCode"] = df["ExioCode"].astype(str)
    df["ExioLabel"] = df["ExioLabel"].astype(str)

    return ExioSectorConcordance(system=system_norm, sectors=df)


def _read_exio_zip_sector_names(source_zip: Path) -> tuple[tuple[str, ...], str]:
    """Read sector names and version metadata from a raw EXIOBASE IOT zip."""

    with zipfile.ZipFile(source_zip) as zf:
        with zf.open("x.txt") as handle:
            x = pd.read_csv(handle, sep="\t", dtype=str, usecols=["sector"]).fillna("")
        version = ""
        if "metadata.json" in zf.namelist():
            try:
                metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
                version = str(metadata.get("version", "")).strip()
            except (json.JSONDecodeError, UnicodeDecodeError):
                version = ""
    names = tuple(dict.fromkeys(x["sector"].astype(str).str.strip().tolist()))
    return names, version


def load_exiobase_metadata(
    *,
    system: str = "pxp",
    year: int | None = None,
    source_zip: str | Path | None = None,
    data_root: str | Path = "data",
    validate_zip_names: bool = True,
) -> ExioMetadata:
    """
    Load EXIOBASE sector metadata through a stable metadata API.

    The stable code list comes from EXIOBASE 3 metadata packaged with ``pymrio``.
    When a raw EXIOBASE table zip is supplied or discoverable from ``year``, the
    loader also streams sector names from ``x.txt`` to verify the table basis.
    No zip extraction or derived cache is written.
    """

    system_norm = _normalize_system(system)
    concordance = load_pymrio_exio3_sectors(system_norm)
    sectors = concordance.sectors.copy()

    resolved_zip: Path | None = None
    if source_zip is not None:
        resolved_zip = Path(source_zip)
    elif year is not None:
        candidate = _default_exiobase_zip(year=year, system=system_norm, data_root=data_root)
        if candidate.exists():
            resolved_zip = candidate

    zip_names: tuple[str, ...] = ()
    version = ""
    source = f"pymrio packaged EXIOBASE3 {system_norm} sectors.tsv"
    if resolved_zip is not None:
        if not resolved_zip.exists():
            raise FileNotFoundError(f"Missing EXIOBASE source zip: {resolved_zip}")
        zip_names, version = _read_exio_zip_sector_names(resolved_zip)
        source = f"{source}; checked against {resolved_zip}"
        if validate_zip_names:
            packaged_names = set(sectors["ExioName"].astype(str).str.strip())
            zip_name_set = set(zip_names)
            missing_in_package = sorted(zip_name_set - packaged_names)
            missing_in_zip = sorted(packaged_names - zip_name_set)
            if missing_in_package or missing_in_zip:
                raise ValueError(
                    "EXIOBASE zip sector names differ from pymrio packaged metadata: "
                    f"missing_in_package={missing_in_package[:20]} missing_in_zip={missing_in_zip[:20]}"
                )

    return ExioMetadata(
        system=system_norm,
        sectors=sectors,
        source=source,
        source_zip=resolved_zip,
        version=version,
        zip_sector_names=zip_names,
    )


def cache_pymrio_exio3_sectors(
    *,
    out_path: str | Path,
    system: str = "pxp",
    force: bool = False,
) -> Path:
    """
    Write the packaged EXIOBASE 3 sector list to a stable local file for reproducibility.

    This does not require network access.
    """
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path

    concordance = load_pymrio_exio3_sectors(system=system)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concordance.sectors.to_csv(out_path, index=False)
    return out_path

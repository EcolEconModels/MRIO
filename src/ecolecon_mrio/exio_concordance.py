from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

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


def _normalize_system(system: str) -> str:
    s = str(system).strip().lower()
    if s in {"pxp", "product", "products", "productxproduct"}:
        return "pxp"
    if s in {"ixi", "industry", "industries", "industryxindustry"}:
        return "ixi"
    raise ValueError("system must be one of: pxp, ixi")


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


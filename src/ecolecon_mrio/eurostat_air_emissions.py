"""
Eurostat air emissions accounts downloader + helpers.

Primary use in this repo:
  - Download CO2 (and other air pollutants) by NACE Rev.2 sector for a country/year.
  - Cache-first, using the same Eurostat SDMX downloader as `eurostat_iot`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.eurostat_iot import (
    EurostatParseError,
    load_eurostat_series_1d,
)


DEFAULT_AIR_EMISSIONS_DATASET = "env_ac_ainah_r2"


def _as_path(p: str | Path | None) -> Path:
    if p is None:
        return default_data_root() / "eurostat"
    return Path(p)


def _convert_unit_to_tonnes(unit: str, values: pd.Series) -> pd.Series:
    u = str(unit).upper()
    factors = {
        "T": 1.0,
        "THS_T": 1_000.0,
        "MIO_T": 1_000_000.0,
        "KG": 0.001,
        "THS_KG": 1.0,
    }
    if u not in factors:
        return values
    return values.astype("float64") * float(factors[u])


@dataclass(frozen=True)
class AirEmissionsResult:
    emissions_tonnes: pd.Series
    meta: dict[str, Any]


def load_eurostat_air_emissions_by_nace(
    *,
    geo: str,
    time: int,
    airpol: str = "CO2",
    unit: str = "THS_T",
    freq: str = "A",
    dataset: str = DEFAULT_AIR_EMISSIONS_DATASET,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    force_reparse: bool = False,
) -> AirEmissionsResult:
    """
    Load air emissions accounts by NACE Rev.2 for a country/year.

    Returns
    -------
    AirEmissionsResult
        emissions_tonnes is indexed by `nace_r2` code.
    """
    filters: dict[str, Any] = {
        "geo": str(geo),
        "time": str(time),
        "airpol": str(airpol),
        "unit": str(unit),
        "freq": str(freq),
    }
    series, meta = load_eurostat_series_1d(
        dataset=dataset,
        filters=filters,
        dim="nace_r2",
        storage_folder=_as_path(storage_folder) if storage_folder is not None else None,
        force_download=force_download,
        force_reparse=force_reparse,
    )
    series_t = _convert_unit_to_tonnes(unit, series)
    meta_out = {"dataset": dataset, "filters": filters, "parsed": meta}
    return AirEmissionsResult(emissions_tonnes=series_t, meta=meta_out)

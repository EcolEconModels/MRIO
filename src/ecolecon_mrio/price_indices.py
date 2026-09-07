"""Official macro price-index acquisition for MRIO monetary sensitivities.

The initial provider is the United Nations Statistics Division (UNSD) Analysis of
Main Aggregates (AMA) API.  The module downloads and caches official GDP implicit
price-deflator observations, normalises them to a small long-form table, and
constructs multiplicative factors for expressing one year's monetary values at a
chosen base-year price level.

These are country-level macro price indices.  Applying them to an MRIO is a
sensitivity calculation, not a reconstruction of product-by-country constant-price
supply-use or input-output tables.  Product-relative price movements within a
country remain unresolved.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import urllib.request

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root


UNSD_AMA_API_BASE = "https://unstats.un.org/unsd/amaapi/api"


class PriceIndexError(RuntimeError):
    """Base exception for price-index acquisition or parsing failures."""


class PriceIndexHTTPError(PriceIndexError):
    """Raised when an official price-index source cannot be downloaded."""


class PriceIndexParseError(PriceIndexError):
    """Raised when downloaded price-index data violate the expected source schema."""


def default_price_index_storage_folder() -> Path:
    """Return the durable cache folder for small official price-index source data."""

    return default_data_root() / "price_indices" / "unsd_ama"


def _mapping_get(record: Mapping[str, Any], *names: str) -> Any:
    """Read one JSON field case-insensitively across UNSD schema casing variants."""

    by_name = {str(key).casefold(): value for key, value in record.items()}
    for name in names:
        key = str(name).casefold()
        if key in by_name:
            return by_name[key]
    return None


def _normalise_json_records(payload: Any, *, context: str) -> list[dict[str, Any]]:
    """Return a list of JSON objects or fail instead of guessing at malformed data."""

    if not isinstance(payload, list):
        raise PriceIndexParseError(f"{context} must be a JSON list, got {type(payload).__name__}.")
    out: list[dict[str, Any]] = []
    for pos, item in enumerate(payload):
        if not isinstance(item, dict):
            raise PriceIndexParseError(f"{context}[{pos}] must be a JSON object.")
        out.append(dict(item))
    return out


def _http_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout_s: int = 120,
) -> Any:
    """Fetch one UNSD AMA JSON endpoint using only the Python standard library."""

    data: bytes | None = None
    headers = {"Accept": "application/json", "User-Agent": "ecolecon-mrio/price-indices"}
    if payload is not None:
        data = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=int(timeout_s)) as response:
            raw = response.read()
    except Exception as exc:  # pragma: no cover - live network failure path
        raise PriceIndexHTTPError(f"UNSD AMA request failed for {url}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PriceIndexParseError(f"UNSD AMA returned invalid JSON for {url}: {exc}") from exc


def _normalise_years(years: Sequence[int]) -> tuple[int, ...]:
    """Return deterministic unique years and reject empty or implausible requests."""

    values = tuple(sorted({int(year) for year in years}))
    if not values:
        raise ValueError("years must contain at least one year")
    if values[0] < 1900 or values[-1] > 2200:
        raise ValueError(f"years outside the supported validation range: {values[0]}..{values[-1]}")
    return values


def _normalise_country_codes(country_codes: Sequence[int] | None) -> tuple[int, ...] | None:
    """Return deterministic unique M49 country codes, preserving None as 'all'."""

    if country_codes is None:
        return None
    values = tuple(sorted({int(code) for code in country_codes}))
    if not values:
        raise ValueError("country_codes must contain at least one M49 code when provided")
    if any(code <= 0 for code in values):
        raise ValueError(f"country_codes must be positive M49 integers, got {values!r}")
    return values


def _request_cache_path(
    *,
    storage_folder: Path,
    years: tuple[int, ...],
    country_codes: tuple[int, ...] | None,
) -> Path:
    """Build a deterministic cache path from the scientific request, not API response order."""

    request_key = {
        "provider": "unsd_ama",
        "indicator": "gdp_implicit_price_deflator",
        "years": list(years),
        "country_codes": "all" if country_codes is None else list(country_codes),
    }
    digest = hashlib.sha256(json.dumps(request_key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return storage_folder / f"gdp_implicit_price_deflators_{years[0]}_{years[-1]}_{digest}.json"


def _series_code(record: Mapping[str, Any]) -> int:
    """Return one integer UNSD series code from a metadata row."""

    value = _mapping_get(record, "serieCode", "seriesCode")
    if value is None:
        raise PriceIndexParseError(f"UNSD series metadata is missing serieCode: {record!r}")
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise PriceIndexParseError(f"Invalid UNSD serieCode {value!r}") from exc


def _series_name(record: Mapping[str, Any]) -> str:
    """Return one UNSD series name, preserving the source wording for provenance."""

    value = _mapping_get(record, "serieName", "seriesName")
    return "" if value is None else str(value).strip()


def _select_deflator_series(series_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select official series whose metadata names GDP implicit-price deflators."""

    selected = [dict(record) for record in series_records if "deflator" in _series_name(record).casefold()]
    if selected:
        return selected
    names = sorted({_series_name(record) for record in series_records if _series_name(record)})
    sample = names[:25]
    raise PriceIndexParseError(
        "UNSD AMA series metadata contains no series name with 'deflator'; "
        f"available sample={sample!r}. The provider schema may have changed."
    )


def _is_deflator_record(record: Mapping[str, Any]) -> bool:
    """Identify a GDP implicit-price-deflator observation inside a selected series."""

    item_name = str(_mapping_get(record, "itemName") or "").strip()
    series_name = str(_mapping_get(record, "serieName", "seriesName") or "").strip()
    # When item names are present, they are more specific than the encompassing
    # series title and prevent other GDP rows from being mistaken for deflators.
    if item_name:
        return "deflator" in item_name.casefold()
    return "deflator" in series_name.casefold()


def download_unsd_ama_gdp_deflators(
    *,
    years: Sequence[int],
    country_codes: Sequence[int] | None = None,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> Path:
    """Download and cache official UNSD AMA GDP implicit-price-deflator records.

    Parameters
    ----------
    years:
        Calendar/fiscal years requested from the official API.
    country_codes:
        Optional UN M49 country codes. ``None`` requests every row marked as a
        country by the AMA country endpoint.
    storage_folder:
        Cache folder. Defaults to ``MRIO_DATA_ROOT/price_indices/unsd_ama``.
    force_download:
        Replace an existing request cache when true.
    timeout_s:
        HTTP timeout for each official API request.

    Returns
    -------
    pathlib.Path
        Path to one self-describing cached JSON payload containing source metadata,
        country metadata, selected series metadata, and raw deflator observations.
    """

    years_norm = _normalise_years(years)
    codes_norm = _normalise_country_codes(country_codes)
    root = default_price_index_storage_folder() if storage_folder is None else Path(storage_folder)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = _request_cache_path(storage_folder=root, years=years_norm, country_codes=codes_norm)
    if raw_path.exists() and not force_download:
        return raw_path

    # Country metadata supplies the authoritative M49 domain and source currency
    # names used later to classify national-currency versus US-dollar index rows.
    countries_payload = _http_json(
        f"{UNSD_AMA_API_BASE}/Country?countriesOnly=true",
        timeout_s=timeout_s,
    )
    countries = _normalise_json_records(countries_payload, context="UNSD countries")
    country_by_code: dict[int, dict[str, Any]] = {}
    for record in countries:
        code_value = _mapping_get(record, "countryCode")
        if code_value is None:
            raise PriceIndexParseError(f"UNSD country metadata is missing countryCode: {record!r}")
        code = int(code_value)
        is_country = _mapping_get(record, "isCountry")
        if is_country is False:
            continue
        country_by_code[code] = record

    requested_codes = tuple(sorted(country_by_code)) if codes_norm is None else codes_norm
    missing_codes = [code for code in requested_codes if code not in country_by_code]
    if missing_codes:
        raise PriceIndexParseError(f"Requested M49 codes are absent from the UNSD country endpoint: {missing_codes}")

    # Series ids are discovered from source metadata every download so the code
    # does not silently depend on a mutable spreadsheet/file id or undocumented id.
    series_payload = _http_json(f"{UNSD_AMA_API_BASE}/Series", timeout_s=timeout_s)
    series_records = _normalise_json_records(series_payload, context="UNSD series")
    selected_series = _select_deflator_series(series_records)

    deflator_records: list[dict[str, Any]] = []
    for series in selected_series:
        code = _series_code(series)
        payload = _http_json(
            f"{UNSD_AMA_API_BASE}/Data/basic/{code}",
            payload={"paramCodes": list(requested_codes), "years": list(years_norm)},
            timeout_s=timeout_s,
        )
        records = _normalise_json_records(payload, context=f"UNSD data series {code}")
        deflator_records.extend(record for record in records if _is_deflator_record(record))

    if not deflator_records:
        names = [_series_name(record) for record in selected_series]
        raise PriceIndexParseError(
            "UNSD deflator series returned no item rows containing 'deflator'; "
            f"selected_series={names!r}. The provider schema may have changed."
        )

    # The source's release date is provenance rather than a numerical input, but
    # persisting it makes later paper runs auditable when UNSD revises history.
    last_updated = _http_json(f"{UNSD_AMA_API_BASE}/Data/lastupdated", timeout_s=timeout_s)
    payload_out = {
        "provider": "UNSD AMA",
        "api_base": UNSD_AMA_API_BASE,
        "indicator": "GDP implicit price deflator",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_last_updated": last_updated,
        "request": {
            "years": list(years_norm),
            "country_codes": "all" if codes_norm is None else list(codes_norm),
        },
        "countries": [country_by_code[code] for code in requested_codes],
        "series": selected_series,
        "records": deflator_records,
    }
    temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(raw_path)
    return raw_path


def _currency_basis_from_text(*, series_name: str, item_name: str, unit: str, currency_name: str) -> str | None:
    """Classify a deflator row from explicit source text, without positional guessing."""

    text = " ".join((series_name, item_name, unit)).casefold()
    # Explicit basis wording is authoritative.  This must precede the currency-name
    # heuristic because a country's national currency can itself be the US dollar.
    if "national curr" in text:
        return "national_currency"
    if "us dollar" in text or "u.s. dollar" in text or " usd" in f" {text}":
        return "usd"
    currency_norm = currency_name.strip().casefold()
    if currency_norm and currency_norm in text:
        return "national_currency"
    return None


def read_unsd_ama_gdp_deflators(raw_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one cached UNSD AMA payload into validated long-form price indices.

    The returned table has one row per ``(m49, year, currency_basis)`` with
    ``currency_basis`` equal to ``national_currency`` or ``usd``.  The source
    wording and ids remain in the table so later diagnostics can trace every
    value back to the official AMA response.
    """

    path = Path(raw_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriceIndexParseError(f"Cannot read cached UNSD price-index JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PriceIndexParseError(f"Cached UNSD price-index payload must be a JSON object: {path}")

    countries = _normalise_json_records(payload.get("countries"), context="cached UNSD countries")
    records = _normalise_json_records(payload.get("records"), context="cached UNSD records")
    currency_by_code: dict[int, str] = {}
    name_by_code: dict[int, str] = {}
    for record in countries:
        code_value = _mapping_get(record, "countryCode")
        if code_value is None:
            raise PriceIndexParseError(f"Cached UNSD country row lacks countryCode: {record!r}")
        code = int(code_value)
        currency_by_code[code] = str(_mapping_get(record, "currencyName") or "").strip()
        name_by_code[code] = str(_mapping_get(record, "countryName") or "").strip()

    parsed: list[dict[str, Any]] = []
    explicit_item_basis: dict[tuple[int, int], str] = {}
    for record in records:
        code_value = _mapping_get(record, "countryCode")
        year_value = _mapping_get(record, "fiscalYear", "year")
        observation_value = _mapping_get(record, "observationValue", "value")
        if code_value is None or year_value is None or observation_value is None:
            raise PriceIndexParseError(f"UNSD deflator record lacks country/year/value fields: {record!r}")
        m49 = int(code_value)
        year = int(year_value)
        try:
            price_index = float(observation_value)
        except (TypeError, ValueError) as exc:
            raise PriceIndexParseError(f"Non-numeric UNSD deflator value {observation_value!r}") from exc
        if not math.isfinite(price_index) or price_index <= 0.0:
            raise PriceIndexParseError(
                f"UNSD deflator must be positive and finite, got {price_index!r} for m49={m49}, year={year}"
            )

        series_id = int(float(_mapping_get(record, "serieCode", "seriesCode") or -1))
        item_id = int(float(_mapping_get(record, "itemId") or -1))
        series_name = str(_mapping_get(record, "serieName", "seriesName") or "").strip()
        item_name = str(_mapping_get(record, "itemName") or "").strip()
        unit = str(_mapping_get(record, "unit") or "").strip()
        country_name = str(_mapping_get(record, "countryName") or name_by_code.get(m49, "")).strip()
        basis = _currency_basis_from_text(
            series_name=series_name,
            item_name=item_name,
            unit=unit,
            currency_name=currency_by_code.get(m49, ""),
        )
        if basis is not None and series_id >= 0 and item_id >= 0:
            key = (series_id, item_id)
            previous = explicit_item_basis.get(key)
            if previous is not None and previous != basis:
                raise PriceIndexParseError(f"UNSD item {key} maps to conflicting currency bases: {previous}, {basis}")
            explicit_item_basis[key] = basis
        parsed.append(
            {
                "m49": m49,
                "country_name": country_name,
                "year": year,
                "currency_basis": basis,
                "price_index": price_index,
                "source_series_id": series_id,
                "source_series_name": series_name,
                "source_item_id": item_id,
                "source_item_name": item_name,
                "source_unit": unit,
                "observation_note": str(_mapping_get(record, "observationNote") or "").strip(),
            }
        )

    # Some AMA payloads may omit basis wording from individual rows.  Reuse a
    # basis learned from the same official series/item id elsewhere before using
    # the final within-country/year two-row complement check below.
    for row in parsed:
        if row["currency_basis"] is None:
            key = (int(row["source_series_id"]), int(row["source_item_id"]))
            row["currency_basis"] = explicit_item_basis.get(key)

    by_country_year: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in parsed:
        by_country_year.setdefault((int(row["m49"]), int(row["year"])), []).append(row)
    for key, rows in by_country_year.items():
        unknown = [row for row in rows if row["currency_basis"] is None]
        known = {str(row["currency_basis"]) for row in rows if row["currency_basis"] is not None}
        # The AMA country profile publishes national-currency and US-dollar GDP
        # deflators side by side.  If exactly one of that pair is explicitly named,
        # the remaining single row is identified as its complement.
        if len(unknown) == 1 and len(rows) == 2 and known == {"usd"}:
            unknown[0]["currency_basis"] = "national_currency"
        elif len(unknown) == 1 and len(rows) == 2 and known == {"national_currency"}:
            unknown[0]["currency_basis"] = "usd"
        elif unknown:
            detail = [(row["source_series_name"], row["source_item_name"], row["source_unit"]) for row in unknown]
            raise PriceIndexParseError(
                f"Cannot classify UNSD deflator currency basis for m49/year={key}: {detail!r}. "
                "No positional fallback is allowed."
            )

    columns = [
        "m49",
        "country_name",
        "year",
        "currency_basis",
        "price_index",
        "source_series_id",
        "source_series_name",
        "source_item_id",
        "source_item_name",
        "source_unit",
        "observation_note",
    ]
    frame = pd.DataFrame(parsed, columns=pd.Index(columns))
    if frame.empty:
        raise PriceIndexParseError(f"Cached UNSD deflator payload contains no observations: {path}")
    allowed_bases = {"national_currency", "usd"}
    observed_bases = set(frame["currency_basis"].astype(str))
    if not observed_bases.issubset(allowed_bases):
        raise PriceIndexParseError(f"Unexpected UNSD currency bases after parsing: {sorted(observed_bases)!r}")
    duplicated = frame.duplicated(subset=["m49", "year", "currency_basis"], keep=False)
    if bool(duplicated.any()):
        sample = frame.loc[duplicated, ["m49", "year", "currency_basis", "source_item_name"]].head(10)
        raise PriceIndexParseError(f"Duplicate UNSD deflator observations:\n{sample.to_string(index=False)}")
    frame = frame.sort_values(["m49", "year", "currency_basis"], kind="stable").reset_index(drop=True)

    metadata = {
        "provider": payload.get("provider", "UNSD AMA"),
        "api_base": payload.get("api_base", UNSD_AMA_API_BASE),
        "indicator": payload.get("indicator", "GDP implicit price deflator"),
        "retrieved_at_utc": payload.get("retrieved_at_utc"),
        "data_last_updated": payload.get("data_last_updated"),
        "raw_path": str(path),
        "currency_bases": sorted(observed_bases),
        "row_count": int(len(frame)),
    }
    return frame, metadata


def load_unsd_ama_gdp_deflators(
    *,
    years: Sequence[int],
    country_codes: Sequence[int] | None = None,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download if necessary, then return validated UNSD GDP deflator observations."""

    raw_path = download_unsd_ama_gdp_deflators(
        years=years,
        country_codes=country_codes,
        storage_folder=storage_folder,
        force_download=force_download,
        timeout_s=timeout_s,
    )
    return read_unsd_ama_gdp_deflators(raw_path)


def price_adjustment_factors(
    price_indices: pd.DataFrame,
    *,
    year: int,
    base_year: int,
    currency_basis: str,
    area_codes: Sequence[int] | None = None,
    area_column: str = "m49",
) -> pd.Series:
    """Return factors ``P_base / P_year`` for one currency basis and area domain.

    Multiplying a current-price monetary value from ``year`` by the returned
    factor expresses it at the selected base-year macro price level under this
    country-level deflator approximation.  If ``area_codes`` is supplied, every
    requested area must have both year and base-year observations; otherwise the
    function returns the intersection available in both years.
    """

    required = {area_column, "year", "currency_basis", "price_index"}
    missing_columns = sorted(required.difference(price_indices.columns))
    if missing_columns:
        raise ValueError(f"price_indices is missing required columns: {missing_columns}")
    basis = str(currency_basis).strip().lower()
    if basis not in {"national_currency", "usd"}:
        raise ValueError("currency_basis must be 'national_currency' or 'usd'")

    subset = price_indices.loc[price_indices["currency_basis"].astype(str) == basis, [area_column, "year", "price_index"]].copy()
    duplicated = subset.duplicated(subset=[area_column, "year"], keep=False)
    if bool(duplicated.any()):
        raise ValueError(f"Duplicate price-index rows for area/year in currency_basis={basis!r}")
    year_values = subset.loc[subset["year"].astype(int) == int(year)].set_index(area_column)["price_index"].astype(float)
    base_values = subset.loc[subset["year"].astype(int) == int(base_year)].set_index(area_column)["price_index"].astype(float)

    if area_codes is None:
        areas = year_values.index.intersection(base_values.index)
    else:
        areas = pd.Index([int(code) for code in area_codes], name=area_column).drop_duplicates()
        missing_year = [code for code in areas if code not in year_values.index]
        missing_base = [code for code in areas if code not in base_values.index]
        if missing_year or missing_base:
            raise ValueError(
                f"Missing price indices for requested areas; year={int(year)} missing={missing_year}, "
                f"base_year={int(base_year)} missing={missing_base}"
            )
    if len(areas) == 0:
        raise ValueError(f"No areas have both year={int(year)} and base_year={int(base_year)} price indices")

    factors = (base_values.reindex(areas) / year_values.reindex(areas)).astype(float)
    invalid = ~factors.map(lambda value: math.isfinite(float(value)) and float(value) > 0.0)
    if bool(invalid.any()):
        raise ValueError(f"Non-positive or non-finite price adjustment factors for areas {list(factors.index[invalid])!r}")
    factors.name = "price_adjustment_factor"
    factors.index.name = area_column
    return factors

# ecolecon-exiobase-eur-price-factors
UNSD_AMA_DOWNLOADS_URL = "https://unstats.un.org/unsd/snaama/downloads"


def _gdp_usd_level_basis(record: Mapping[str, Any]) -> str | None:
    """Classify aggregate GDP-level observations in USD as current or constant prices."""

    item_name = str(_mapping_get(record, "itemName") or "").strip()
    unit = str(_mapping_get(record, "unit") or "").strip()
    text = f"{item_name} {unit}".casefold()
    gdp_named = "gross domestic product" in text or "gdp" in text
    usd_named = "us dollar" in text or "u.s. dollar" in text or " usd" in f" {text}"
    excluded = any(token in text for token in ("per capita", "deflator", "growth rate", "index", "gni"))
    if not (gdp_named and usd_named and not excluded):
        return None
    if "constant" in text:
        return "constant"
    if "current" in text:
        return "current"
    return None


def _gdp_usd_levels_cache_path(
    *, storage_folder: Path, years: tuple[int, ...], country_codes: tuple[int, ...] | None
) -> Path:
    """Build one deterministic cache key for paired current/constant USD GDP levels."""

    request_key = {
        "provider": "unsd_ama",
        "indicator": "gdp_current_and_constant_usd_levels",
        "schema": 1,
        "years": list(years),
        "country_codes": "all" if country_codes is None else list(country_codes),
    }
    digest = hashlib.sha256(json.dumps(request_key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return storage_folder / f"gdp_usd_levels_{years[0]}_{years[-1]}_{digest}.json"


def download_unsd_ama_gdp_usd_levels(
    *,
    years: Sequence[int],
    country_codes: Sequence[int] | None = None,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> Path:
    """Download/cache aggregate GDP at current and constant prices in US dollars.

    Both level families are selected from the same dynamically discovered broad
    UNSD AMA GDP series used by the deflator API. The provider's constant-price
    reference is common across countries in one release (currently 2020); the
    runtime schema deliberately calls the column ``constant_usd_gdp`` because the
    aggregate deflator ratio is invariant to the common fixed-price reference year.
    """

    years_norm = _normalise_years(years)
    codes_norm = _normalise_country_codes(country_codes)
    root = default_price_index_storage_folder() if storage_folder is None else Path(storage_folder)
    root.mkdir(parents=True, exist_ok=True)
    path = _gdp_usd_levels_cache_path(storage_folder=root, years=years_norm, country_codes=codes_norm)
    if path.exists() and not force_download:
        return path

    # Reuse the existing authoritative country/series discovery rather than owning
    # a second mutable list of provider series ids.
    deflator_path = download_unsd_ama_gdp_deflators(
        years=years_norm,
        country_codes=codes_norm,
        storage_folder=root,
        force_download=force_download,
        timeout_s=timeout_s,
    )
    try:
        deflator_payload = json.loads(deflator_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriceIndexParseError(f"Cannot read supporting UNSD deflator cache {deflator_path}: {exc}") from exc
    if not isinstance(deflator_payload, dict):
        raise PriceIndexParseError("Supporting UNSD deflator cache must be a JSON object")
    countries = _normalise_json_records(deflator_payload.get("countries"), context="cached UNSD countries")
    selected_series = _normalise_json_records(deflator_payload.get("series"), context="cached UNSD series")
    requested_codes = tuple(sorted(int(_mapping_get(record, "countryCode")) for record in countries))
    if not requested_codes:
        raise PriceIndexParseError("Supporting UNSD deflator cache contains no countries")

    level_records: list[dict[str, Any]] = []
    observed_bases: set[str] = set()
    for series in selected_series:
        series_code = _series_code(series)
        response = _http_json(
            f"{UNSD_AMA_API_BASE}/Data/basic/{series_code}",
            payload={"paramCodes": list(requested_codes), "years": list(years_norm)},
            timeout_s=timeout_s,
        )
        records = _normalise_json_records(response, context=f"UNSD GDP-level series {series_code}")
        for record in records:
            basis = _gdp_usd_level_basis(record)
            if basis is not None:
                level_records.append(record)
                observed_bases.add(basis)
    if observed_bases != {"current", "constant"}:
        raise PriceIndexParseError(
            "UNSD broad GDP series did not expose both current- and constant-price USD GDP levels; "
            f"observed={sorted(observed_bases)!r}"
        )

    fiscal_year_notes = _http_json(f"{UNSD_AMA_API_BASE}/Metadata/fiscalyearnotes", timeout_s=timeout_s)
    payload_out = {
        "provider": "UNSD AMA",
        "api_base": UNSD_AMA_API_BASE,
        "indicator": "GDP at current and constant prices, US dollars",
        "downloads_page": UNSD_AMA_DOWNLOADS_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": {
            "years": list(years_norm),
            "country_codes": "all" if codes_norm is None else list(codes_norm),
        },
        "countries": countries,
        "series": selected_series,
        "fiscal_year_notes": fiscal_year_notes,
        "records": level_records,
        "source_deflator_cache": str(deflator_path),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def read_unsd_ama_gdp_usd_levels(raw_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one paired UNSD GDP-level cache without inventing missing observations."""

    path = Path(raw_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriceIndexParseError(f"Cannot read cached UNSD GDP-level JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PriceIndexParseError(f"Cached UNSD GDP-level payload must be a JSON object: {path}")
    records = _normalise_json_records(payload.get("records"), context="cached UNSD GDP-level records")

    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        basis = _gdp_usd_level_basis(record)
        if basis is None:
            raise PriceIndexParseError(f"Cached GDP-level row is no longer classifiable: {record!r}")
        code_raw = _mapping_get(record, "countryCode")
        year_raw = _mapping_get(record, "fiscalYear", "year")
        value_raw = _mapping_get(record, "observationValue", "value")
        if code_raw is None or year_raw is None or value_raw is None:
            raise PriceIndexParseError(f"UNSD GDP-level row lacks country/year/value: {record!r}")
        m49 = int(code_raw)
        year = int(year_raw)
        try:
            value = float(value_raw)
        except (TypeError, ValueError) as exc:
            raise PriceIndexParseError(f"UNSD GDP-level value is non-numeric: {value_raw!r}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise PriceIndexParseError(
                f"UNSD GDP-level value must be positive and finite, got {value!r} for m49={m49}, year={year}, basis={basis}"
            )
        key = (m49, year)
        row = by_key.setdefault(
            key,
            {
                "m49": m49,
                "country_name": str(_mapping_get(record, "countryName") or "").strip(),
                "year": year,
                "current_usd_gdp": np.nan,
                "constant_usd_gdp": np.nan,
                "current_source_item_name": "",
                "constant_source_item_name": "",
                "current_source_unit": "",
                "constant_source_unit": "",
            },
        )
        column = f"{basis}_usd_gdp"
        if math.isfinite(float(row[column])):
            raise PriceIndexParseError(f"Duplicate UNSD GDP-level observation for m49={m49}, year={year}, basis={basis}")
        row[column] = value
        row[f"{basis}_source_item_name"] = str(_mapping_get(record, "itemName") or "").strip()
        row[f"{basis}_source_unit"] = str(_mapping_get(record, "unit") or "").strip()

    frame = pd.DataFrame(by_key.values())
    if frame.empty:
        raise PriceIndexParseError(f"Cached UNSD GDP-level payload contains no observations: {path}")
    if frame.duplicated(subset=["m49", "year"]).any():
        raise PriceIndexParseError("Parsed UNSD GDP levels duplicate an M49/year row")
    frame = frame.sort_values(["m49", "year"], kind="stable").reset_index(drop=True)
    metadata = {
        "provider": payload.get("provider", "UNSD AMA"),
        "api_base": payload.get("api_base", UNSD_AMA_API_BASE),
        "indicator": payload.get("indicator", "GDP at current and constant prices, US dollars"),
        "downloads_page": payload.get("downloads_page", UNSD_AMA_DOWNLOADS_URL),
        "retrieved_at_utc": payload.get("retrieved_at_utc"),
        "fiscal_year_notes": payload.get("fiscal_year_notes"),
        "raw_path": str(path),
        "row_count": int(len(frame)),
        "source_deflator_cache": payload.get("source_deflator_cache"),
    }
    return frame, metadata


def load_unsd_ama_gdp_usd_levels(
    *,
    years: Sequence[int],
    country_codes: Sequence[int] | None = None,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download if necessary, then return paired UNSD current/constant USD GDP levels."""

    raw_path = download_unsd_ama_gdp_usd_levels(
        years=years,
        country_codes=country_codes,
        storage_folder=storage_folder,
        force_download=force_download,
        timeout_s=timeout_s,
    )
    return read_unsd_ama_gdp_usd_levels(raw_path)

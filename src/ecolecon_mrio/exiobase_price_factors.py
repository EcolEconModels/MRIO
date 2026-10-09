"""Build complete EXIOBASE3 macro price-adjustment factors in a common EUR basis.

The published Bjelle et al. EXIOBASE3rx 214-to-49 bridge owns Rest-of-World
membership. Ordinary countries and RoW aggregates use paired UNSD AMA current and
constant USD GDP levels; RoW levels are summed before the implicit deflator is
formed. The bridge's 44 explicit targets are ISO3 codes and are translated to the
runtime two-letter EXIOBASE3 region codes through a pinned 00-class crosswalk.
Taiwan keeps the separately audited DGBAS/CBC route selected by mapping policy.

The result remains a country-level macro sensitivity. It does not reconstruct
product-by-country constant-price EXIOBASE tables and cannot remove within-country
relative-price movements.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from importlib import resources
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, cast
import urllib.request

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.mapping_csv import read_mapping_csv
from ecolecon_mrio.price_indices import (
    PriceIndexHTTPError,
    PriceIndexParseError,
    load_unsd_ama_gdp_usd_levels,
)

ECB_USD_EUR_SERIES = "EXR.A.USD.EUR.SP00.A"
ECB_USD_EUR_URL = (
    "https://data-api.ecb.europa.eu/service/data/EXR/A.USD.EUR.SP00.A"
    "?format=csvdata"
)
UNSD_M49_URL = "https://unstats.un.org/unsd/methodology/m49/overview/"
EXIOBASE_META_COMMIT = "768ff12b3c1e449e89d4310a44db9c5dc2266652"
EXIOBASE_COUNTRY_COVERAGE_URL = (
    "https://raw.githubusercontent.com/EXIOBASE/00-class/"
    f"{EXIOBASE_META_COMMIT}/class/country_axes/country_coverage_matrix.csv"
)
EXIOBASE3RX_BRIDGE_SHA256 = "92e9b95d240b338c0a6a8c06f2c4768149d81092839ef4581d780a1ce119f972"
EXIOBASE3RX_BRIDGE_DOI = "10.1186/s40008-020-0182-y"
EXIOBASE3RX_BRIDGE_SUPPLEMENT = "40008_2020_182_MOESM2_ESM.xlsx"
EXIOBASE3RX_BRIDGE_LICENSE = "CC BY 4.0"
EXIOBASE3RX_BRIDGE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DGBAS_TABLE_ID = "A018102050"
DGBAS_METADATA_URL = f"https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx?sdmx/{DGBAS_TABLE_ID}/all"
DGBAS_SNAPSHOT_RESOURCE = "reference_data/taiwan_dgbas_gdp_deflator.json"
PRICE_FACTOR_SOURCE_POLICY_RESOURCE = "mappings/exiobase_price_factor_source_policy.csv"
PRICE_FACTOR_SOURCE_IDENTITY_RESOURCE = "mappings/exiobase_price_factor_source_identity.csv"
EXIOBASE3RX_BRIDGE_RESOURCE = "mappings/exiobase3rx_to_exiobase3_regions.csv"
PRICE_FACTOR_METHOD_UNSD = "unsd_ama_usd_levels_common_eur"
PRICE_FACTOR_METHOD_TAIWAN = "taiwan_dgbas_cbc_common_eur"
PRICE_FACTOR_SOURCE_POLICY_COLUMNS = (
    "kind", "source_code", "source_label", "target_code", "target_label",
    "start_year", "end_year", "enabled", "provenance", "certainty",
    "official_source", "notes",
)
PRICE_FACTOR_SOURCE_IDENTITY_COLUMNS = (
    "kind", "source_code", "source_label", "target_code", "target_label",
    "start_year", "end_year", "provider_system", "provider_code",
    "resolution_status", "enabled", "provenance", "certainty",
    "official_source", "notes",
)
EXIOBASE3RX_BRIDGE_COLUMNS = (
    "source_country", "source_iso3", "target_exio_region", "weight",
)
CBC_TWD_USD_ANNUAL_URL = "https://www.cbc.gov.tw/en/cp-480-1879-66035-2.html"
EXPECTED_ROW_REGIONS = frozenset({"WA", "WL", "WE", "WF", "WM"})
EXPECTED_EXIOBASE3_REGION_COUNT = 49


def default_exiobase3_price_factor_storage_folder() -> Path:
    """Return the durable cache root for small common-EUR factor source files."""

    return default_data_root() / "price_indices" / "exiobase3_eur"


def _http_text(url: str, *, timeout_s: int = 120) -> str:
    """Fetch one small authoritative text source and decode it fail-closed."""

    request = urllib.request.Request(
        url,
        headers={"Accept": "text/html,text/csv,application/xml,text/xml,*/*", "User-Agent": "ecolecon-mrio/price-indices"},
    )
    try:
        with urllib.request.urlopen(request, timeout=int(timeout_s)) as response:
            raw = response.read()
            content_type = str(response.headers.get("Content-Type", ""))
    except Exception as exc:  # pragma: no cover - live network failure path
        raise PriceIndexHTTPError(f"Price-factor source request failed for {url}: {exc}") from exc
    charset_match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
    candidates = [charset_match.group(1).strip('"\'') if charset_match else "", "utf-8-sig", "utf-8", "big5", "cp950"]
    for encoding in candidates:
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    raise PriceIndexParseError(f"Could not decode price-factor source response from {url}")


def _cached_text(
    url: str,
    *,
    path: Path,
    force_download: bool,
    timeout_s: int,
) -> Path:
    """Cache one small text source atomically; existing cache is reused by default."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_download:
        return path
    text = _http_text(url, timeout_s=timeout_s)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text.rstrip() + "\n", encoding="utf-8")
    temp.replace(path)
    return path


class _TableAndLinkParser(HTMLParser):
    """Collect simple HTML table cells and links without third-party HTML parsers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self.links: list[str] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.casefold()
        if tag_lower == "a":
            for key, value in attrs:
                if key.casefold() == "href" and value:
                    self.links.append(value)
        elif tag_lower == "table":
            self._table = []
        elif tag_lower == "tr" and self._table is not None:
            self._row = []
        elif tag_lower in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.casefold()
        if tag_lower in {"td", "th"} and self._cell_parts is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell_parts).split()))
            self._cell_parts = None
        elif tag_lower == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag_lower == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def _parse_html(text: str) -> _TableAndLinkParser:
    parser = _TableAndLinkParser()
    parser.feed(text)
    parser.close()
    return parser


def _positive_series(values: Mapping[int, float], *, name: str) -> pd.Series:
    """Return one unique positive annual numeric series."""

    series = pd.Series(dict(sorted(values.items())), dtype=float, name=name)
    series.index = pd.Index(series.index.astype(int), name="year")
    if series.empty or not np.isfinite(series.to_numpy(dtype=float)).all() or bool((series <= 0.0).any()):
        raise PriceIndexParseError(f"{name} must contain positive finite annual values")
    return series


def _read_ecb_usd_per_eur(path: Path) -> pd.Series:
    """Parse the exact ECB annual USD-per-EUR average reference-rate series."""

    frame = pd.read_csv(path)
    required = {"TIME_PERIOD", "OBS_VALUE"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PriceIndexParseError(f"ECB CSV missing columns: {missing}")
    if "KEY" in frame.columns:
        keys = set(frame["KEY"].dropna().astype(str))
        if keys and keys != {ECB_USD_EUR_SERIES}:
            raise PriceIndexParseError(f"ECB CSV contains unexpected series keys: {sorted(keys)!r}")
    values: dict[int, float] = {}
    for _, row in frame.iterrows():
        token = str(row["TIME_PERIOD"]).strip()
        if not re.fullmatch(r"\d{4}", token):
            continue
        year = int(token)
        value = float(cast(float | str, row["OBS_VALUE"]))
        if year in values:
            raise PriceIndexParseError(f"ECB annual USD/EUR series duplicates year {year}")
        values[year] = value
    return _positive_series(values, name="usd_per_eur")


def _read_unsd_m49_iso3(path: Path) -> pd.DataFrame:
    """Parse ISO-alpha3 to M49 from the official UN M49 overview table."""

    parser = _parse_html(path.read_text(encoding="utf-8"))
    pairs: dict[str, int] = {}
    for table in parser.tables:
        header_pos = next(
            (
                pos
                for pos, row in enumerate(table)
                if "M49 Code" in row and "ISO-alpha3 Code" in row
            ),
            None,
        )
        if header_pos is None:
            continue
        header = table[header_pos]
        m49_col = header.index("M49 Code")
        iso_col = header.index("ISO-alpha3 Code")
        for row in table[header_pos + 1 :]:
            if max(m49_col, iso_col) >= len(row):
                continue
            m49_token = row[m49_col].strip()
            iso3 = row[iso_col].strip().upper()
            if not re.fullmatch(r"\d{3}", m49_token) or not re.fullmatch(r"[A-Z]{3}", iso3):
                continue
            m49 = int(m49_token)
            previous = pairs.get(iso3)
            if previous is not None and previous != m49:
                raise PriceIndexParseError(f"UN M49 table maps {iso3} to conflicting codes {previous} and {m49}")
            pairs[iso3] = m49
    if len(pairs) < 150:
        raise PriceIndexParseError(f"UN M49 parser found only {len(pairs)} ISO3 mappings")
    return pd.DataFrame({"iso3": list(pairs), "m49": list(pairs.values())}).sort_values("iso3").reset_index(drop=True)

def _mapping_enabled_column(frame: pd.DataFrame, *, mapping_name: str) -> pd.DataFrame:
    """Normalize an explicit mapping ``enabled`` column without permissive blanks."""

    result = frame.copy()
    tokens = result["enabled"].astype(str).str.strip().str.casefold()
    invalid = sorted(set(tokens).difference({"true", "false"}))
    if invalid:
        raise PriceIndexParseError(f"{mapping_name} has invalid enabled values: {invalid}")
    result["enabled"] = tokens.eq("true")
    return result


def _mapping_year_columns(frame: pd.DataFrame, *, mapping_name: str) -> pd.DataFrame:
    """Parse inclusive mapping validity intervals and reject reversed ranges."""

    result = frame.copy()
    try:
        start_years = cast(pd.Series, pd.to_numeric(result["start_year"], errors="raise"))
        end_years = cast(pd.Series, pd.to_numeric(result["end_year"], errors="raise"))
        result["start_year"] = start_years.astype(int)
        result["end_year"] = end_years.astype(int)
    except (TypeError, ValueError) as exc:
        raise PriceIndexParseError(f"{mapping_name} has non-integer validity years") from exc
    if bool((result["start_year"] > result["end_year"]).any()):
        raise PriceIndexParseError(f"{mapping_name} contains start_year > end_year")
    return result


def _reject_overlapping_mapping_intervals(
    frame: pd.DataFrame, *, source_column: str, mapping_name: str
) -> None:
    """Require at most one enabled rule for one source key in any year."""

    enabled = frame.loc[frame["enabled"].astype(bool)]
    for source_code, group in enabled.groupby(source_column, sort=False):
        ordered = group.sort_values(["start_year", "end_year"], kind="stable")
        previous_end: int | None = None
        starts = ordered["start_year"].astype(int).tolist()
        ends = ordered["end_year"].astype(int).tolist()
        for start_year, end_year in zip(starts, ends, strict=True):
            if previous_end is not None and start_year <= previous_end:
                raise PriceIndexParseError(
                    f"{mapping_name} has overlapping enabled intervals for {source_code!r}"
                )
            previous_end = end_year


def _read_price_factor_source_policy(path: Path) -> pd.DataFrame:
    """Read the data-owned explicit-country price-factor source/method policy."""

    frame = read_mapping_csv(
        path,
        expected_kind="price-factor-policy",
        required_columns=PRICE_FACTOR_SOURCE_POLICY_COLUMNS,
    )
    frame = _mapping_enabled_column(frame, mapping_name="price-factor source policy")
    frame = _mapping_year_columns(frame, mapping_name="price-factor source policy")
    frame["source_code"] = frame["source_code"].astype(str).str.strip().str.upper()
    frame["target_code"] = frame["target_code"].astype(str).str.strip()
    allowed_methods = {PRICE_FACTOR_METHOD_UNSD, PRICE_FACTOR_METHOD_TAIWAN}
    invalid_methods = sorted(set(frame["target_code"]).difference(allowed_methods))
    if invalid_methods:
        raise PriceIndexParseError(
            f"price-factor source policy has unsupported method codes: {invalid_methods}"
        )
    invalid_regions = sorted(
        code for code in set(frame["source_code"]) if code != "*" and not re.fullmatch(r"[A-Z]{2}", code)
    )
    if invalid_regions:
        raise PriceIndexParseError(
            f"price-factor source policy has invalid EXIOBASE region codes: {invalid_regions}"
        )
    _reject_overlapping_mapping_intervals(
        frame, source_column="source_code", mapping_name="price-factor source policy"
    )
    if not bool((frame.loc[frame["enabled"], "source_code"] == "*").any()):
        raise PriceIndexParseError("price-factor source policy requires an enabled wildcard default")
    return frame.reset_index(drop=True)


def _load_price_factor_source_policy() -> pd.DataFrame:
    """Load the packaged single-source price-factor policy mapping."""

    resource = resources.files("ecolecon_mrio").joinpath(PRICE_FACTOR_SOURCE_POLICY_RESOURCE)
    with resources.as_file(resource) as path:
        return _read_price_factor_source_policy(path)


def _resolve_price_factor_source_policy(
    policy: pd.DataFrame, *, exio_region: str, year: int
) -> pd.Series:
    """Resolve one exact-over-wildcard source-method rule for a region and year."""

    region = str(exio_region).strip().upper()
    year_int = int(year)
    active = policy.loc[
        policy["enabled"].astype(bool)
        & (policy["start_year"].astype(int) <= year_int)
        & (policy["end_year"].astype(int) >= year_int)
    ]
    exact = active.loc[active["source_code"].astype(str) == region]
    selected = exact if not exact.empty else active.loc[active["source_code"].astype(str) == "*"]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one price-factor source rule for region {region!r}, year {year_int}; "
            f"got {len(selected)}"
        )
    return selected.iloc[0]

def _read_price_factor_source_identity_mapping(path: Path) -> pd.DataFrame:
    """Read year-bounded provider identities and explicit inactive historical aliases."""

    frame = read_mapping_csv(
        path,
        expected_kind="price-factor-source-identity",
        required_columns=PRICE_FACTOR_SOURCE_IDENTITY_COLUMNS,
    )
    frame = _mapping_enabled_column(frame, mapping_name="price-factor source identity mapping")
    frame = _mapping_year_columns(frame, mapping_name="price-factor source identity mapping")
    frame["source_code"] = frame["source_code"].astype(str).str.strip().str.upper()
    frame["target_code"] = frame["target_code"].astype(str).str.strip().str.upper()
    frame["provider_system"] = frame["provider_system"].astype(str).str.strip()
    frame["provider_code"] = frame["provider_code"].astype(str).str.strip()
    frame["resolution_status"] = frame["resolution_status"].astype(str).str.strip().str.casefold()
    allowed_status = {"resolved", "inactive", "unresolved"}
    invalid_status = sorted(set(frame["resolution_status"]).difference(allowed_status))
    if invalid_status:
        raise PriceIndexParseError(
            f"price-factor source identity mapping has unsupported resolution_status values: {invalid_status}"
        )
    invalid_targets = sorted(set(frame["target_code"]).difference(EXPECTED_ROW_REGIONS))
    if invalid_targets:
        raise PriceIndexParseError(f"price-factor source identity mapping has invalid RoW targets: {invalid_targets}")
    unresolved_enabled = frame["enabled"].astype(bool) & frame["resolution_status"].eq("unresolved")
    if bool(unresolved_enabled.any()):
        labels = frame.loc[unresolved_enabled, "source_label"].astype(str).tolist()
        raise PriceIndexParseError(f"unresolved price-factor source identities cannot be enabled: {labels}")
    inactive = frame["resolution_status"].eq("inactive")
    if bool((inactive & (frame["provider_system"].ne("") | frame["provider_code"].ne(""))).any()):
        raise PriceIndexParseError("inactive source identities must not carry provider identities")
    resolved_enabled = frame["enabled"].astype(bool) & frame["resolution_status"].eq("resolved")
    incomplete = resolved_enabled & (frame["provider_system"].eq("") | frame["provider_code"].eq(""))
    if bool(incomplete.any()):
        labels = frame.loc[incomplete, "source_label"].astype(str).tolist()
        raise PriceIndexParseError(
            f"enabled resolved source identities require provider_system and provider_code: {labels}"
        )
    unsd = resolved_enabled & frame["provider_system"].eq("UNSD_AMA")
    bad_unsd = unsd & ~frame["provider_code"].str.fullmatch(r"\d{3}")
    if bool(bad_unsd.any()):
        raise PriceIndexParseError("resolved UNSD_AMA source identities require three-digit provider codes")
    _reject_overlapping_mapping_intervals(
        frame, source_column="source_code", mapping_name="price-factor source identity mapping"
    )
    return frame.reset_index(drop=True)

def _load_price_factor_source_identity_mapping() -> pd.DataFrame:
    """Load audited source-identity mappings, including disabled unresolved evidence rows."""

    resource = resources.files("ecolecon_mrio").joinpath(PRICE_FACTOR_SOURCE_IDENTITY_RESOURCE)
    with resources.as_file(resource) as path:
        return _read_price_factor_source_identity_mapping(path)


def _read_taiwan_dgbas_gdp_deflator_snapshot(path: Path) -> tuple[pd.Series, dict[str, Any]]:
    """Read the audited DGBAS annual GDP-deflator snapshot and its source identity.

    The package resource declares its audited annual observation domain. All
    observations were checked against one official publication, preserving a
    coherent source vintage without runtime dependence on the DGBAS transport.
    Its table, measure, item, frequency,
    and index-base metadata are validated before the values enter the Taiwan EUR
    conversion, so a future source update cannot silently change the statistic.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriceIndexParseError(f"Cannot parse pinned DGBAS Taiwan GDP-deflator snapshot {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PriceIndexParseError("Pinned DGBAS Taiwan GDP-deflator snapshot must be a JSON object")
    required_identity = {
        "provider": "DGBAS",
        "country": "Taiwan",
        "dgbas_table": DGBAS_TABLE_ID,
        "measure": "fldid=3",
        "item": "code1=14",
        "frequency": "annual",
        "reference_year": 2021,
    }
    for key, expected in required_identity.items():
        if payload.get(key) != expected:
            raise PriceIndexParseError(
                f"Pinned DGBAS Taiwan GDP-deflator snapshot has unexpected {key}: {payload.get(key)!r}"
            )
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise PriceIndexParseError("Pinned DGBAS Taiwan GDP-deflator snapshot observations must be a list")
    values: dict[int, float] = {}
    for row in observations:
        if not isinstance(row, dict) or set(row) != {"year", "gdp_deflator"}:
            raise PriceIndexParseError("Pinned DGBAS Taiwan GDP-deflator snapshot rows must contain only year and gdp_deflator")
        try:
            year = int(row["year"])
            value = float(row["gdp_deflator"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PriceIndexParseError(f"Pinned DGBAS Taiwan GDP-deflator snapshot has non-numeric row {row!r}") from exc
        if year in values:
            raise PriceIndexParseError(f"Pinned DGBAS Taiwan GDP-deflator snapshot duplicates year {year}")
        values[year] = value
    # Source metadata owns the admitted years; extending a verified snapshot
    # requires no Python year constant. Gaps never become interpolated prices.
    try:
        start, end = (int(year) for year in str(payload["snapshot_domain"]).split("--"))
    except (KeyError, TypeError, ValueError) as exc:
        raise PriceIndexParseError("Pinned DGBAS snapshot requires an annual snapshot_domain start--end") from exc
    if end < start or tuple(sorted(values)) != tuple(range(start, end + 1)):
        raise PriceIndexParseError("Pinned DGBAS snapshot observations must cover exactly the declared annual domain")
    series = _positive_series(values, name="taiwan_gdp_deflator")
    provenance = {key: value for key, value in payload.items() if key != "observations"}
    provenance["snapshot_resource"] = DGBAS_SNAPSHOT_RESOURCE
    return series, provenance


def _load_taiwan_dgbas_gdp_deflator(years: Sequence[int]) -> tuple[pd.Series, dict[str, Any]]:
    """Load the packaged DGBAS snapshot and reject requests outside its audited domain."""

    requested = tuple(sorted({int(year) for year in years}))
    resource = resources.files("ecolecon_mrio").joinpath(DGBAS_SNAPSHOT_RESOURCE)
    with resources.as_file(resource) as path:
        series, provenance = _read_taiwan_dgbas_gdp_deflator_snapshot(path)
    # Reject years without actual observations, including an unavailable base year.
    # This remains fail-closed as the pinned observation domain is extended.
    return _require_years(series, requested, name="Pinned DGBAS Taiwan GDP deflator"), provenance


def _read_taiwan_ntd_per_usd(path: Path) -> pd.Series:
    """Parse annual-average NTD/USD from the Taiwan central bank HTML table."""

    parser = _parse_html(path.read_text(encoding="utf-8"))
    values: dict[int, float] = {}
    for table in parser.tables:
        for row in table:
            if len(row) < 2 or not re.fullmatch(r"\d{4}", row[0].strip()):
                continue
            year = int(row[0].strip())
            try:
                value = float(row[1].replace(",", "").strip())
            except ValueError:
                continue
            previous = values.get(year)
            if previous is not None and not math.isclose(previous, value, rel_tol=0.0, abs_tol=1.0e-12):
                raise PriceIndexParseError(f"Taiwan central-bank page has conflicting NTD/USD values for {year}")
            values[year] = value
    return _positive_series(values, name="ntd_per_usd")


def _require_years(series: pd.Series, years: Sequence[int], *, name: str) -> pd.Series:
    """Select an exact requested year domain without interpolation or carry-forward."""

    requested = pd.Index(sorted({int(year) for year in years}), name="year")
    missing = [int(year) for year in requested if int(year) not in series.index]
    if missing:
        raise ValueError(f"{name} is missing requested years {missing}")
    return series.reindex(requested).astype(float)


def _taiwan_common_eur_factor(
    deflator: pd.Series,
    ntd_per_usd: pd.Series,
    usd_per_eur: pd.Series,
    *,
    year: int,
    base_year: int,
) -> float:
    """Convert Taiwan's national-currency GDP deflator to the same base-year EUR gauge."""

    p = _require_years(deflator, [year, base_year], name="Taiwan GDP deflator")
    ntd_usd = _require_years(ntd_per_usd, [year, base_year], name="Taiwan NTD/USD")
    usd_eur = _require_years(usd_per_eur, [year, base_year], name="ECB USD/EUR")
    exchange_t = float(ntd_usd.loc[year]) * float(usd_eur.loc[year])
    exchange_b = float(ntd_usd.loc[base_year]) * float(usd_eur.loc[base_year])
    factor = (float(p.loc[base_year]) / float(p.loc[year])) * (exchange_t / exchange_b)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"Taiwan common-EUR factor is invalid for year {year}: {factor!r}")
    return factor

def _read_exiobase3rx_region_bridge(path: Path) -> pd.DataFrame:
    """Read and normalize the recovered four-column EXIOBASE3rx 214-to-49 bridge.

    The installed CSV is intentionally the exact audited extraction artifact:
    ``source_country, source_iso3, target_exio_region, weight``. The loader derives
    an internal source key from ISO3 when present and otherwise from the unique
    historical source label; provider exceptions remain in the separate
    source-identity mapping.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != list(EXIOBASE3RX_BRIDGE_COLUMNS):
        raise PriceIndexParseError(
            "EXIOBASE3rx bridge must retain the audited four-column extraction schema"
        )
    if len(frame) != 214 or frame["source_country"].astype(str).nunique() != 214:
        raise PriceIndexParseError(
            "EXIOBASE3rx bridge must contain exactly 214 unique source-country labels"
        )
    source_iso3 = frame["source_iso3"].astype(str).str.strip().str.upper()
    nonblank_iso3 = source_iso3.loc[source_iso3.ne("")]
    if len(nonblank_iso3) != 212 or nonblank_iso3.nunique() != 212:
        raise PriceIndexParseError(
            "EXIOBASE3rx bridge must retain 212 unique nonblank ISO3 annotations"
        )
    blank_labels = set(frame.loc[source_iso3.eq(""), "source_country"].astype(str))
    if blank_labels != {"Netherlands Antilles", "Zanzibar"}:
        raise PriceIndexParseError(
            f"EXIOBASE3rx bridge has unexpected blank source_iso3 identities: {sorted(blank_labels)!r}"
        )
    weights = cast(pd.Series, pd.to_numeric(frame["weight"], errors="raise")).astype(float)
    if (
        not np.isfinite(weights.to_numpy()).all()
        or not np.allclose(weights.to_numpy(), 1.0, rtol=0.0, atol=0.0)
    ):
        raise PriceIndexParseError(
            "EXIOBASE3rx bridge must remain a one-to-one weight=1 concordance"
        )

    target_code = frame["target_exio_region"].astype(str).str.strip().str.upper()
    if target_code.nunique() != EXPECTED_EXIOBASE3_REGION_COUNT:
        raise PriceIndexParseError(
            "EXIOBASE3rx bridge must resolve to exactly 49 legacy target regions"
        )
    row_counts = (
        target_code.loc[target_code.isin(list(EXPECTED_ROW_REGIONS))]
        .value_counts()
        .to_dict()
    )
    expected = {"WA": 45, "WL": 43, "WE": 14, "WF": 53, "WM": 15}
    if row_counts != expected:
        raise PriceIndexParseError(
            f"EXIOBASE3rx bridge RoW counts differ from the audited bridge: {row_counts}"
        )
    explicit = target_code.loc[~target_code.isin(list(EXPECTED_ROW_REGIONS))]
    if explicit.nunique() != 44 or not bool((explicit.value_counts() == 1).all()):
        raise PriceIndexParseError(
            "EXIOBASE3rx bridge must contain 44 singleton explicit target regions"
        )

    source_label = frame["source_country"].astype(str).str.strip()
    source_code = source_iso3.where(source_iso3.ne(""), source_label.str.upper())
    normalized = pd.DataFrame(
        {
            "source_code": source_code,
            "source_label": source_label,
            "source_iso3": source_iso3,
            "target_code": target_code,
            "target_label": target_code,
        }
    )
    if normalized["source_code"].nunique() != 214:
        raise PriceIndexParseError("EXIOBASE3rx normalized source keys must remain unique")
    return normalized.sort_values(
        ["target_code", "source_code"], kind="stable"
    ).reset_index(drop=True)


def _load_exiobase3rx_region_bridge() -> pd.DataFrame:
    """Load the exact installed Bjelle-derived regional bridge used by production factors."""

    resource = resources.files("ecolecon_mrio").joinpath(EXIOBASE3RX_BRIDGE_RESOURCE)
    with resources.as_file(resource) as path:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXIOBASE3RX_BRIDGE_SHA256:
            raise PriceIndexParseError(
                "Packaged EXIOBASE3rx bridge differs from the audited Bjelle-derived artifact: "
                f"expected {EXIOBASE3RX_BRIDGE_SHA256}, got {digest}"
            )
        return _read_exiobase3rx_region_bridge(path)




def _read_exiobase3_explicit_region_crosswalk(path: Path) -> pd.DataFrame:
    """Read only the pinned 00-class ISO3-to-runtime-EXIO explicit-country crosswalk.

    The published EXIOBASE3rx bridge owns WA/WL/WE/WF/WM membership. This helper
    intentionally ignores 00-class RoW classification fields and returns only rows
    with a non-empty two-letter ``EXIO3`` runtime region code.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"ISO3", "EXIO3"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PriceIndexParseError(f"EXIOBASE 00-class country coverage CSV missing columns: {missing}")
    selected = frame.loc[
        frame["ISO3"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)
        & frame["EXIO3"].astype(str).str.fullmatch(r"[A-Z]{2}", na=False),
        ["ISO3", "EXIO3"],
    ].copy()
    selected.columns = ["iso3", "exio_region"]
    selected["iso3"] = selected["iso3"].astype(str).str.strip().str.upper()
    selected["exio_region"] = selected["exio_region"].astype(str).str.strip().str.upper()
    if len(selected) != 44 or selected["iso3"].nunique() != 44 or selected["exio_region"].nunique() != 44:
        raise PriceIndexParseError(
            "Pinned EXIOBASE 00-class explicit-country crosswalk must contain exactly 44 one-to-one ISO3->EXIO3 rows"
        )
    if selected.loc[selected["iso3"].eq("TWN"), "exio_region"].tolist() != ["TW"]:
        raise PriceIndexParseError("Pinned EXIOBASE 00-class crosswalk must map TWN to TW")
    return selected.sort_values("iso3", kind="stable").reset_index(drop=True)


def _compile_exiobase3_bridge_runtime_targets(
    bridge: pd.DataFrame,
    explicit_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Translate the bridge's 44 explicit ISO3 targets to runtime EXIO region codes."""

    explicit = dict(
        zip(
            explicit_crosswalk["iso3"].astype(str),
            explicit_crosswalk["exio_region"].astype(str),
            strict=True,
        )
    )
    published_explicit = set(
        bridge.loc[~bridge["target_code"].isin(list(EXPECTED_ROW_REGIONS)), "target_code"].astype(str)
    )
    if published_explicit != set(explicit):
        missing = sorted(published_explicit.difference(explicit))
        extra = sorted(set(explicit).difference(published_explicit))
        raise PriceIndexParseError(
            "Published EXIOBASE3rx explicit targets and pinned 00-class crosswalk disagree; "
            f"missing={missing}, extra={extra}"
        )
    result = bridge.copy()
    is_row = result["target_code"].isin(list(EXPECTED_ROW_REGIONS))
    result["published_target_code"] = result["target_code"].astype(str)
    result.loc[~is_row, "target_code"] = result.loc[~is_row, "target_code"].map(explicit)
    result.loc[~is_row, "target_label"] = result.loc[~is_row, "target_code"]
    if bool(result["target_code"].isna().any()):
        raise PriceIndexParseError("Compiled bridge contains missing runtime EXIO target codes")
    if result["target_code"].nunique() != EXPECTED_EXIOBASE3_REGION_COUNT:
        raise PriceIndexParseError("Compiled EXIOBASE3rx bridge must resolve to exactly 49 runtime regions")
    explicit_groups = result.loc[~is_row].groupby("target_code").size()
    if len(explicit_groups) != 44 or not bool((explicit_groups == 1).all()):
        raise PriceIndexParseError("Compiled bridge must contain 44 singleton explicit runtime EXIO regions")
    return result.sort_values(["target_code", "source_code"], kind="stable").reset_index(drop=True)




def _mapping_resource_sha256(resource_name: str) -> str:
    """Hash one installed mapping so result provenance identifies its exact scientific content."""

    resource = resources.files("ecolecon_mrio").joinpath(resource_name)
    with resources.as_file(resource) as path:
        return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_source_policy_domain(policy: pd.DataFrame, *, explicit_regions: set[str], years: Sequence[int]) -> None:
    """Require every exact policy row to name a real explicit region and every year to resolve once."""

    enabled = policy.loc[policy["enabled"].astype(bool)]
    exact = set(enabled.loc[enabled["source_code"].astype(str) != "*", "source_code"].astype(str))
    unknown = sorted(exact.difference(explicit_regions))
    if unknown:
        raise PriceIndexParseError(f"price-factor source policy names unknown explicit regions: {unknown}")
    taiwan_rows = enabled.loc[enabled["target_code"].astype(str) == PRICE_FACTOR_METHOD_TAIWAN]
    if set(taiwan_rows["source_code"].astype(str)) != {"TW"}:
        raise PriceIndexParseError("Taiwan DGBAS/CBC method must be selected only by the exact TW policy row")
    for region in sorted(explicit_regions):
        for year in years:
            _resolve_price_factor_source_policy(policy, exio_region=region, year=int(year))


def _effective_row_members(
    bridge: pd.DataFrame,
    identity: pd.DataFrame,
    iso3_to_m49: Mapping[str, int],
    *,
    year: int,
) -> pd.DataFrame:
    """Resolve one year's effective published RoW source identities to UNSD statistical units."""

    rows: list[dict[str, object]] = []
    static = bridge.loc[bridge["target_code"].isin(list(EXPECTED_ROW_REGIONS))]
    for member in static.itertuples(index=False):
        source_code = str(member.source_code).strip().upper()
        target_code = str(member.target_code).strip().upper()
        candidates = identity.loc[
            identity["enabled"].astype(bool)
            & identity["source_code"].astype(str).eq(source_code)
            & (identity["start_year"].astype(int) <= int(year))
            & (identity["end_year"].astype(int) >= int(year))
        ]
        if len(candidates) > 1:
            raise ValueError(f"Multiple source-identity rules apply to {source_code!r}, year={year}")
        if len(candidates) == 1:
            rule = candidates.iloc[0]
            if str(rule["target_code"]) != target_code:
                raise ValueError(
                    f"source-identity rule moves {source_code!r} from published {target_code} to {rule['target_code']!r}"
                )
            status = str(rule["resolution_status"])
            if status == "inactive":
                continue
            if status != "resolved" or str(rule["provider_system"]) != "UNSD_AMA":
                raise ValueError(f"Unsupported active source-identity rule for {source_code!r}: {status!r}")
            m49 = int(str(rule["provider_code"]))
        else:
            m49_value = iso3_to_m49.get(source_code)
            if m49_value is None:
                raise ValueError(
                    f"Published RoW source {source_code!r} has no UN M49 identity and no year-specific mapping rule"
                )
            m49 = int(m49_value)
        rows.append(
            {
                "target_code": target_code,
                "source_code": source_code,
                "source_label": str(member.source_label),
                "m49": m49,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(subset=["target_code", "source_code"]).any():
        raise ValueError("Effective RoW membership is empty or duplicates source identities")
    if result.duplicated(subset=["target_code", "m49"]).any():
        dup = result.loc[result.duplicated(subset=["target_code", "m49"], keep=False)]
        raise ValueError(f"Effective RoW membership double-counts one UNSD statistical unit:\n{dup.to_string(index=False)}")
    return result.sort_values(["target_code", "source_code"], kind="stable").reset_index(drop=True)


def _require_gdp_level_rows(levels: pd.DataFrame, *, m49_codes: Sequence[int], year: int) -> pd.DataFrame:
    """Select complete positive current/constant USD GDP levels for an exact member set."""

    required = {"m49", "year", "current_usd_gdp", "constant_usd_gdp"}
    missing_columns = sorted(required.difference(levels.columns))
    if missing_columns:
        raise ValueError(f"UNSD GDP-level table is missing columns {missing_columns}")
    subset = levels.loc[levels["year"].astype(int) == int(year)].copy()
    if subset.duplicated(subset=["m49"]).any():
        raise ValueError(f"UNSD GDP-level table duplicates M49 rows for year {year}")
    subset = subset.set_index("m49")
    requested = [int(code) for code in m49_codes]
    missing = [code for code in requested if code not in subset.index]
    if missing:
        raise ValueError(f"UNSD GDP levels are missing required M49 rows for year {year}: {missing}")
    selected = subset.reindex(requested).loc[:, ["current_usd_gdp", "constant_usd_gdp"]].astype(float)
    invalid = (~np.isfinite(selected.to_numpy())).any(axis=1) | (selected <= 0.0).any(axis=1)
    if bool(invalid.any()):
        bad = selected.index[invalid].tolist()
        raise ValueError(f"UNSD GDP levels are missing/non-positive/non-finite for year {year}, M49={bad}")
    return selected


def _common_eur_factor_from_usd_levels(
    levels: pd.DataFrame,
    usd_per_eur: pd.Series,
    *,
    m49_codes: Sequence[int],
    year: int,
    base_year: int,
) -> float:
    """Form one implicit deflator from extensive USD GDP levels, then convert to a common-EUR gauge."""

    codes = tuple(dict.fromkeys(int(code) for code in m49_codes))
    if not codes:
        raise ValueError("GDP-level factor requires at least one source statistical unit")
    current = _require_gdp_level_rows(levels, m49_codes=codes, year=int(year))
    base = _require_gdp_level_rows(levels, m49_codes=codes, year=int(base_year))
    q_t = float(cast(float, current["current_usd_gdp"].sum())) / float(cast(float, current["constant_usd_gdp"].sum()))
    q_b = float(cast(float, base["current_usd_gdp"].sum())) / float(cast(float, base["constant_usd_gdp"].sum()))
    rates = _require_years(usd_per_eur, [year, base_year], name="ECB USD/EUR")
    factor = (q_b / q_t) * (float(rates.loc[int(year)]) / float(rates.loc[int(base_year)]))
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"Invalid common-EUR GDP-level factor for year {year}: {factor!r}")
    return factor


def build_exiobase3_eur_price_factors(
    *,
    years: Sequence[int],
    base_year: int,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build complete legacy EXIOBASE3 factors from audited GDP-level source geography.

    The Bjelle-derived bridge owns the published 214-country membership. Its 44
    explicit ISO3 targets are translated through the pinned 00-class ``ISO3`` to
    ``EXIO3`` crosswalk; 00-class never supplies RoW membership. Ordinary explicit
    countries and the five RoW aggregates use UNSD current/constant USD GDP levels
    plus the ECB annual USD/EUR correction. RoW levels are summed before implicit
    deflation. Taiwan remains on its mapped DGBAS/CBC route.
    """

    requested_years = tuple(sorted({int(year) for year in years}))
    if not requested_years:
        raise ValueError("years must contain at least one year")
    all_years = tuple(sorted(set(requested_years) | {int(base_year)}))
    root = default_exiobase3_price_factor_storage_folder() if storage_folder is None else Path(storage_folder)
    root.mkdir(parents=True, exist_ok=True)

    source_policy = _load_price_factor_source_policy()
    identity = _load_price_factor_source_identity_mapping()
    published_bridge = _load_exiobase3rx_region_bridge()

    # External source files are small, cached provenance inputs. The 00-class file
    # is used only to translate explicit ISO3 targets to runtime two-letter EXIO
    # codes; its EXIO3_3L / RoW fields have no membership authority here.
    ecb_path = _cached_text(
        ECB_USD_EUR_URL,
        path=root / "ecb" / "usd_per_eur_annual.csv",
        force_download=force_download,
        timeout_s=timeout_s,
    )
    m49_path = _cached_text(
        UNSD_M49_URL,
        path=root / "unsd_m49" / "overview.html",
        force_download=force_download,
        timeout_s=timeout_s,
    )
    exio_path = _cached_text(
        EXIOBASE_COUNTRY_COVERAGE_URL,
        path=root / "exiobase_meta" / f"country_coverage_matrix_{EXIOBASE_META_COMMIT[:12]}.csv",
        force_download=force_download,
        timeout_s=timeout_s,
    )

    usd_per_eur = _require_years(_read_ecb_usd_per_eur(ecb_path), all_years, name="ECB USD/EUR")
    m49_table = _read_unsd_m49_iso3(m49_path)
    explicit_crosswalk = _read_exiobase3_explicit_region_crosswalk(exio_path)
    bridge = _compile_exiobase3_bridge_runtime_targets(published_bridge, explicit_crosswalk)
    explicit_bridge = bridge.loc[~bridge["target_code"].isin(list(EXPECTED_ROW_REGIONS))].copy()
    explicit_regions = set(explicit_bridge["target_code"].astype(str))
    _validate_source_policy_domain(source_policy, explicit_regions=explicit_regions, years=all_years)

    iso3_to_m49 = dict(
        zip(
            m49_table["iso3"].astype(str).tolist(),
            m49_table["m49"].astype(int).tolist(),
            strict=True,
        )
    )
    effective_by_year = {
        year: _effective_row_members(bridge, identity, iso3_to_m49, year=year)
        for year in all_years
    }
    reference_members = {
        (str(target_code), str(source_code), int(m49))
        for target_code, source_code, m49 in effective_by_year[int(base_year)].loc[
            :, ["target_code", "source_code", "m49"]
        ].itertuples(index=False, name=None)
    }
    for year, members in effective_by_year.items():
        observed = {
            (str(target_code), str(source_code), int(m49))
            for target_code, source_code, m49 in members.loc[
                :, ["target_code", "source_code", "m49"]
            ].itertuples(index=False, name=None)
        }
        if observed != reference_members:
            raise ValueError(
                "Effective RoW source membership changes inside the requested/base-year domain; "
                f"year={year}. Add an explicitly audited common identity regime before comparing these years."
            )

    needed_m49 = {int(m49) for m49 in effective_by_year[int(base_year)]["m49"].tolist()}
    explicit_m49: dict[str, int] = {}
    active_methods: dict[int, dict[str, str]] = {}
    for year in all_years:
        active_methods[year] = {}
        for member in explicit_bridge.itertuples(index=False):
            region = str(member.target_code)
            method = str(
                _resolve_price_factor_source_policy(source_policy, exio_region=region, year=year)["target_code"]
            )
            active_methods[year][region] = method
            if method == PRICE_FACTOR_METHOD_UNSD:
                m49 = iso3_to_m49.get(str(member.source_code))
                if m49 is None:
                    raise ValueError(f"Explicit EXIOBASE source {member.source_code!r} has no UN M49 mapping")
                explicit_m49[region] = int(m49)
                needed_m49.add(int(m49))

    levels, unsd_levels_meta = load_unsd_ama_gdp_usd_levels(
        years=all_years,
        country_codes=sorted(needed_m49),
        storage_folder=root / "unsd_ama",
        force_download=force_download,
        timeout_s=timeout_s,
    )

    needs_taiwan = any(
        method == PRICE_FACTOR_METHOD_TAIWAN
        for by_region in active_methods.values()
        for method in by_region.values()
    )
    taiwan_deflator: pd.Series | None = None
    ntd_per_usd: pd.Series | None = None
    taiwan_dgbas_metadata: dict[str, Any] | None = None
    cbc_path: Path | None = None
    if needs_taiwan:
        taiwan_deflator, taiwan_dgbas_metadata = _load_taiwan_dgbas_gdp_deflator(all_years)
        cbc_path = _cached_text(
            CBC_TWD_USD_ANNUAL_URL,
            path=root / "taiwan_cbc" / "ntd_per_usd_annual.html",
            force_download=force_download,
            timeout_s=timeout_s,
        )
        ntd_per_usd = _require_years(_read_taiwan_ntd_per_usd(cbc_path), all_years, name="Taiwan NTD/USD")

    outputs: list[pd.DataFrame] = []
    for year in all_years:
        rows: list[dict[str, object]] = []
        for member in explicit_bridge.sort_values("target_code").itertuples(index=False):
            region = str(member.target_code)
            method = active_methods[year][region]
            if method == PRICE_FACTOR_METHOD_UNSD:
                m49 = explicit_m49[region]
                factor = _common_eur_factor_from_usd_levels(
                    levels, usd_per_eur, m49_codes=[m49], year=year, base_year=int(base_year)
                )
                provider = "UNSD AMA current/constant USD GDP levels + ECB"
            elif method == PRICE_FACTOR_METHOD_TAIWAN:
                if taiwan_deflator is None or ntd_per_usd is None:
                    raise ValueError("Mapped Taiwan price-factor method is active but its source inputs were not acquired")
                factor = _taiwan_common_eur_factor(
                    taiwan_deflator, ntd_per_usd, usd_per_eur, year=year, base_year=int(base_year)
                )
                provider = "DGBAS GDP deflator + CBC NTD/USD + ECB"
            else:
                raise ValueError(f"Unsupported mapped explicit-country price-factor method {method!r}")
            rows.append(
                {
                    "exio_region": region,
                    "region_kind": "country",
                    "price_adjustment_factor": factor,
                    "source_country_count": 1,
                    "source_country_iso3": str(member.source_code),
                    "source_provider": provider,
                }
            )

        effective = effective_by_year[year]
        for region in sorted(EXPECTED_ROW_REGIONS):
            group = effective.loc[effective["target_code"].astype(str) == region]
            if group.empty:
                raise ValueError(f"Effective RoW region {region} has no source members")
            factor = _common_eur_factor_from_usd_levels(
                levels,
                usd_per_eur,
                m49_codes=group["m49"].astype(int).tolist(),
                year=year,
                base_year=int(base_year),
            )
            rows.append(
                {
                    "exio_region": region,
                    "region_kind": "row",
                    "price_adjustment_factor": factor,
                    "source_country_count": int(len(group)),
                    "source_country_iso3": ";".join(group["source_code"].astype(str).tolist()),
                    "source_provider": "UNSD AMA aggregate current/constant USD GDP levels + ECB",
                }
            )
        region_frame = pd.DataFrame(rows).sort_values("exio_region", kind="stable").reset_index(drop=True)
        if len(region_frame) != EXPECTED_EXIOBASE3_REGION_COUNT or region_frame["exio_region"].nunique() != EXPECTED_EXIOBASE3_REGION_COUNT:
            raise ValueError(f"Expected exactly 49 EXIOBASE factors for year {year}, got {len(region_frame)}")
        if any(len(str(code)) != 2 for code in region_frame["exio_region"]):
            raise ValueError("Production factor table must expose runtime two-letter EXIOBASE region codes")
        region_frame.insert(0, "year", int(year))
        region_frame.insert(1, "base_year", int(base_year))
        region_frame.insert(4, "factor_basis", "country_gdp_deflator_common_eur")
        outputs.append(region_frame)

    result = pd.concat(outputs, ignore_index=True)
    result = result.sort_values(["year", "exio_region"], kind="stable").reset_index(drop=True)
    base = result.loc[result["year"] == int(base_year), "price_adjustment_factor"].to_numpy(dtype=float)
    if len(base) != EXPECTED_EXIOBASE3_REGION_COUNT or not np.allclose(base, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("EXIOBASE common-EUR base-year factors must equal one for all 49 regions")
    expected_rows = len(all_years) * EXPECTED_EXIOBASE3_REGION_COUNT
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} year-region factor rows, got {len(result)}")

    raw_gdp_path = Path(str(unsd_levels_meta.get("raw_path", "")))
    if not raw_gdp_path.is_file():
        raise ValueError("UNSD GDP-level metadata must identify the exact cached source bytes through raw_path")
    published_counts = published_bridge.loc[
        published_bridge["target_code"].isin(list(EXPECTED_ROW_REGIONS))
    ].groupby("target_code").size().astype(int).to_dict()
    effective_counts = {
        int(year): members.groupby("target_code").size().astype(int).to_dict()
        for year, members in effective_by_year.items()
    }
    source_sha256: dict[str, str] = {
        "bjelle_exiobase3rx_bridge": _mapping_resource_sha256(EXIOBASE3RX_BRIDGE_RESOURCE),
        "price_factor_source_policy": _mapping_resource_sha256(PRICE_FACTOR_SOURCE_POLICY_RESOURCE),
        "price_factor_source_identity": _mapping_resource_sha256(PRICE_FACTOR_SOURCE_IDENTITY_RESOURCE),
        "unsd_gdp_usd_levels_cache": hashlib.sha256(raw_gdp_path.read_bytes()).hexdigest(),
        "ecb_usd_per_eur": hashlib.sha256(ecb_path.read_bytes()).hexdigest(),
        "un_m49_overview": hashlib.sha256(m49_path.read_bytes()).hexdigest(),
        "exiobase_00_class_explicit_crosswalk": hashlib.sha256(exio_path.read_bytes()).hexdigest(),
    }
    if needs_taiwan:
        if cbc_path is None:
            raise AssertionError("Taiwan CBC path is missing after conditional source acquisition")
        source_sha256["taiwan_dgbas_snapshot"] = _mapping_resource_sha256(DGBAS_SNAPSHOT_RESOURCE)
        source_sha256["taiwan_cbc_ntd_per_usd"] = hashlib.sha256(cbc_path.read_bytes()).hexdigest()

    factor_bytes = result.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metadata: dict[str, Any] = {
        "factor_basis": "country_gdp_deflator_common_eur",
        "row_aggregation": "aggregate current-USD and constant-price-USD GDP levels before implicit deflation",
        "requested_years": list(requested_years),
        "returned_years": list(all_years),
        "base_year": int(base_year),
        "region_count": EXPECTED_EXIOBASE3_REGION_COUNT,
        "factor_row_count": int(len(result)),
        "factor_table_sha256": hashlib.sha256(factor_bytes).hexdigest(),
        "unsd_gdp_usd_levels_metadata": unsd_levels_meta,
        "ecb_series": ECB_USD_EUR_SERIES,
        "un_m49_url": UNSD_M49_URL,
        "exiobase_meta_commit": EXIOBASE_META_COMMIT,
        "exiobase_country_coverage_url": EXIOBASE_COUNTRY_COVERAGE_URL,
        "exiobase_00_class_role": "explicit ISO3 target to runtime EXIO code only; not RoW membership",
        "exiobase3rx_bridge_resource": EXIOBASE3RX_BRIDGE_RESOURCE,
        "exiobase3rx_bridge_sha256": EXIOBASE3RX_BRIDGE_SHA256,
        "exiobase3rx_bridge_doi": EXIOBASE3RX_BRIDGE_DOI,
        "exiobase3rx_bridge_supplement": EXIOBASE3RX_BRIDGE_SUPPLEMENT,
        "exiobase3rx_bridge_license": EXIOBASE3RX_BRIDGE_LICENSE,
        "exiobase3rx_bridge_license_url": EXIOBASE3RX_BRIDGE_LICENSE_URL,
        "price_factor_source_policy_resource": PRICE_FACTOR_SOURCE_POLICY_RESOURCE,
        "price_factor_source_policy_sha256": _mapping_resource_sha256(PRICE_FACTOR_SOURCE_POLICY_RESOURCE),
        "price_factor_source_identity_resource": PRICE_FACTOR_SOURCE_IDENTITY_RESOURCE,
        "price_factor_source_identity_sha256": _mapping_resource_sha256(PRICE_FACTOR_SOURCE_IDENTITY_RESOURCE),
        "published_row_member_counts": published_counts,
        "effective_row_member_counts": effective_counts,
        "active_methods_by_year": active_methods,
        "taiwan_sources_used": bool(needs_taiwan),
        "taiwan_dgbas_snapshot": taiwan_dgbas_metadata,
        "taiwan_dgbas_metadata_url": DGBAS_METADATA_URL if needs_taiwan else None,
        "taiwan_cbc_exchange_rate": CBC_TWD_USD_ANNUAL_URL if needs_taiwan else None,
        "source_sha256": source_sha256,
        "interpretation": "country-level macro GDP-deflator sensitivity; not product-country constant-price EXIOBASE",
    }
    return result, metadata

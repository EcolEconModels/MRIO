"""Build complete EXIOBASE3 macro price-adjustment factors in a common EUR basis.

This module completes the country-level GDP-deflator sensitivity boundary without
changing the primitive monetary row-scaling algorithm.  It combines the UNSD AMA
USD-basis GDP deflator with the ECB annual USD/EUR rate for ordinary countries,
uses the official Taiwan DGBAS GDP deflator plus Taiwan central-bank NTD/USD rate
for Taiwan, and aggregates country factors to the five legacy EXIOBASE Rest-of-
World buckets with current-USD GDP weights.

The result remains a country-level macro sensitivity.  It does not reconstruct
product-by-country constant-price EXIOBASE tables and cannot remove within-country
relative-price movements.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ecolecon_mrio.data import default_data_root
from ecolecon_mrio.price_indices import (
    PriceIndexHTTPError,
    PriceIndexParseError,
    load_unsd_ama_current_usd_gdp,
    load_unsd_ama_gdp_deflators,
    price_adjustment_factors,
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
DGBAS_DATASET_PAGE = "https://data.gov.tw/en/datasets/44224"
CBC_TWD_USD_ANNUAL_URL = "https://www.cbc.gov.tw/en/cp-480-1879-66035-2.html"
ROW_CODE_MAP = {"WWA": "WA", "WWL": "WL", "WWE": "WE", "WWF": "WF", "WWM": "WM"}
EXPECTED_ROW_REGIONS = frozenset(ROW_CODE_MAP.values())
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
        value = float(row["OBS_VALUE"])
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


def _read_exiobase3_membership(path: Path) -> pd.DataFrame:
    """Read the pinned EXIOBASE 44-country plus five-RoW legacy geography."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"ISO3", "EXIO3", "EXIO3_3L"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PriceIndexParseError(f"EXIOBASE country coverage CSV missing columns: {missing}")
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        iso3 = str(row["ISO3"]).strip().upper()
        explicit = str(row["EXIO3"]).strip().upper()
        level3 = str(row["EXIO3_3L"]).strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", iso3):
            continue
        if explicit:
            if not re.fullmatch(r"[A-Z]{2}", explicit):
                raise PriceIndexParseError(f"Invalid EXIOBASE explicit region code {explicit!r} for {iso3}")
            exio_region = explicit
            region_kind = "country"
        elif level3 in ROW_CODE_MAP:
            exio_region = ROW_CODE_MAP[level3]
            region_kind = "row"
        else:
            continue
        rows.append({"iso3": iso3, "exio_region": exio_region, "region_kind": region_kind})
    result = pd.DataFrame(rows)
    if result.empty or result["iso3"].duplicated().any():
        raise PriceIndexParseError("EXIOBASE legacy membership must contain unique ISO3 rows")
    country_counts = result.loc[result["region_kind"] == "country"].groupby("exio_region")["iso3"].nunique()
    if len(country_counts) != 44 or not bool((country_counts == 1).all()):
        raise PriceIndexParseError(f"Expected 44 singleton EXIOBASE country regions, got {len(country_counts)}")
    row_regions = set(result.loc[result["region_kind"] == "row", "exio_region"])
    if row_regions != set(EXPECTED_ROW_REGIONS):
        raise PriceIndexParseError(f"Expected RoW regions {sorted(EXPECTED_ROW_REGIONS)}, got {sorted(row_regions)}")
    if result["exio_region"].nunique() != EXPECTED_EXIOBASE3_REGION_COUNT:
        raise PriceIndexParseError("EXIOBASE membership does not resolve to exactly 49 regions")
    return result.sort_values(["region_kind", "exio_region", "iso3"], kind="stable").reset_index(drop=True)


def _read_taiwan_dgbas_gdp_deflator(path: Path) -> pd.Series:
    """Parse annual Taiwan aggregate GDP implicit-price-deflator rows from DGBAS XML."""

    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError) as exc:
        raise PriceIndexParseError(f"Cannot parse Taiwan DGBAS XML {path}: {exc}") from exc
    records: list[dict[str, str]] = []
    for element in root.iter():
        fields: dict[str, str] = {}
        for child in list(element):
            tag = child.tag.rsplit("}", 1)[-1].strip().casefold()
            if tag in {"item", "time_period", "freq", "type", "item_value"}:
                fields[tag] = " ".join("".join(child.itertext()).split())
        if {"item", "time_period", "freq", "type", "item_value"}.issubset(fields):
            records.append(fields)
    if not records:
        raise PriceIndexParseError("Taiwan DGBAS XML contains no standard Item/TIME_PERIOD/FREQ/TYPE/Item_VALUE records")
    values: dict[int, float] = {}
    for record in records:
        combined = f"{record['item']} {record['type']}".casefold()
        is_gdp = bool(re.search(r"\bgdp\b", combined)) or "gross domestic product" in combined or "國內生產毛額" in combined
        is_deflator = "deflator" in combined or "平減" in combined
        freq = record["freq"].strip().casefold()
        annual = freq in {"a", "annual", "year", "yearly", "年"} or "annual" in freq
        if not (is_gdp and is_deflator and annual):
            continue
        year_match = re.fullmatch(r"(19|20|21)\d{2}", record["time_period"].strip())
        if year_match is None:
            raise PriceIndexParseError(f"Taiwan DGBAS GDP-deflator row has non-Gregorian annual period {record['time_period']!r}")
        year = int(year_match.group(0))
        try:
            value = float(record["item_value"].replace(",", ""))
        except ValueError as exc:
            raise PriceIndexParseError(f"Taiwan DGBAS GDP deflator is non-numeric for {year}") from exc
        if year in values:
            raise PriceIndexParseError(f"Taiwan DGBAS XML has multiple aggregate GDP deflator rows for {year}")
        values[year] = value
    return _positive_series(values, name="taiwan_gdp_deflator")


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


def _discover_dgbas_xml_url(page_path: Path) -> str:
    """Discover the current DGBAS XML resource from Taiwan's official data portal."""

    parser = _parse_html(page_path.read_text(encoding="utf-8"))
    candidates = [urljoin(DGBAS_DATASET_PAGE, href) for href in parser.links if href.casefold().endswith(".xml")]
    candidates = [url for url in candidates if "dgbas.gov.tw" in url.casefold()]
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise PriceIndexParseError(f"Expected exactly one DGBAS XML link on dataset 44224, got {unique!r}")
    return unique[0]


def _require_years(series: pd.Series, years: Sequence[int], *, name: str) -> pd.Series:
    """Select an exact requested year domain without interpolation or carry-forward."""

    requested = pd.Index(sorted({int(year) for year in years}), name="year")
    missing = [int(year) for year in requested if int(year) not in series.index]
    if missing:
        raise ValueError(f"{name} is missing requested years {missing}")
    return series.reindex(requested).astype(float)


def common_eur_price_factors_from_usd(
    price_indices: pd.DataFrame,
    usd_per_eur: pd.Series,
    *,
    year: int,
    base_year: int,
    area_codes: Sequence[int],
) -> pd.Series:
    """Convert UNSD USD-basis GDP-deflator factors into base-year EUR factors.

    If ``u_t`` is annual-average USD per EUR, the common-EUR macro factor is
    ``(P_usd,b/P_usd,t) * (u_t/u_b)``.  This removes the exchange-rate movement
    embedded in the USD-basis current-price series while retaining the aggregate
    GDP-deflator approximation.
    """

    rates = _require_years(usd_per_eur, [year, base_year], name="ECB USD/EUR")
    factors = price_adjustment_factors(
        price_indices,
        year=int(year),
        base_year=int(base_year),
        currency_basis="usd",
        area_codes=area_codes,
    )
    result = factors * (float(rates.loc[int(year)]) / float(rates.loc[int(base_year)]))
    result.name = "price_adjustment_factor"
    return result


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


def aggregate_country_price_factors_to_exiobase3_regions(
    country_factors: pd.Series,
    current_usd_gdp: pd.Series,
    membership: pd.DataFrame,
    iso3_to_m49: Mapping[str, int],
    *,
    taiwan_factor: float,
) -> pd.DataFrame:
    """Map country factors to all 49 EXIOBASE3 regions, GDP-weighting each RoW bucket.

    Rest-of-World factors use current-USD GDP weights for the same source year.
    Every RoW member must have both a factor and a positive GDP weight; missing
    members fail rather than being dropped and implicitly renormalized.
    """

    required = {"iso3", "exio_region", "region_kind"}
    missing_columns = sorted(required.difference(membership.columns))
    if missing_columns:
        raise ValueError(f"membership is missing columns {missing_columns}")
    rows: list[dict[str, object]] = []
    for region, group in membership.groupby("exio_region", sort=True):
        kind_values = set(group["region_kind"].astype(str))
        if len(kind_values) != 1:
            raise ValueError(f"EXIOBASE region {region!r} mixes country and RoW membership")
        kind = next(iter(kind_values))
        members = tuple(group["iso3"].astype(str))
        if kind == "country":
            if len(members) != 1:
                raise ValueError(f"Explicit EXIOBASE region {region!r} must have one country member")
            iso3 = members[0]
            if iso3 == "TWN":
                factor = float(taiwan_factor)
                provider = "Taiwan DGBAS + CBC + ECB"
            else:
                m49 = iso3_to_m49.get(iso3)
                if m49 is None or m49 not in country_factors.index:
                    raise ValueError(f"Missing UNSD common-EUR factor for explicit EXIOBASE member {iso3}")
                factor = float(country_factors.loc[m49])
                provider = "UNSD AMA + ECB"
        elif kind == "row":
            missing_map = [iso3 for iso3 in members if iso3 not in iso3_to_m49]
            if missing_map:
                raise ValueError(f"RoW region {region} has ISO3 members absent from UN M49 mapping: {missing_map}")
            m49_codes = [int(iso3_to_m49[iso3]) for iso3 in members]
            missing_factor = [iso3 for iso3, m49 in zip(members, m49_codes, strict=True) if m49 not in country_factors.index]
            missing_gdp = [iso3 for iso3, m49 in zip(members, m49_codes, strict=True) if m49 not in current_usd_gdp.index]
            if missing_factor or missing_gdp:
                raise ValueError(
                    f"RoW region {region} has incomplete source coverage; "
                    f"missing_factor={missing_factor}, missing_current_usd_gdp={missing_gdp}"
                )
            weights = current_usd_gdp.reindex(m49_codes).astype(float)
            factors = country_factors.reindex(m49_codes).astype(float)
            if not np.isfinite(weights.to_numpy()).all() or bool((weights <= 0.0).any()):
                raise ValueError(f"RoW region {region} has non-positive or non-finite current-USD GDP weights")
            factor = float(np.average(factors.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))
            provider = "UNSD AMA + ECB; current-USD-GDP weighted"
        else:
            raise ValueError(f"Unsupported EXIOBASE region_kind {kind!r}")
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"Invalid EXIOBASE factor for region {region!r}: {factor!r}")
        rows.append(
            {
                "exio_region": str(region),
                "region_kind": kind,
                "price_adjustment_factor": factor,
                "source_country_count": len(members),
                "source_country_iso3": ";".join(members),
                "source_provider": provider,
            }
        )
    result = pd.DataFrame(rows).sort_values("exio_region", kind="stable").reset_index(drop=True)
    if len(result) != EXPECTED_EXIOBASE3_REGION_COUNT or result["exio_region"].nunique() != EXPECTED_EXIOBASE3_REGION_COUNT:
        raise ValueError(f"Expected exactly 49 EXIOBASE factors, got {len(result)}")
    return result


def build_exiobase3_eur_price_factors(
    *,
    years: Sequence[int],
    base_year: int,
    storage_folder: str | Path | None = None,
    force_download: bool = False,
    timeout_s: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build complete annual EXIOBASE3 factors for a country-GDP-deflator EUR sensitivity.

    The returned table includes the requested years plus ``base_year`` and exactly
    49 EXIOBASE regions per year.  No source member is interpolated or omitted.
    The base-year factors are required to equal one within numerical precision.
    """

    requested_years = tuple(sorted({int(year) for year in years}))
    if not requested_years:
        raise ValueError("years must contain at least one year")
    all_years = tuple(sorted(set(requested_years) | {int(base_year)}))
    root = default_exiobase3_price_factor_storage_folder() if storage_folder is None else Path(storage_folder)
    root.mkdir(parents=True, exist_ok=True)

    # UNSD owns ordinary-country GDP deflators and current-USD GDP weights.  The
    # current-GDP helper reuses the deflator request metadata but keeps a separate
    # small cache so the first public deflator cache contract remains unchanged.
    price_indices, unsd_deflator_meta = load_unsd_ama_gdp_deflators(
        years=all_years,
        country_codes=None,
        storage_folder=root / "unsd_ama",
        force_download=force_download,
        timeout_s=timeout_s,
    )
    current_gdp, unsd_gdp_meta = load_unsd_ama_current_usd_gdp(
        years=all_years,
        country_codes=None,
        storage_folder=root / "unsd_ama",
        force_download=force_download,
        timeout_s=timeout_s,
    )

    # Small authoritative external geography/exchange-rate sources are cached as
    # source text, never as derived EXIOBASE matrices or factor tables.
    ecb_path = _cached_text(ECB_USD_EUR_URL, path=root / "ecb" / "usd_per_eur_annual.csv", force_download=force_download, timeout_s=timeout_s)
    m49_path = _cached_text(UNSD_M49_URL, path=root / "unsd_m49" / "overview.html", force_download=force_download, timeout_s=timeout_s)
    exio_path = _cached_text(EXIOBASE_COUNTRY_COVERAGE_URL, path=root / "exiobase_meta" / f"country_coverage_matrix_{EXIOBASE_META_COMMIT[:12]}.csv", force_download=force_download, timeout_s=timeout_s)
    dgbas_page = _cached_text(DGBAS_DATASET_PAGE, path=root / "taiwan_dgbas" / "dataset_44224.html", force_download=force_download, timeout_s=timeout_s)
    dgbas_xml_url = _discover_dgbas_xml_url(dgbas_page)
    dgbas_xml = _cached_text(dgbas_xml_url, path=root / "taiwan_dgbas" / "gdp_deflator_annual.xml", force_download=force_download, timeout_s=timeout_s)
    cbc_path = _cached_text(CBC_TWD_USD_ANNUAL_URL, path=root / "taiwan_cbc" / "ntd_per_usd_annual.html", force_download=force_download, timeout_s=timeout_s)

    usd_per_eur = _require_years(_read_ecb_usd_per_eur(ecb_path), all_years, name="ECB USD/EUR")
    m49_table = _read_unsd_m49_iso3(m49_path)
    membership = _read_exiobase3_membership(exio_path)
    taiwan_deflator = _require_years(_read_taiwan_dgbas_gdp_deflator(dgbas_xml), all_years, name="Taiwan GDP deflator")
    ntd_per_usd = _require_years(_read_taiwan_ntd_per_usd(cbc_path), all_years, name="Taiwan NTD/USD")

    iso3_to_m49 = dict(zip(
        m49_table["iso3"].astype(str).tolist(),
        m49_table["m49"].astype(int).tolist(),
        strict=True,
    ))
    needed_iso3 = set(membership.loc[membership["iso3"] != "TWN", "iso3"].astype(str))
    missing_m49 = sorted(needed_iso3.difference(iso3_to_m49))
    if missing_m49:
        raise ValueError(f"EXIOBASE source members absent from official UN ISO3/M49 table: {missing_m49}")
    needed_m49 = sorted({iso3_to_m49[iso3] for iso3 in needed_iso3})

    outputs: list[pd.DataFrame] = []
    for year in all_years:
        country_factors = common_eur_price_factors_from_usd(
            price_indices,
            usd_per_eur,
            year=year,
            base_year=int(base_year),
            area_codes=needed_m49,
        )
        gdp_year = current_gdp.loc[current_gdp["year"].astype(int) == year].set_index("m49")["current_usd_gdp"].astype(float)
        if gdp_year.index.duplicated().any():
            raise ValueError(f"UNSD current-USD GDP duplicates M49 rows for year {year}")
        taiwan_factor = _taiwan_common_eur_factor(
            taiwan_deflator,
            ntd_per_usd,
            usd_per_eur,
            year=year,
            base_year=int(base_year),
        )
        region = aggregate_country_price_factors_to_exiobase3_regions(
            country_factors,
            gdp_year,
            membership,
            iso3_to_m49,
            taiwan_factor=taiwan_factor,
        )
        region.insert(0, "year", int(year))
        region.insert(1, "base_year", int(base_year))
        region.insert(4, "factor_basis", "country_gdp_deflator_common_eur")
        outputs.append(region)

    result = pd.concat(outputs, ignore_index=True)
    base = result.loc[result["year"] == int(base_year), "price_adjustment_factor"].to_numpy(dtype=float)
    if len(base) != EXPECTED_EXIOBASE3_REGION_COUNT or not np.allclose(base, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("EXIOBASE common-EUR base-year factors must equal one for all 49 regions")
    expected_rows = len(all_years) * EXPECTED_EXIOBASE3_REGION_COUNT
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} year-region factor rows, got {len(result)}")

    metadata: dict[str, Any] = {
        "factor_basis": "country_gdp_deflator_common_eur",
        "requested_years": list(requested_years),
        "returned_years": list(all_years),
        "base_year": int(base_year),
        "region_count": EXPECTED_EXIOBASE3_REGION_COUNT,
        "unsd_deflator_metadata": unsd_deflator_meta,
        "unsd_current_usd_gdp_metadata": unsd_gdp_meta,
        "ecb_series": ECB_USD_EUR_SERIES,
        "un_m49_url": UNSD_M49_URL,
        "exiobase_meta_commit": EXIOBASE_META_COMMIT,
        "exiobase_country_coverage_url": EXIOBASE_COUNTRY_COVERAGE_URL,
        "taiwan_dgbas_dataset": DGBAS_DATASET_PAGE,
        "taiwan_dgbas_resource": dgbas_xml_url,
        "taiwan_cbc_exchange_rate": CBC_TWD_USD_ANNUAL_URL,
        "interpretation": "country-level macro GDP-deflator sensitivity; not product-country constant-price EXIOBASE",
    }
    return result.sort_values(["year", "exio_region"], kind="stable").reset_index(drop=True), metadata

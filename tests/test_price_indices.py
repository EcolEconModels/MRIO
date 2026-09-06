"""Regression tests for official macro price-index acquisition and rebasing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from ecolecon_mrio import price_indices


def _fake_unsd_http_factory() -> tuple[Callable[..., Any], list[tuple[str, object]]]:
    """Return an AMA-schema fake that protects source discovery and cache semantics."""

    calls: list[tuple[str, object]] = []

    def fake_http(url: str, *, payload=None, timeout_s: int = 120):
        calls.append((url, payload))
        if url.endswith("/Country?countriesOnly=true"):
            return [
                {"countryCode": 40, "countryName": "Austria", "currencyName": "Euro", "isCountry": True},
                {"countryCode": 840, "countryName": "United States", "currencyName": "US Dollar", "isCountry": True},
            ]
        if url.endswith("/Series"):
            return [
                {
                    "serieCode": 7,
                    "serieName": "Gross Domestic Product (GDP), Per capita GDP, GDP growth rate, Implicite Price Deflators",
                    "unitMeasureType": "Mixed",
                },
                {"serieCode": 8, "serieName": "GDP by kind of Expenditure"},
            ]
        if url.endswith("/Data/basic/7"):
            assert payload == {"paramCodes": [40, 840], "years": [2020, 2022]}
            rows = []
            values = {
                40: {2020: (100.0, 100.0), 2022: (118.0, 104.0)},
                840: {2020: (100.0, 100.0), 2022: (116.0, 116.0)},
            }
            names = {40: "Austria", 840: "United States"}
            currencies = {40: "Euro", 840: "US Dollar"}
            for code, by_year in values.items():
                for year, (national, usd) in by_year.items():
                    rows.append(
                        {
                            "countryCode": code,
                            "countryName": names[code],
                            "fiscalYear": year,
                            "serieCode": 7,
                            "serieName": "Gross Domestic Product (GDP), Per capita GDP, GDP growth rate, Implicite Price Deflators",
                            "itemId": 70,
                            "itemName": "GDP - Implicit Price Deflators - National currency",
                            "observationValue": national,
                            "unit": currencies[code],
                        }
                    )
                    rows.append(
                        {
                            "countryCode": code,
                            "countryName": names[code],
                            "fiscalYear": year,
                            "serieCode": 7,
                            "serieName": "Gross Domestic Product (GDP), Per capita GDP, GDP growth rate, Implicite Price Deflators",
                            "itemId": 71,
                            "itemName": "GDP - Implicit Price Deflators - US Dollars",
                            "observationValue": usd,
                            "unit": "US Dollars",
                        }
                    )
            return rows
        if url.endswith("/Data/lastupdated"):
            return "2026-01-15"
        raise AssertionError(f"Unexpected fake UNSD URL: {url}")

    return fake_http, calls


def test_unsd_download_parse_and_cache_are_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Official metadata discovery must select deflators, classify bases, and reuse the raw cache."""

    fake_http, calls = _fake_unsd_http_factory()
    monkeypatch.setattr(price_indices, "_http_json", fake_http)

    raw_path = price_indices.download_unsd_ama_gdp_deflators(
        years=[2022, 2020],
        storage_folder=tmp_path,
    )
    assert raw_path.exists()
    frame, metadata = price_indices.read_unsd_ama_gdp_deflators(raw_path)
    assert len(frame) == 8
    assert set(frame["currency_basis"]) == {"national_currency", "usd"}
    assert metadata["provider"] == "UNSD AMA"
    assert metadata["data_last_updated"] == "2026-01-15"

    # A repeated identical request must be cache-only and make no source calls.
    first_call_count = len(calls)
    again = price_indices.download_unsd_ama_gdp_deflators(years=[2020, 2022], storage_folder=tmp_path)
    assert again == raw_path
    assert len(calls) == first_call_count

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    assert payload["request"]["country_codes"] == "all"
    assert payload["request"]["years"] == [2020, 2022]


def test_price_adjustment_factors_rebase_selected_basis_and_require_requested_coverage() -> None:
    """Factors must equal P_base/P_year and fail when an explicitly requested area is missing."""

    frame = pd.DataFrame(
        [
            {"m49": 40, "year": 2020, "currency_basis": "national_currency", "price_index": 100.0},
            {"m49": 40, "year": 2022, "currency_basis": "national_currency", "price_index": 118.0},
            {"m49": 840, "year": 2020, "currency_basis": "national_currency", "price_index": 100.0},
            {"m49": 840, "year": 2022, "currency_basis": "national_currency", "price_index": 116.0},
            {"m49": 40, "year": 2020, "currency_basis": "usd", "price_index": 100.0},
            {"m49": 40, "year": 2022, "currency_basis": "usd", "price_index": 104.0},
        ]
    )
    factors = price_indices.price_adjustment_factors(
        frame,
        year=2020,
        base_year=2022,
        currency_basis="national_currency",
        area_codes=[40, 840],
    )
    assert factors.loc[40] == pytest.approx(1.18)
    assert factors.loc[840] == pytest.approx(1.16)

    with pytest.raises(ValueError, match="Missing price indices"):
        price_indices.price_adjustment_factors(
            frame,
            year=2020,
            base_year=2022,
            currency_basis="usd",
            area_codes=[40, 840],
        )


def test_unsd_parser_rejects_unclassified_currency_basis(tmp_path: Path) -> None:
    """Ambiguous source rows must fail rather than be assigned by row position."""

    raw = {
        "provider": "UNSD AMA",
        "countries": [{"countryCode": 40, "countryName": "Austria", "currencyName": "Euro"}],
        "records": [
            {
                "countryCode": 40,
                "countryName": "Austria",
                "fiscalYear": 2022,
                "serieCode": 7,
                "serieName": "Implicit Price Deflators",
                "itemId": 70,
                "itemName": "GDP deflator",
                "observationValue": 118.0,
                "unit": "Index",
            }
        ],
    }
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(price_indices.PriceIndexParseError, match="Cannot classify"):
        price_indices.read_unsd_ama_gdp_deflators(path)

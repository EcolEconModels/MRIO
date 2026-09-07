"""Scientific regressions for complete EXIOBASE3 common-EUR macro price factors."""
from __future__ import annotations

from collections.abc import Callable
import copy
import json
from pathlib import Path
import string
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ecolecon_mrio import exiobase_price_factors as factors


def _valid_iso3_codes(count: int) -> list[str]:
    codes: list[str] = []
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            for c in string.ascii_uppercase:
                token = f"{a}{b}{c}"
                if token != "TWN":
                    codes.append(token)
                    if len(codes) == count:
                        return codes
    raise AssertionError("not enough synthetic ISO3 codes")


def _valid_region_codes(count: int) -> list[str]:
    excluded = {"TW", "WA", "WL", "WE", "WF", "WM"}
    codes = [f"{a}{b}" for a in string.ascii_uppercase for b in string.ascii_uppercase if f"{a}{b}" not in excluded]
    return codes[:count]


def _write_source_fixture(tmp_path: Path) -> tuple[dict[str, Path], dict[str, int]]:
    ordinary = _valid_iso3_codes(48)
    explicit_iso = ordinary[:43]
    row_iso = ordinary[43:48]
    explicit_regions = _valid_region_codes(43)
    iso_to_m49 = {iso: pos + 1 for pos, iso in enumerate(ordinary)}

    exio_rows = ["ISO3,EXIO3,EXIO3_3L"]
    for iso, region in zip(explicit_iso, explicit_regions, strict=True):
        exio_rows.append(f"{iso},{region},{iso}")
    exio_rows.append("TWN,TW,TWN")
    for iso, legacy in zip(row_iso, ("WWA", "WWL", "WWE", "WWF", "WWM"), strict=True):
        exio_rows.append(f"{iso},,{legacy}")
    exio_path = tmp_path / "exio.csv"
    exio_path.write_text("\n".join(exio_rows) + "\n", encoding="utf-8")

    # The production parser requires a realistically sized official table.  Add
    # irrelevant dummy rows so this synthetic source also exercises that guard.
    m49_rows = [
        "<table><tr><th>Country or Area</th><th>M49 Code</th><th>ISO-alpha3 Code</th></tr>"
    ]
    for iso, m49 in iso_to_m49.items():
        m49_rows.append(f"<tr><td>{iso}</td><td>{m49:03d}</td><td>{iso}</td></tr>")
    extra = _valid_iso3_codes(220)
    used = set(ordinary)
    next_code = 300
    for iso in extra:
        if iso in used:
            continue
        m49_rows.append(f"<tr><td>{iso}</td><td>{next_code:03d}</td><td>{iso}</td></tr>")
        next_code += 1
        if next_code >= 430:
            break
    m49_rows.append("</table>")
    m49_path = tmp_path / "m49.html"
    m49_path.write_text("".join(m49_rows), encoding="utf-8")

    ecb_path = tmp_path / "ecb.csv"
    ecb_path.write_text(
        "KEY,TIME_PERIOD,OBS_VALUE\n"
        "EXR.A.USD.EUR.SP00.A,2020,1.14\n"
        "EXR.A.USD.EUR.SP00.A,2022,1.05\n",
        encoding="utf-8",
    )
    cbc_path = tmp_path / "cbc.html"
    cbc_path.write_text(
        "<table><tr><th>Period Average</th><th>NTD/USD</th></tr>"
        "<tr><td>2020</td><td>29.578</td></tr><tr><td>2022</td><td>29.777</td></tr></table>",
        encoding="utf-8",
    )
    return {
        "exio": exio_path,
        "m49": m49_path,
        "ecb": ecb_path,
        "cbc": cbc_path,
    }, iso_to_m49


def test_common_eur_formula_removes_usd_eur_movement() -> None:
    """USD-basis deflation and ECB conversion must implement phi_usd*u_t/u_b."""

    table = pd.DataFrame(
        [
            {"m49": 40, "year": 2020, "currency_basis": "usd", "price_index": 100.0},
            {"m49": 40, "year": 2022, "currency_basis": "usd", "price_index": 110.0},
        ]
    )
    rates = pd.Series({2020: 1.14, 2022: 1.05})
    result = factors.common_eur_price_factors_from_usd(
        table, rates, year=2020, base_year=2022, area_codes=[40]
    )
    assert result.loc[40] == pytest.approx(1.10 * 1.14 / 1.05)


def test_row_factor_is_current_usd_gdp_weighted_not_unweighted() -> None:
    """A RoW factor must represent the aggregate monetary row, not an average index number."""

    membership_rows = [
        {"iso3": "AAA", "exio_region": "ZZ", "region_kind": "country"},
        {"iso3": "TWN", "exio_region": "TW", "region_kind": "country"},
        {"iso3": "BBB", "exio_region": "WA", "region_kind": "row"},
        {"iso3": "CCC", "exio_region": "WA", "region_kind": "row"},
    ]
    mapping = {"AAA": 1, "BBB": 2, "CCC": 3}
    country_factors = {1: 1.0, 2: 0.5, 3: 1.5}
    gdp = {2: 90.0, 3: 10.0}
    # Pad the synthetic table to the public 49-region contract.
    for pos, region in enumerate(_valid_region_codes(42), start=10):
        iso = _valid_iso3_codes(100)[pos]
        membership_rows.append({"iso3": iso, "exio_region": region, "region_kind": "country"})
        mapping[iso] = 1000 + pos
        country_factors[1000 + pos] = 1.0
    for pos, (region, iso) in enumerate(zip(("WL", "WE", "WF", "WM"), ("DDD", "EEE", "FFF", "GGG"), strict=True), start=2000):
        membership_rows.append({"iso3": iso, "exio_region": region, "region_kind": "row"})
        mapping[iso] = pos
        country_factors[pos] = 1.0
        gdp[pos] = 1.0
    result = factors.aggregate_country_price_factors_to_exiobase3_regions(
        pd.Series(country_factors),
        pd.Series(gdp),
        pd.DataFrame(membership_rows),
        mapping,
        year=2022,
        source_policy=factors._load_price_factor_source_policy(),
        taiwan_factor=1.0,
    )
    row_factor = float(result.loc[result["exio_region"] == "WA", "price_adjustment_factor"].iloc[0])
    assert row_factor == pytest.approx(0.6)
    assert row_factor != pytest.approx(1.0)


def test_row_factor_fails_if_one_member_lacks_factor_or_gdp() -> None:
    """RoW aggregation must not silently drop a member and renormalize the weights."""

    membership = pd.DataFrame(
        [
            {"iso3": iso, "exio_region": region, "region_kind": "country"}
            for iso, region in zip(_valid_iso3_codes(44)[:43], _valid_region_codes(43), strict=True)
        ]
        + [{"iso3": "TWN", "exio_region": "TW", "region_kind": "country"}]
        + [
            {"iso3": iso, "exio_region": region, "region_kind": "row"}
            for iso, region in zip(("BBB", "CCC", "DDD", "EEE", "FFF"), ("WA", "WL", "WE", "WF", "WM"), strict=True)
        ]
    )
    all_iso = [iso for iso in membership["iso3"] if iso != "TWN"]
    mapping = {iso: pos + 1 for pos, iso in enumerate(all_iso)}
    country = pd.Series({code: 1.0 for code in mapping.values()})
    gdp = pd.Series({code: 1.0 for code in mapping.values()})
    missing_code = mapping["BBB"]
    gdp = gdp.drop(missing_code)
    with pytest.raises(ValueError, match="incomplete source coverage"):
        factors.aggregate_country_price_factors_to_exiobase3_regions(
            country,
            gdp,
            membership,
            mapping,
            year=2022,
            source_policy=factors._load_price_factor_source_policy(),
            taiwan_factor=1.0,
        )


def _snapshot_payload() -> dict[str, Any]:
    """Return an independently mutable valid fixture for snapshot validation tests."""

    observations = [
        {"year": year, "gdp_deflator": float(year - 1900)}
        for year in factors.PINNED_TAIWAN_DGBAS_YEARS
    ]
    return {
        "provider": "DGBAS",
        "country": "Taiwan",
        "dgbas_table": factors.DGBAS_TABLE_ID,
        "measure": "fldid=3",
        "item": "code1=14",
        "frequency": "annual",
        "reference_year": 2021,
        "observations": observations,
    }


def _write_snapshot(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "taiwan_dgbas_snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_taiwan_pinned_dgbas_snapshot_has_audited_official_annual_series() -> None:
    """Taiwan uses the audited official DGBAS annual GDP-deflator source, not China data."""

    series, provenance = factors._load_taiwan_dgbas_gdp_deflator(range(2012, 2023))
    assert len(series) == 11
    assert series.loc[2012] == pytest.approx(90.99)
    assert series.loc[2021] == pytest.approx(100.00)
    assert series.loc[2022] == pytest.approx(102.07)
    assert np.isfinite(series.to_numpy(dtype=float)).all()
    assert bool((series > 0.0).all())
    assert provenance["provider"] == "DGBAS"
    assert provenance["dgbas_table"] == "A018102050"
    assert provenance["snapshot_resource"] == factors.DGBAS_SNAPSHOT_RESOURCE


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda payload: payload["observations"].append(copy.deepcopy(payload["observations"][0])), "duplicates year"),
        (lambda payload: payload["observations"].pop(), "cover exactly 2012--2022"),
        (lambda payload: payload["observations"].__setitem__(0, {"year": 2012, "gdp_deflator": float("nan")}), "positive finite"),
        (lambda payload: payload["observations"].__setitem__(0, {"year": 2012, "gdp_deflator": 0.0}), "positive finite"),
    ],
)
def test_taiwan_pinned_snapshot_rejects_malformed_observations(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], error: str
) -> None:
    """Duplicate, incomplete, non-finite, and non-positive snapshot values fail closed."""

    payload = _snapshot_payload()
    mutate(payload)
    with pytest.raises(factors.PriceIndexParseError, match=error):
        factors._read_taiwan_dgbas_gdp_deflator_snapshot(_write_snapshot(tmp_path, payload))


def test_taiwan_pinned_snapshot_rejects_outside_audited_years() -> None:
    with pytest.raises(ValueError, match="Pinned DGBAS Taiwan GDP-deflator snapshot covers 2012--2022 only"):
        factors._load_taiwan_dgbas_gdp_deflator([2011, 2022])


def test_taiwan_pinned_snapshot_load_never_performs_runtime_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_http(*args: object, **kwargs: object) -> str:
        raise AssertionError("the packaged Taiwan snapshot must not make an HTTP request")

    monkeypatch.setattr(factors, "_http_text", forbidden_http)
    series, _ = factors._load_taiwan_dgbas_gdp_deflator([2020, 2022])
    assert series.index.tolist() == [2020, 2022]


def test_taiwan_common_eur_formula_and_base_year_identity_are_unchanged() -> None:
    deflator = pd.Series({2020: 98.15, 2022: 102.07})
    ntd_per_usd = pd.Series({2020: 29.578, 2022: 29.777})
    usd_per_eur = pd.Series({2020: 1.14, 2022: 1.05})
    result = factors._taiwan_common_eur_factor(
        deflator, ntd_per_usd, usd_per_eur, year=2020, base_year=2022
    )
    expected = (102.07 / 98.15) * ((29.578 * 1.14) / (29.777 * 1.05))
    assert result == pytest.approx(expected)
    assert factors._taiwan_common_eur_factor(
        deflator, ntd_per_usd, usd_per_eur, year=2022, base_year=2022
    ) == 1.0


def test_complete_builder_returns_exact_49_region_base_year_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The production builder must be complete on 44 countries + 5 RoW and exact at the base year."""

    sources, iso_to_m49 = _write_source_fixture(tmp_path)
    all_m49 = sorted(iso_to_m49.values())
    price_rows: list[dict[str, object]] = []
    gdp_rows: list[dict[str, object]] = []
    for m49 in all_m49:
        price_rows.extend(
            [
                {"m49": m49, "year": 2020, "currency_basis": "usd", "price_index": 100.0},
                {"m49": m49, "year": 2022, "currency_basis": "usd", "price_index": 110.0},
            ]
        )
        gdp_rows.extend(
            [
                {"m49": m49, "year": 2020, "current_usd_gdp": float(100 + m49)},
                {"m49": m49, "year": 2022, "current_usd_gdp": float(110 + m49)},
            ]
        )
    price_table = pd.DataFrame(price_rows)
    gdp_table = pd.DataFrame(gdp_rows)
    monkeypatch.setattr(factors, "load_unsd_ama_gdp_deflators", lambda **_: (price_table, {"provider": "fake"}))
    monkeypatch.setattr(factors, "load_unsd_ama_current_usd_gdp", lambda **_: (gdp_table, {"provider": "fake", "fiscal_year_notes": []}))

    def fake_cached(url: str, *, path: Path, force_download: bool, timeout_s: int) -> Path:
        del path, force_download, timeout_s
        if url == factors.ECB_USD_EUR_URL:
            return sources["ecb"]
        if url == factors.UNSD_M49_URL:
            return sources["m49"]
        if url == factors.EXIOBASE_COUNTRY_COVERAGE_URL:
            return sources["exio"]
        if url == factors.CBC_TWD_USD_ANNUAL_URL:
            return sources["cbc"]
        raise AssertionError(url)

    monkeypatch.setattr(factors, "_cached_text", fake_cached)
    result, metadata = factors.build_exiobase3_eur_price_factors(
        years=[2020], base_year=2022, storage_folder=tmp_path / "cache"
    )
    assert len(result) == 98
    assert result.groupby("year")["exio_region"].nunique().to_dict() == {2020: 49, 2022: 49}
    assert __import__("numpy").allclose(result.loc[result["year"] == 2022, "price_adjustment_factor"].to_numpy(dtype=float), 1.0)
    assert set(result["factor_basis"]) == {"country_gdp_deflator_common_eur"}
    assert metadata["exiobase_meta_commit"] == factors.EXIOBASE_META_COMMIT
    assert metadata["region_count"] == 49



def test_price_factor_source_policy_exact_rule_overrides_wildcard() -> None:
    """Country-specific source choice is mapping data, with exact rows overriding the default."""

    policy = factors._load_price_factor_source_policy()
    default = factors._resolve_price_factor_source_policy(policy, exio_region="AT", year=2022)
    taiwan = factors._resolve_price_factor_source_policy(policy, exio_region="TW", year=2022)
    assert default["target_code"] == factors.PRICE_FACTOR_METHOD_UNSD
    assert taiwan["target_code"] == factors.PRICE_FACTOR_METHOD_TAIWAN


def test_historical_source_identity_mapping_records_resolved_and_unresolved_evidence() -> None:
    """Provider identities are data rows; unresolved historical geography stays disabled."""

    mapping = factors._load_price_factor_source_identity_mapping()
    by_source = mapping.set_index("source_code")
    assert str(by_source.loc["Tanzania", "provider_code"]) == "835"
    assert str(by_source.loc["Zanzibar", "provider_code"]) == "836"
    antilles = by_source.loc["Netherlands Antilles"]
    assert antilles["resolution_status"] == "unresolved"
    assert not bool(antilles["enabled"])


def test_unresolved_historical_identity_cannot_be_enabled(tmp_path: Path) -> None:
    """A mapping edit cannot turn unresolved scientific evidence into an executable rule."""

    mapping = factors._load_price_factor_source_identity_mapping()
    mapping.loc[mapping["source_code"] == "Netherlands Antilles", "enabled"] = True
    path = tmp_path / "identity.csv"
    mapping.to_csv(path, index=False)
    with pytest.raises(factors.PriceIndexParseError, match="unresolved.*cannot be enabled"):
        factors._read_price_factor_source_identity_mapping(path)


def test_unsd_current_usd_gdp_support_reuses_discovered_series_and_keeps_fiscal_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RoW weights must come from the same discovered UNSD GDP series with fiscal provenance retained."""

    from ecolecon_mrio import price_indices

    deflator = tmp_path / "deflator.json"
    deflator.write_text(
        __import__("json").dumps(
            {
                "countries": [{"countryCode": 40, "countryName": "Austria"}],
                "series": [{"serieCode": 7, "serieName": "GDP and Implicit Price Deflators"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(price_indices, "download_unsd_ama_gdp_deflators", lambda **_: deflator)
    calls: list[str] = []

    def fake_http(url: str, *, payload=None, timeout_s: int = 120):
        del timeout_s
        calls.append(url)
        if url.endswith("/Data/basic/7"):
            assert payload == {"paramCodes": [40], "years": [2020, 2022]}
            return [
                {
                    "countryCode": 40,
                    "countryName": "Austria",
                    "fiscalYear": year,
                    "serieCode": 7,
                    "serieName": "GDP and Implicit Price Deflators",
                    "itemId": 1,
                    "itemName": "GDP at current prices - US Dollars",
                    "observationValue": value,
                    "unit": "Million US Dollars",
                }
                for year, value in ((2020, 430000.0), (2022, 470000.0))
            ]
        if url.endswith("/Metadata/fiscalyearnotes"):
            return [{"countryCode": 356, "note": "Fiscal year starts 1 April"}]
        raise AssertionError(url)

    monkeypatch.setattr(price_indices, "_http_json", fake_http)
    path = price_indices.download_unsd_ama_current_usd_gdp(
        years=[2022, 2020], country_codes=[40], storage_folder=tmp_path / "cache"
    )
    frame, metadata = price_indices.read_unsd_ama_current_usd_gdp(path)
    assert frame[["year", "current_usd_gdp"]].values.tolist() == [[2020.0, 430000.0], [2022.0, 470000.0]]
    assert metadata["fiscal_year_notes"] == [{"countryCode": 356, "note": "Fiscal year starts 1 April"}]
    assert any(url.endswith("/Metadata/fiscalyearnotes") for url in calls)


def test_price_adjustment_rejects_negative_output_even_when_called_directly() -> None:
    """The public transformation must preserve the MRIO non-negative gross-output invariant."""

    from ecolecon_mrio.price_adjustment import adjust_mrio_iot_prices

    index = pd.MultiIndex.from_tuples([("AA", "s1")], names=["region", "sector"])
    x = pd.Series([-1.0], index=index, name="x")
    row_factors = pd.Series([1.1], index=index, name="price_adjustment_factor")
    with pytest.raises(ValueError, match="x must be non-negative"):
        adjust_mrio_iot_prices({"x": x}, basis_index=index, row_factors=row_factors)

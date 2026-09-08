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

def test_row_factor_sums_current_and_constant_levels_before_deflation() -> None:
    """RoW implicit deflation must aggregate extensive GDP levels before forming C/K."""

    levels = pd.DataFrame(
        [
            {"m49": 1, "year": 2020, "current_usd_gdp": 90.0, "constant_usd_gdp": 100.0},
            {"m49": 2, "year": 2020, "current_usd_gdp": 10.0, "constant_usd_gdp": 5.0},
            {"m49": 1, "year": 2022, "current_usd_gdp": 120.0, "constant_usd_gdp": 100.0},
            {"m49": 2, "year": 2022, "current_usd_gdp": 30.0, "constant_usd_gdp": 10.0},
        ]
    )
    rates = pd.Series({2020: 1.0, 2022: 1.0})
    factor = factors._common_eur_factor_from_usd_levels(
        levels, rates, m49_codes=[1, 2], year=2020, base_year=2022
    )
    expected = ((150.0 / 110.0) / (100.0 / 105.0))
    individual = np.array([(120.0 / 100.0) / (90.0 / 100.0), (30.0 / 10.0) / (10.0 / 5.0)])
    old_weighted = float(np.average(individual, weights=np.array([90.0, 10.0])))
    assert factor == pytest.approx(expected)
    assert factor != pytest.approx(old_weighted)


def test_row_factor_fails_if_one_required_level_is_missing() -> None:
    """Missing constant/current GDP for one member must fail rather than drop and renormalize it."""

    levels = pd.DataFrame(
        [
            {"m49": 1, "year": 2020, "current_usd_gdp": 90.0, "constant_usd_gdp": 100.0},
            {"m49": 2, "year": 2020, "current_usd_gdp": 10.0, "constant_usd_gdp": np.nan},
            {"m49": 1, "year": 2022, "current_usd_gdp": 120.0, "constant_usd_gdp": 100.0},
            {"m49": 2, "year": 2022, "current_usd_gdp": 30.0, "constant_usd_gdp": 10.0},
        ]
    )
    with pytest.raises(ValueError, match="missing/non-positive/non-finite"):
        factors._common_eur_factor_from_usd_levels(
            levels, pd.Series({2020: 1.0, 2022: 1.0}), m49_codes=[1, 2], year=2020, base_year=2022
        )


def test_published_bridge_and_effective_antilles_identity_are_data_owned() -> None:
    """The packaged bridge is 214->49; ANT is an inactive WL alias while CUW/SXM remain distinct."""

    bridge = factors._load_exiobase3rx_region_bridge()
    assert len(bridge) == 214
    assert bridge["target_code"].nunique() == 49
    counts = bridge.loc[
        bridge["target_code"].isin(list(factors.EXPECTED_ROW_REGIONS))
    ].groupby("target_code").size().to_dict()
    assert counts == {"WA": 45, "WL": 43, "WE": 14, "WF": 53, "WM": 15}
    assert bridge.loc[bridge["source_code"].isin(["CUW", "SXM"]), "target_code"].tolist().count("WL") == 2
    ant_bridge = bridge.loc[bridge["source_label"] == "Netherlands Antilles"]
    assert len(ant_bridge) == 1
    assert ant_bridge.iloc[0]["source_code"] == "NETHERLANDS ANTILLES"
    assert ant_bridge.iloc[0]["target_code"] == "WL"
    assert "BES" not in set(bridge["source_code"])
    assert factors._mapping_resource_sha256(factors.EXIOBASE3RX_BRIDGE_RESOURCE) == "92e9b95d240b338c0a6a8c06f2c4768149d81092839ef4581d780a1ce119f972"

    identity = factors._load_price_factor_source_identity_mapping()
    ant = identity.loc[identity["source_code"] == "NETHERLANDS ANTILLES"]
    assert len(ant) == 1
    assert ant.iloc[0]["resolution_status"] == "inactive"
    assert int(ant.iloc[0]["start_year"]) == 2012
    assert int(ant.iloc[0]["end_year"]) == 2022


def test_pinned_00_class_crosswalk_is_explicit_country_only(tmp_path: Path) -> None:
    """The pinned 00-class parser returns exactly 44 explicit runtime-code mappings."""

    sources, _ = _write_source_fixture(tmp_path)
    crosswalk = factors._read_exiobase3_explicit_region_crosswalk(sources["exio"])
    assert len(crosswalk) == 44
    assert crosswalk["iso3"].nunique() == 44
    assert crosswalk["exio_region"].nunique() == 44
    assert crosswalk.loc[crosswalk["iso3"].eq("TWN"), "exio_region"].tolist() == ["TW"]


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
    """The production builder compiles published ISO3 targets to runtime EXIO codes."""

    sources, iso_to_m49 = _write_source_fixture(tmp_path)
    ordinary = list(iso_to_m49)
    explicit_iso = ordinary[:43]
    row_iso = ordinary[43:48]
    bridge_rows = [
        {"source_code": iso, "source_label": iso, "source_iso3": iso, "target_code": iso, "target_label": iso}
        for iso in explicit_iso
    ]
    bridge_rows.append(
        {"source_code": "TWN", "source_label": "Taiwan", "source_iso3": "TWN", "target_code": "TWN", "target_label": "TWN"}
    )
    bridge_rows.extend(
        {"source_code": iso, "source_label": iso, "source_iso3": iso, "target_code": region, "target_label": region}
        for iso, region in zip(row_iso, ("WA", "WL", "WE", "WF", "WM"), strict=True)
    )
    bridge = pd.DataFrame(bridge_rows)
    empty_identity = pd.DataFrame(
        columns=list(factors.PRICE_FACTOR_SOURCE_IDENTITY_COLUMNS)  # pyright: ignore[reportArgumentType]
    )
    monkeypatch.setattr(factors, "_load_exiobase3rx_region_bridge", lambda: bridge)
    monkeypatch.setattr(factors, "_load_price_factor_source_identity_mapping", lambda: empty_identity)

    level_rows: list[dict[str, object]] = []
    for m49 in sorted(iso_to_m49.values()):
        level_rows.extend(
            [
                {"m49": m49, "year": 2020, "current_usd_gdp": float(100 + m49), "constant_usd_gdp": float(90 + m49)},
                {"m49": m49, "year": 2022, "current_usd_gdp": float(110 + m49), "constant_usd_gdp": float(95 + m49)},
            ]
        )
    level_table = pd.DataFrame(level_rows)
    raw_levels = tmp_path / "gdp_levels_source.json"
    raw_levels.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        factors,
        "load_unsd_ama_gdp_usd_levels",
        lambda **_: (
            level_table,
            {"provider": "fake", "fiscal_year_notes": [], "raw_path": str(raw_levels)},
        ),
    )

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
    assert all(len(str(code)) == 2 for code in result["exio_region"])
    assert "TW" in set(result["exio_region"])
    assert "TWN" not in set(result["exio_region"])
    assert np.allclose(
        result.loc[result["year"] == 2022, "price_adjustment_factor"].to_numpy(dtype=float),
        1.0,
    )
    assert set(result["factor_basis"]) == {"country_gdp_deflator_common_eur"}
    assert metadata["region_count"] == 49
    assert metadata["exiobase_00_class_role"].endswith("not RoW membership")
    assert metadata["factor_table_sha256"]
    assert metadata["source_sha256"]["exiobase_00_class_explicit_crosswalk"]





def test_price_factor_source_policy_exact_rule_overrides_wildcard() -> None:
    """Country-specific source choice is mapping data, with exact rows overriding the default."""

    policy = factors._load_price_factor_source_policy()
    default = factors._resolve_price_factor_source_policy(policy, exio_region="AT", year=2022)
    taiwan = factors._resolve_price_factor_source_policy(policy, exio_region="TW", year=2022)
    assert default["target_code"] == factors.PRICE_FACTOR_METHOD_UNSD
    assert taiwan["target_code"] == factors.PRICE_FACTOR_METHOD_TAIWAN


def test_source_identity_mapping_resolves_tanzania_zanzibar_and_inactivates_ant() -> None:
    """Provider statistical-unit decisions and historical alias inactivity live in the mapping CSV."""

    mapping = factors._load_price_factor_source_identity_mapping()
    tanzania = mapping.loc[mapping["source_code"] == "TZA"].iloc[0]
    assert str(tanzania["provider_code"]) == "835"
    zanzibar = mapping.loc[mapping["provider_code"].astype(str) == "836"].iloc[0]
    assert str(zanzibar["target_code"]) == "WF"
    ant = mapping.loc[mapping["source_code"] == "NETHERLANDS ANTILLES"].iloc[0]
    assert ant["resolution_status"] == "inactive"
    assert str(ant["provider_code"]) == ""
    assert str(ant["target_code"]) == "WL"


def test_effective_row_members_drop_ant_without_dropping_cuw_or_sxm() -> None:
    """The 2012-2022 identity normalization removes only the duplicate ANT alias."""

    bridge = pd.DataFrame(
        [
            {"source_code": "NETHERLANDS ANTILLES", "source_label": "Netherlands Antilles", "target_code": "WL"},
            {"source_code": "CUW", "source_label": "Curacao", "target_code": "WL"},
            {"source_code": "SXM", "source_label": "Sint Maarten", "target_code": "WL"},
        ]
    )
    identity = factors._load_price_factor_source_identity_mapping()
    result = factors._effective_row_members(bridge, identity, {"CUW": 531, "SXM": 534}, year=2012)
    assert set(result["source_code"]) == {"CUW", "SXM"}
    assert set(result["m49"]) == {531, 534}


def test_unsd_gdp_level_support_reuses_discovered_series_and_keeps_fiscal_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Current and constant USD GDP levels come from one discovered UNSD series with fiscal provenance."""

    from ecolecon_mrio import price_indices

    deflator = tmp_path / "deflator.json"
    deflator.write_text(
        json.dumps(
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
            rows = []
            for year, current, constant in ((2020, 430000.0, 410000.0), (2022, 470000.0, 430000.0)):
                rows.extend(
                    [
                        {
                            "countryCode": 40, "countryName": "Austria", "fiscalYear": year,
                            "serieCode": 7, "serieName": "GDP and Implicit Price Deflators",
                            "itemId": 1, "itemName": "GDP at current prices - US Dollars",
                            "observationValue": current, "unit": "Million US Dollars",
                        },
                        {
                            "countryCode": 40, "countryName": "Austria", "fiscalYear": year,
                            "serieCode": 7, "serieName": "GDP and Implicit Price Deflators",
                            "itemId": 2, "itemName": "GDP at constant prices - US Dollars",
                            "observationValue": constant, "unit": "Million US Dollars",
                        },
                    ]
                )
            return rows
        if url.endswith("/Metadata/fiscalyearnotes"):
            return [{"countryCode": 356, "note": "Fiscal year starts 1 April"}]
        raise AssertionError(url)

    monkeypatch.setattr(price_indices, "_http_json", fake_http)
    path = price_indices.download_unsd_ama_gdp_usd_levels(
        years=[2022, 2020], country_codes=[40], storage_folder=tmp_path / "cache"
    )
    frame, metadata = price_indices.read_unsd_ama_gdp_usd_levels(path)
    assert frame[["year", "current_usd_gdp", "constant_usd_gdp"]].values.tolist() == [
        [2020.0, 430000.0, 410000.0], [2022.0, 470000.0, 430000.0]
    ]
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

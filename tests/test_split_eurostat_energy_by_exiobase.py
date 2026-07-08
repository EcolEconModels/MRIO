from __future__ import annotations

import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest


def test_mapping_integrity_check_passes():
    from scripts.split_eurostat_energy_by_exiobase import _assert_mapping_integrity

    euro_Z = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["CPA_A", "CPA_B"], columns=["CPA_A", "CPA_B"])
    euro_Y = pd.DataFrame({"FD": [100.0, 100.0]}, index=["CPA_A", "CPA_B"])

    mapping = {"CPA_A": {"S1": 1.0}, "CPA_B": {"S2": 1.0}}

    _assert_mapping_integrity(
        sector_mapping=mapping,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        label="test",
    )


def test_output_consistency_check_uses_destination_use_not_x_region():
    from types import SimpleNamespace
    from scripts.split_eurostat_energy_by_exiobase import _assert_output_consistency

    euro_Z = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["CPA_A", "CPA_B"], columns=["CPA_A", "CPA_B"])
    euro_Y = pd.DataFrame({"P3": [50.0, 150.0]}, index=["CPA_A", "CPA_B"])

    exio_slice = SimpleNamespace(
        Z_domestic=pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["S1", "S2"], columns=["S1", "S2"]),
        Y_region=pd.DataFrame({"P3": [100.0, 100.0]}, index=["S1", "S2"]),
        # Deliberately wrong producer-output values. The boundary-correct check
        # must use Z_domestic + Y_region, not x_region.
        x_region=pd.Series([40.0, 40.0], index=["S1", "S2"], dtype="float64"),
    )
    mapping = {"CPA_A": {"S1": 1.0}, "CPA_B": {"S2": 1.0}}

    _assert_output_consistency(
        sector_mapping=mapping,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        rtol=0.0,
        atol=0.0,
        label="destination-use-boundary-test",
    )


def test_output_consistency_strict_helper_still_fails_on_destination_use_mismatch():
    from types import SimpleNamespace
    from scripts.split_eurostat_energy_by_exiobase import _assert_output_consistency

    euro_Z = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["CPA_A", "CPA_B"], columns=["CPA_A", "CPA_B"])
    euro_Y = pd.DataFrame({"P3": [50.0, 150.0]}, index=["CPA_A", "CPA_B"])

    exio_slice = SimpleNamespace(
        Z_domestic=pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["S1", "S2"], columns=["S1", "S2"]),
        Y_region=pd.DataFrame({"P3": [50.0, 100.0]}, index=["S1", "S2"]),
        x_region=pd.Series([999.0, 999.0], index=["S1", "S2"], dtype="float64"),
    )
    mapping = {"CPA_A": {"S1": 1.0}, "CPA_B": {"S2": 1.0}}

    with pytest.raises(SystemExit):
        _assert_output_consistency(
            sector_mapping=mapping,
            eurostat_Z=euro_Z,
            eurostat_Y=euro_Y,
            exio_slice=exio_slice,
            rtol=0.0,
            atol=0.0,
            label="strict-destination-use-mismatch-test",
        )


def test_output_consistency_check_allows_sector_mix_differences_within_total_tolerance():
    from types import SimpleNamespace
    from scripts.split_eurostat_energy_by_exiobase import _assert_output_consistency

    euro_Z = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["CPA_A", "CPA_B"], columns=["CPA_A", "CPA_B"])
    euro_Y = pd.DataFrame({"P3": [50.0, 150.0]}, index=["CPA_A", "CPA_B"])

    exio_slice = SimpleNamespace(
        Z_domestic=pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["S1", "S2"], columns=["S1", "S2"]),
        Y_region=pd.DataFrame({"P3": [100.0, 100.0]}, index=["S1", "S2"]),
        x_region=pd.Series([150.0, 50.0], index=["S1", "S2"], dtype="float64"),
    )
    mapping = {"CPA_A": {"S1": 1.0}, "CPA_B": {"S2": 1.0}}

    _assert_output_consistency(
        sector_mapping=mapping,
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        rtol=0.2,
        atol=1e-6,
        label="test",
    )


def test_split_metadata_helpers_include_fallback_count_and_legacy_aliases() -> None:
    from scripts.split_eurostat_energy_by_exiobase import (
        SPLIT_OUTPUT_NAME_ALIASES,
        _legacy_split_info_payload,
        _split_info_payload,
    )

    fake_result = SimpleNamespace(
        balancing_method="DONOR_BLOCK_WITH_SHARE_FALLBACK",
        balancing_stats=SimpleNamespace(iterations=7, max_abs_row_gap=0.0, max_abs_col_gap=0.0),
        fallback_count=3,
        fallback_diagnostics=pd.DataFrame([{"row_sector": "A"}, {"row_sector": "B"}, {"row_sector": "C"}]),
    )

    split_info = _split_info_payload(fake_result, variant_dim="stk_flow", variant_value="TOTAL")
    assert split_info["method"] == "DONOR_BLOCK_WITH_SHARE_FALLBACK"
    assert split_info["is_ras_balanced"] is False
    assert split_info["is_reconstruction_validated"] is True
    assert split_info["fallback_count"] == 3
    assert split_info["has_fallback_diagnostics"] is True
    assert split_info["variant_dim"] == "stk_flow"
    assert split_info["variant_value"] == "TOTAL"

    legacy_info = _legacy_split_info_payload(split_info)
    assert legacy_info["deprecated_name"] is True
    assert legacy_info["actual_method"] == split_info["method"]
    assert legacy_info["is_ras_balanced"] is False

    assert SPLIT_OUTPUT_NAME_ALIASES == (
        ("Z_split.csv", "Z_split_balanced.csv"),
        ("Y_split.csv", "Y_split_balanced.csv"),
        ("x_split.csv", "x_split_balanced.csv"),
        ("A_split.csv", "A_split_balanced.csv"),
    )


def test_assert_mapping_integrity_only_requires_explicit_targets() -> None:
    from scripts.split_eurostat_energy_by_exiobase import _assert_mapping_integrity

    eurostat_Z = pd.DataFrame([[10.0, 1.0], [2.0, 3.0]], index=["CPA_C19", "CPA_C21"], columns=["CPA_C19", "CPA_C21"])
    eurostat_Y = pd.DataFrame({"FD": [4.0, 5.0]}, index=["CPA_C19", "CPA_C21"])
    sector_mapping = {"CPA_C19": {"p24.g": 1.0}}

    _assert_mapping_integrity(
        sector_mapping=sector_mapping,
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        required_target_codes=["CPA_C19"],
        label="test",
    )



def _write_member(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, text.encode("utf-8"))


def test_split_eurostat_energy_sectors_extract_quantity_and_price(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["coke oven products", "refined petroleum products", "electricity from coal", "gas supply"]

    # Minimal EXIO slice (domestic block + one final-demand column).
    Z_domestic = pd.DataFrame(
        [
            [1.0, 2.0, 0.5, 0.1],
            [0.2, 1.0, 0.4, 0.2],
            [0.1, 0.3, 1.5, 0.4],
            [0.0, 0.2, 0.6, 1.0],
        ],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame(
        {"FD": [1.0, 2.0, 3.0, 4.0]},
        index=exio_sectors,
    )
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    # Minimal extension zip for quantity/price extraction.
    zp = tmp_path / "IOT_2020_pxp.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        _write_member(zf, "energy/unit.txt", "stressor\tunit\nEnergy use - Gross\tTJ\n")
        _write_member(
            zf,
            "energy/F.txt",
            "\n".join(
                [
                    "region\tAT\tAT\tAT\tAT",
                    "sector\tcoke oven products\trefined petroleum products\telectricity from coal\tgas supply",
                    "stressor\t\t\t\t",
                    "Energy use - Gross\t1\t2\t4\t3",
                    "",
                ]
            ),
        )

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=zp,
    )

    euro_sectors = ["A", "C19", "D35"]
    euro_Z = pd.DataFrame(
        [
            [10.0, 1.0, 2.0],
            [1.0, 5.0, 1.0],
            [2.0, 1.0, 6.0],
        ],
        index=euro_sectors,
        columns=euro_sectors,
    )
    euro_Y = pd.DataFrame({"FD": [3.0, 2.0, 1.0]}, index=euro_sectors)

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_patterns={"C19": ("coke", "refined"), "D35": ("electricity", "gas")},
        extract_quantity_and_price=True,
        quantity_price_extensions=("energy",),
        quantity_price_stressor_keywords=("Energy use -",),
        quantity_price_cache_dir=tmp_path / "cache",
        ras_max_iter=5_000,
        ras_tol=1e-12,
    )

    assert "C19" not in res.Z.index
    assert "D35" not in res.Z.index
    assert len(res.Z.index) > len(euro_sectors)

    assert res.exio_quantity_and_price is not None
    assert set(res.exio_quantity_and_price["exio_sector"]) == set(exio_sectors)
    assert (res.exio_quantity_and_price["unit"] == "TJ").all()


def test_write_energy_split_report_xlsx(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import (
        split_iot_using_exiobase_split_profiles,
        write_energy_split_report_xlsx,
    )

    exio_sectors = ["coke oven products", "refined petroleum products", "electricity", "gas"]
    Z_domestic = pd.DataFrame(
        1.0,
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame({"FD": [1.0, 1.0, 1.0, 1.0]}, index=exio_sectors)
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    zp = tmp_path / "IOT_2020_pxp.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        _write_member(zf, "energy/unit.txt", "stressor\tunit\nEnergy use - Gross\tTJ\n")
        _write_member(
            zf,
            "energy/F.txt",
            "\n".join(
                [
                    "region\tAT\tAT\tAT\tAT",
                    "sector\tcoke oven products\trefined petroleum products\telectricity\tgas",
                    "stressor\t\t\t\t",
                    "Energy use - Gross\t1\t1\t1\t1",
                    "",
                ]
            ),
        )

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=zp,
    )

    euro_sectors = ["A", "C19", "D35"]
    euro_Z = pd.DataFrame(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 3.0],
        ],
        index=euro_sectors,
        columns=euro_sectors,
    )
    euro_Y = pd.DataFrame({"FD": [1.0, 1.0, 1.0]}, index=euro_sectors)

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_patterns={"C19": ("coke", "refined"), "D35": ("electricity", "gas")},
        extract_quantity_and_price=True,
        quantity_price_extensions=("energy",),
        quantity_price_stressor_keywords=("Energy use -",),
        quantity_price_cache_dir=tmp_path / "cache",
        ras_max_iter=5_000,
        ras_tol=1e-12,
    )

    out_xlsx = write_energy_split_report_xlsx(
        tmp_path / "energy_split_report.xlsx",
        result=res,
        aggregates=("C19", "D35"),
    )
    assert out_xlsx.exists()


def test_split_with_grouped_mapping_collapses_electricity_sources(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["Electricity by coal", "Electricity by gas", "Transmission services of electricity"]
    Z_domestic = pd.DataFrame(
        [
            [1.0, 0.2, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame({"FD": [1.0, 1.0, 1.0]}, index=exio_sectors)
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )

    euro_sectors = ["CPA_D", "A"]
    euro_Z = pd.DataFrame([[10.0, 1.0], [1.0, 5.0]], index=euro_sectors, columns=euro_sectors)
    euro_Y = pd.DataFrame({"FD": [3.0, 2.0]}, index=euro_sectors)

    sector_to_exio = {
        "CPA_D": {
            "Electricity generation": ["Electricity by coal", "Electricity by gas"],
            "Transmission services of electricity": ["Transmission services of electricity"],
        }
    }

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors=sector_to_exio,
        ras_max_iter=2_000,
        ras_tol=1e-12,
    )

    assert "CPA_D" not in res.Z.index
    assert any(s.endswith("Electricity generation") for s in res.Z.index)
    assert not any("Electricity by coal" in s for s in res.Z.index)
    assert not any("Electricity by gas" in s for s in res.Z.index)


def test_donor_split_preserves_nonseparable_cross_block(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["a1", "a2", "b1", "b2"]
    Z_domestic = pd.DataFrame(
        [
            [1.0, 0.0, 90.0, 10.0],
            [0.0, 1.0, 10.0, 90.0],
            [5.0, 0.0, 1.0, 0.0],
            [0.0, 5.0, 0.0, 1.0],
        ],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame({"Final consumption expenditure by households": [1.0, 1.0, 1.0, 1.0]}, index=exio_sectors)
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )

    euro_Z = pd.DataFrame([[0.0, 200.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    euro_Y = pd.DataFrame({"P3": [2.0, 2.0]}, index=["A", "B"])

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"A": ["a1", "a2"], "B": ["b1", "b2"]},
        create_io_system=False,
    )

    assert res.balancing_method == "DONOR_BLOCK"
    assert res.Z.loc["a1", "b1"] == pytest.approx(90.0)
    assert res.Z.loc["a1", "b2"] == pytest.approx(10.0)
    assert res.Z.loc["a2", "b1"] == pytest.approx(10.0)
    assert res.Z.loc["a2", "b2"] == pytest.approx(90.0)


def test_donor_split_uses_mapping_for_unchanged_to_split_block(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["u", "v", "a1", "a2"]
    Z_domestic = pd.DataFrame(
        [
            [1.0, 0.0, 90.0, 10.0],
            [0.0, 1.0, 10.0, 90.0],
            [5.0, 5.0, 1.0, 0.0],
            [5.0, 5.0, 0.0, 1.0],
        ],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame({"Final consumption expenditure by households": [1.0, 1.0, 1.0, 1.0]}, index=exio_sectors)
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )

    euro_Z = pd.DataFrame(
        [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0], [0.0, 0.0, 0.0]],
        index=["U", "V", "A"],
        columns=["U", "V", "A"],
    )
    euro_Y = pd.DataFrame({"P3": [1.0, 1.0, 2.0]}, index=["U", "V", "A"])

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"A": ["a1", "a2"]},
        eurostat_sector_mapping={"U": {"u": 1.0}, "V": {"v": 1.0}},
        create_io_system=False,
    )

    assert res.Z.loc["U", "a1"] == pytest.approx(90.0)
    assert res.Z.loc["U", "a2"] == pytest.approx(10.0)
    assert res.Z.loc["V", "a1"] == pytest.approx(10.0)
    assert res.Z.loc["V", "a2"] == pytest.approx(90.0)


def test_donor_split_uses_category_specific_final_demand_shares(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["a1", "a2"]
    Z_domestic = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=exio_sectors, columns=exio_sectors)
    Y_region = pd.DataFrame(
        {
            "Final consumption expenditure by households": [90.0, 10.0],
            "Exports: Total (fob)": [10.0, 90.0],
        },
        index=exio_sectors,
    )
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")

    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )

    euro_Z = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    euro_Y = pd.DataFrame({"P3": [100.0], "P6": [100.0]}, index=["A"])

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"A": ["a1", "a2"]},
        create_io_system=False,
    )

    assert res.Y.loc["a1", "P3"] == pytest.approx(90.0)
    assert res.Y.loc["a2", "P3"] == pytest.approx(10.0)
    assert res.Y.loc["a1", "P6"] == pytest.approx(10.0)
    assert res.Y.loc["a2", "P6"] == pytest.approx(90.0)


def test_edge_aware_split_emits_source_target_child_and_preserves_mass(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["p24.d", "p24.g", "c21"]
    Z_domestic = pd.DataFrame(
        [
            [10.0, 5.0, 2.0],
            [3.0, 8.0, 1.0],
            [4.0, 2.0, 6.0],
        ],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [20.0, 10.0, 5.0]},
        index=exio_sectors,
    )
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")
    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )
    euro_Z = pd.DataFrame(
        [[100.0, 20.0], [30.0, 40.0]],
        index=["CPA_C20", "CPA_C21"],
        columns=["CPA_C20", "CPA_C21"],
    )
    euro_Y = pd.DataFrame({"P3": [50.0, 60.0]}, index=["CPA_C20", "CPA_C21"])

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"CPA_C20": ["p24.d", "p24.g"]},
        eurostat_sector_mapping={
            "CPA_C20": {"p24.d": 0.75, "p24.g": 1.0},
            "CPA_C21": {"p24.d": 0.25, "c21": 1.0},
        },
        create_io_system=False,
    )

    assert "p24.d@CPA_C20" in res.Z.index
    assert "p24.d" not in res.Z.index
    assert "p24.g" in res.Z.index
    assert res.exio_selected_sectors["CPA_C20"] == ["p24.d@CPA_C20", "p24.g"]

    child_codes = ["p24.d@CPA_C20", "p24.g"]
    assert float(res.Z.loc[child_codes, child_codes].to_numpy().sum()) == pytest.approx(euro_Z.loc["CPA_C20", "CPA_C20"])
    assert float(res.Z.loc[child_codes, "CPA_C21"].sum()) == pytest.approx(euro_Z.loc["CPA_C20", "CPA_C21"])
    assert float(res.Z.loc["CPA_C21", child_codes].sum()) == pytest.approx(euro_Z.loc["CPA_C21", "CPA_C20"])
    assert float(res.Y.loc[child_codes, "P3"].sum()) == pytest.approx(euro_Y.loc["CPA_C20", "P3"])
    pd.testing.assert_series_equal(
        (res.Z.sum(axis=1) + res.Y.sum(axis=1)).astype("float64"),
        res.x.astype("float64"),
        check_names=False,
    )


def test_edge_aware_final_demand_shares_use_edge_weights(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["p24.d", "p24.g"]
    Z_domestic = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=exio_sectors, columns=exio_sectors)
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [100.0, 50.0]},
        index=exio_sectors,
    )
    x_region = Y_region.sum(axis=1).astype("float64")
    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )
    euro_Z = pd.DataFrame([[0.0]], index=["CPA_C20"], columns=["CPA_C20"])
    euro_Y = pd.DataFrame({"P3": [100.0]}, index=["CPA_C20"])

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"CPA_C20": ["p24.d", "p24.g"]},
        eurostat_sector_mapping={
            "CPA_C20": {"p24.d": 0.25, "p24.g": 1.0},
            "CPA_C21": {"p24.d": 0.75},
        },
        create_io_system=False,
    )

    # Edge-weighted donor values are p24.d: 25 and p24.g: 50.
    assert res.Y.loc["p24.d@CPA_C20", "P3"] == pytest.approx(100.0 / 3.0)
    assert res.Y.loc["p24.g", "P3"] == pytest.approx(200.0 / 3.0)


def test_supplementary_rows_split_columnwise_with_edge_aware_input_shares(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["p24.d", "p24.g"]
    Z_domestic = pd.DataFrame(
        [[4.0, 0.0], [0.0, 8.0]],
        index=exio_sectors,
        columns=exio_sectors,
    )
    Y_region = pd.DataFrame({"Final consumption expenditure by households": [1.0, 1.0]}, index=exio_sectors)
    x_region = (Z_domestic.sum(axis=1) + Y_region.sum(axis=1)).astype("float64")
    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=x_region,
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )
    euro_Z = pd.DataFrame([[0.0]], index=["CPA_C20"], columns=["CPA_C20"])
    euro_Y = pd.DataFrame({"P3": [1.0]}, index=["CPA_C20"])
    euro_V = pd.DataFrame(
        [[90.0], [999.0]],
        index=["Value added, gross", "Total intermediate consumption /final use"],
        columns=["CPA_C20"],
    )
    euro_M = pd.DataFrame(
        [[7.0], [999.0]],
        index=["Value added, gross", "Total intermediate consumption /final use"],
        columns=["P3"],
    )

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        eurostat_V=euro_V,
        eurostat_M=euro_M,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"CPA_C20": ["p24.d", "p24.g"]},
        eurostat_sector_mapping={
            "CPA_C20": {"p24.d": 0.25, "p24.g": 1.0},
            "CPA_C21": {"p24.d": 0.75},
        },
        create_io_system=False,
    )

    assert res.V is not None
    assert res.M is not None
    expected_intermediate = res.Z.sum(axis=0).astype("float64")
    expected_value_added = (res.x - expected_intermediate).astype("float64")
    pd.testing.assert_series_equal(
        res.V.loc["Value added, gross"].astype("float64"),
        expected_value_added,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        res.V.loc["Total intermediate consumption /final use"].astype("float64"),
        expected_intermediate,
        check_names=False,
    )
    assert res.M.loc["Value added, gross", "P3"] == pytest.approx(7.0)
    assert res.M.loc["Total intermediate consumption /final use", "P3"] == pytest.approx(float(res.Y.sum(axis=0)["P3"]))


def test_validate_split_reconstruction_preserves_original_final_demand_columns():
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import _validate_split_reconstruction

    euro_Z = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["A", "CPA_C19"],
        columns=["A", "CPA_C19"],
    )
    euro_Y = pd.DataFrame(
        [[5.0, 1.0, 0.5], [6.0, 2.0, 1.5]],
        index=["A", "CPA_C19"],
        columns=["P3", "P5", "P6"],
    )
    split_profiles = {
        "CPA_C19": SplitProfiles(
            aggregate_sector="CPA_C19",
            subsectors=("CPA_C19::s1", "CPA_C19::s2"),
            output_share=pd.Series([0.4, 0.6], index=["CPA_C19::s1", "CPA_C19::s2"]),
            intermediate_sales_share=pd.Series([0.2, 0.8], index=["CPA_C19::s1", "CPA_C19::s2"]),
            intermediate_input_share=pd.Series([0.7, 0.3], index=["CPA_C19::s1", "CPA_C19::s2"]),
            final_demand_share=pd.Series([0.25, 0.75], index=["CPA_C19::s1", "CPA_C19::s2"]),
        )
    }
    Z_split = pd.DataFrame(
        [[1.0, 1.4, 0.6], [0.6, 0.24, 0.56], [2.4, 0.96, 2.24]],
        index=["A", "CPA_C19::s1", "CPA_C19::s2"],
        columns=["A", "CPA_C19::s1", "CPA_C19::s2"],
    )
    Y_split = pd.DataFrame(
        [[5.0, 1.0, 0.5], [1.5, 0.5, 0.375], [4.5, 1.5, 1.125]],
        index=["A", "CPA_C19::s1", "CPA_C19::s2"],
        columns=["P3", "P5", "P6"],
    )
    x_split = (Z_split.sum(axis=1) + Y_split.sum(axis=1)).astype("float64")

    _validate_split_reconstruction(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        split_profiles=split_profiles,
        Z_split=Z_split,
        Y_split=Y_split,
        x_split=x_split,
    )


def test_validate_split_reconstruction_raises_on_final_demand_drift():
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import _validate_split_reconstruction

    euro_Z = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["A", "CPA_C19"],
        columns=["A", "CPA_C19"],
    )
    euro_Y = pd.DataFrame(
        [[5.0, 1.0, 0.5], [6.0, 2.0, 1.5]],
        index=["A", "CPA_C19"],
        columns=["P3", "P5", "P6"],
    )
    split_profiles = {
        "CPA_C19": SplitProfiles(
            aggregate_sector="CPA_C19",
            subsectors=("CPA_C19::s1", "CPA_C19::s2"),
            output_share=pd.Series([0.4, 0.6], index=["CPA_C19::s1", "CPA_C19::s2"]),
            intermediate_sales_share=pd.Series([0.2, 0.8], index=["CPA_C19::s1", "CPA_C19::s2"]),
            intermediate_input_share=pd.Series([0.7, 0.3], index=["CPA_C19::s1", "CPA_C19::s2"]),
            final_demand_share=pd.Series([0.25, 0.75], index=["CPA_C19::s1", "CPA_C19::s2"]),
        )
    }
    Z_split = pd.DataFrame(
        [[1.0, 1.4, 0.6], [0.6, 0.24, 0.56], [2.4, 0.96, 2.24]],
        index=["A", "CPA_C19::s1", "CPA_C19::s2"],
        columns=["A", "CPA_C19::s1", "CPA_C19::s2"],
    )
    Y_split = pd.DataFrame(
        [[5.0, 1.0, 0.5], [1.5, 0.5, 0.375], [99.0, 1.5, 1.125]],
        index=["A", "CPA_C19::s1", "CPA_C19::s2"],
        columns=["P3", "P5", "P6"],
    )
    x_split = (Z_split.sum(axis=1) + Y_split.sum(axis=1)).astype("float64")

    with pytest.raises(ValueError, match="Disaggregation validation failed"):
        _validate_split_reconstruction(
            eurostat_Z=euro_Z,
            eurostat_Y=euro_Y,
            split_profiles=split_profiles,
            Z_split=Z_split,
            Y_split=Y_split,
            x_split=x_split,
        )

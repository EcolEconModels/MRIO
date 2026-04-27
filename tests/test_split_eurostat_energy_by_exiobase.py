from __future__ import annotations

import zipfile

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


def test_output_consistency_check_fails_on_total_mismatch():
    from types import SimpleNamespace
    from scripts.split_eurostat_energy_by_exiobase import _assert_output_consistency

    euro_Z = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["CPA_A", "CPA_B"], columns=["CPA_A", "CPA_B"])
    euro_Y = pd.DataFrame({"P3": [50.0, 150.0]}, index=["CPA_A", "CPA_B"])

    exio_slice = SimpleNamespace(
        Z_domestic=pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["S1", "S2"], columns=["S1", "S2"]),
        Y_region=pd.DataFrame({"P3": [100.0, 100.0]}, index=["S1", "S2"]),
        x_region=pd.Series([40.0, 40.0], index=["S1", "S2"], dtype="float64"),
    )
    mapping = {"CPA_A": {"S1": 1.0}, "CPA_B": {"S2": 1.0}}

    with pytest.raises(SystemExit):
        _assert_output_consistency(
            sector_mapping=mapping,
            eurostat_Z=euro_Z,
            eurostat_Y=euro_Y,
            exio_slice=exio_slice,
            rtol=0.01,
            atol=1e-6,
            label="test",
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
    assert any(s.startswith("C19::") for s in res.Z.index)
    assert any(s.startswith("D35::") for s in res.Z.index)

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
    assert res.Z.loc["A::a1", "B::b1"] == pytest.approx(90.0)
    assert res.Z.loc["A::a1", "B::b2"] == pytest.approx(10.0)
    assert res.Z.loc["A::a2", "B::b1"] == pytest.approx(10.0)
    assert res.Z.loc["A::a2", "B::b2"] == pytest.approx(90.0)


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

    assert res.Z.loc["U", "A::a1"] == pytest.approx(90.0)
    assert res.Z.loc["U", "A::a2"] == pytest.approx(10.0)
    assert res.Z.loc["V", "A::a1"] == pytest.approx(10.0)
    assert res.Z.loc["V", "A::a2"] == pytest.approx(90.0)


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

    assert res.Y.loc["A::a1", "P3"] == pytest.approx(90.0)
    assert res.Y.loc["A::a2", "P3"] == pytest.approx(10.0)
    assert res.Y.loc["A::a1", "P6"] == pytest.approx(10.0)
    assert res.Y.loc["A::a2", "P6"] == pytest.approx(90.0)


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

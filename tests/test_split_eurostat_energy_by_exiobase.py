from __future__ import annotations

import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_mapping_integrity_check_passes():
    from scripts.split_eurostat_energy_by_exiobase import _assert_mapping_integrity

    euro_Z = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]], index=pd.Index(["CPA_A", "CPA_B"]), columns=pd.Index(["CPA_A", "CPA_B"])
    )
    euro_Y = pd.DataFrame({"FD": [100.0, 100.0]}, index=pd.Index(["CPA_A", "CPA_B"]))

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

    euro_Z = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]], index=pd.Index(["CPA_A", "CPA_B"]), columns=pd.Index(["CPA_A", "CPA_B"])
    )
    euro_Y = pd.DataFrame({"P3": [50.0, 150.0]}, index=pd.Index(["CPA_A", "CPA_B"]))

    exio_slice = SimpleNamespace(
        Z_domestic=pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]], index=pd.Index(["S1", "S2"]), columns=pd.Index(["S1", "S2"])
        ),
        Y_region=pd.DataFrame({"P3": [100.0, 100.0]}, index=pd.Index(["S1", "S2"])),
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

    euro_Z = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]], index=pd.Index(["CPA_A", "CPA_B"]), columns=pd.Index(["CPA_A", "CPA_B"])
    )
    euro_Y = pd.DataFrame({"P3": [50.0, 150.0]}, index=pd.Index(["CPA_A", "CPA_B"]))

    exio_slice = SimpleNamespace(
        Z_domestic=pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]], index=pd.Index(["S1", "S2"]), columns=pd.Index(["S1", "S2"])
        ),
        Y_region=pd.DataFrame({"P3": [50.0, 100.0]}, index=pd.Index(["S1", "S2"])),
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

    euro_Z = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]], index=pd.Index(["CPA_A", "CPA_B"]), columns=pd.Index(["CPA_A", "CPA_B"])
    )
    euro_Y = pd.DataFrame({"P3": [50.0, 150.0]}, index=pd.Index(["CPA_A", "CPA_B"]))

    exio_slice = SimpleNamespace(
        Z_domestic=pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]], index=pd.Index(["S1", "S2"]), columns=pd.Index(["S1", "S2"])
        ),
        Y_region=pd.DataFrame({"P3": [100.0, 100.0]}, index=pd.Index(["S1", "S2"])),
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


def test_split_metadata_includes_fallback_count() -> None:
    from scripts.split_eurostat_energy_by_exiobase import (
        _split_info_payload,
    )

    fake_result = SimpleNamespace(
        balancing_method="DONOR_BLOCK_WITH_SHARE_FALLBACK",
        balancing_stats=SimpleNamespace(
            iterations=7, max_abs_row_gap=0.0, max_abs_col_gap=0.0
        ),
        fallback_count=3,
        fallback_diagnostics=pd.DataFrame(
            [{"row_sector": "A"}, {"row_sector": "B"}, {"row_sector": "C"}]
        ),
    )

    split_info = _split_info_payload(
        fake_result, variant_dim="stk_flow", variant_value="TOTAL"
    )
    assert split_info["method"] == "DONOR_BLOCK_WITH_SHARE_FALLBACK"
    assert split_info["is_reconstruction_validated"] is True
    assert split_info["fallback_count"] == 3
    assert split_info["has_fallback_diagnostics"] is True
    assert split_info["variant_dim"] == "stk_flow"
    assert split_info["variant_value"] == "TOTAL"

def test_assert_mapping_integrity_only_requires_explicit_targets() -> None:
    from scripts.split_eurostat_energy_by_exiobase import _assert_mapping_integrity

    eurostat_Z = pd.DataFrame(
        [[10.0, 1.0], [2.0, 3.0]],
        index=pd.Index(["CPA_C19", "CPA_C21"]),
        columns=pd.Index(["CPA_C19", "CPA_C21"]),
    )
    eurostat_Y = pd.DataFrame({"FD": [4.0, 5.0]}, index=pd.Index(["CPA_C19", "CPA_C21"]))
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

    exio_sectors = [
        "coke oven products",
        "refined petroleum products",
        "electricity from coal",
        "gas supply",
    ]
    exio_sectors_index = pd.Index(exio_sectors)

    # Minimal EXIO slice (domestic block + one final-demand column).
    Z_domestic = pd.DataFrame(
        [
            [1.0, 2.0, 0.5, 0.1],
            [0.2, 1.0, 0.4, 0.2],
            [0.1, 0.3, 1.5, 0.4],
            [0.0, 0.2, 0.6, 1.0],
        ],
        index=exio_sectors_index,
        columns=exio_sectors_index,
    )
    Y_region = pd.DataFrame(
        {"FD": [1.0, 2.0, 3.0, 4.0]},
        index=exio_sectors_index,
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
    euro_sectors_index = pd.Index(euro_sectors)
    euro_Z = pd.DataFrame(
        [
            [10.0, 1.0, 2.0],
            [1.0, 5.0, 1.0],
            [2.0, 1.0, 6.0],
        ],
        index=euro_sectors_index,
        columns=euro_sectors_index,
    )
    # Reduced Eurostat fixture: use a recognized aggregate final-demand family.
    euro_Y = pd.DataFrame({"P3": [3.0, 2.0, 1.0]}, index=euro_sectors_index)

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_patterns={"C19": ("coke", "refined"), "D35": ("electricity", "gas")},
        extract_quantity_and_price=True,
        quantity_price_extensions=("energy",),
        quantity_price_stressor_keywords=("Energy use -",),
        quantity_price_cache_dir=tmp_path / "cache",
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

    exio_sectors = [
        "coke oven products",
        "refined petroleum products",
        "electricity",
        "gas",
    ]
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        1.0,
        index=exio_sectors_index,
        columns=exio_sectors_index,
    )
    Y_region = pd.DataFrame({"FD": [1.0, 1.0, 1.0, 1.0]}, index=exio_sectors_index)
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
    euro_sectors_index = pd.Index(euro_sectors)
    euro_Z = pd.DataFrame(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 3.0],
        ],
        index=euro_sectors_index,
        columns=euro_sectors_index,
    )
    # Reduced Eurostat fixture: use a recognized aggregate final-demand family.
    euro_Y = pd.DataFrame({"P3": [1.0, 1.0, 1.0]}, index=euro_sectors_index)

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_patterns={"C19": ("coke", "refined"), "D35": ("electricity", "gas")},
        extract_quantity_and_price=True,
        quantity_price_extensions=("energy",),
        quantity_price_stressor_keywords=("Energy use -",),
        quantity_price_cache_dir=tmp_path / "cache",
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

    exio_sectors = [
        "Electricity by coal",
        "Electricity by gas",
        "Transmission services of electricity",
    ]
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [
            [1.0, 0.2, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ],
        index=exio_sectors_index,
        columns=exio_sectors_index,
    )
    Y_region = pd.DataFrame({"FD": [1.0, 1.0, 1.0]}, index=exio_sectors_index)
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
    euro_sectors_index = pd.Index(euro_sectors)
    euro_Z = pd.DataFrame(
        [[10.0, 1.0], [1.0, 5.0]], index=euro_sectors_index, columns=euro_sectors_index
    )
    # Reduced Eurostat fixture: use a recognized aggregate final-demand family.
    euro_Y = pd.DataFrame({"P3": [3.0, 2.0]}, index=euro_sectors_index)

    sector_to_exio = {
        "CPA_D": {
            "Electricity generation": ["Electricity by coal", "Electricity by gas"],
            "Transmission services of electricity": [
                "Transmission services of electricity"
            ],
        }
    }

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors=sector_to_exio,
    )

    assert "CPA_D" not in res.Z.index
    assert any(s.endswith("Electricity generation") for s in res.Z.index)
    assert not any("Electricity by coal" in s for s in res.Z.index)
    assert not any("Electricity by gas" in s for s in res.Z.index)


def test_donor_split_preserves_nonseparable_cross_block(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["a1", "a2", "b1", "b2"]
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [
            [1.0, 0.0, 90.0, 10.0],
            [0.0, 1.0, 10.0, 90.0],
            [5.0, 0.0, 1.0, 0.0],
            [0.0, 5.0, 0.0, 1.0],
        ],
        index=exio_sectors_index,
        columns=exio_sectors_index,
    )
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [1.0, 1.0, 1.0, 1.0]},
        index=exio_sectors_index,
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
        [[0.0, 200.0], [0.0, 0.0]], index=pd.Index(["A", "B"]), columns=pd.Index(["A", "B"])
    )
    euro_Y = pd.DataFrame({"P3": [2.0, 2.0]}, index=pd.Index(["A", "B"]))

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
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [
            [1.0, 0.0, 90.0, 10.0],
            [0.0, 1.0, 10.0, 90.0],
            [5.0, 5.0, 1.0, 0.0],
            [5.0, 5.0, 0.0, 1.0],
        ],
        index=exio_sectors_index,
        columns=exio_sectors_index,
    )
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [1.0, 1.0, 1.0, 1.0]},
        index=exio_sectors_index,
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
        [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0], [0.0, 0.0, 0.0]],
        index=pd.Index(["U", "V", "A"]),
        columns=pd.Index(["U", "V", "A"]),
    )
    euro_Y = pd.DataFrame({"P3": [1.0, 1.0, 2.0]}, index=pd.Index(["U", "V", "A"]))

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
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]], index=exio_sectors_index, columns=exio_sectors_index
    )
    Y_region = pd.DataFrame(
        {
            "Final consumption expenditure by households": [90.0, 10.0],
            "Exports: Total (fob)": [10.0, 90.0],
        },
        index=exio_sectors_index,
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

    euro_Z = pd.DataFrame([[0.0]], index=pd.Index(["A"]), columns=pd.Index(["A"]))
    euro_Y = pd.DataFrame({"P3": [100.0], "P6": [100.0]}, index=pd.Index(["A"]))

    def injected_export_loader(*, source_zip, region):
        assert source_zip == tmp_path / "IOT_2020_pxp.zip"
        assert region == "AT"
        return pd.Series([10.0, 90.0], index=exio_sectors, dtype="float64")

    res = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"A": ["a1", "a2"]},
        export_vector_loader=injected_export_loader,
        create_io_system=False,
    )

    assert res.Y.loc["a1", "P3"] == pytest.approx(90.0)
    assert res.Y.loc["a2", "P3"] == pytest.approx(10.0)
    assert res.Y.loc["a1", "P6"] == pytest.approx(10.0)
    assert res.Y.loc["a2", "P6"] == pytest.approx(90.0)


def test_import_donor_uses_documented_import_use_proxy(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import build_split_profiles_from_exiobase_domestic_block, split_iot_using_exiobase_split_profiles

    exio_sectors = ["a1", "a2"]
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [[3.0, 0.0], [0.0, 1.0]], index=exio_sectors_index, columns=exio_sectors_index
    )
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [1.0, 1.0]}, index=exio_sectors_index
    )
    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=Z_domestic.sum(axis=1) + Y_region.sum(axis=1),
        source_zip=tmp_path / "not-needed.zip",
        flow="imports",
    )
    dom_profiles, _ = build_split_profiles_from_exiobase_domestic_block(
        exio_slice, sector_to_exio_sectors={"A": exio_sectors}
    )
    result = split_iot_using_exiobase_split_profiles(
        eurostat_Z=pd.DataFrame([[0.0]], index=pd.Index(["A"]), columns=pd.Index(["A"])),
        eurostat_Y=pd.DataFrame({"P6": [100.0]}, index=pd.Index(["A"])),
        exio_slice=exio_slice,
        sector_to_exio_sectors={"A": exio_sectors},
        column_split_profiles=dom_profiles,
        create_io_system=False,
    )
    assert result.Y.loc["a1", "P6"] == pytest.approx(100.0 * 4.0 / 6.0)
    assert result.Y.loc["a2", "P6"] == pytest.approx(100.0 * 2.0 / 6.0)


def test_domestic_export_donor_converts_exio_labels_to_codes(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    sectors = pd.Index(["p23.20.a", "p23.20.m"])
    exio_slice = ExioRegionSlice(
        year=2021,
        system="pxp",
        region="AT",
        sectors=sectors,
        Z_domestic=pd.DataFrame(np.eye(2), index=sectors, columns=sectors),
        Y_region=pd.DataFrame({"Households": [1.0, 1.0]}, index=sectors),
        x_region=pd.Series([2.0, 2.0], index=sectors),
        source_zip=tmp_path / "IOT_2021_pxp.zip",
    )

    result = split_iot_using_exiobase_split_profiles(
        eurostat_Z=pd.DataFrame([[0.0]], index=pd.Index(["CPA_C19"]), columns=pd.Index(["CPA_C19"])),
        eurostat_Y=pd.DataFrame({"P6": [100.0]}, index=pd.Index(["CPA_C19"])),
        exio_slice=exio_slice,
        sector_to_exio_sectors={"CPA_C19": sectors.tolist()},
        export_vector_loader=lambda **_: pd.Series(
            [10.0, 90.0], index=["Motor Gasoline", "White Spirit & SBP"]
        ),
        create_io_system=False,
    )

    assert result.Y.loc["p23.20.a", "P6"] == pytest.approx(10.0)
    assert result.Y.loc["p23.20.m", "P6"] == pytest.approx(90.0)


def test_zero_export_donor_uses_explicit_final_demand_fallback(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["a1", "a2"]
    exio_sectors_index = pd.Index(exio_sectors)
    # Keep profile construction valid; the injected export donor is the zero
    # mass condition under test.
    Z_domestic = pd.DataFrame(np.eye(2), index=exio_sectors_index, columns=exio_sectors_index)
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [1.0, 1.0]}, index=exio_sectors_index
    )
    exio_slice = ExioRegionSlice(
        year=2020,
        system="pxp",
        region="AT",
        sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic,
        Y_region=Y_region,
        x_region=Y_region.sum(axis=1),
        source_zip=tmp_path / "not-needed.zip",
    )

    def zero_export_loader(*, source_zip, region):
        return pd.Series([0.0, 0.0], index=exio_sectors, dtype="float64")

    result = split_iot_using_exiobase_split_profiles(
        eurostat_Z=pd.DataFrame([[0.0]], index=pd.Index(["A"]), columns=pd.Index(["A"])),
        eurostat_Y=pd.DataFrame({"P6": [100.0]}, index=pd.Index(["A"])),
        exio_slice=exio_slice,
        sector_to_exio_sectors={"A": exio_sectors},
        export_vector_loader=zero_export_loader,
        create_io_system=False,
    )
    assert result.Y.loc["a1", "P6"] == pytest.approx(50.0)
    assert result.Y.loc["a2", "P6"] == pytest.approx(50.0)
    assert result.fallback_diagnostics is not None
    assert "final_demand_share" in set(
        result.fallback_diagnostics["fallback_type"].astype(str)
    )


def test_edge_aware_split_emits_source_target_child_and_preserves_mass(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles
    from ecolecon_mrio.iot_flow_semantics import flow_use_total

    exio_sectors = ["p24.d", "p24.g", "c21"]
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [
            [10.0, 5.0, 2.0],
            [3.0, 8.0, 1.0],
            [4.0, 2.0, 6.0],
        ],
        index=exio_sectors_index,
        columns=exio_sectors_index,
    )
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [20.0, 10.0, 5.0]},
        index=exio_sectors_index,
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
        index=pd.Index(["CPA_C20", "CPA_C21"]),
        columns=pd.Index(["CPA_C20", "CPA_C21"]),
    )
    euro_Y = pd.DataFrame({"P3": [50.0, 60.0]}, index=pd.Index(["CPA_C20", "CPA_C21"]))

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
    assert float(res.Z.loc[child_codes, child_codes].to_numpy().sum()) == pytest.approx(
        euro_Z.loc["CPA_C20", "CPA_C20"]
    )
    assert float(res.Z.loc[child_codes, "CPA_C21"].sum()) == pytest.approx(
        euro_Z.loc["CPA_C20", "CPA_C21"]
    )
    assert float(res.Z.loc["CPA_C21", child_codes].sum()) == pytest.approx(
        euro_Z.loc["CPA_C21", "CPA_C20"]
    )
    assert float(res.Y.loc[child_codes, "P3"].sum()) == pytest.approx(
        euro_Y.loc["CPA_C20", "P3"]
    )
    pd.testing.assert_series_equal(
        flow_use_total(res.Z, res.Y),
        res.use_total.astype("float64"),
        check_names=False,
    )
    assert res.x is None and res.A is None


def test_edge_aware_final_demand_shares_use_edge_weights(tmp_path):
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    exio_sectors = ["p24.d", "p24.g"]
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]], index=exio_sectors_index, columns=exio_sectors_index
    )
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [100.0, 50.0]},
        index=exio_sectors_index,
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
    euro_Z = pd.DataFrame([[0.0]], index=pd.Index(["CPA_C20"]), columns=pd.Index(["CPA_C20"]))
    euro_Y = pd.DataFrame({"P3": [100.0]}, index=pd.Index(["CPA_C20"]))

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
    exio_sectors_index = pd.Index(exio_sectors)
    Z_domestic = pd.DataFrame(
        [[4.0, 0.0], [0.0, 8.0]],
        index=exio_sectors_index,
        columns=exio_sectors_index,
    )
    Y_region = pd.DataFrame(
        {"Final consumption expenditure by households": [1.0, 1.0]}, index=exio_sectors_index
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
    euro_Z = pd.DataFrame([[0.0]], index=pd.Index(["CPA_C20"]), columns=pd.Index(["CPA_C20"]))
    euro_Y = pd.DataFrame({"P3": [1.0]}, index=pd.Index(["CPA_C20"]))
    euro_V = pd.DataFrame(
        [[90.0], [999.0], [888.0]],
        index=pd.Index([
            "Value added, gross",
            "Total intermediate consumption /final use",
            "Total supply at basic prices",
        ]),
        columns=pd.Index(["CPA_C20"]),
    )
    euro_M = pd.DataFrame(
        [[7.0], [999.0]],
        index=pd.Index(["Value added, gross", "Total intermediate consumption /final use"]),
        columns=pd.Index(["P3"]),
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

    assert res.M is not None
    assert res.V is None
    pd.testing.assert_frame_equal(res.M, euro_M)


def test_validate_split_reconstruction_preserves_original_final_demand_columns():
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import _validate_split_reconstruction

    euro_Z = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["A", "CPA_C19"]),
        columns=pd.Index(["A", "CPA_C19"]),
    )
    euro_Y = pd.DataFrame(
        [[5.0, 1.0, 0.5], [6.0, 2.0, 1.5]],
        index=pd.Index(["A", "CPA_C19"]),
        columns=pd.Index(["P3", "P5", "P6"]),
    )
    split_profiles = {
        "CPA_C19": SplitProfiles(
            aggregate_sector="CPA_C19",
            subsectors=("CPA_C19::s1", "CPA_C19::s2"),
            output_share=pd.Series([0.4, 0.6], index=["CPA_C19::s1", "CPA_C19::s2"]),
            intermediate_sales_share=pd.Series(
                [0.2, 0.8], index=["CPA_C19::s1", "CPA_C19::s2"]
            ),
            intermediate_input_share=pd.Series(
                [0.7, 0.3], index=["CPA_C19::s1", "CPA_C19::s2"]
            ),
            final_demand_share=pd.Series(
                [0.25, 0.75], index=["CPA_C19::s1", "CPA_C19::s2"]
            ),
        )
    }
    Z_split = pd.DataFrame(
        [[1.0, 1.4, 0.6], [0.6, 0.24, 0.56], [2.4, 0.96, 2.24]],
        index=pd.Index(["A", "CPA_C19::s1", "CPA_C19::s2"]),
        columns=pd.Index(["A", "CPA_C19::s1", "CPA_C19::s2"]),
    )
    Y_split = pd.DataFrame(
        [[5.0, 1.0, 0.5], [1.5, 0.5, 0.375], [4.5, 1.5, 1.125]],
        index=pd.Index(["A", "CPA_C19::s1", "CPA_C19::s2"]),
        columns=pd.Index(["P3", "P5", "P6"]),
    )
    _validate_split_reconstruction(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        split_profiles=split_profiles,
        Z_split=Z_split,
        Y_split=Y_split,
    )


def test_validate_split_reconstruction_raises_on_final_demand_drift():
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import _validate_split_reconstruction

    euro_Z = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index(["A", "CPA_C19"]),
        columns=pd.Index(["A", "CPA_C19"]),
    )
    euro_Y = pd.DataFrame(
        [[5.0, 1.0, 0.5], [6.0, 2.0, 1.5]],
        index=pd.Index(["A", "CPA_C19"]),
        columns=pd.Index(["P3", "P5", "P6"]),
    )
    split_profiles = {
        "CPA_C19": SplitProfiles(
            aggregate_sector="CPA_C19",
            subsectors=("CPA_C19::s1", "CPA_C19::s2"),
            output_share=pd.Series([0.4, 0.6], index=["CPA_C19::s1", "CPA_C19::s2"]),
            intermediate_sales_share=pd.Series(
                [0.2, 0.8], index=["CPA_C19::s1", "CPA_C19::s2"]
            ),
            intermediate_input_share=pd.Series(
                [0.7, 0.3], index=["CPA_C19::s1", "CPA_C19::s2"]
            ),
            final_demand_share=pd.Series(
                [0.25, 0.75], index=["CPA_C19::s1", "CPA_C19::s2"]
            ),
        )
    }
    Z_split = pd.DataFrame(
        [[1.0, 1.4, 0.6], [0.6, 0.24, 0.56], [2.4, 0.96, 2.24]],
        index=pd.Index(["A", "CPA_C19::s1", "CPA_C19::s2"]),
        columns=pd.Index(["A", "CPA_C19::s1", "CPA_C19::s2"]),
    )
    Y_split = pd.DataFrame(
        [[5.0, 1.0, 0.5], [1.5, 0.5, 0.375], [99.0, 1.5, 1.125]],
        index=pd.Index(["A", "CPA_C19::s1", "CPA_C19::s2"]),
        columns=pd.Index(["P3", "P5", "P6"]),
    )
    with pytest.raises(ValueError, match="Disaggregation validation failed"):
        _validate_split_reconstruction(
            eurostat_Z=euro_Z,
            eurostat_Y=euro_Y,
            split_profiles=split_profiles,
            Z_split=Z_split,
            Y_split=Y_split,
        )


def test_write_split_core_outputs_removes_stale_optional_sidecars(tmp_path) -> None:
    from scripts.split_eurostat_energy_by_exiobase import (
        _clear_managed_split_outputs,
        write_split_core_outputs,
    )

    stale_names = [
        "M_split.csv",
        "M_split_balanced.csv",
        "environmental_extensions.csv",
        "exio_environmental_extensions.csv",
        "disaggregated_sector_co2_intensity.csv",
        "exio_final_demand_co2_intensity.csv",
        "eurostat_sector_co2_intensity.csv",
        "split_child_edges.csv",
    ]
    for name in stale_names:
        (tmp_path / name).write_text("stale\\n", encoding="utf-8")

    sectors = pd.Index(["A"])
    result = SimpleNamespace(
        Z=pd.DataFrame([[1.0]], index=sectors, columns=sectors),
        Y=pd.DataFrame([[2.0]], index=sectors, columns=pd.Index(["P3"])),
        use_total=pd.Series([3.0], index=sectors, dtype="float64"),
        split_profiles={},
        exio_selected_sectors={"A": ["A"]},
        balancing_method="DONOR_BLOCK",
        fallback_count=0,
        fallback_diagnostics=None,
        V=None,
        M=None,
        supplementary_diagnostics=None,
    )

    _clear_managed_split_outputs(tmp_path)
    write_split_core_outputs(result, tmp_path, verbose=False)

    assert (tmp_path / "Z_split.csv").is_file()
    assert (tmp_path / "Y_split.csv").is_file()
    assert all(not (tmp_path / name).exists() for name in stale_names)


def test_mapped_output_identity_allows_only_published_source_rounding() -> None:
    from scripts.split_eurostat_energy_by_exiobase import (
        _build_mapped_output_flow_identity_frame,
    )

    totals = {
        "DOM": {
            "eurostat_output_total": 10.0,
            "exio_mapped_output_total": 10.0,
            "eurostat_rounding_bound": 0.04,
        },
        "IMP": {
            "eurostat_output_total": 5.0,
            "exio_mapped_output_total": 5.0,
            "eurostat_rounding_bound": 0.03,
        },
        "TOTAL": {
            "eurostat_output_total": 15.1,
            "exio_mapped_output_total": 15.0,
            "eurostat_rounding_bound": 0.04,
        },
    }
    result = _build_mapped_output_flow_identity_frame(
        totals, rtol=0.0, atol=1e-6
    ).set_index("quantity")

    assert bool(result.loc["eurostat_output_total", "ok"]) is True
    assert result.loc[
        "eurostat_output_total", "source_rounding_allowance"
    ] == pytest.approx(0.11)
    assert bool(result.loc["exio_mapped_output_total", "ok"]) is True
    assert result.loc["exio_mapped_output_total", "source_rounding_allowance"] == 0.0



def test_environmental_extension_request_parser_accepts_list_and_mapping() -> None:
    from scripts.split_eurostat_energy_by_exiobase import (
        _parse_environmental_extension_requests,
    )

    assert _parse_environmental_extension_requests(
        '["air_emissions", "materials"]'
    ) == {"air_emissions": (), "materials": ()}
    assert _parse_environmental_extension_requests(
        '{"air_emissions":["CO2"],"materials":[]}'
    ) == {"air_emissions": ("CO2",), "materials": ()}


def test_split_api_projects_multiple_environmental_extensions(tmp_path) -> None:
    """Selected extensions retain F_Y and project only production F to sectors."""
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    source = tmp_path / "IOT_2021_pxp.zip"
    with zipfile.ZipFile(source, "w") as zf:
        for extension, stressor, f_value, fy_value in (
            ("air_emissions", "CO2 - combustion - air", 10, 5),
            ("material", "Material input", 30, 15),
        ):
            _write_member(
                zf,
                f"{extension}/unit.txt",
                f"stressor\tunit\n{stressor}\tkg\n",
            )
            _write_member(
                zf,
                f"{extension}/F.txt",
                "\n".join(
                    [
                        "region\tAT",
                        "sector\tS1",
                        "stressor\t",
                        f"{stressor}\t{f_value}",
                        "",
                    ]
                ),
            )
            _write_member(
                zf,
                f"{extension}/F_Y.txt",
                "\n".join(
                    [
                        "region\tAT",
                        "final_demand_category\tP3",
                        "stressor\t",
                        f"{stressor}\t{fy_value}",
                        "",
                    ]
                ),
            )
        _write_member(
            zf,
            "Y.txt",
            "\n".join(
                [
                    "\t\tAT",
                    "\t\tP3",
                    "\t\tM.EUR",
                    "AT\tS1\t4",
                    "DE\tS1\t6",
                    "",
                ]
            ),
        )

    exio_Z = pd.DataFrame(
        [[1.0]], index=pd.Index(["S1"]), columns=pd.Index(["S1"])
    )
    exio_Y = pd.DataFrame({"P3": [4.0]}, index=pd.Index(["S1"]))
    exio_slice = ExioRegionSlice(
        year=2021,
        system="pxp",
        region="AT",
        sectors=pd.Index(["S1"]),
        Z_domestic=exio_Z,
        Y_region=exio_Y,
        x_region=pd.Series({"S1": 5.0}, dtype="float64"),
        source_zip=source,
        flow="domestic",
    )
    euro_Z = pd.DataFrame(
        [[2.0]], index=pd.Index(["A"]), columns=pd.Index(["A"])
    )
    euro_Y = pd.DataFrame({"P3": [8.0]}, index=pd.Index(["A"]))

    result = split_iot_using_exiobase_split_profiles(
        eurostat_Z=euro_Z,
        eurostat_Y=euro_Y,
        exio_slice=exio_slice,
        sector_to_exio_sectors={"A": ["S1"]},
        environmental_extension_requests={
            "air_emissions": ("CO2",),
            "material": (),
        },
        create_io_system=False,
    )

    assert result.exio_environmental_extensions is not None
    assert set(result.exio_environmental_extensions["extension"]) == {
        "air_emissions",
        "material",
    }
    final_demand_rows = result.exio_environmental_extensions.loc[
        result.exio_environmental_extensions["kind"].astype(str) == "F_Y"
    ].set_index("extension")
    assert set(final_demand_rows.index) == {"air_emissions", "material"}
    co2_final_demand = final_demand_rows.loc["air_emissions"]
    assert float(co2_final_demand["value"]) == pytest.approx(5.0)
    # The denominator is destination final demand from all producer origins: 4 + 6.
    assert float(co2_final_demand["monetary_denominator"]) == pytest.approx(10.0)
    assert float(co2_final_demand["intensity"]) == pytest.approx(0.5)

    assert result.environmental_extensions is not None
    satellite = result.environmental_extensions.set_index("extension")
    assert float(satellite.loc["air_emissions", "monetary_output"]) == 10.0
    assert float(satellite.loc["air_emissions", "value"]) == 20.0
    assert float(satellite.loc["material", "value"]) == 60.0

def test_environmental_extension_cli_selection_prefers_readable_official_names() -> None:
    """Explicit selectors are exact archive directory names, not maintained aliases."""
    from scripts.split_eurostat_energy_by_exiobase import (
        _resolve_environmental_extension_requests,
    )

    assert _resolve_environmental_extension_requests(
        preset="combustion-co2", specs=None, raw_json=None
    ) == {"air_emissions": ("CO2 - combustion - air",)}
    assert _resolve_environmental_extension_requests(
        preset="none", specs=None, raw_json=None
    ) == {}
    assert _resolve_environmental_extension_requests(
        preset="combustion-co2",
        specs=[
            "material",
            "air_emissions=CH4",
            "air_emissions=CO2 - combustion - air",
        ],
        raw_json=None,
    ) == {
        "material": (),
        "air_emissions": ("CH4", "CO2 - combustion - air"),
    }
    assert _resolve_environmental_extension_requests(
        preset="combustion-co2", specs=None, raw_json='{"water_scarcity":[]}'
    ) == {"water_scarcity": ()}

def test_environmental_extension_pattern_matches_multiple_indicator_rows() -> None:
    from ecolecon_mrio.exiobase3_extensions import _select_rows_by_keywords

    rows = [
        "Employment: Low-skilled male",
        "Employment: Low-skilled female",
        "Employment: High-skilled male",
    ]
    assert _select_rows_by_keywords(rows, ("Low-skilled",)) == rows[:2]
    assert _select_rows_by_keywords(rows, ()) == rows


def test_extension_loader_rejects_malformed_existing_fy(tmp_path) -> None:
    from ecolecon_mrio.exiobase3_extensions import load_exiobase3_extension

    malformed = tmp_path / "malformed_fy.zip"
    with zipfile.ZipFile(malformed, "w") as zf:
        _write_member(
            zf,
            "air_emissions/unit.txt",
            "stressor\tunit\nCO2 - combustion - air\tkg\n",
        )
        _write_member(
            zf,
            "air_emissions/F.txt",
            "\n".join(
                [
                    "region\tAT",
                    "sector\tS1",
                    "stressor\t",
                    "CO2 - combustion - air\t10",
                    "",
                ]
            ),
        )
        _write_member(
            zf,
            "air_emissions/F_Y.txt",
            "\n".join(
                [
                    "region\tAT",
                    "final_demand_category\tP3",
                    "stressor\t",
                    "CO2 - combustion - air\tnot-a-number",
                    "",
                ]
            ),
        )

    load_exiobase3_extension.cache_clear()
    with pytest.raises(ValueError):
        load_exiobase3_extension(
            malformed, extension="air_emissions", region="AT"
        )


def test_extension_extraction_rejects_blank_selected_value(tmp_path) -> None:
    """A blank physical satellite cell is unavailable data, never physical zero."""
    from ecolecon_mrio.exiobase3_extensions import (
        extract_extension_values,
        load_exiobase3_extension,
    )

    source = tmp_path / "blank_extension_value.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(
            zf,
            "air_emissions/unit.txt",
            "stressor\tunit\nCO2 - combustion - air\tkg\n",
        )
        _write_member(
            zf,
            "air_emissions/F.txt",
            "\n".join(
                [
                    "region\tAT",
                    "sector\tS1",
                    "stressor\t",
                    "CO2 - combustion - air\t",
                    "",
                ]
            ),
        )

    load_exiobase3_extension.cache_clear()
    with pytest.raises(ValueError, match="Non-finite values in EXIO extension"):
        extract_extension_values(
            container=source,
            region="AT",
            source_labels=["S1"],
            monetary_denominator=pd.Series({"S1": 10.0}, dtype="float64"),
            extension_requests={"air_emissions": ("CO2",)},
            kind="F",
            required=True,
        )
    load_exiobase3_extension.cache_clear()


def test_archive_driven_environmental_selection_uses_exact_complete_directories(
    tmp_path,
) -> None:
    """All-extension selection follows the archive manifest, including beta names."""
    from ecolecon_mrio.exiobase3_extensions import discover_exiobase3_extensions
    from scripts.split_eurostat_energy_by_exiobase import (
        _environmental_extension_requests_from_archive,
    )

    source = tmp_path / "extensions.zip"
    with zipfile.ZipFile(source, "w") as zf:
        for extension in ("material", "water_scarcity"):
            for member in ("F.txt", "F_Y.txt", "unit.txt"):
                _write_member(zf, f"{extension}/{member}", "fixture\n")
        _write_member(zf, "unrelated/readme.txt", "not an extension\n")

    assert discover_exiobase3_extensions(source) == ("material", "water_scarcity")
    assert _environmental_extension_requests_from_archive(source) == {
        "material": (),
        "water_scarcity": (),
    }


def test_archive_driven_environmental_selection_rejects_partial_extension(
    tmp_path,
) -> None:
    """A candidate extension with F but no F_Y/unit cannot disappear silently."""
    from ecolecon_mrio.exiobase3_extensions import discover_exiobase3_extensions

    source = tmp_path / "partial_extensions.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(zf, "material/F.txt", "fixture\n")
    with pytest.raises(ValueError, match="extension directories are incomplete"):
        discover_exiobase3_extensions(source)


def test_selected_environmental_extension_requires_F_Y(tmp_path) -> None:
    """Every selected extension must provide direct-final-demand F_Y as well as F."""
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    source = tmp_path / "missing_fy.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(
            zf, "air_emissions/unit.txt",
            "stressor\tunit\nCO2 - combustion - air\tkg\n",
        )
        _write_member(
            zf, "air_emissions/F.txt",
            "region\tAT\nsector\tS1\nstressor\t\nCO2 - combustion - air\t10\n",
        )
        _write_member(
            zf, "Y.txt",
            "\t\tAT\n\t\tP3\n\t\tM.EUR\nAT\tS1\t5\n",
        )

    exio_slice = ExioRegionSlice(
        year=2021, system="pxp", region="AT", sectors=pd.Index(["S1"]),
        Z_domestic=pd.DataFrame([[1.0]], index=["S1"], columns=["S1"]),
        Y_region=pd.DataFrame({"P3": [5.0]}, index=["S1"]),
        x_region=pd.Series({"S1": 6.0}, dtype="float64"),
        source_zip=source, flow="domestic",
    )
    with pytest.raises(ValueError, match="kind='F_Y'"):
        split_iot_using_exiobase_split_profiles(
            eurostat_Z=pd.DataFrame([[1.0]], index=["A"], columns=["A"]),
            eurostat_Y=pd.DataFrame({"P3": [5.0]}, index=["A"]),
            exio_slice=exio_slice,
            sector_to_exio_sectors={"A": ["S1"]},
            environmental_extension_requests={"air_emissions": ("CO2",)},
            create_io_system=False,
        )


def test_production_zero_raw_output_is_retained_for_mapping(tmp_path) -> None:
    """Production F with zero raw x remains extensive donor evidence for later mapping."""
    from ecolecon_mrio.exiobase3_extensions import (
        extract_extension_values,
        load_exiobase3_extension,
    )

    source = tmp_path / "zero_raw_output_f.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(zf, "air_emissions/unit.txt", "stressor\tunit\nCH4\tkg\n")
        _write_member(
            zf,
            "air_emissions/F.txt",
            "region\tAT\tAT\nsector\tS0\tS1\nstressor\t\t\nCH4\t2\t8\n",
        )

    load_exiobase3_extension.cache_clear()
    raw = extract_extension_values(
        container=source, region="AT", source_labels=["S0", "S1"],
        monetary_denominator=pd.Series({"S0": 0.0, "S1": 10.0}, dtype="float64"),
        extension_requests={"air_emissions": ()}, kind="F", required=True,
    )
    assert raw is not None
    by_source = raw.set_index("source_label")
    assert float(by_source.loc["S0", "value"]) == pytest.approx(2.0)
    assert pd.isna(by_source.loc["S0", "intensity"])
    load_exiobase3_extension.cache_clear()


def test_extension_nonzero_value_with_zero_denominator_fails(tmp_path) -> None:
    """A positive F_Y observation over zero monetary final demand is inconsistent."""
    from ecolecon_mrio.exiobase3_extensions import (
        extract_extension_values,
        load_exiobase3_extension,
    )

    source = tmp_path / "zero_denominator.zip"
    with zipfile.ZipFile(source, "w") as zf:
        _write_member(
            zf, "air_emissions/unit.txt",
            "stressor\tunit\nCO2 - combustion - air\tkg\n",
        )
        _write_member(
            zf, "air_emissions/F_Y.txt",
            "region\tAT\nfinal_demand_category\tP3\nstressor\t\nCO2 - combustion - air\t1\n",
        )
    load_exiobase3_extension.cache_clear()
    with pytest.raises(ValueError, match="nonzero physical value with zero monetary"):
        extract_extension_values(
            container=source, region="AT", source_labels=["P3"],
            monetary_denominator=pd.Series({"P3": 0.0}, dtype="float64"),
            extension_requests={"air_emissions": ()}, kind="F_Y", required=True,
        )
    load_exiobase3_extension.cache_clear()

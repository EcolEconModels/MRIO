from __future__ import annotations

from pathlib import Path
import importlib.util

import pandas as pd
import pytest


def test_output_share_uses_weighted_producer_output_for_canonical_p1(tmp_path: Path) -> None:
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import build_split_profiles_from_exiobase_domestic_block

    sectors = pd.Index(["a", "b"])
    # Destination use is equal (20/20), whereas producer output is deliberately
    # asymmetric (80/20).  The edge-child labels exercise concordance weights.
    Z = pd.DataFrame([[10.0, 0.0], [0.0, 10.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame({"FD": [10.0, 10.0]}, index=sectors)
    exio = ExioRegionSlice(
        year=2021, system="pxp", region="AT", sectors=sectors,
        Z_domestic=Z, Y_region=Y,
        x_region=pd.Series([80.0, 20.0], index=sectors),
        source_zip=tmp_path / "donor.zip",
    )
    profiles, _ = build_split_profiles_from_exiobase_domestic_block(
        exio,
        sector_to_exio_sectors={"CPA": {"a@CPA": ["a"], "b@CPA": ["b"]}},
        eurostat_sector_mapping={"CPA": {"a": 0.5, "b": 1.0}},
    )

    shares = profiles["CPA"].output_share
    assert shares.loc["a@CPA"] == pytest.approx(2.0 / 3.0)
    assert shares.loc["b@CPA"] == pytest.approx(1.0 / 3.0)
    assert shares.loc["a@CPA"] != pytest.approx(0.5)
    assert shares.loc["b@CPA"] != pytest.approx(0.5)
    child_p1 = shares * 120.0
    assert child_p1.loc["a@CPA"] == pytest.approx(80.0)
    assert child_p1.loc["b@CPA"] == pytest.approx(40.0)


def test_donor_block_fallback_diagnostics_count_and_columns(tmp_path: Path) -> None:
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import _split_iot_using_donor_blocks

    exio_sectors = ["a1", "a2", "b1", "b2"]
    Z_domestic = pd.DataFrame(0.0, index=exio_sectors, columns=exio_sectors)
    Y_region = pd.DataFrame({"FD": [1.0, 1.0, 1.0, 1.0]}, index=exio_sectors)
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

    eurostat_Z = pd.DataFrame([[12.0, 3.0], [4.0, 5.0]], index=["A", "B"], columns=["A", "B"])
    # Reduced Eurostat fixture: use a recognized aggregate final-demand family.
    eurostat_Y = pd.DataFrame({"P3": [1.0, 2.0]}, index=["A", "B"])
    shares = pd.Series([0.5, 0.5], index=["a1", "a2"], dtype="float64")
    split_profiles = {
        "A": SplitProfiles(
            aggregate_sector="A",
            subsectors=("a1", "a2"),
            output_share=shares,
            intermediate_sales_share=shares,
            intermediate_input_share=shares,
            final_demand_share=shares,
            subsector_to_exio_sectors={"a1": ["a1"], "a2": ["a2"]},
        )
    }

    res = _split_iot_using_donor_blocks(
        eurostat_Z=eurostat_Z,
        eurostat_Y=eurostat_Y,
        exio_slice=exio_slice,
        split_profiles=split_profiles,
        eurostat_sector_mapping=None,
        create_io_system=False,
    )

    assert res.fallback_diagnostics is not None
    assert not res.fallback_diagnostics.empty
    assert res.fallback_count == len(res.fallback_diagnostics)
    required_cols = {
        "row_sector",
        "col_sector",
        "cell_total",
        "row_labels",
        "col_labels",
        "donor_total",
        "fallback_type",
    }
    assert required_cols.issubset(set(res.fallback_diagnostics.columns))
    assert (res.fallback_diagnostics["donor_total"].fillna(0.0) == 0.0).any()

def test_donor_block_derives_controls_instead_of_allocating_them(tmp_path: Path) -> None:
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import _split_iot_using_donor_blocks

    exio_sectors = ["a1", "a2", "b"]
    Z_domestic = pd.DataFrame(0.0, index=exio_sectors, columns=exio_sectors)
    Y_region = pd.DataFrame({"FD": [1.0, 1.0, 1.0]}, index=exio_sectors)
    exio_slice = ExioRegionSlice(
        year=2020, system="pxp", region="AT", sectors=pd.Index(exio_sectors),
        Z_domestic=Z_domestic, Y_region=Y_region,
        x_region=Y_region.sum(axis=1).astype("float64"),
        source_zip=tmp_path / "IOT_2020_pxp.zip",
    )
    eurostat_Z = pd.DataFrame([[2.0, 0.0], [0.0, 3.0]], index=["A", "B"], columns=["A", "B"])
    eurostat_Y = pd.DataFrame(
        {
            "P3": [999.0, 999.0],
            "P3_S13": [1.0, 2.0],
            "P3_S14": [2.0, 3.0],
            "P3_S15": [3.0, 4.0],
            "TFU": [999.0, 999.0],
            "TU": [999.0, 999.0],
        },
        index=["A", "B"],
    )
    shares = pd.Series([0.5, 0.5], index=["a1", "a2"], dtype="float64")
    profiles = {
        "A": SplitProfiles(
            aggregate_sector="A", subsectors=("a1", "a2"), output_share=shares,
            intermediate_sales_share=shares, intermediate_input_share=shares,
            final_demand_share=shares,
            subsector_to_exio_sectors={"a1": ["a1"], "a2": ["a2"]},
        )
    }

    result = _split_iot_using_donor_blocks(
        eurostat_Z=eurostat_Z, eurostat_Y=eurostat_Y, exio_slice=exio_slice,
        split_profiles=profiles, eurostat_sector_mapping=None, create_io_system=False,
    )

    assert result.Y["P3"].equals(result.Y[["P3_S13", "P3_S14", "P3_S15"]].sum(axis=1))
    assert result.Y["TFU"].equals(result.Y["P3"])
    assert result.Y["TU"].equals(result.Z.sum(axis=1) + result.Y["TFU"])
    if result.fallback_diagnostics is not None:
        assert not set(result.fallback_diagnostics["col_sector"].astype(str)).intersection(
            {"P3", "TFU", "TU"}
        )


def _primitive_fixture(tmp_path: Path, *, dom_output: tuple[float, float] = (0.0, 1.0)):
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exiobase3 import ExioRegionSlice

    children = ("a0", "a1")
    dom = SplitProfiles(
        aggregate_sector="A", subsectors=children,
        output_share=pd.Series(dom_output, index=children),
        intermediate_sales_share=pd.Series([0.5, 0.5], index=children),
        intermediate_input_share=pd.Series([0.8, 0.2], index=children),
        final_demand_share=pd.Series([0.5, 0.5], index=children),
        subsector_to_exio_sectors={"a0": ["a0"], "a1": ["a1"]},
    )
    imp = SplitProfiles(
        aggregate_sector="A", subsectors=children,
        output_share=pd.Series([1.0, 0.0], index=children),
        intermediate_sales_share=pd.Series([0.5, 0.5], index=children),
        intermediate_input_share=pd.Series([0.1, 0.9], index=children),
        final_demand_share=pd.Series([0.5, 0.5], index=children),
        subsector_to_exio_sectors={"a0": ["a0"], "a1": ["a1"]},
    )
    raw = ["a0", "a1", "B"]
    donor = pd.DataFrame(0.0, index=raw, columns=raw)
    y = pd.DataFrame({"P3": [1.0, 1.0, 1.0]}, index=raw)
    exio = ExioRegionSlice(
        year=2021, system="pxp", region="AT", sectors=pd.Index(raw),
        Z_domestic=donor, Y_region=y, x_region=y.sum(axis=1),
        source_zip=tmp_path / "donors.zip", flow="imports",
    )
    Z = pd.DataFrame([[5.0, 8.0], [2.0, 3.0]], index=["A", "B"], columns=["A", "B"])
    Y = pd.DataFrame({"P3": [2.0, 1.0]}, index=["A", "B"])
    return Z, Y, exio, {"A": dom}, {"A": imp}


def test_primitive_imp_uses_dom_columns_and_defers_x_a(tmp_path: Path) -> None:
    from ecolecon_mrio.exio_share_split import _split_iot_using_donor_blocks

    Z, Y, exio, dom_profiles, imp_profiles = _primitive_fixture(tmp_path)
    dom_result = _split_iot_using_donor_blocks(
        eurostat_Z=Z, eurostat_Y=Y, exio_slice=exio,
        split_profiles=dom_profiles, column_split_profiles=dom_profiles,
        eurostat_sector_mapping=None, create_io_system=False,
    )
    result = _split_iot_using_donor_blocks(
        eurostat_Z=Z, eurostat_Y=Y, exio_slice=exio,
        split_profiles=imp_profiles, column_split_profiles=dom_profiles,
        eurostat_sector_mapping=None, create_io_system=False,
    )
    assert list(dom_result.Z.index) == list(result.Z.index)
    assert list(result.Z.index) == ["a0", "a1", "B"]
    assert list(result.Z.columns) == ["a0", "a1", "B"]
    assert float(dom_result.Z["a0"].sum()) == 0.0
    assert float(result.Z["a0"].sum()) == 0.0
    assert float(result.Z["a1"].sum()) > 0.0
    assert result.x is None and result.A is None
    assert result.use_total.equals(result.Z.sum(axis=1) + result.Y["P3"])


def test_donor_projection_uses_separate_row_and_column_profiles() -> None:
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import project_exio_donor_blocks_for_split_profiles

    children = ("c0", "c1")
    shares = pd.Series([0.5, 0.5], index=children)
    row = SplitProfiles("A", children, shares, shares, shares, shares, {"c0": ["r0"], "c1": ["r1"]})
    column = SplitProfiles("A", children, shares, shares, shares, shares, {"c0": ["r1"], "c1": ["r0"]})
    raw_z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["r0", "r1"], columns=["r0", "r1"])
    raw_y = pd.DataFrame({"P3": [5.0, 6.0]}, index=["r0", "r1"])
    _, donor_z, donor_y = project_exio_donor_blocks_for_split_profiles(
        exio_Z=raw_z, exio_Y=raw_y, split_profiles={"A": row},
        column_split_profiles={"A": column}, raw_exio_sectors=["r0", "r1"],
        emitted_labels=children,
    )
    pd.testing.assert_frame_equal(donor_z, pd.DataFrame([[2.0, 1.0], [4.0, 3.0]], index=children, columns=children))
    pd.testing.assert_frame_equal(donor_y, pd.DataFrame({"P3": [5.0, 6.0]}, index=children))


def test_mapped_imp_rows_use_imp_weights_and_columns_use_dom_weights() -> None:
    from ecolecon_mrio.disaggregation import SplitProfiles
    from ecolecon_mrio.exio_share_split import project_exio_donor_blocks_for_split_profiles

    children = ("s@A", "s@B")
    shares = pd.Series([0.5, 0.5], index=children)
    profiles = {"A": SplitProfiles("A", children, shares, shares, shares, shares, {"s@A": ["s"], "s@B": ["s"]})}
    dom_mapping = {"A": {"s": 0.9}, "B": {"s": 0.1}}
    imp_mapping = {"A": {"s": 0.2}, "B": {"s": 0.8}}
    raw_z = pd.DataFrame([[10.0]], index=["s"], columns=["s"])
    raw_y = pd.DataFrame({"P3": [4.0]}, index=["s"])
    _, donor_z, donor_y = project_exio_donor_blocks_for_split_profiles(
        exio_Z=raw_z, exio_Y=raw_y, split_profiles=profiles,
        column_split_profiles=profiles, eurostat_sector_mapping=imp_mapping,
        column_eurostat_sector_mapping=dom_mapping, raw_exio_sectors=["s"],
        emitted_labels=children,
    )
    expected = pd.DataFrame([[1.8, 0.2], [7.2, 0.8]], index=children, columns=children)
    pd.testing.assert_frame_equal(donor_z, expected)
    pd.testing.assert_frame_equal(donor_y, pd.DataFrame({"P3": [0.8, 3.2]}, index=children))
    assert not donor_z.equals(pd.DataFrame([[0.4, 1.6], [1.6, 6.4]], index=children, columns=children))


def test_public_import_split_requires_domestic_column_profiles(tmp_path: Path) -> None:
    from ecolecon_mrio.exiobase3 import ExioRegionSlice
    from ecolecon_mrio.exio_share_split import split_iot_using_exiobase_split_profiles

    raw = pd.Index(["s"])
    exio = ExioRegionSlice(
        year=2021, system="pxp", region="AT", flow="imports", sectors=raw,
        Z_domestic=pd.DataFrame([[1.0]], index=raw, columns=raw),
        Y_region=pd.DataFrame({"P3": [1.0]}, index=raw),
        x_region=pd.Series([2.0], index=raw), source_zip=tmp_path / "donors.zip",
    )
    with pytest.raises(ValueError, match="requires domestic production-column profiles"):
        split_iot_using_exiobase_split_profiles(
            eurostat_Z=pd.DataFrame([[1.0]], index=["A"], columns=["A"]),
            eurostat_Y=pd.DataFrame({"P3": [1.0]}, index=["A"]), exio_slice=exio,
            sector_to_exio_sectors={"A": ["s"]}, create_io_system=False,
        )


def test_primitive_zero_output_parent_with_inputs_fails_contextually(tmp_path: Path) -> None:
    from ecolecon_mrio.exio_share_split import _split_iot_using_donor_blocks

    Z, Y, exio, dom_profiles, _ = _primitive_fixture(tmp_path, dom_output=(0.0, 0.0))
    with pytest.raises(ValueError, match="parent='A'.*source column total=.*flow='imports'"):
        _split_iot_using_donor_blocks(
            eurostat_Z=Z, eurostat_Y=Y, exio_slice=exio,
            split_profiles=dom_profiles, column_split_profiles=dom_profiles,
            eurostat_sector_mapping=None, create_io_system=False,
        )


def test_primitive_writer_emits_use_total_but_not_x_or_a(tmp_path: Path) -> None:
    from ecolecon_mrio.exio_share_split import _split_iot_using_donor_blocks

    Z, Y, exio, dom_profiles, _ = _primitive_fixture(tmp_path)
    result = _split_iot_using_donor_blocks(
        eurostat_Z=Z, eurostat_Y=Y, eurostat_M=pd.DataFrame([[1.0]], index=["m"], columns=["P3"]),
        exio_slice=exio, split_profiles=dom_profiles, column_split_profiles=dom_profiles,
        eurostat_sector_mapping=None, create_io_system=False,
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "split_eurostat_energy_by_exiobase.py"
    spec = importlib.util.spec_from_file_location("split_writer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_split_core_outputs(result, tmp_path / "out", verbose=False)
    out = tmp_path / "out"
    assert (out / "Z_split.csv").is_file() and (out / "Y_split.csv").is_file()
    assert (out / "use_total_split.csv").is_file() and (out / "M_split.csv").is_file()
    assert not any((out / name).exists() for name in ("x_split.csv", "A_split.csv", "x_split_balanced.csv", "A_split_balanced.csv"))

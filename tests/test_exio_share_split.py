from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    eurostat_Y = pd.DataFrame({"FD": [1.0, 2.0]}, index=["A", "B"])
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
        sector_to_exio_sectors={"A": ["a1", "a2"]},
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

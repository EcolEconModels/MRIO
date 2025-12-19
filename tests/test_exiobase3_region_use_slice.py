from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_build_exio_region_use_slice_from_pymrio_test_data(tmp_path):
    import pymrio

    from ecolecon_mrio.exiobase3 import build_exio_region_use_slice_from_mrio

    mrio = pymrio.load_test()
    Z = mrio.Z
    Y = mrio.Y
    region = "reg1"
    dummy_zip = Path(tmp_path) / "dummy.zip"

    # Expected use-side slices into region.
    Z_to_r = Z.xs(region, level=0, axis=1)
    Y_to_r = Y.xs(region, level=0, axis=1)

    Z_dom = Z_to_r.xs(region, level=0, axis=0)
    Y_dom = Y_to_r.xs(region, level=0, axis=0)

    Z_foreign = Z_to_r.loc[Z_to_r.index.get_level_values(0) != region]
    Y_foreign = Y_to_r.loc[Y_to_r.index.get_level_values(0) != region]
    Z_imp = Z_foreign.groupby(level=1).sum()
    Y_imp = Y_foreign.groupby(level=1).sum()

    Z_tot = Z_to_r.groupby(level=1).sum()
    Y_tot = Y_to_r.groupby(level=1).sum()

    # Domestic
    s_dom = build_exio_region_use_slice_from_mrio(
        year=0, system="test", region=region, Z=Z, Y=Y, source_zip=dummy_zip, flow="domestic"
    )
    assert s_dom.flow == "domestic"
    pd.testing.assert_frame_equal(s_dom.Z_domestic, Z_dom)
    pd.testing.assert_frame_equal(s_dom.Y_region, Y_dom)
    pd.testing.assert_series_equal(s_dom.x_region, (Z_dom.sum(axis=1) + Y_dom.sum(axis=1)).astype("float64"))

    # Imports
    s_imp = build_exio_region_use_slice_from_mrio(
        year=0, system="test", region=region, Z=Z, Y=Y, source_zip=dummy_zip, flow="imports"
    )
    assert s_imp.flow == "imports"
    pd.testing.assert_frame_equal(s_imp.Z_domestic, Z_imp)
    pd.testing.assert_frame_equal(s_imp.Y_region, Y_imp)
    pd.testing.assert_series_equal(s_imp.x_region, (Z_imp.sum(axis=1) + Y_imp.sum(axis=1)).astype("float64"))

    # Total
    s_tot = build_exio_region_use_slice_from_mrio(
        year=0, system="test", region=region, Z=Z, Y=Y, source_zip=dummy_zip, flow="total"
    )
    assert s_tot.flow == "total"
    pd.testing.assert_frame_equal(s_tot.Z_domestic, Z_tot)
    pd.testing.assert_frame_equal(s_tot.Y_region, Y_tot)
    pd.testing.assert_series_equal(s_tot.x_region, (Z_tot.sum(axis=1) + Y_tot.sum(axis=1)).astype("float64"))

    # Additivity check (pure accounting identity on use side)
    pd.testing.assert_frame_equal(Z_tot, Z_dom + Z_imp)
    pd.testing.assert_frame_equal(Y_tot, Y_dom + Y_imp)


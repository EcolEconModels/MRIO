from __future__ import annotations

import numpy as np
import pandas as pd


def test_aggregate_iot_keeps_unmapped_and_sums_correctly():
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    sectors = ["A", "B", "C"]
    Z = pd.DataFrame(
        [[1.0, 2.0, 3.0], [0.5, 1.0, 0.0], [4.0, 0.0, 2.0]],
        index=sectors,
        columns=sectors,
    )
    Y = pd.DataFrame({"FD": [1.0, 2.0, 3.0]}, index=sectors)

    # Aggregate B and C into X; keep A unmapped.
    mapping = {"B": "X", "C": "X"}
    res = aggregate_iot(Z, Y, mapping, keep_unmapped=True, create_io_system=False)

    assert list(res.Z.index) == ["A", "X"]
    assert list(res.Z.columns) == ["A", "X"]
    assert list(res.Y.index) == ["A", "X"]

    # Manual aggregation check.
    Zm = Z.copy()
    Zm.index = ["A", "X", "X"]
    Zm.columns = ["A", "X", "X"]
    Z_expected = Zm.groupby(level=0).sum().T.groupby(level=0).sum().T

    Ym = Y.copy()
    Ym.index = ["A", "X", "X"]
    Y_expected = Ym.groupby(level=0).sum()

    pd.testing.assert_frame_equal(res.Z, Z_expected)
    pd.testing.assert_frame_equal(res.Y, Y_expected)

    x_expected = (Z_expected.sum(axis=1) + Y_expected.sum(axis=1)).astype("float64")
    pd.testing.assert_series_equal(res.x, x_expected)

    # A = Z / x (column-wise)
    A_expected = Z_expected.div(x_expected.replace(0.0, np.nan), axis=1).fillna(0.0)
    pd.testing.assert_frame_equal(res.A, A_expected)


def test_aggregate_square_matrix_multiindex_sector_level():
    from ecolecon_mrio.iot_aggregation import aggregate_square_matrix

    idx = pd.MultiIndex.from_product([["AT"], ["s1", "s2", "s3"]], names=["region", "sector"])
    Z = pd.DataFrame(np.eye(3), index=idx, columns=idx)
    mapping = {"s2": "S", "s3": "S"}

    Z_agg = aggregate_square_matrix(Z, mapping, level=1, keep_unmapped=True)
    assert isinstance(Z_agg.index, pd.MultiIndex)
    assert Z_agg.index.names == ["region", "sector"]
    assert ("AT", "S") in Z_agg.index
    assert ("AT", "s1") in Z_agg.index

# pyright: reportArgumentType=false
from __future__ import annotations

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_flow_semantics import flow_use_total


def test_aggregate_iot_keeps_unmapped_and_sums_correctly():
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    sectors = ["A", "B", "C"]
    Z = pd.DataFrame(
        [[1.0, 2.0, 3.0], [0.5, 1.0, 0.0], [4.0, 0.0, 2.0]],
        index=sectors,
        columns=sectors,
    )
    # P3 is a non-overlapping final-demand fixture, so the explicit production
    # helper is numerically the familiar Z + Y total here.
    Y = pd.DataFrame({"P3": [1.0, 2.0, 3.0]}, index=sectors)

    # Aggregate B and C into X; keep A unmapped.
    mapping = {"B": "X", "C": "X"}
    res = aggregate_iot(Z, Y, mapping, output=flow_use_total(Z, Y), keep_unmapped=True, create_io_system=False)

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

    x_expected = (Z_expected.sum(axis=1) + Y_expected.sum(axis=1)).astype("float64").rename("x")
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


def test_aggregate_iot_preserves_final_demand_column_totals():
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    sectors = ["A", "B", "C"]
    Z = pd.DataFrame(
        [[1.0, 2.0, 3.0], [0.5, 1.0, 0.0], [4.0, 0.0, 2.0]],
        index=sectors,
        columns=sectors,
    )
    Y = pd.DataFrame(
        {
            "P3": [1.0, 2.0, 3.0],
            "P5": [0.5, 1.0, 1.5],
            "P6": [2.0, 0.0, 1.0],
        },
        index=sectors,
    )

    mapping = {"B": "X", "C": "X"}
    res = aggregate_iot(Z, Y, mapping, output=flow_use_total(Z, Y), keep_unmapped=True, create_io_system=False)

    pd.testing.assert_series_equal(res.Y.sum(axis=0).astype("float64"), Y.sum(axis=0).astype("float64"))
    pd.testing.assert_series_equal(
        res.x.astype("float64"),
        (res.Z.sum(axis=1) + res.Y.sum(axis=1)).astype("float64").rename("x"),
    )


def test_aggregate_iot_reaggregates_supplementary_rows_columnwise():
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    sectors = ["p24.d@CPA_C20", "p24.g", "CPA_C21"]
    Z = pd.DataFrame(np.eye(3), index=sectors, columns=sectors)
    Y = pd.DataFrame({"P3": [1.0, 2.0, 3.0]}, index=sectors)
    V = pd.DataFrame(
        [[10.0, 80.0, 5.0], [4.0, 5.0, 6.0], [999.0, 999.0, 999.0], [1.0, 2.0, 3.0]],
        index=["Value added, gross", "Imports of goods and services", "Total supply at basic prices", "Taxes less subsidies on products"],
        columns=sectors,
    )
    M = pd.DataFrame({"P3": [7.0, 9.0, 8.0, 8.0]}, index=V.index)
    mapping = {"p24.d@CPA_C20": "CPA_C20_others", "p24.g": "CPA_C20_BIOGASOLINE"}

    res = aggregate_iot(Z, Y, mapping, V=V, M=M, output=flow_use_total(Z, Y), keep_unmapped=True, create_io_system=False)

    assert res.V is not None
    assert res.M is not None
    expected_v = V.copy()
    expected_v.columns = ["CPA_C20_others", "CPA_C20_BIOGASOLINE", "CPA_C21"]
    expected_v = expected_v.T.groupby(level=0).sum().T
    pd.testing.assert_frame_equal(res.V, expected_v)
    pd.testing.assert_frame_equal(res.M, M)


def test_aggregate_iot_preserves_m_exactly_including_zero_and_nan():
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    sectors = ["A", "B"]
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=sectors, columns=sectors)
    Y = pd.DataFrame(
        {
            "P3": [10.0, 20.0],
            "P5": [1.0, 2.0],
            "P6": [3.0, 4.0],
        },
        index=sectors,
    )
    M = pd.DataFrame(
        [[float("nan"), 0.0, -1.0, -1.0, -1.0], [5.0, 6.0, 7.0, 8.0, 9.0]],
        index=["Total intermediate consumption / final use", "Other supplementary row"],
        columns=["P3", "P5", "P6", "TFU", "TU"],
    )

    res = aggregate_iot(Z, Y, {}, M=M, output=flow_use_total(Z, Y), keep_unmapped=True, create_io_system=False)

    assert res.M is not None
    pd.testing.assert_frame_equal(res.M, M)
    assert pd.isna(res.M.iloc[0, 0])
    assert res.M.iloc[0, 1] == 0.0

def test_aggregate_iot_rejects_misaligned_output_instead_of_zero_filling() -> None:
    import pytest
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    Z = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "B"])
    Y = pd.DataFrame({"P3": [1.0, 1.0]}, index=["A", "B"])
    bad_output = pd.Series([2.0], index=["A"], name="x")
    with pytest.raises(ValueError, match="exactly match"):
        aggregate_iot(Z, Y, {}, output=bad_output, create_io_system=False)


def test_aggregate_iot_rejects_extra_output_sector() -> None:
    import pytest
    from ecolecon_mrio.iot_aggregation import aggregate_iot

    Z = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    Y = pd.DataFrame({"P3": [1.0]}, index=["A"])
    bad_output = pd.Series([2.0, 10.0], index=["A", "EXTRA"], name="x")
    with pytest.raises(ValueError, match="exactly match"):
        aggregate_iot(Z, Y, {}, output=bad_output, create_io_system=False)

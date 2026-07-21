# pyright: reportArgumentType=false
from __future__ import annotations

import numpy as np
import pandas as pd

from ecolecon_mrio.iot_flow_semantics import (
    atomic_final_demand_columns,
    flow_use_total,
    refresh_supplementary_rows,
    technical_coefficients,
)


def test_atomic_final_demand_excludes_overlapping_aggregates_and_summaries() -> None:
    cols = pd.Index(["P3", "P3_S13", "P3_S14", "P3_S15", "P5", "P5M", "P51G", "P52", "P53", "P6", "P6_B0", "TU", "TFU"])
    assert atomic_final_demand_columns(cols) == ["P3_S13", "P3_S14", "P3_S15", "P51G", "P52", "P53", "P6"]


def test_output_and_coefficients_use_atomic_final_demand() -> None:
    idx = pd.Index(["a", "b"])
    Z = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=idx, columns=idx)
    Y = pd.DataFrame({
        "P3": [100.0, 200.0], "P3_S13": [10.0, 20.0], "P3_S14": [30.0, 40.0], "P3_S15": [60.0, 140.0],
        "P5": [50.0, 60.0], "P51G": [20.0, 30.0], "P52": [15.0, 15.0], "P53": [15.0, 15.0],
        "P6": [5.0, 6.0], "P6_B0": [5.0, 6.0], "TU": [999.0, 999.0], "TFU": [999.0, 999.0],
    }, index=idx)
    output = flow_use_total(Z, Y)
    expected = Z.sum(axis=1) + Y[["P3_S13", "P3_S14", "P3_S15", "P51G", "P52", "P53", "P6"]].sum(axis=1)
    pd.testing.assert_series_equal(output, expected.rename("use_total"))
    A = technical_coefficients(Z, output)
    np.testing.assert_allclose(A.to_numpy(), Z.divide(output, axis="columns").to_numpy())


def test_partial_flow_retains_flow_specific_value_added_and_total_supply_rows() -> None:
    idx = pd.Index(["a"])
    Z = pd.DataFrame([[2.0]], index=idx, columns=idx)
    Y = pd.DataFrame({"P3": [3.0], "P5": [0.0], "P6": [0.0]}, index=idx)
    V = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=["P1", "P2_ADJ", "B1G", "TS_BP"])
    out = pd.Series([5.0], index=idx, name="x")
    V_imp, _ = refresh_supplementary_rows(V, None, Z=Z, Y=Y, output=out, flow_role="IMP")
    assert V_imp is not None
    assert float(V_imp.loc["P1", "a"]) == 5.0
    assert float(V_imp.loc["P2_ADJ", "a"]) == 2.0
    assert float(V_imp.loc["B1G", "a"]) == 3.0
    assert float(V_imp.loc["TS_BP", "a"]) == 4.0

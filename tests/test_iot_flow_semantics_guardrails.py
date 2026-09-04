from __future__ import annotations

import pandas as pd
import pytest

from ecolecon_mrio.iot_flow_semantics import atomic_final_demand, technical_coefficients


def test_atomic_final_demand_accepts_only_p6() -> None:
    assert list(atomic_final_demand(pd.DataFrame({"P6": [1.0]})).columns) == ["P6"]


def test_atomic_final_demand_uses_non_overlapping_p6_details() -> None:
    Y = pd.DataFrame({"P6_B0": [5.0], "P6_D0": [2.0], "P6_U2": [3.0], "P6_U3": [2.0]})
    assert list(atomic_final_demand(Y).columns) == ["P6_B0", "P6_D0"]


def test_atomic_final_demand_accepts_only_p3() -> None:
    assert list(atomic_final_demand(pd.DataFrame({"P3": [1.0]})).columns) == ["P3"]


def test_atomic_final_demand_skips_an_entirely_absent_p3_family() -> None:
    assert list(atomic_final_demand(pd.DataFrame({"P5": [1.0]})).columns) == ["P5"]


def test_atomic_final_demand_uses_p3_aggregate_when_detail_partition_is_partial() -> None:
    Y = pd.DataFrame({"P3": [10.0], "P3_S13": [3.0]})
    assert list(atomic_final_demand(Y).columns) == ["P3"]


def test_atomic_final_demand_rejects_partial_p3_partition_without_aggregate() -> None:
    with pytest.raises(ValueError, match="Incomplete P3 detail partition without aggregate"):
        atomic_final_demand(pd.DataFrame({"P3_S13": [3.0]}))


def test_atomic_final_demand_uses_complete_consistent_p3_detail_partition() -> None:
    Y = pd.DataFrame({"P3": [9.0], "P3_S13": [3.0], "P3_S14": [4.0], "P3_S15": [2.0]})
    assert list(atomic_final_demand(Y).columns) == ["P3_S13", "P3_S14", "P3_S15"]


def test_atomic_final_demand_rejects_inconsistent_complete_p3_detail_partition() -> None:
    Y = pd.DataFrame({"P3": [10.0], "P3_S13": [3.0], "P3_S14": [4.0], "P3_S15": [2.0]})
    with pytest.raises(ValueError, match="P3 aggregate/detail identity"):
        atomic_final_demand(Y)


def test_atomic_final_demand_rejects_no_recognised_columns() -> None:
    with pytest.raises(ValueError, match="no recognised non-overlapping"):
        atomic_final_demand(pd.DataFrame({"TU": [1.0]}))


def test_atomic_final_demand_rejects_partial_p5_partition_even_with_aggregate() -> None:
    Y = pd.DataFrame({"P5": [10.0], "P51G": [3.0]})
    with pytest.raises(ValueError, match="Incomplete P5 investment partition"):
        atomic_final_demand(Y)


def test_atomic_final_demand_rejects_partial_p5_partition_without_aggregate() -> None:
    with pytest.raises(ValueError, match="Incomplete P5 investment partition"):
        atomic_final_demand(pd.DataFrame({"P51G": [3.0]}))


def test_atomic_final_demand_uses_complete_nested_p5_partition() -> None:
    Y = pd.DataFrame({"P5": [9.0], "P51G": [3.0], "P5M": [6.0]})
    assert list(atomic_final_demand(Y).columns) == ["P51G", "P5M"]


def test_atomic_final_demand_accepts_aggregate_only_p5() -> None:
    assert list(atomic_final_demand(pd.DataFrame({"P5": [9.0]})).columns) == ["P5"]


def test_atomic_final_demand_uses_complete_consistent_p5_detail_partition() -> None:
    Y = pd.DataFrame({"P5": [9.0], "P51G": [3.0], "P52": [4.0], "P53": [2.0]})
    assert list(atomic_final_demand(Y).columns) == ["P51G", "P52", "P53"]


def test_atomic_final_demand_rejects_inconsistent_complete_p5_detail_partition() -> None:
    Y = pd.DataFrame({"P5": [10.0], "P51G": [3.0], "P52": [4.0], "P53": [2.0]})
    with pytest.raises(ValueError, match="P5 aggregate/detail identity"):
        atomic_final_demand(Y)


def test_technical_coefficients_reject_zero_output_with_nonzero_inputs() -> None:
    Z = pd.DataFrame([[1.0]], index=pd.Index(["a"]), columns=pd.Index(["a"]))
    with pytest.raises(ValueError, match=r"(?i)zero output with non-zero intermediate inputs"):
        technical_coefficients(Z, pd.Series([0.0], index=["a"]))


def test_technical_coefficients_allows_structurally_zero_activity() -> None:
    Z = pd.DataFrame([[0.0]], index=pd.Index(["a"]), columns=pd.Index(["a"]))
    A = technical_coefficients(Z, pd.Series([0.0], index=["a"]))
    assert float(A.loc["a", "a"]) == 0.0


def test_display_aggregates_are_derived_from_atomic_details() -> None:
    from ecolecon_mrio.iot_flow_semantics import enforce_final_demand_display_identities

    Y = pd.DataFrame(
        {
            "P3": [999.0],
            "P3_S13": [1.0],
            "P3_S14": [2.0],
            "P3_S15": [3.0],
            "P5": [999.0],
            "P51G": [4.0],
            "P52": [5.0],
            "P53": [6.0],
            "P6": [7.0],
            "P6_B0": [5.0],
            "P6_D0": [999.0],
            "P6_U2": [3.0],
            "P6_U3": [999.0],
        },
        index=pd.Index(["a"]),
    )
    out = enforce_final_demand_display_identities(Y)
    assert out.loc["a", "P3"] == 6.0
    assert out.loc["a", "P5"] == 15.0
    assert out.loc["a", "P6"] == 7.0
    assert out.loc["a", "P6_D0"] == 2.0
    assert out.loc["a", "P6_U3"] == 2.0


def test_display_aggregates_derive_nested_p5_without_fabricating_flat_details() -> None:
    from ecolecon_mrio.iot_flow_semantics import enforce_final_demand_display_identities

    out = enforce_final_demand_display_identities(
        pd.DataFrame({"P5": [999.0], "P51G": [4.0], "P5M": [6.0]}, index=pd.Index(["a"]))
    )
    assert out.loc["a", "P5"] == 10.0
    assert "P52" not in out.columns
    assert "P53" not in out.columns


def test_display_export_residuals_allow_rounded_geographic_crossing() -> None:
    """Rounded geographic controls may cross while atomic P6 remains unchanged."""
    from ecolecon_mrio.iot_flow_semantics import (
        atomic_final_demand,
        enforce_final_demand_display_identities,
    )

    Y = pd.DataFrame(
        {
            "P6": [3.0],
            "P6_B0": [4.0],
            "P6_D0": [999.0],
            "P6_U2": [5.0],
            "P6_U3": [999.0],
        },
        index=pd.Index(["a"]),
    )
    out = enforce_final_demand_display_identities(Y)

    assert float(out.loc["a", "P6"]) == 3.0
    assert float(out.loc["a", "P6_B0"]) == 4.0
    assert float(out.loc["a", "P6_U2"]) == 5.0
    assert float(out.loc["a", "P6_D0"]) == -1.0
    assert float(out.loc["a", "P6_U3"]) == -1.0
    assert list(atomic_final_demand(out).columns) == ["P6"]


def test_derived_final_demand_controls_and_tu_are_rebuilt() -> None:
    from ecolecon_mrio.iot_flow_semantics import (
        derived_final_demand_control_columns,
        enforce_final_demand_display_identities,
    )

    idx = pd.Index(["a"])
    Z = pd.DataFrame([[10.0]], index=idx, columns=idx)
    Y = pd.DataFrame(
        {
            "P3": [999.0],
            "P3_S13": [1.0],
            "P3_S14": [2.0],
            "P3_S15": [3.0],
            "P6": [4.0],
            "TFU": [999.0],
            "TU": [999.0],
        },
        index=idx,
    )

    assert derived_final_demand_control_columns(Y.columns) == ["P3", "TFU", "TU"]
    out = enforce_final_demand_display_identities(Y, Z=Z)
    assert out.loc["a", "P3"] == 6.0
    assert out.loc["a", "TFU"] == 10.0
    assert out.loc["a", "TU"] == 20.0

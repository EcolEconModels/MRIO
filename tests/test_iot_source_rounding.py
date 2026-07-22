from __future__ import annotations

import pandas as pd
import pytest

from ecolecon_mrio.eurostat_iot import infer_source_rounding_quantum
from ecolecon_mrio.iot_flow_semantics import (
    atomic_final_demand,
    rounded_source_control_diagnostic,
    source_atomic_final_demand,
)


def test_infer_two_decimal_source_quantum() -> None:
    payload = {"value": {"0": 1, "1": 2.5, "2": 3.14}}
    assert infer_source_rounding_quantum(payload) == pytest.approx(0.01)


def test_rounded_control_accepts_accumulated_quantisation_gap() -> None:
    diagnostic = rounded_source_control_diagnostic(
        identity="P3",
        published=6.01,
        derived=6.0,
        component_term_count=3,
        source_rounding_quantum=0.01,
        rtol=1e-9,
        atol=1e-6,
    )
    assert diagnostic["ok"] is True
    assert diagnostic["rounding_bound"] == pytest.approx(0.020001)


def test_rounded_control_rejects_gap_beyond_quantisation_bound() -> None:
    with pytest.raises(ValueError, match="Published Eurostat P3 control failed"):
        rounded_source_control_diagnostic(
            identity="P3",
            published=6.03,
            derived=6.0,
            component_term_count=3,
            source_rounding_quantum=0.01,
            rtol=1e-9,
            atol=1e-6,
        )


def test_rounding_bound_uses_underlying_component_count() -> None:
    diagnostic = rounded_source_control_diagnostic(
        identity="TS_BP",
        published=10.03,
        derived=10.0,
        component_term_count=7,
        source_rounding_quantum=0.01,
        rtol=0.0,
        atol=0.0,
    )
    assert diagnostic["rounding_bound"] == pytest.approx(0.04)

def test_published_source_basis_allows_rounded_aggregate_controls() -> None:
    Y = pd.DataFrame(
        {
            "P3": [6.01],
            "P3_S13": [1.0],
            "P3_S14": [2.0],
            "P3_S15": [3.0],
            "P5": [15.01],
            "P51G": [4.0],
            "P52": [5.0],
            "P53": [6.0],
            "P6": [7.0],
        },
        index=pd.Index(["a"]),
    )

    with pytest.raises(ValueError, match="P3 aggregate/detail identity"):
        atomic_final_demand(Y)

    source_atomic = source_atomic_final_demand(Y)
    assert list(source_atomic.columns) == [
        "P3_S13", "P3_S14", "P3_S15", "P51G", "P52", "P53", "P6"
    ]
    assert float(source_atomic.sum(axis=1).iloc[0]) == pytest.approx(28.0)

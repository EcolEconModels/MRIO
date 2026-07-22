from __future__ import annotations

import pytest

from ecolecon_mrio.eurostat_iot import infer_source_rounding_quantum
from ecolecon_mrio.iot_flow_semantics import rounded_source_control_diagnostic


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

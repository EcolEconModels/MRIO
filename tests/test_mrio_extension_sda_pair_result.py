from __future__ import annotations

import pandas as pd
import pytest
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import spsolve

from ecolecon_mrio.exiobase3_extensions import ExtensionRowResult
from ecolecon_mrio.mrio_extension_sda import (
    compute_extension_intensity_state,
    compute_extension_sda_pair_result_from_states,
)
from ecolecon_mrio.mrio_sda import MrioYearState


def _idx() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([("R1", "s1"), ("R1", "s2")], names=["region", "sector"])


def _series(values: list[float], index: pd.MultiIndex, name: str) -> pd.Series:
    return pd.Series(values, index=index, dtype=float, name=name)


def _state(year: int, A: csr_matrix, y_values: list[float]) -> MrioYearState:
    index = _idx()
    y = _series(y_values, index, f"y{year}")
    x_values = spsolve(sparse_eye(2, format="csr") - A, y.to_numpy(dtype=float))
    x = pd.Series(x_values, index=index, dtype=float, name=f"x{year}")
    return MrioYearState(
        year=year,
        A=A,
        x=x,
        y=y,
        index=index,
        monetary_unit="MEUR",
        monetary_unit_source="test_fixture",
        metadata={"fixture": True},
    )


def _extension(values: pd.Series, *, year: int, unit: str = "kg", stressor: str = "CO2") -> ExtensionRowResult:
    return ExtensionRowResult(
        year=year,
        values=values,
        stressor=stressor,
        unit=unit,
        unit_source="test/unit.txt",
        extension="air_emissions",
        kind="F",
        metadata={"fixture": True},
    )


def test_extension_sda_six_polar_components_sum_to_stabilized_delta() -> None:
    """Protect the scientific identity Δf~=Δq+ΔA+Δy+r, not monetary post-valuation."""

    index = _idx()
    A0 = csr_matrix([[0.10, 0.02], [0.03, 0.12]], dtype=float)
    A1 = csr_matrix([[0.11, 0.01], [0.04, 0.10]], dtype=float)
    start = _state(2000, A0, [10.0, 7.0])
    end = _state(2001, A1, [11.0, 8.5])
    q0 = _series([2.0, 3.0], index, "q0")
    q1 = _series([2.5, 2.7], index, "q1")
    f0 = (q0 * start.x).rename("CO2")
    f1 = (q1 * end.x).rename("CO2")

    result = compute_extension_sda_pair_result_from_states(
        start,
        end,
        _extension(f0, year=2000),
        _extension(f1, year=2001),
        stabilization="raw",
    )

    reconstructed = (
        result.sda_intensity_delta
        + result.sda_structure_delta
        + result.sda_demand_delta
        + result.sda_residual
    )
    pd.testing.assert_series_equal(result.stabilized_extension_delta, reconstructed, check_names=False)
    assert result.residual_l2 < 1.0e-10
    assert result.residual_share_l2 < 1.0e-10
    assert result.metadata["identity_mode"] == "six_polar_three_factor_extension_sda"


def test_extension_sda_has_nonzero_intensity_component_when_intensity_changes() -> None:
    """Catch regressions to average-intensity times monetary SDA components."""

    index = _idx()
    A = csr_matrix([[0.05, 0.0], [0.0, 0.05]], dtype=float)
    start = _state(2000, A, [10.0, 10.0])
    end = _state(2001, A, [10.0, 10.0])
    f0 = (_series([1.0, 1.0], index, "q0") * start.x).rename("CO2")
    f1 = (_series([3.0, 1.0], index, "q1") * end.x).rename("CO2")

    result = compute_extension_sda_pair_result_from_states(
        start,
        end,
        _extension(f0, year=2000),
        _extension(f1, year=2001),
        stabilization="raw",
    )

    assert float(result.sda_intensity_delta.abs().sum()) > 0.0
    assert float(result.sda_structure_delta.abs().sum()) == pytest.approx(0.0)
    assert float(result.sda_demand_delta.abs().sum()) == pytest.approx(0.0)


def test_threshold_stabilization_reports_raw_gap() -> None:
    """Small-output stabilization must be visible as a gap to raw observed data."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0e-12, 10.0])
    end = _state(2001, A, [1.0e-12, 12.0])
    f0 = _series([1.0, 20.0], index, "CO2")
    f1 = _series([2.0, 24.0], index, "CO2")

    result = compute_extension_sda_pair_result_from_states(
        start,
        end,
        _extension(f0, year=2000),
        _extension(f1, year=2001),
        stabilization="threshold",
        tau=1.0e-8,
    )

    bridge = result.stabilized_extension_delta + result.stabilization_gap_delta
    pd.testing.assert_series_equal(result.observed_extension_delta, bridge, check_names=False)
    assert float(result.stabilization_gap_delta.abs().sum()) > 0.0


def test_extension_sda_rejects_f_y_index() -> None:
    """Production SDA must not silently treat F_Y categories as sector intensities."""

    fd_index = pd.MultiIndex.from_tuples(
        [("R1", "households"), ("R1", "government")], names=["region", "final_demand_category"]
    )
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    f_y = ExtensionRowResult(
        year=2000,
        values=pd.Series([1.0, 2.0], index=fd_index, dtype=float, name="CO2"),
        stressor="CO2",
        unit="kg",
        unit_source="test/unit.txt",
        extension="air_emissions",
        kind="F_Y",
        metadata={"fixture": True},
    )

    with pytest.raises(ValueError, match="kind='F'"):
        compute_extension_intensity_state(mrio_state=start, extension_row=f_y)


def test_extension_sda_rejects_misaligned_extension_index() -> None:
    """Wrong sector order must fail instead of being silently reindexed."""

    index = _idx()
    reversed_index = pd.MultiIndex.from_tuples(list(reversed(list(index))), names=["region", "sector"])
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    end = _state(2001, A, [1.0, 2.0])
    good = _extension(_series([1.0, 2.0], index, "CO2"), year=2000)
    bad = _extension(pd.Series([1.0, 2.0], index=reversed_index, dtype=float, name="CO2"), year=2001)

    with pytest.raises(ValueError, match="does not exactly match"):
        compute_extension_sda_pair_result_from_states(start, end, good, bad)


def test_extension_sda_rejects_unit_mismatch() -> None:
    """Extension rows with different physical units cannot be combined silently."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    end = _state(2001, A, [1.0, 2.0])
    f0 = _extension(_series([1.0, 2.0], index, "CO2"), year=2000, unit="kg")
    f1 = _extension(_series([1.0, 2.0], index, "CO2"), year=2001, unit="tonnes")

    with pytest.raises(ValueError, match="Unit mismatch"):
        compute_extension_sda_pair_result_from_states(start, end, f0, f1)


def test_extension_sda_progress_and_low_memory_metadata() -> None:
    """Progress callbacks and metadata make sparse resource behavior auditable."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    end = _state(2001, A, [2.0, 3.0])
    f0 = _extension(_series([1.0, 2.0], index, "CO2"), year=2000)
    f1 = _extension(_series([2.0, 3.0], index, "CO2"), year=2001)
    messages: list[str] = []

    result = compute_extension_sda_pair_result_from_states(
        start,
        end,
        f0,
        f1,
        stabilization="raw",
        progress=messages.append,
    )

    assert any("validate" in message for message in messages)
    assert any("Leontief" in message for message in messages)
    assert result.metadata["forms_explicit_leontief_inverse"] is False
    assert result.metadata["stores_solver_factorization"] is False
    assert result.metadata["forms_sparse_delta_A"] is False
    assert result.metadata["rhs_columns_per_factorization"] == 2

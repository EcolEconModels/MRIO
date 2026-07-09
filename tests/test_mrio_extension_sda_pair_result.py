from __future__ import annotations

import pandas as pd
import pytest
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import spsolve

from ecolecon_mrio.exiobase3_extensions import ExtensionRowResult
from ecolecon_mrio.mrio_extension_sda import (
    MrioExtensionYearState,
    compute_extension_intensity_state,
    compute_extension_sda_pair_result_from_extension_year_states,
    compute_extension_sda_pair_result_from_states,
)
from ecolecon_mrio.mrio_sda import MrioYearState


def _idx() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([("R1", "s1"), ("R1", "s2")], names=["region", "sector"])


def _series(values: list[float], index: pd.MultiIndex, name: str) -> pd.Series:
    return pd.Series(values, index=index, dtype=float, name=name)


def _state(year: int, A: csr_matrix, y_values: list[float], x_override: list[float] | None = None) -> MrioYearState:
    index = _idx()
    y = _series(y_values, index, f"y{year}")
    if x_override is None:
        x_values = spsolve(sparse_eye(2, format="csr") - A, y.to_numpy(dtype=float))
    else:
        x_values = x_override
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


def _series_with_unit(values: pd.Series, *, unit: str = "kg", stressor: str = "CO2") -> pd.Series:
    values.attrs["unit"] = unit
    values.attrs["unit_source"] = "test/unit.txt"
    values.attrs["extension"] = "air_emissions"
    values.attrs["kind"] = "F"
    values.attrs["stressor"] = stressor
    return values


def test_extension_sda_six_polar_components_sum_to_leontief_delta() -> None:
    """Protect the scientific identity Δ(qL y)=Δq+ΔA+Δy+r, not monetary post-valuation."""

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

    reconstructed_leontief = (
        result.sda_intensity_delta
        + result.sda_structure_delta
        + result.sda_demand_delta
        + result.sda_residual
    )
    pd.testing.assert_series_equal(result.leontief_extension_delta, reconstructed_leontief, check_names=False)
    reconstructed_stabilized = result.leontief_extension_delta + result.reported_x_identity_gap_delta
    pd.testing.assert_series_equal(result.stabilized_extension_delta, reconstructed_stabilized, check_names=False)
    assert result.residual_l2 < 1.0e-10
    assert result.residual_share_l2 < 1.0e-10
    assert result.metadata["identity_mode"] == "six_polar_three_factor_extension_sda"


def test_extension_sda_has_nonzero_intensity_component_when_intensity_changes() -> None:
    """Catch regressions away from physical extension-SDA components."""

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


def test_reported_x_identity_gap_is_separate_from_sda_residual() -> None:
    """Reported x != L y must be a separate diagnostic, not hidden in SDA residual."""

    index = _idx()
    A = csr_matrix([[0.10, 0.0], [0.0, 0.10]], dtype=float)
    start = _state(2000, A, [9.0, 9.0], x_override=[12.0, 10.0])
    end = _state(2001, A, [9.0, 9.0], x_override=[10.0, 12.0])
    f0 = (_series([2.0, 2.0], index, "q0") * start.x).rename("CO2")
    f1 = (_series([2.0, 2.0], index, "q1") * end.x).rename("CO2")

    result = compute_extension_sda_pair_result_from_states(
        start,
        end,
        _extension(f0, year=2000),
        _extension(f1, year=2001),
        stabilization="raw",
    )

    reconstructed_leontief = (
        result.sda_intensity_delta
        + result.sda_structure_delta
        + result.sda_demand_delta
        + result.sda_residual
    )
    pd.testing.assert_series_equal(result.leontief_extension_delta, reconstructed_leontief, check_names=False)
    assert result.residual_l2 < 1.0e-10
    assert float(result.reported_x_identity_gap_delta.abs().sum()) > 0.0
    pd.testing.assert_series_equal(
        result.stabilized_extension_delta,
        result.leontief_extension_delta + result.reported_x_identity_gap_delta,
        check_names=False,
    )


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


def test_plain_series_requires_explicit_unit_metadata() -> None:
    """Bare Series must not get fake matching unknown units at the pair boundary."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    with pytest.raises(ValueError, match=r"attrs\['unit'\]"):
        compute_extension_intensity_state(mrio_state=start, extension_row=_series([1.0, 2.0], index, "CO2"))

    state = compute_extension_intensity_state(
        mrio_state=start,
        extension_row=_series_with_unit(_series([1.0, 2.0], index, "CO2")),
        stabilization="raw",
    )
    assert state.unit == "kg"
    assert state.stressor == "CO2"


def test_all_zero_extension_row_rejected_when_output_has_mass() -> None:
    """A missing selected extension row must not yield a scientifically bogus zero SDA."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    zero = _extension(_series([0.0, 0.0], index, "CO2"), year=2000)

    with pytest.raises(ValueError, match="all-zero extension row"):
        compute_extension_intensity_state(mrio_state=start, extension_row=zero)

    allowed = compute_extension_intensity_state(mrio_state=start, extension_row=zero, reject_all_zero=False)
    assert float(allowed.values.abs().sum()) == pytest.approx(0.0)


def test_pair_api_rejects_intensity_state_instead_of_raw_row() -> None:
    """The pair API must fail loudly if caller passes loader's intensity state field."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    end = _state(2001, A, [1.0, 2.0])
    f0 = _extension(_series([1.0, 2.0], index, "CO2"), year=2000)
    f1 = _extension(_series([1.0, 2.0], index, "CO2"), year=2001)
    intensity_state = compute_extension_intensity_state(mrio_state=start, extension_row=f0)

    with pytest.raises(TypeError, match="ExtensionIntensityState"):
        compute_extension_sda_pair_result_from_states(start, end, intensity_state, f1)  # type: ignore[arg-type]


def test_extension_year_state_helper_preserves_raw_row_for_pair_api() -> None:
    """Convenience-loaded year bundles must compose cleanly into the pair API."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    end = _state(2001, A, [2.0, 3.0])
    f0 = _extension(_series([1.0, 2.0], index, "CO2"), year=2000)
    f1 = _extension(_series([2.0, 3.0], index, "CO2"), year=2001)
    start_bundle = MrioExtensionYearState(
        mrio_state=start,
        extension_row=f0,
        intensity_state=compute_extension_intensity_state(mrio_state=start, extension_row=f0, stabilization="raw"),
    )
    end_bundle = MrioExtensionYearState(
        mrio_state=end,
        extension_row=f1,
        intensity_state=compute_extension_intensity_state(mrio_state=end, extension_row=f1, stabilization="raw"),
    )

    result = compute_extension_sda_pair_result_from_extension_year_states(start_bundle, end_bundle)

    pd.testing.assert_series_equal(result.observed_extension_delta, _series([1.0, 1.0], index, "expected"), check_names=False)


def test_extension_year_state_pair_reuses_precomputed_all_zero_override() -> None:
    """Loaded bundle pairs must not recompute all-zero rows with stricter defaults."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    end = _state(2001, A, [2.0, 3.0])
    f0 = _extension(_series([0.0, 0.0], index, "CO2"), year=2000)
    f1 = _extension(_series([0.0, 0.0], index, "CO2"), year=2001)
    start_bundle = MrioExtensionYearState(
        mrio_state=start,
        extension_row=f0,
        intensity_state=compute_extension_intensity_state(
            mrio_state=start,
            extension_row=f0,
            stabilization="raw",
            reject_all_zero=False,
        ),
    )
    end_bundle = MrioExtensionYearState(
        mrio_state=end,
        extension_row=f1,
        intensity_state=compute_extension_intensity_state(
            mrio_state=end,
            extension_row=f1,
            stabilization="raw",
            reject_all_zero=False,
        ),
    )

    result = compute_extension_sda_pair_result_from_extension_year_states(start_bundle, end_bundle)

    assert float(result.observed_extension_delta.abs().sum()) == pytest.approx(0.0)
    assert result.metadata["reject_all_zero"] == {"start": False, "end": False}


def test_extension_year_state_pair_uses_isclose_for_loaded_tau() -> None:
    """Tiny metadata roundoff in loaded tau must not force a false mismatch."""

    index = _idx()
    A = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    start = _state(2000, A, [1.0, 2.0])
    end = _state(2001, A, [2.0, 3.0])
    f0 = _extension(_series([1.0, 2.0], index, "CO2"), year=2000)
    f1 = _extension(_series([2.0, 3.0], index, "CO2"), year=2001)
    start_bundle = MrioExtensionYearState(
        mrio_state=start,
        extension_row=f0,
        intensity_state=compute_extension_intensity_state(
            mrio_state=start,
            extension_row=f0,
            stabilization="threshold",
            tau=1.0e-8,
        ),
    )
    end_bundle = MrioExtensionYearState(
        mrio_state=end,
        extension_row=f1,
        intensity_state=compute_extension_intensity_state(
            mrio_state=end,
            extension_row=f1,
            stabilization="threshold",
            tau=1.0e-8 * (1.0 + 1.0e-13),
        ),
    )

    result = compute_extension_sda_pair_result_from_extension_year_states(start_bundle, end_bundle)

    assert result.metadata["stabilization"]["method"] == "threshold"


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
    assert any("raw direct extension intensity" in message for message in messages)
    assert any("stabilization" in message for message in messages)
    assert any("Leontief" in message for message in messages)
    assert any("identity-gap" in message for message in messages)
    assert result.metadata["forms_explicit_leontief_inverse"] is False
    assert result.metadata["stores_solver_factorization"] is False
    assert result.metadata["forms_sparse_delta_A"] is False
    assert result.metadata["rhs_columns_per_factorization"] == 2

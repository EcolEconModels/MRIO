from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import spsolve

from ecolecon_mrio.mrio_sda import (
    _strict_csr_matrix,
    compute_output_sda_pair_result,
    require_region_sector_multiindex,
)


def _idx() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("R1", "s1"), ("R1", "s2")], names=["region", "sector"]
    )


def _series(values: list[float], index: pd.MultiIndex, name: str) -> pd.Series:
    return pd.Series(values, index=index, dtype=float, name=name)


def test_require_region_sector_multiindex_rejects_flat_index() -> None:
    with pytest.raises(TypeError, match="MultiIndex"):
        require_region_sector_multiindex(pd.Index(["s1", "s2"]))


def test_require_region_sector_multiindex_rejects_wrong_names() -> None:
    bad = pd.MultiIndex.from_tuples([("R1", "s1")], names=["country", "sector"])
    with pytest.raises(ValueError, match="region"):
        require_region_sector_multiindex(bad)


def test_require_region_sector_multiindex_rejects_duplicates() -> None:
    duplicate = pd.MultiIndex.from_tuples(
        [("R1", "s1"), ("R1", "s1")], names=["region", "sector"]
    )
    with pytest.raises(ValueError, match="unique"):
        require_region_sector_multiindex(duplicate)


def test_average_polar_sda_preserves_index_and_identity() -> None:
    index = _idx()
    A0 = csr_matrix([[0.10, 0.02], [0.03, 0.12]], dtype=float)
    A1 = csr_matrix([[0.11, 0.01], [0.04, 0.10]], dtype=float)
    y0 = _series([10.0, 7.0], index, "y0")
    y1 = _series([11.0, 8.5], index, "y1")
    x0_values = spsolve(sparse_eye(2, format="csr") - A0, y0.to_numpy(dtype=float))
    x1_values = spsolve(sparse_eye(2, format="csr") - A1, y1.to_numpy(dtype=float))
    x0 = pd.Series(x0_values, index=index, dtype=float, name="x0")
    x1 = pd.Series(x1_values, index=index, dtype=float, name="x1")

    result = compute_output_sda_pair_result(
        start_year=2000,
        end_year=2001,
        A_start=A0,
        A_end=A1,
        x_start=x0,
        x_end=x1,
        y_start=y0,
        y_end=y1,
        index=index,
        monetary_unit="MEUR",
        monetary_unit_source="test_fixture",
    )

    for vector in (
        result.observed_output_delta,
        result.sda_intermediate_delta,
        result.sda_demand_delta,
        result.sda_residual,
    ):
        assert vector.index.equals(index)
    reconstructed = result.sda_intermediate_delta + result.sda_demand_delta + result.sda_residual
    pd.testing.assert_series_equal(result.observed_output_delta, reconstructed, check_names=False)
    assert result.residual_l2 == pytest.approx(float(np.linalg.norm(result.sda_residual.to_numpy(dtype=float))))
    assert result.residual_l2 < 1.0e-10
    assert result.residual_share_l2 < 1.0e-10
    assert result.monetary_unit == "MEUR"
    assert result.metadata["identity_mode"] == "average_polar_forward_backward"


def test_forward_solve_failure_raises() -> None:
    index = _idx()
    A0 = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    A1 = csr_matrix(np.eye(2), dtype=float)  # singular I - A_end
    x0 = _series([1.0, 2.0], index, "x0")
    x1 = _series([1.5, 2.5], index, "x1")
    y0 = _series([1.0, 2.0], index, "y0")
    y1 = _series([1.0, 2.0], index, "y1")

    with pytest.raises(RuntimeError, match="forward"):
        compute_output_sda_pair_result(
            start_year=2000,
            end_year=2001,
            A_start=A0,
            A_end=A1,
            x_start=x0,
            x_end=x1,
            y_start=y0,
            y_end=y1,
            index=index,
            monetary_unit="MEUR",
            monetary_unit_source="test_fixture",
        )


def test_backward_solve_failure_raises_when_average_requested() -> None:
    index = _idx()
    A0 = csr_matrix(np.eye(2), dtype=float)  # singular I - A_start
    A1 = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    x0 = _series([1.0, 2.0], index, "x0")
    x1 = _series([1.5, 2.5], index, "x1")
    y0 = _series([1.0, 2.0], index, "y0")
    y1 = _series([1.2, 2.3], index, "y1")

    with pytest.raises(RuntimeError, match="backward"):
        compute_output_sda_pair_result(
            start_year=2000,
            end_year=2001,
            A_start=A0,
            A_end=A1,
            x_start=x0,
            x_end=x1,
            y_start=y0,
            y_end=y1,
            index=index,
            monetary_unit="MEUR",
            monetary_unit_source="test_fixture",
        )


def test_forward_only_does_not_attempt_backward_solve() -> None:
    index = _idx()
    A0 = csr_matrix(np.eye(2), dtype=float)  # would fail only if backward solve is attempted
    A1 = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    x0 = _series([1.0, 2.0], index, "x0")
    x1 = _series([1.5, 2.5], index, "x1")
    y0 = _series([1.0, 2.0], index, "y0")
    y1 = _series([1.2, 2.3], index, "y1")

    result = compute_output_sda_pair_result(
        start_year=2000,
        end_year=2001,
        A_start=A0,
        A_end=A1,
        x_start=x0,
        x_end=x1,
        y_start=y0,
        y_end=y1,
        index=index,
        monetary_unit="MEUR",
        monetary_unit_source="test_fixture",
        average_identities=False,
    )
    assert result.metadata["identity_mode"] == "forward_only"

def test_strict_sda_rejects_nonfinite_vectors() -> None:
    index = _idx()
    A0 = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    A1 = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    x0 = _series([1.0, 2.0], index, "x0")
    x1 = _series([1.5, 2.5], index, "x1")
    y0 = _series([1.0, np.nan], index, "y0")
    y1 = _series([1.2, 2.3], index, "y1")

    with pytest.raises(ValueError, match="y_start"):
        compute_output_sda_pair_result(
            start_year=2000,
            end_year=2001,
            A_start=A0,
            A_end=A1,
            x_start=x0,
            x_end=x1,
            y_start=y0,
            y_end=y1,
            index=index,
            monetary_unit="MEUR",
            monetary_unit_source="test_fixture",
        )


def test_strict_sda_rejects_nonfinite_sparse_matrix_data() -> None:
    index = _idx()
    A0 = csr_matrix([[0.0, np.nan], [0.0, 0.0]], dtype=float)
    A1 = csr_matrix([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    x0 = _series([1.0, 2.0], index, "x0")
    x1 = _series([1.5, 2.5], index, "x1")
    y0 = _series([1.0, 2.0], index, "y0")
    y1 = _series([1.2, 2.3], index, "y1")

    with pytest.raises(ValueError, match="A_start"):
        compute_output_sda_pair_result(
            start_year=2000,
            end_year=2001,
            A_start=A0,
            A_end=A1,
            x_start=x0,
            x_end=x1,
            y_start=y0,
            y_end=y1,
            index=index,
            monetary_unit="MEUR",
            monetary_unit_source="test_fixture",
        )



def test_strict_csr_matrix_reuses_float64_csr_without_copy() -> None:
    """Strict validation must not copy an already-float64 CSR matrix."""

    index = _idx()
    matrix = csr_matrix([[0.10, 0.02], [0.03, 0.12]], dtype=np.float64)
    checked = _strict_csr_matrix(matrix, index=index, name="A")

    assert checked is matrix

# PUBLIC_YEAR_STATE_LEONTIEF_RHS_TESTS_V1
def test_year_state_leontief_rhs_preserves_index_columns_and_matches_direct_solve() -> None:
    """The public state solver must reuse the exact Leontief operator for many RHS."""

    from ecolecon_mrio import MrioYearState, solve_mrio_year_state_leontief_rhs

    index = _idx()
    A = csr_matrix([[0.10, 0.02], [0.03, 0.12]], dtype=float)
    state = MrioYearState(
        year=2020,
        A=A,
        x=_series([10.0, 8.0], index, "x"),
        y=_series([9.0, 7.0], index, "y"),
        index=index,
        monetary_unit="MEUR",
        monetary_unit_source="test_fixture",
        metadata={"valuation_basis": "current_price"},
    )
    rhs = pd.DataFrame(
        {"source": [1.0, -0.5], "input": [0.25, 2.0], "total": [1.25, 1.5]},
        index=index,
        dtype=float,
    )
    actual = solve_mrio_year_state_leontief_rhs(state, rhs, label="test multi RHS")
    expected = np.asarray(
        spsolve(sparse_eye(2, format="csr") - A, rhs.to_numpy(dtype=float)),
        dtype=float,
    )
    pd.testing.assert_index_equal(actual.index, index)
    pd.testing.assert_index_equal(actual.columns, rhs.columns)
    np.testing.assert_allclose(actual.to_numpy(dtype=float), expected, rtol=1e-13, atol=1e-13)


def test_year_state_leontief_rhs_rejects_misaligned_rhs() -> None:
    """Arbitrary RHS solves must not silently reorder or fill a scientific basis."""

    from ecolecon_mrio import MrioYearState, solve_mrio_year_state_leontief_rhs

    index = _idx()
    state = MrioYearState(
        year=2020,
        A=csr_matrix([[0.10, 0.02], [0.03, 0.12]], dtype=float),
        x=_series([10.0, 8.0], index, "x"),
        y=_series([9.0, 7.0], index, "y"),
        index=index,
        monetary_unit="MEUR",
        monetary_unit_source="test_fixture",
        metadata={"valuation_basis": "current_price"},
    )
    bad_index = pd.MultiIndex.from_tuples(
        [("R1", "s2"), ("R1", "s1")], names=["region", "sector"]
    )
    rhs = pd.DataFrame({"shock": [1.0, 2.0]}, index=bad_index)
    with pytest.raises(ValueError, match="exactly match"):
        solve_mrio_year_state_leontief_rhs(state, rhs)

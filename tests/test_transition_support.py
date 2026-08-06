from __future__ import annotations

import numpy as np
import pandas as pd

from ecolecon_mrio.transition_support import (
    baseline_productivity_certificate,
    adjacent_coefficient_relative_changes,
    benchmark_step_for_emissions_cut,
    coefficient_change_bounds,
    coefficient_edge_counts,
    emissions_cut_from_step,
    emissions_direction_slope,
    max_demand_nonnegative_emissions_cut,
    max_demand_nonnegative_step,
    max_output_nonnegative_emissions_cut,
    max_output_nonnegative_step,
    normalize_productive_certificate_vector,
    productive_certificate_row_rhs,
    productive_certificate_slack,
    satisfies_productive_certificate,
)


def test_coefficient_change_bounds_keep_nonnegativity_and_relative_cap() -> None:
    lower, upper = coefficient_change_bounds(
        pd.Series([0.2, 0.0, 0.5], index=["a", "b", "c"]),
        relative_change_cap=0.5,
        relative_change_cap_floor=1.0e-4,
    )
    assert np.allclose(lower, np.array([-0.1, 0.0, -0.25]))
    assert np.allclose(upper, np.array([0.1, 5.0e-5, 0.25]))


def test_productive_certificate_helpers_report_slack_and_row_rhs() -> None:
    idx = pd.Index(["a", "b"])
    A = pd.DataFrame([[0.2, 0.1], [0.05, 0.15]], index=idx, columns=idx, dtype=float)
    q = normalize_productive_certificate_vector(pd.Series([0.6, 0.4], index=idx), idx)
    slack = productive_certificate_slack(A, q, 0.5)
    assert bool((slack > 0.0).all())
    assert satisfies_productive_certificate(A, q, 0.5)
    rhs = productive_certificate_row_rhs(A.loc["a"], q, 0.5, row_label="a")
    assert rhs is not None
    assert rhs > 0.0


def test_emissions_cut_and_step_conversion_round_trip() -> None:
    baseline = 100.0
    slope = -5.0
    step = benchmark_step_for_emissions_cut(0.01, baseline_emissions=baseline, emissions_slope=slope)
    assert step == 0.2
    cut = emissions_cut_from_step(step, baseline_emissions=baseline, emissions_slope=slope)
    assert cut == 0.01


def test_output_nonnegative_frontier_matches_manual_bound() -> None:
    idx = pd.Index(["a", "b", "c"])
    x = pd.Series([10.0, 5.0, 8.0], index=idx)
    d = pd.Series([2.0, -1.0, -4.0], index=idx)
    step = max_output_nonnegative_step(x, d)
    assert step == 2.0
    cut = max_output_nonnegative_emissions_cut(x, d, baseline_emissions=100.0, emissions_slope=-10.0)
    assert cut == 0.2


def test_demand_nonnegative_frontier_matches_manual_bound() -> None:
    idx = pd.Index(["a", "b"])
    A = pd.DataFrame([[0.2, 0.1], [0.1, 0.2]], index=idx, columns=idx, dtype=float)
    y = pd.Series([4.0, 3.0], index=idx, dtype=float)
    d = pd.Series([-1.0, 2.0], index=idx, dtype=float)
    # (I - A) d = [-1.0, 1.5], so the first component binds at step 4.
    step = max_demand_nonnegative_step(A, y, d)
    assert np.isclose(step, 4.0)
    cut = max_demand_nonnegative_emissions_cut(A, y, d, baseline_emissions=200.0, emissions_slope=-20.0)
    assert np.isclose(cut, 0.4)


def test_emissions_direction_slope_aligns_series_by_index() -> None:
    e = pd.Series([3.0, 1.0], index=["b", "a"])
    d = pd.Series([2.0, -4.0], index=["a", "b"])
    slope = emissions_direction_slope(e, d)
    # aligned order gives 3*(-4) + 1*2 = -10
    assert slope == -10.0

def test_baseline_productivity_certificate_matches_leontief_identity() -> None:
    A = np.diag([0.2, 0.5])
    vector, epsilon, slack = baseline_productivity_certificate(A)
    np.testing.assert_allclose((np.eye(2) - A) @ vector, np.ones(2))
    assert np.isclose(epsilon, 0.25)
    assert bool((slack >= 0.0).all())
    assert np.max((A @ vector) / vector) <= 1.0 - epsilon


def test_baseline_productivity_certificate_rejects_nonproductive_matrix() -> None:
    with np.testing.assert_raises(ValueError):
        baseline_productivity_certificate(np.diag([0.2, 1.1]))


def test_sparse_edge_counts_and_adjacent_change_distribution() -> None:
    start = np.array([[0.2, 0.0], [0.01, 0.0]])
    end = np.array([[0.1, 0.3], [0.01, 0.0]])
    assert coefficient_edge_counts(end, [0.0, 0.15]) == {0.0: 3, 0.15: 1}
    relative, appearing, disappearing = adjacent_coefficient_relative_changes(
        start, end, floor=0.05
    )
    np.testing.assert_allclose(np.sort(relative), np.array([0.5, 6.0]))
    assert appearing == 1
    assert disappearing == 0

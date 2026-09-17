from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, diags

from ecolecon_mrio.transition_support import (
    baseline_productivity_certificate,
    adjacent_coefficient_change_diagnostics,
    adjacent_coefficient_change_quantiles,
    benchmark_step_for_emissions_cut,
    certificate_safe_coefficient_capacity_products,
    certificate_safe_coefficient_headroom,
    coefficient_eligible_mask,
    reconstruct_coefficient_change_witness,
    complete_directions_with_fixed_final_demand,
    merge_panel_coefficient_envelope,
    coefficient_change_bounds,
    coefficient_edge_counts,
    emissions_cut_from_step,
    emissions_direction_slope,
    max_demand_nonnegative_emissions_cut,
    max_demand_nonnegative_step,
    max_output_nonnegative_emissions_cut,
    max_output_nonnegative_step,
    merge_panel_observed_coefficient_mask,
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


def test_panel_observed_coefficient_mask_is_strict_union() -> None:
    """Panel support is the strict-threshold sparse union across annual A matrices."""

    first = csr_matrix(np.array([[0.0, 0.2], [0.1, 0.0]], dtype=float))
    second = csr_matrix(np.array([[0.3, 0.0], [0.05, 0.4]], dtype=float))
    panel = merge_panel_observed_coefficient_mask(None, first, threshold=0.1)
    panel = merge_panel_observed_coefficient_mask(panel, second, threshold=0.1)
    np.testing.assert_array_equal(
        panel.toarray(), np.array([[1, 1], [0, 1]], dtype=np.int8)
    )


def test_sparse_edge_counts_and_adjacent_change_distribution() -> None:
    """Protect exact pair-active quantiles, turnover counts, and unchanged active links."""

    start = np.array([[0.2, 0.0], [0.1, 0.0]])
    end = np.array([[0.1, 0.3], [0.1, 0.0]])
    assert coefficient_edge_counts(end, [0.0, 0.15]) == {0.0: 3, 0.15: 1}
    count, appearing, disappearing, quantiles = adjacent_coefficient_change_quantiles(
        start,
        end,
        floor=0.05,
        quantiles=(0.5, 0.9),
    )
    assert count == 3
    assert appearing == 1
    assert disappearing == 0
    assert np.isclose(quantiles[0.5], 0.5)
    assert np.isclose(quantiles[0.9], 4.9)




def test_adjacent_coefficient_diagnostics_report_transition_magnitudes() -> None:
    """Appearing/disappearing links retain their observed positive coefficient scale."""

    start = np.array([[0.2, 0.0], [0.1, 0.4]], dtype=float)
    end = np.array([[0.1, 0.3], [0.1, 0.0]], dtype=float)
    count, appearing, disappearing, _, appearing_q, disappearing_q = (
        adjacent_coefficient_change_diagnostics(
            start,
            end,
            floor=0.05,
            quantiles=(0.25, 0.5, 0.9),
        )
    )
    assert count == 4
    assert appearing == 1
    assert disappearing == 1
    assert all(np.isclose(value, 0.3) for value in appearing_q.values())
    assert all(np.isclose(value, 0.4) for value in disappearing_q.values())


def test_certificate_safe_coefficient_bounds_are_masked_and_productive() -> None:
    """Historical level bounds must produce exact compact products and a productive upper box."""

    A = np.array([[0.2, 0.0], [0.0, 0.1]], dtype=float)
    vector, epsilon, _ = baseline_productivity_certificate(A)
    allowed = np.array([[True, True], [False, True]], dtype=bool)
    historical_minimum = np.array([[0.1, 0.0], [0.0, 0.05]], dtype=float)
    historical_maximum = np.array([[0.3, 0.05], [0.0, 0.15]], dtype=float)
    rhs = np.array([[10.0, 1.0], [4.0, -2.0]], dtype=float)
    lower_products, upper_products, scale, margin = (
        certificate_safe_coefficient_capacity_products(
            A,
            allowed,
            historical_minimum,
            historical_maximum,
            rhs,
            certificate_vector=vector,
            productivity_epsilon=epsilon,
        )
    )

    raw_lower = np.array([[-0.1, 0.0], [0.0, -0.05]], dtype=float)
    raw_upper = np.array([[0.1, 0.05], [0.0, 0.05]], dtype=float)
    baseline_slack = (1.0 - epsilon) * vector - A @ vector
    positive_use = raw_upper @ vector
    expected_scale = np.ones(2, dtype=float)
    active = positive_use > 1.0e-12
    expected_scale[active] = np.minimum(
        1.0, np.maximum(baseline_slack[active], 0.0) / positive_use[active]
    )
    safe_upper = raw_upper * expected_scale[:, None]

    np.testing.assert_allclose(scale, expected_scale)
    np.testing.assert_allclose(
        lower_products, raw_lower @ rhs, rtol=1.0e-12, atol=1.0e-14
    )
    np.testing.assert_allclose(
        upper_products, safe_upper @ rhs, rtol=1.0e-12, atol=1.0e-14
    )
    assert float(margin.min()) >= -1.0e-12
    assert bool(((A + safe_upper) @ vector <= (1.0 - epsilon) * vector + 1.0e-12).all())


def test_sparse_certificate_headroom_matches_compact_products_and_mask_semantics() -> None:
    """LP headroom must exactly match compact products and reject/ignore invalid mask storage."""

    A = csr_matrix(np.array([[0.2, 0.0], [0.0, 0.1]], dtype=float))
    vector, epsilon, _ = baseline_productivity_certificate(A)
    # Keep an explicit stored zero in an otherwise canonical CSR row.  That entry
    # is not an allowed edge and must be removed before either API sees columns.
    allowed = csr_matrix(
        (
            np.array([1, 0, 1], dtype=np.int8),
            np.array([0, 1, 1], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    assert allowed.has_canonical_format and allowed.nnz == 3
    minimum = csr_matrix(np.array([[0.1, 0.0], [0.0, 0.05]], dtype=float))
    maximum = csr_matrix(np.array([[0.3, 9.0], [0.0, 0.15]], dtype=float))
    rhs = np.array([[10.0, 1.0, -2.0], [4.0, -2.0, 3.0]], dtype=float)
    lower_products, upper_products, scale, margin = certificate_safe_coefficient_capacity_products(
        A,
        allowed,
        minimum,
        maximum,
        rhs,
        certificate_vector=vector,
        productivity_epsilon=epsilon,
    )
    lower, upper, returned_scale, returned_margin = certificate_safe_coefficient_headroom(
        A,
        allowed,
        minimum,
        maximum,
        certificate_vector=vector,
        productivity_epsilon=epsilon,
    )
    assert lower.has_canonical_format and upper.has_canonical_format
    assert float(lower[0, 1]) == 0.0 and float(upper[0, 1]) == 0.0
    np.testing.assert_allclose(lower_products[0], -0.1 * rhs[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(upper_products[0], 0.1 * rhs[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(lower @ rhs, lower_products, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(upper @ rhs, upper_products, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(returned_scale, scale)
    np.testing.assert_allclose(returned_margin, margin)

    negative_mask = csr_matrix(np.array([[1, -1], [0, 1]], dtype=np.int8))
    with pytest.raises(ValueError, match="mask must be non-negative"):
        certificate_safe_coefficient_capacity_products(
            A,
            negative_mask,
            minimum,
            maximum,
            rhs,
            certificate_vector=vector,
            productivity_epsilon=epsilon,
        )
    with pytest.raises(ValueError, match="mask must be non-negative"):
        certificate_safe_coefficient_headroom(
            A,
            negative_mask,
            minimum,
            maximum,
            certificate_vector=vector,
            productivity_epsilon=epsilon,
        )


def test_productivity_certificate_iterative_sparse_residual() -> None:
    A = diags(np.full(2000, 0.4, dtype=float), format="csr")
    vector, epsilon, slack = baseline_productivity_certificate(A)
    residual = vector - A.dot(vector) - np.ones(2000, dtype=float)
    assert float(np.max(np.abs(residual))) <= 1.0e-9
    assert np.isclose(epsilon, 0.3)
    assert float(slack.min()) >= -1.0e-12


def test_support_validation_does_not_mutate_sparse_input() -> None:
    A = csr_matrix(
        (
            np.array([0.2, 0.0, 0.1], dtype=float),
            np.array([0, 1, 1], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    before = (A.data.copy(), A.indices.copy(), A.indptr.copy())
    coefficient_edge_counts(A, [0.0])
    np.testing.assert_array_equal(A.data, before[0])
    np.testing.assert_array_equal(A.indices, before[1])
    np.testing.assert_array_equal(A.indptr, before[2])


def test_panel_coefficient_envelope_tracks_thresholded_counts_minimum_and_maximum() -> None:
    """Panel history must retain persistence and the exact link-specific observed range."""

    annual = (
        np.array([[0.20, 0.00, 0.40], [0.00, 0.30, 0.00], [0.00, 0.00, 0.10]]),
        np.array([[0.10, 0.25, 0.00], [0.00, 0.40, 0.00], [0.00, 0.00, 0.10]]),
        np.array([[0.15, 0.20, 0.00], [0.00, 0.35, 0.00], [0.00, 0.00, 0.10]]),
    )
    counts = minimum = maximum = None
    for A in annual:
        counts, minimum, maximum = merge_panel_coefficient_envelope(
            counts, minimum, maximum, A, threshold=0.05
        )

    assert counts is not None
    assert minimum is not None
    assert maximum is not None
    np.testing.assert_array_equal(
        counts.toarray(), np.array([[3, 2, 1], [0, 3, 0], [0, 0, 3]], dtype=np.uint8)
    )
    np.testing.assert_allclose(
        minimum.toarray(), np.array([[0.10, 0.0, 0.0], [0.0, 0.30, 0.0], [0.0, 0.0, 0.10]])
    )
    np.testing.assert_allclose(
        maximum.toarray(), np.array([[0.20, 0.25, 0.40], [0.0, 0.40, 0.0], [0.0, 0.0, 0.10]])
    )


def test_panel_coefficient_envelope_matches_dense_thresholded_history() -> None:
    """CSR streaming must preserve zero-inclusive minima and union count/max semantics."""

    threshold = 0.1
    annual = (
        np.array([[0.2, 0.1, 0.0], [0.5, 0.0, 0.0], [0.0, 0.3, 0.0]]),
        np.array([[0.4, 0.0, 0.2], [0.0, 0.0, 0.0], [0.0, 0.6, 0.0]]),
        np.array([[0.3, 0.2, 0.0], [0.7, 0.0, 0.0], [0.0, 0.4, 0.9]]),
    )
    thresholded = np.stack([np.where(matrix > threshold, matrix, 0.0) for matrix in annual])
    counts = minimum = maximum = None
    for matrix in annual:
        counts, minimum, maximum = merge_panel_coefficient_envelope(
            counts, minimum, maximum, csr_matrix(matrix), threshold=threshold
        )

    assert counts is not None and minimum is not None and maximum is not None
    np.testing.assert_array_equal(counts.toarray(), (thresholded > 0.0).sum(axis=0).astype(np.uint8))
    np.testing.assert_allclose(minimum.toarray(), thresholded.min(axis=0))
    np.testing.assert_allclose(maximum.toarray(), thresholded.max(axis=0))
    for summary, dtype in ((counts, np.uint8), (minimum, np.float64), (maximum, np.float64)):
        assert isinstance(summary, csr_matrix)
        assert summary.has_canonical_format
        assert summary.dtype == dtype
        assert not bool((summary.data == 0).any())


def test_fixed_final_demand_completion_preserves_prescribed_rows() -> None:
    """The generic completion must enforce [(I-A)D]_N=0 without altering D_S."""

    A = csr_matrix(
        np.array([[0.10, 0.00, 0.00], [0.20, 0.10, 0.00], [0.00, 0.30, 0.10]])
    )
    prescribed = np.array([True, False, False])
    D_S = np.array([[1.0, 2.0]])
    completed = complete_directions_with_fixed_final_demand(
        A, prescribed, D_S, tolerance=1.0e-10
    )
    np.testing.assert_allclose(completed[prescribed, :], D_S)
    residual = completed - A.dot(completed)
    assert float(np.max(np.abs(residual[~prescribed, :]))) <= 1.0e-10


# INVERSE_FRONTIER_REALIZABILITY_TESTS_V1
def test_coefficient_eligible_mask_keeps_current_links_and_recurrent_dormant_links() -> None:
    """The accepted topology allows current links plus dormant links seen in >=2 years."""

    A0 = csr_matrix(np.array([[0.10, 0.0, 0.0], [0.15, 0.10, 0.0], [0.0, 0.0, 0.10]]))
    counts = csr_matrix(np.array([[3, 2, 1], [1, 3, 0], [0, 0, 3]], dtype=np.uint8))
    eligible = coefficient_eligible_mask(
        A0, counts, edge_threshold=0.05, minimum_active_years=2
    )
    expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int8)
    np.testing.assert_array_equal(eligible.toarray(), expected)


def test_reconstruct_coefficient_change_witness_exactly_realizes_rowwise_capacity() -> None:
    """Every flow inside Lx<=u<=Ux has a sparse coefficient witness for x>=0."""

    lower = csr_matrix(
        np.array([[-0.10, -0.05, 0.0], [0.0, -0.20, -0.10], [-0.05, 0.0, 0.0]])
    )
    upper = csr_matrix(
        np.array([[0.20, 0.15, 0.0], [0.10, 0.10, 0.20], [0.05, 0.0, 0.30]])
    )
    x1 = np.array([2.0, 0.0, 4.0])
    low_flow = np.asarray(lower.dot(x1)).reshape(-1)
    high_flow = np.asarray(upper.dot(x1)).reshape(-1)
    row_theta = np.array([0.25, 0.70, 0.40])
    target = low_flow + row_theta * (high_flow - low_flow)

    witness = reconstruct_coefficient_change_witness(lower, upper, x1, target)
    np.testing.assert_allclose(witness.dot(x1), target, rtol=1.0e-12, atol=1.0e-12)
    dense = witness.toarray()
    assert bool((dense >= lower.toarray() - 1.0e-12).all())
    assert bool((dense <= upper.toarray() + 1.0e-12).all())

    outside = target.copy()
    outside[0] = high_flow[0] + 1.0
    with pytest.raises(ValueError, match="outside exact coefficient capacity"):
        reconstruct_coefficient_change_witness(lower, upper, x1, outside)

import numpy as np
import pytest


def test_ras_balance_hits_margins():
    from ecolecon_mrio.balancing import ras_balance

    M0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    row_targets = np.array([10.0, 20.0])
    col_targets = np.array([12.0, 18.0])

    M, stats = ras_balance(M0, row_targets=row_targets, col_targets=col_targets, max_iter=50_000, tol=1e-12)

    assert np.allclose(M.sum(axis=1), row_targets, atol=1e-9)
    assert np.allclose(M.sum(axis=0), col_targets, atol=1e-9)
    assert stats.max_abs_row_gap <= 1e-9
    assert stats.max_abs_col_gap <= 1e-9


def test_gras_balance_handles_negatives():
    from ecolecon_mrio.balancing import gras_balance

    M0 = np.array([[1.0, -0.5], [0.5, 1.0]])
    row_targets = np.array([1.0, 1.0])
    col_targets = np.array([1.0, 1.0])

    M, stats = gras_balance(M0, row_targets=row_targets, col_targets=col_targets, max_iter=50_000, tol=1e-12)

    assert np.allclose(M.sum(axis=1), row_targets, atol=1e-9)
    assert np.allclose(M.sum(axis=0), col_targets, atol=1e-9)
    # GRAS converges more slowly than RAS in small toy cases; still requires tight column fit.
    assert stats.max_abs_row_gap <= 2e-5
    assert stats.max_abs_col_gap <= 1e-9


def test_balance_biproportional_selects_method():
    from ecolecon_mrio.balancing import balance_biproportional

    M0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    row_targets = np.array([10.0, 20.0])
    col_targets = np.array([12.0, 18.0])

    _, _, method = balance_biproportional(M0, row_targets=row_targets, col_targets=col_targets, max_iter=10_000, tol=1e-12)
    assert method == "RAS"

    M0n = np.array([[1.0, -0.5], [0.5, 1.0]])
    row_targets = np.array([1.0, 1.0])
    col_targets = np.array([1.0, 1.0])
    _, _, method = balance_biproportional(M0n, row_targets=row_targets, col_targets=col_targets, max_iter=10_000, tol=1e-12)
    assert method == "GRAS"


def test_balance_biproportional_masked_clips_tiny_negative_residuals():
    from ecolecon_mrio.balancing import balance_biproportional_masked

    M0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    adjustable = np.array([[False, False], [True, True]])
    row_targets = np.array([3.0 - 1e-11, 7.0])
    col_targets = np.array([4.0, 6.0 - 1e-11])

    M, stats, method = balance_biproportional_masked(
        M0,
        row_targets=row_targets,
        col_targets=col_targets,
        adjustable=adjustable,
        max_iter=10_000,
        tol=1e-10,
    )

    assert method == "RAS"
    assert np.allclose(M.sum(axis=1), np.array([3.0, 7.0]), atol=1e-9)
    assert np.allclose(M.sum(axis=0), np.array([4.0, 6.0]), atol=1e-9)
    assert stats.max_abs_row_gap <= 1e-9
    assert stats.max_abs_col_gap <= 1e-9

# MULTI_MARGIN_IPFP_TESTS_V1
def test_multi_margin_ipfp_specializes_to_existing_ras() -> None:
    """The generalized solver must reproduce the trusted two-margin RAS contract."""
    from ecolecon_mrio.balancing import IPFPMargin, ipfp_balance_partitions, ras_balance

    prior = np.array([[1.0, 2.0], [3.0, 4.0]])
    row_targets = np.array([10.0, 20.0])
    col_targets = np.array([12.0, 18.0])
    row_groups = np.broadcast_to(np.arange(2)[:, None], prior.shape)
    col_groups = np.broadcast_to(np.arange(2)[None, :], prior.shape)

    expected, _ = ras_balance(
        prior, row_targets=row_targets, col_targets=col_targets, tol=1e-12
    )
    observed, stats = ipfp_balance_partitions(
        prior,
        (
            IPFPMargin("rows", row_groups, row_targets),
            IPFPMargin("columns", col_groups, col_targets),
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-9)
    assert stats.max_normalized_gap <= 1.0


def test_multi_margin_ipfp_hits_parent_partition_and_preserves_zero_support() -> None:
    """A third partition must fit jointly without fabricating cells outside prior support."""
    from ecolecon_mrio.balancing import IPFPMargin, ipfp_balance_partitions

    prior = np.array(
        [
            [1.0, 2.0, 3.0, 0.0],
            [2.0, 3.0, 1.0, 4.0],
            [3.0, 1.0, 4.0, 2.0],
            [1.0, 2.0, 2.0, 5.0],
        ]
    )
    parent = np.array([0, 0, 1, 1], dtype=np.int64)
    row_factor = np.array([1.2, 0.8, 1.1, 0.9])
    col_factor = np.array([0.7, 1.3, 0.9, 1.1])
    block_factor = np.array([[1.0, 1.4], [0.6, 1.2]])
    target_matrix = (
        prior
        * row_factor[:, None]
        * col_factor[None, :]
        * block_factor[parent[:, None], parent[None, :]]
    )
    row_targets = target_matrix.sum(axis=1)
    col_targets = target_matrix.sum(axis=0)
    block_groups = parent[:, None] * 2 + parent[None, :]
    block_targets = np.bincount(
        block_groups.ravel(), weights=target_matrix.ravel(), minlength=4
    )
    observed, _ = ipfp_balance_partitions(
        prior,
        (
            IPFPMargin("rows", np.broadcast_to(np.arange(4)[:, None], prior.shape), row_targets),
            IPFPMargin("columns", np.broadcast_to(np.arange(4)[None, :], prior.shape), col_targets),
            IPFPMargin("blocks", block_groups, block_targets),
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    reversed_order, _ = ipfp_balance_partitions(
        prior,
        (
            IPFPMargin("blocks", block_groups, block_targets),
            IPFPMargin("columns", np.broadcast_to(np.arange(4)[None, :], prior.shape), col_targets),
            IPFPMargin("rows", np.broadcast_to(np.arange(4)[:, None], prior.shape), row_targets),
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(observed, reversed_order, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(observed.sum(axis=1), row_targets, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(observed.sum(axis=0), col_targets, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.bincount(block_groups.ravel(), weights=observed.ravel(), minlength=4),
        block_targets,
        rtol=1e-10,
        atol=1e-10,
    )
    assert observed[0, 3] == 0.0


def test_multi_margin_ipfp_rejects_positive_target_without_support() -> None:
    """Positive hard mass with no current support is scientific infeasibility, not zero."""
    from ecolecon_mrio.balancing import IPFPMargin, ipfp_balance_partitions

    prior = np.array([[1.0, 0.0], [0.0, 0.0]])
    row_groups = np.broadcast_to(np.arange(2)[:, None], prior.shape)
    col_groups = np.broadcast_to(np.arange(2)[None, :], prior.shape)
    with np.testing.assert_raises_regex(ValueError, "structural support"):
        ipfp_balance_partitions(
            prior,
            (
                IPFPMargin("rows", row_groups, np.array([0.5, 0.5])),
                IPFPMargin("columns", col_groups, np.array([0.5, 0.5])),
            ),
        )


# IPFP_PARTITION_BOUNDS_TESTS_V2
def test_partition_ipfp_upper_bound_is_joint_kl_constraint() -> None:
    from ecolecon_mrio.balancing import IPFPMargin, ipfp_balance_partitions

    prior = np.array([[1.0, 2.0], [3.0, 4.0]])
    rows = np.broadcast_to(np.arange(2)[:, None], prior.shape)
    cols = np.broadcast_to(np.arange(2)[None, :], prior.shape)
    result, stats = ipfp_balance_partitions(
        prior,
        (
            IPFPMargin("rows", rows, np.array([4.0, 6.0])),
            IPFPMargin("column_caps", cols, np.array([3.5, 10.0]), relation="upper"),
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    reversed_result, _ = ipfp_balance_partitions(
        prior,
        (
            IPFPMargin("column_caps", cols, np.array([3.5, 10.0]), relation="upper"),
            IPFPMargin("rows", rows, np.array([4.0, 6.0])),
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.sum(axis=1), np.array([4.0, 6.0]), atol=1e-10)
    assert float(result[:, 0].sum()) <= 3.5 + 1e-10
    np.testing.assert_allclose(result, reversed_result, rtol=1e-9, atol=1e-9)
    assert stats.max_normalized_gap <= 1.0


def test_partition_ipfp_rejects_unknown_relation() -> None:
    from ecolecon_mrio.balancing import IPFPMargin, ipfp_balance_partitions

    prior = np.ones((2, 2), dtype="float64")
    groups = np.broadcast_to(np.arange(2)[:, None], prior.shape)
    with np.testing.assert_raises_regex(ValueError, "relation"):
        ipfp_balance_partitions(
            prior,
            (IPFPMargin("bad", groups, np.array([2.0, 2.0]), relation="lower"),),
        )


# KL_LINEAR_EQUALITY_TESTS_V1
def test_linear_kl_reconciliation_hits_overlapping_equalities() -> None:
    from ecolecon_mrio.balancing import kl_balance_linear_equalities

    prior = np.array([1.0, 1.0, 1.0])
    constraints = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    result, stats = kl_balance_linear_equalities(
        prior, constraints, np.array([4.0, 4.0]), rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(constraints @ result, np.array([4.0, 4.0]), atol=1e-10)
    assert result[0] == pytest.approx(result[2])
    assert bool((result > 0.0).all())
    assert stats.max_normalized_gap <= 1.0


def test_linear_kl_reconciliation_supports_signed_coefficients() -> None:
    from ecolecon_mrio.balancing import kl_balance_linear_equalities

    prior = np.array([2.0, 1.0])
    constraints = np.array([[1.0, -1.0]])
    result, stats = kl_balance_linear_equalities(
        prior, constraints, np.array([1.0]), rtol=1e-12, atol=1e-10
    )
    assert float(result[0] - result[1]) == pytest.approx(1.0, abs=1e-10)
    assert bool((result >= 0.0).all())
    assert stats.max_normalized_gap <= 1.0


# KL_LINEAR_FEASIBILITY_FIRST_TEST_V2
def test_linear_kl_skips_root_polish_when_lbfgs_candidate_is_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ecolecon_mrio.balancing as balancing

    prior = np.array([1.0, 2.0])
    constraints = np.eye(2, dtype="float64")

    def unexpected_root(*_args, **_kwargs):
        raise AssertionError("Krylov root polish must not run for an already feasible KL candidate")

    monkeypatch.setattr(balancing, "root", unexpected_root)
    result, stats = balancing.kl_balance_linear_equalities(
        prior,
        constraints,
        prior.copy(),
        rtol=1e-12,
        atol=1e-10,
    )
    np.testing.assert_allclose(result, prior, rtol=0.0, atol=1e-12)
    assert stats.max_normalized_gap <= 1.0



# KL_LINEAR_LARGE_PRIOR_CONDITIONING_TEST_V3
def test_linear_kl_handles_donor_prior_far_above_authoritative_targets() -> None:
    """Large donor-prior mass must not destroy exact target-country equalities."""
    from ecolecon_mrio.balancing import kl_balance_linear_equalities

    # The first two prior entries mimic an output-conditioned donor column whose
    # monetary scale is many orders above the authoritative target-country cells.
    # The target system itself is modest and feasible. Solver conditioning must
    # therefore follow the equalities, not the donor-prior activity level.
    prior = np.array([1.0e8, 1.0e6, 10.0, 5.0], dtype="float64")
    constraints = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype="float64",
    )
    feasible = np.array([2.0, 3.0, 4.0, 5.0], dtype="float64")
    targets = constraints @ feasible

    result, stats = kl_balance_linear_equalities(
        prior,
        constraints,
        targets,
        rtol=1e-12,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        constraints @ result, targets, rtol=1e-12, atol=1e-10
    )
    assert np.isfinite(result).all()
    assert bool((result > 0.0).all())
    assert stats.max_normalized_gap <= 1.0

def test_linear_kl_reconciliation_fails_without_structural_support() -> None:
    from ecolecon_mrio.balancing import kl_balance_linear_equalities

    prior = np.array([1.0, 0.0])
    constraints = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="structural support"):
        kl_balance_linear_equalities(prior, constraints, np.array([1.0]))


# SEPARABLE_LINEAR_EQUALITY_TESTS_V2
def test_separable_linear_reconciliation_allows_signed_variable_to_cross_zero() -> None:
    """Signed inventory-like variables may change sign without epsilon support."""
    from ecolecon_mrio.balancing import balance_separable_linear_equalities

    prior = np.array([1.0, 2.0, 0.5], dtype="float64")
    constraints = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float64",
    )
    targets = np.array([0.0, 1.0, 2.0], dtype="float64")
    signed_mask = np.array([False, False, True])
    signed_scale = np.array([0.0, 0.0, 1.0])

    result, stats = balance_separable_linear_equalities(
        prior,
        constraints,
        targets,
        signed_quadratic_mask=signed_mask,
        signed_quadratic_scale=signed_scale,
        rtol=1e-12,
        atol=1e-10,
    )

    np.testing.assert_allclose(constraints @ result, targets, rtol=1e-12, atol=1e-10)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(2.0)
    assert result[2] == pytest.approx(-1.0)
    assert stats.max_normalized_gap <= 1.0


def test_separable_linear_reconciliation_keeps_zero_scale_signed_prior_fixed() -> None:
    """Zero signed adjustment scale is an exact fixed value, not hidden support."""
    from ecolecon_mrio.balancing import balance_separable_linear_equalities

    prior = np.array([2.0, -3.0], dtype="float64")
    constraints = np.array([[1.0, 1.0]], dtype="float64")
    result, _ = balance_separable_linear_equalities(
        prior,
        constraints,
        np.array([1.0], dtype="float64"),
        signed_quadratic_mask=np.array([False, True]),
        signed_quadratic_scale=np.array([0.0, 0.0]),
    )
    assert result[1] == pytest.approx(-3.0)
    assert result[0] == pytest.approx(4.0)

# SEPARABLE_MONETARY_UNIT_INVARIANCE_TEST_V3
def test_separable_linear_reconciliation_is_invariant_to_monetary_unit() -> None:
    """Mixed KL/signed priors must not change allocation when money units change."""
    from ecolecon_mrio.balancing import balance_separable_linear_equalities

    prior = np.array([2.0, 1.0], dtype="float64")
    constraints = np.array([[1.0, 1.0]], dtype="float64")
    target = np.array([1.0], dtype="float64")
    signed_mask = np.array([False, True])
    signed_scale = np.array([0.0, 1.0], dtype="float64")

    base, _ = balance_separable_linear_equalities(
        prior,
        constraints,
        target,
        signed_quadratic_mask=signed_mask,
        signed_quadratic_scale=signed_scale,
        rtol=1e-12,
        atol=1e-10,
    )
    factor = 1.0e6
    scaled, stats = balance_separable_linear_equalities(
        prior * factor,
        constraints,
        target * factor,
        signed_quadratic_mask=signed_mask,
        signed_quadratic_scale=signed_scale * factor,
        equality_conditioning_scale=np.array([factor], dtype="float64"),
        rtol=1e-12,
        atol=1e-10 * factor,
    )

    np.testing.assert_allclose(scaled / factor, base, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(constraints @ scaled, target * factor, rtol=1e-12, atol=1e-6)
    assert stats.max_normalized_gap <= 1.0


def test_separable_equality_conditioning_scale_is_algebraically_neutral() -> None:
    """A numerical row-scale floor must not alter the reconciled primal optimum."""
    from ecolecon_mrio.balancing import balance_separable_linear_equalities

    prior = np.array([4.0, 2.0, 1.0], dtype="float64")
    constraints = np.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, -1.0]], dtype="float64"
    )
    targets = np.array([5.0, 0.0], dtype="float64")
    signed_mask = np.array([False, False, True])
    signed_scale = np.array([0.0, 0.0, 1.0], dtype="float64")

    base, _ = balance_separable_linear_equalities(
        prior,
        constraints,
        targets,
        signed_quadratic_mask=signed_mask,
        signed_quadratic_scale=signed_scale,
    )
    conditioned, _ = balance_separable_linear_equalities(
        prior,
        constraints,
        targets,
        signed_quadratic_mask=signed_mask,
        signed_quadratic_scale=signed_scale,
        equality_conditioning_scale=np.array([5.0, 1.0e6], dtype="float64"),
    )
    np.testing.assert_allclose(conditioned, base, rtol=1e-9, atol=1e-9)

# SEPARABLE_HOMOGENEOUS_TOLERANCE_SCALE_TEST_V4
def test_separable_homogeneous_tolerance_scale_is_distinct_from_conditioning() -> None:
    """Conditioning must not relax source controls; homogeneous scale is explicit."""
    from ecolecon_mrio.balancing import balance_separable_linear_equalities

    # Two fixed signed values leave a small residual after subtractive cancellation.
    # A large conditioning scale alone must not change acceptance. Declaring the
    # monetary reference scale separately restores the relative-tolerance meaning of
    # an equivalent nonzero-RHS accounting identity.
    prior = np.array([100_000.0, 99_999.99999995], dtype="float64")
    constraints = np.array([[1.0, -1.0]], dtype="float64")
    signed_mask = np.array([True, True])
    signed_scale = np.zeros(2, dtype="float64")

    with pytest.raises(ValueError, match="structural support"):
        balance_separable_linear_equalities(
            prior,
            constraints,
            np.array([0.0]),
            signed_quadratic_mask=signed_mask,
            signed_quadratic_scale=signed_scale,
            equality_conditioning_scale=np.array([100_000.0]),
            rtol=1e-12,
            atol=1e-10,
        )

    result, stats = balance_separable_linear_equalities(
        prior,
        constraints,
        np.array([0.0]),
        signed_quadratic_mask=signed_mask,
        signed_quadratic_scale=signed_scale,
        equality_conditioning_scale=np.array([100_000.0]),
        equality_tolerance_scale=np.array([100_000.0]),
        rtol=1e-12,
        atol=1e-10,
    )
    np.testing.assert_array_equal(result, prior)
    assert stats.max_normalized_gap <= 1.0

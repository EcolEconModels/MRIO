import numpy as np


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

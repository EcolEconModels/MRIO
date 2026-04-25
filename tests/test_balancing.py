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

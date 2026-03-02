from __future__ import annotations

import numpy as np
import pandas as pd

from ecolecon_mrio.inverse_primitives import (
    calibrate_relative_change_caps,
    compute_row_interval_bounds,
    dominant_mode_diagnostics,
    matrix_min_norm_inverse_step,
)


def test_row_interval_bounds_detects_infeasible_rhs() -> None:
    a_row = np.array([0.2, 0.1, 0.3], dtype=float)
    d = np.array([0.2, 0.5, 0.3], dtype=float)
    interval, lower, upper = compute_row_interval_bounds(
        a_row=a_row,
        d=d,
        lambda_target=1.9,
        d_i=0.8,
        relative_change_cap=0.05,
        relative_change_cap_floor=1e-4,
        nonnegativity=True,
    )
    assert lower.shape == upper.shape == a_row.shape
    assert interval.feasible is False
    assert interval.rhs > interval.upper_sum


def test_matrix_inverse_step_hits_constraint_without_cap() -> None:
    idx = pd.Index(["a", "b", "c"])
    A = pd.DataFrame(
        [[0.4, 0.2, 0.1], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]],
        index=idx,
        columns=idx,
        dtype=float,
    )
    d = pd.Series([0.05, 0.60, 0.35], index=idx)
    A_new, delta, row_diags, intervals, mdiag = matrix_min_norm_inverse_step(
        A=A,
        target_d=d,
        lambda_target=0.60,
        solver_cfg={"method_primary": "trust-constr", "method_fallback": "SLSQP", "maxiter": 2000, "tol": 1e-10},
        constraints_cfg={"nonnegativity": True, "relative_change_cap": None, "relative_change_cap_floor": 1e-4},
    )
    assert A_new.shape == A.shape
    assert delta.shape == A.shape
    assert len(row_diags) == len(A)
    assert len(intervals) == len(A)
    assert mdiag.success is True

    d_norm = d / d.sum()
    lhs = A_new.to_numpy(dtype=float) @ d_norm.to_numpy(dtype=float)
    rhs = 0.60 * d_norm.to_numpy(dtype=float)
    rel_res = float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
    assert rel_res < 1e-6


def test_dominant_mode_diagnostics_has_core_fields() -> None:
    A = pd.DataFrame(np.eye(4) * 0.6)
    d = pd.Series([0.4, 0.2, 0.1, 0.3])
    diag = dominant_mode_diagnostics(A, d)
    assert np.isfinite(diag.lambda1)
    assert np.isfinite(diag.cosine_target)


def test_cap_calibration_quantiles_present() -> None:
    idx = pd.Index(["a", "b"])
    A_by_year = {
        2020: pd.DataFrame([[0.2, 0.1], [0.05, 0.25]], index=idx, columns=idx),
        2021: pd.DataFrame([[0.25, 0.12], [0.06, 0.24]], index=idx, columns=idx),
        2022: pd.DataFrame([[0.23, 0.14], [0.07, 0.20]], index=idx, columns=idx),
    }
    out = calibrate_relative_change_caps(A_by_year)
    assert out["n_pairs"] == 2
    assert out["n_obs"] > 0
    for key in ["P50", "P75", "P90", "P95", "P99"]:
        assert key in out["quantiles"]

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from ecolecon_mrio.diagnostics import solve_safe_just_lp


def _tiny_problem():
    idx = pd.Index(["a", "b"], name="sector")
    x = pd.Series([1.0, 2.0], index=idx)
    y = pd.Series([0.1, 0.1], index=idx)
    E = pd.DataFrame([[0.0, 0.0]], index=["eco"], columns=idx)
    S = pd.DataFrame([[0.0, 0.0]], index=["soc"], columns=idx)
    e_bar = pd.Series([10.0], index=E.index)
    s_floor = pd.Series([0.0], index=S.index)
    return idx, x, y, E, S, e_bar, s_floor


def test_safe_just_lp_accepts_unlabelled_sparse_A_full_basis() -> None:
    _idx, x, y, E, S, e_bar, s_floor = _tiny_problem()
    A = csr_matrix(np.array([[0.2, 0.0], [0.0, 0.1]], dtype=float))
    result = solve_safe_just_lp(x, y, A, E, S, e_bar, s_floor)
    assert result.x_opt.index.equals(x.index)
    assert result.objective >= -1e-10
    assert result.delta_plus.index.equals(x.index)


def test_safe_just_lp_accepts_unlabelled_sparse_A_working_set() -> None:
    _idx, x, y, E, S, e_bar, s_floor = _tiny_problem()
    A = csr_matrix(np.array([[0.2, 0.0], [0.0, 0.1]], dtype=float))
    result = solve_safe_just_lp(
        x,
        y,
        A,
        E,
        S,
        e_bar,
        s_floor,
        working_set_n=1,
        working_set_mode="output_weighted",
    )
    assert result.x_opt.index.equals(x.index)
    assert result.delta_minus.index.equals(x.index)


def test_safe_just_lp_rejects_sparse_A_with_mismatched_labels() -> None:
    _idx, x, y, E, S, e_bar, s_floor = _tiny_problem()
    A = csr_matrix(np.array([[0.2, 0.0], [0.0, 0.1]], dtype=float))
    A.index = pd.Index(["b", "a"])
    A.columns = pd.Index(["b", "a"])
    with pytest.raises(ValueError, match="Sparse A .* labels do not match x.index"):
        solve_safe_just_lp(x, y, A, E, S, e_bar, s_floor)

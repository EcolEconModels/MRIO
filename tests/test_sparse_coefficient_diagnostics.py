from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ecolecon_mrio.diagnostics import calc_coefficient_diagnostics


def test_calc_coefficient_diagnostics_accepts_sparse_without_densifying() -> None:
    A = csr_matrix(np.array([[0.0, 0.2], [0.1, 0.0]], dtype=float))
    out = calc_coefficient_diagnostics(A)
    assert out["a_min"] == 0.0
    assert out["a_max"] == 0.2
    assert out["share_neg"] == 0.0
    assert out["share_gt_upper"] == 0.0
    assert out["share_zero_or_near_zero"] == 0.5
"""Scientific invariants for sparse-safe MRIO price adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from ecolecon_mrio.price_adjustment import adjust_mrio_iot_prices, expand_region_price_factors


def _fixture() -> tuple[pd.MultiIndex, csr_matrix, pd.Series, pd.DataFrame]:
    """Construct a small balanced two-region IOT with exact row accounting."""

    index = pd.MultiIndex.from_tuples(
        [("AA", "s1"), ("AA", "s2"), ("BB", "s1")],
        names=["region", "sector"],
    )
    z_dense = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0],
        ]
    )
    y = pd.DataFrame({("FD", "hh"): [4.0, 2.0, 5.0]}, index=index)
    y.columns = pd.MultiIndex.from_tuples(y.columns, names=["region", "category"])
    x = pd.Series(z_dense.sum(axis=1) + y.sum(axis=1).to_numpy(), index=index, name="x")
    return index, csr_matrix(z_dense), x, y


def test_price_adjustment_preserves_row_accounting_and_similarity_transform() -> None:
    """Row-scaling x/Z/Y must preserve balance and imply A*=D A D^-1 exactly."""

    index, Z, x, Y = _fixture()
    region_factors = pd.Series({"AA": 1.2, "BB": 0.8}, dtype=float)
    factors = expand_region_price_factors(index, region_factors)
    adjusted = adjust_mrio_iot_prices({"Z": Z, "x": x, "Y": Y}, basis_index=index, row_factors=factors)

    Z_star = adjusted["Z"].toarray()
    x_star = adjusted["x"].to_numpy(dtype=float)
    Y_star = adjusted["Y"].to_numpy(dtype=float)
    np.testing.assert_allclose(x_star, Z_star.sum(axis=1) + Y_star.sum(axis=1))

    Z_dense = Z.toarray()
    x_values = x.to_numpy(dtype=float)
    A = Z_dense / x_values[np.newaxis, :]
    A_star = Z_star / x_star[np.newaxis, :]
    D = np.diag(factors.to_numpy(dtype=float))
    expected = D @ A @ np.diag(1.0 / factors.to_numpy(dtype=float))
    np.testing.assert_allclose(A_star, expected)

    # A common country factor cancels exactly inside the AA domestic block.
    np.testing.assert_allclose(A_star[:2, :2], A[:2, :2])


def test_price_adjustment_is_sparse_and_does_not_mutate_inputs() -> None:
    """EXIO-scale row scaling must keep Z sparse and leave the caller's matrix unchanged."""

    index, Z, x, Y = _fixture()
    before = Z.copy()
    factors = expand_region_price_factors(index, pd.Series({"AA": 1.1, "BB": 0.9}))
    adjusted = adjust_mrio_iot_prices({"Z": Z, "x": x, "Y": Y}, basis_index=index, row_factors=factors)
    assert isinstance(adjusted["Z"], csr_matrix)
    np.testing.assert_allclose(Z.toarray(), before.toarray())
    assert adjusted["Z"] is not Z


def test_price_adjustment_rejects_derived_or_physical_accounts() -> None:
    """The API must not silently deflate technical coefficients or physical extensions."""

    index, Z, _x, _Y = _fixture()
    factors = expand_region_price_factors(index, pd.Series({"AA": 1.0, "BB": 1.0}))
    with pytest.raises(ValueError, match="only primitive monetary x/Z/Y"):
        adjust_mrio_iot_prices({"A": Z}, basis_index=index, row_factors=factors)
    with pytest.raises(ValueError, match="only primitive monetary x/Z/Y"):
        adjust_mrio_iot_prices({"F": pd.DataFrame(index=index)}, basis_index=index, row_factors=factors)


def test_expand_region_factors_requires_complete_exact_region_domain() -> None:
    """Missing MRIO region factors must fail before any monetary row is rescaled."""

    index, _Z, _x, _Y = _fixture()
    with pytest.raises(ValueError, match="Missing price factors"):
        expand_region_price_factors(index, pd.Series({"AA": 1.1}))

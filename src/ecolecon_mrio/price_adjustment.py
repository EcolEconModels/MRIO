"""Sparse-safe row price adjustment for primitive MRIO monetary accounts.

The transformation operates on extensive current-price accounts before any
technical coefficient is derived.  With positive row factors ``d_i`` and
``D = diag(d)``, it applies ``x* = D x``, ``Z* = D Z``, and ``Y* = D Y``.  A
caller that needs technical coefficients must derive them from the adjusted
``Z*`` and ``x*``; algebraically ``A* = D A D^-1``.

Only monetary ``x``, ``Z``, and ``Y`` are accepted.  Derived ``A`` matrices and
physical environmental extensions must not be passed through this API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, spmatrix


_ALLOWED_MONETARY_ACCOUNTS = frozenset({"x", "Z", "Y"})


def _require_basis_index(index: pd.Index) -> pd.MultiIndex:
    """Require the canonical two-level MRIO ``(region, sector)`` index."""

    if not isinstance(index, pd.MultiIndex):
        raise TypeError("basis_index must be a pandas MultiIndex")
    if list(index.names) != ["region", "sector"]:
        raise ValueError(f"basis_index names must be ['region', 'sector'], got {list(index.names)!r}")
    if not index.is_unique:
        raise ValueError("basis_index must contain unique (region, sector) labels")
    return index


def _require_full_row_factors(row_factors: pd.Series, *, basis_index: pd.MultiIndex) -> pd.Series:
    """Validate one positive finite factor for every region-sector row."""

    if not isinstance(row_factors, pd.Series):
        raise TypeError("row_factors must be a pandas Series")
    if not row_factors.index.equals(basis_index):
        raise ValueError("row_factors index must exactly equal basis_index")
    try:
        factors = row_factors.astype("float64", copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("row_factors contains non-numeric values") from exc
    values = factors.to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0.0)
    if bool(invalid.any()):
        labels = [repr(basis_index[int(pos)]) for pos in np.flatnonzero(invalid)[:5]]
        raise ValueError(f"row_factors must be positive and finite; invalid labels={labels}")
    return factors


def expand_region_price_factors(
    basis_index: pd.MultiIndex,
    region_factors: pd.Series,
) -> pd.Series:
    """Expand one positive price factor per MRIO region over its sector rows.

    This helper performs no geographic aggregation or country-code mapping.  The
    caller must first construct factors on the exact region labels used by the
    MRIO.  Keeping that boundary explicit prevents a price-index provider from
    silently inventing EXIOBASE Rest-of-World aggregation semantics.
    """

    index = _require_basis_index(basis_index)
    if not isinstance(region_factors, pd.Series):
        raise TypeError("region_factors must be a pandas Series")
    if not region_factors.index.is_unique:
        raise ValueError("region_factors index must contain unique region labels")
    try:
        numeric = region_factors.astype("float64", copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("region_factors contains non-numeric values") from exc
    if bool((~np.isfinite(numeric.to_numpy(dtype=float)) | (numeric.to_numpy(dtype=float) <= 0.0)).any()):
        raise ValueError("region_factors must be positive and finite")

    # Match on canonical region labels exactly.  No fuzzy country-name matching is
    # allowed at the numerical transformation boundary.
    lookup = {str(label): float(value) for label, value in numeric.items()}
    regions = [str(label) for label in index.get_level_values("region")]
    missing = sorted({region for region in regions if region not in lookup})
    if missing:
        raise ValueError(f"Missing price factors for MRIO regions: {missing}")
    out = pd.Series([lookup[region] for region in regions], index=index, dtype=float, name="price_adjustment_factor")
    return _require_full_row_factors(out, basis_index=index)


def _scale_sparse_rows(matrix: spmatrix, *, factors: np.ndarray, expected_rows: int, name: str) -> csr_matrix:
    """Return a float64 CSR copy with each stored row multiplied in bounded memory."""

    if matrix.shape[0] != expected_rows:
        raise ValueError(f"{name} row count {matrix.shape[0]} does not match basis length {expected_rows}")
    out: csr_matrix = cast(csr_matrix, matrix).tocsr(copy=True).astype(np.float64, copy=False)
    if out.data.size and not np.isfinite(out.data).all():
        raise ValueError(f"{name} contains non-finite sparse values")
    # EXIOBASE-scale matrices stay sparse: row slices are multiplied in place and
    # no nnz-length repeated factor vector or dense diagonal matrix is allocated.
    for row, factor in enumerate(factors):
        start = int(out.indptr[row])
        stop = int(out.indptr[row + 1])
        if stop > start:
            out.data[start:stop] *= float(factor)
    return out


def _scale_dataframe_rows(
    frame: pd.DataFrame,
    *,
    factors: pd.Series,
    basis_index: pd.MultiIndex,
    name: str,
    require_square_columns: bool,
) -> pd.DataFrame:
    """Scale a labelled dense monetary table after strict row/column validation."""

    if not frame.index.equals(basis_index):
        raise ValueError(f"{name} row index must exactly equal basis_index")
    if require_square_columns and not frame.columns.equals(basis_index):
        raise ValueError(f"{name} columns must exactly equal basis_index")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite or non-numeric values")
    return cast(pd.DataFrame, numeric.mul(factors, axis="index"))


def adjust_mrio_iot_prices(
    matrices: Mapping[str, Any],
    *,
    basis_index: pd.MultiIndex,
    row_factors: pd.Series,
) -> dict[str, Any]:
    """Apply one coherent row-price transform to primitive MRIO ``x``, ``Z``, ``Y``.

    ``matrices`` may contain any non-empty subset of ``x``, ``Z``, and ``Y``.
    Unsupported keys fail loudly so derived technical coefficients ``A`` or
    physical extensions cannot be deflated accidentally.  Inputs are never
    mutated; sparse inputs are returned as CSR matrices and pandas inputs preserve
    their labels.
    """

    index = _require_basis_index(basis_index)
    factors = _require_full_row_factors(row_factors, basis_index=index)
    if not matrices:
        raise ValueError("matrices must contain at least one of 'x', 'Z', or 'Y'")
    unsupported = sorted(set(matrices).difference(_ALLOWED_MONETARY_ACCOUNTS))
    if unsupported:
        raise ValueError(
            f"Unsupported price-adjustment accounts {unsupported}; only primitive monetary x/Z/Y are allowed"
        )

    factor_values = factors.to_numpy(dtype=float)
    adjusted: dict[str, Any] = {}
    for name, value in matrices.items():
        if name == "x":
            if not isinstance(value, pd.Series):
                raise TypeError("x must be a pandas Series")
            if not value.index.equals(index):
                raise ValueError("x index must exactly equal basis_index")
            numeric: pd.Series = cast(pd.Series, pd.to_numeric(value, errors="coerce")).astype(float)
            numeric_values = numeric.to_numpy(dtype=float)
            if not np.isfinite(numeric_values).all():
                raise ValueError("x contains non-finite or non-numeric values")
            if bool((numeric_values < 0.0).any()):
                raise ValueError("x must be non-negative")
            adjusted[name] = numeric.mul(factors)
        elif name == "Z":
            if issparse(value):
                matrix = _scale_sparse_rows(value, factors=factor_values, expected_rows=len(index), name="Z")
                matrix_shape = matrix.shape
                if matrix_shape is None:
                    raise TypeError("Z sparse matrix must expose a two-dimensional shape")
                if matrix_shape[1] != len(index):
                    raise ValueError(f"Z column count {matrix_shape[1]} does not match basis length {len(index)}")
                adjusted[name] = matrix
            elif isinstance(value, pd.DataFrame):
                adjusted[name] = _scale_dataframe_rows(
                    value,
                    factors=factors,
                    basis_index=index,
                    name="Z",
                    require_square_columns=True,
                )
            else:
                raise TypeError("Z must be a scipy sparse matrix or pandas DataFrame")
        elif name == "Y":
            if issparse(value):
                adjusted[name] = _scale_sparse_rows(value, factors=factor_values, expected_rows=len(index), name="Y")
            elif isinstance(value, pd.DataFrame):
                adjusted[name] = _scale_dataframe_rows(
                    value,
                    factors=factors,
                    basis_index=index,
                    name="Y",
                    require_square_columns=False,
                )
            else:
                raise TypeError("Y must be a scipy sparse matrix or pandas DataFrame")
    return adjusted

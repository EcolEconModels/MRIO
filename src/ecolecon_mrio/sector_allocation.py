"""Allocate product-parent vectors onto a final IOT product structure."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def parent_code(sector: str) -> str:
    """Return the unsplit product code from a final IOT sector label."""
    return str(sector).split(":", 1)[0]


def allocate_parent_matrix_by_output(
    parent_values: pd.DataFrame,
    *,
    final_output: pd.Series,
    parent_of: Callable[[str], str] = parent_code,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate parent-product values by final IOT P1 output shares.

    ``parent_values`` must be indexed by the official, unsplit product codes.
    ``final_output`` is the final fuels dis/re-aggregated IOT output vector, so
    its labels already include every route-specific and edge-child decision.
    """
    values = parent_values.apply(pd.to_numeric, errors="raise").astype("float64")
    values.index = values.index.astype(str)
    output = pd.to_numeric(final_output, errors="raise").astype("float64").copy()
    output.index = output.index.astype(str)
    if not values.index.is_unique or not output.index.is_unique:
        raise ValueError("Parent values and final IOT output must have unique sector labels.")
    if not np.isfinite(values.to_numpy(dtype="float64")).all() or not np.isfinite(output.to_numpy()).all():
        raise ValueError("Parent values and final IOT output must be finite.")
    if (output < 0).any():
        raise ValueError("Final IOT output cannot be negative when used as allocation weights.")

    parents = pd.Index([parent_of(sector) for sector in output.index], name="parent_sector")
    missing = sorted(set(parents) - set(values.index))
    extra = sorted(set(values.index) - set(parents))
    if missing or extra:
        raise ValueError(f"Product parent coverage mismatch; missing={missing!r}, extra={extra!r}.")

    denominators = output.groupby(parents).transform("sum")
    if (denominators <= 0).any():
        bad = sorted(set(parents[denominators <= 0]))
        raise ValueError(f"Cannot allocate parent product(s) with zero final IOT output: {bad!r}.")
    shares = output / denominators
    allocated = values.loc[parents].set_axis(output.index)
    allocated = allocated.mul(shares, axis=0)
    audit = pd.DataFrame({"target_sector": output.index, "parent_sector": parents, "iot_output": output, "share": shares})
    return allocated, audit

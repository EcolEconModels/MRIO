from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def embodied_pressure(
    leontief: pd.DataFrame,
    final_demand: pd.Series,
    intensity: Optional[pd.Series] = None,
    normalize: bool = True,
) -> pd.Series:
    """
    Compute embodied pressure (e.g., CO2, energy, water) by upstream sector.

    This applies an intensity vector to the Leontief-based supply chain response.
    If no intensity is provided, the result is the standard embodied output that
    underpins final demand.
    """
    response = leontief.values @ final_demand.values  # upstream output required

    if intensity is not None:
        upstream = intensity.values * response
    else:
        upstream = response

    result = pd.Series(upstream, index=leontief.index, name="embodied_pressure")
    if normalize and result.sum() != 0:
        result = result / result.sum()
    return result


def forward_linkages(ghosh: pd.DataFrame) -> pd.Series:
    """
    Compute standard forward linkages (row sums of Ghosh inverse).
    Represents the total output generated throughout the economy by a unit increase
    in primary inputs of sector i.
    """
    return pd.Series(ghosh.sum(axis=1), index=ghosh.index, name="forward_linkages")


def demand_weighted_linkages(
    ghosh: pd.DataFrame, final_demand: pd.Series
) -> pd.Series:
    """
    Compute demand-weighted forward linkages (Ghosh inverse * Final Demand).
    This is the supply-side equivalent of the footprint, showing which sectors'
    supply enables the current final demand structure.
    """
    response = ghosh.values @ final_demand.values
    return pd.Series(response, index=ghosh.index, name="demand_weighted_linkages")



def dependency_contributions(
    leontief: pd.DataFrame,
    final_demand: pd.Series,
    intensity: Optional[pd.Series] = None,
    focus_sectors: Optional[Iterable[str]] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Allocate embodied pressure from each upstream sector to each focus sector.

    Parameters
    ----------
    leontief : pd.DataFrame
        Leontief inverse (L).
    final_demand : pd.Series
        Final demand vector (y).
    intensity : pd.Series, optional
        Environmental intensity.
    focus_sectors : Iterable[str], optional
        Subset of sectors for which to compute upstream allocations. Defaults
        to all sectors in the Leontief matrix.
    normalize : bool
        If True, normalise allocations to each focus sector.
    """
    L = leontief

    if focus_sectors is None:
        focus_sectors = L.index

    focus_sectors = list(focus_sectors)
    missing = set(focus_sectors) - set(L.index)
    if missing:
        raise KeyError(f"focus sectors not in system: {missing}")

    y = final_demand.loc[focus_sectors]
    response = L.values[:, L.columns.get_indexer(focus_sectors)] @ y.values

    if intensity is not None:
        response = intensity.values.reshape(-1, 1) * response

    allocations = pd.DataFrame(
        response,
        index=L.index,
        columns=focus_sectors,
    )
    if normalize:
        allocations = allocations.div(allocations.sum(axis=0), axis=1)
    return allocations


def rank_foundational_sectors(
    embodied_series: pd.Series, top_n: int = 15
) -> pd.Series:
    """Return the top contributing sectors (largest bases of the pyramid)."""
    if not isinstance(embodied_series, pd.Series):
        raise TypeError("embodied_series must be a pandas Series")
    return embodied_series.sort_values(ascending=False).head(top_n)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class PyramidInputs:
    """Container for the key matrices used in inverted pyramid analysis."""

    leontief: pd.DataFrame
    final_demand: pd.Series
    intensity: Optional[pd.Series] = None  # environmental intensity per sector

    def validate(self) -> None:
        if not isinstance(self.leontief, pd.DataFrame):
            raise TypeError("leontief must be a pandas DataFrame")
        if not isinstance(self.final_demand, pd.Series):
            raise TypeError("final_demand must be a pandas Series")
        if self.intensity is not None and not isinstance(self.intensity, pd.Series):
            raise TypeError("intensity must be a pandas Series or None")
        # Align indices to catch mismatches early
        if not self.leontief.index.equals(self.leontief.columns):
            raise ValueError("Leontief matrix must have matching index and columns")
        if not self.final_demand.index.equals(self.leontief.index):
            raise ValueError("final_demand index must match leontief dimensions")
        if self.intensity is not None and not self.intensity.index.equals(
            self.leontief.index
        ):
            raise ValueError("intensity index must match leontief dimensions")


def compute_leontief(
    Z: pd.DataFrame, output: pd.Series | pd.DataFrame, epsilon: float = 1e-15
) -> pd.DataFrame:
    """
    Compute the Leontief inverse from intermediate flows and gross output.

    Parameters
    ----------
    Z : pd.DataFrame
        Intermediate transaction matrix (sectors x sectors).
    output : pd.Series
        Gross output by sector (matching Z columns).
    epsilon : float
        Small value to avoid division by zero when normalising.
    """
    if isinstance(output, pd.DataFrame):
        if output.shape[1] != 1:
            raise ValueError("output DataFrame must have a single column")
        output = output.iloc[:, 0]

    if not Z.index.equals(Z.columns):
        raise ValueError("Z must have identical row/column labels for sectors")
    if not output.index.equals(Z.columns):
        raise ValueError("output index must match Z columns")

    output_safe = output.replace(0, epsilon)
    A = Z.divide(output_safe, axis=1)
    identity = np.eye(len(A))
    L = np.linalg.inv(identity - A.values)
    return pd.DataFrame(L, index=Z.index, columns=Z.columns)


def compute_ghosh(
    Z: pd.DataFrame, output: pd.Series | pd.DataFrame, epsilon: float = 1e-15
) -> pd.DataFrame:
    """
    Compute the Ghosh inverse (supply-driven) from intermediate flows and gross output.

    Parameters
    ----------
    Z : pd.DataFrame
        Intermediate transaction matrix (sectors x sectors).
    output : pd.Series
        Gross output by sector (matching Z index).
    epsilon : float
        Small value to avoid division by zero when normalising.

    Notes
    -----
    The allocation matrix B is row-normalised (each sector's output distributed
    across purchasing sectors): B_ij = z_ij / x_i. Ghosh inverse = (I - B)^{-1}.
    """
    if isinstance(output, pd.DataFrame):
        if output.shape[1] != 1:
            raise ValueError("output DataFrame must have a single column")
        output = output.iloc[:, 0]

    if not Z.index.equals(Z.columns):
        raise ValueError("Z must have identical row/column labels for sectors")
    if not output.index.equals(Z.index):
        raise ValueError("output index must match Z index")

    output_safe = output.replace(0, epsilon)
    B = Z.divide(output_safe, axis=0)
    identity = np.eye(len(B))
    G = np.linalg.inv(identity - B.values)
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def embodied_pressure(
    pyramid_inputs: PyramidInputs,
    normalize: bool = True,
) -> pd.Series:
    """
    Compute embodied pressure (e.g., CO2, energy, water) by upstream sector.

    This applies an intensity vector to the Leontief-based supply chain response.
    If no intensity is provided, the result is the standard embodied output that
    underpins final demand.
    """
    pyramid_inputs.validate()
    L = pyramid_inputs.leontief
    y = pyramid_inputs.final_demand
    response = L.values @ y.values  # upstream output required

    if pyramid_inputs.intensity is not None:
        upstream = pyramid_inputs.intensity.values * response
    else:
        upstream = response

    result = pd.Series(upstream, index=L.index, name="embodied_pressure")
    if normalize and result.sum() != 0:
        result = result / result.sum()
    return result


def dependency_contributions(
    pyramid_inputs: PyramidInputs,
    focus_sectors: Optional[Iterable[str]] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Allocate embodied pressure from each upstream sector to each focus sector.

    Parameters
    ----------
    pyramid_inputs : PyramidInputs
        Contains Leontief inverse, final demand vector, and optional intensity.
    focus_sectors : Iterable[str], optional
        Subset of sectors for which to compute upstream allocations. Defaults
        to all sectors in the Leontief matrix.
    normalize : bool
        If True, normalise allocations to each focus sector.
    """
    pyramid_inputs.validate()
    L = pyramid_inputs.leontief

    if focus_sectors is None:
        focus_sectors = L.index

    focus_sectors = list(focus_sectors)
    missing = set(focus_sectors) - set(L.index)
    if missing:
        raise KeyError(f"focus sectors not in system: {missing}")

    y = pyramid_inputs.final_demand.loc[focus_sectors]
    response = L.values[:, L.columns.get_indexer(focus_sectors)] @ y.values

    if pyramid_inputs.intensity is not None:
        response = pyramid_inputs.intensity.values.reshape(-1, 1) * response

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

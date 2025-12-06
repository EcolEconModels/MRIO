import pandas as pd
import numpy as np
import pytest

from mrio_pyramids.metrics import (
    embodied_pressure,
    rank_foundational_sectors,
    forward_linkages,
    demand_weighted_linkages,
)

def test_embodied_pressure_normalises():
    sectors = ["a", "b"]
    L = pd.DataFrame([[1, 0.5], [0, 1]], index=sectors, columns=sectors)
    y = pd.Series([1, 1], index=sectors)
    intensity = pd.Series([2, 1], index=sectors)
    
    # Old way used PyramidInputs, new way passes df direct
    result = embodied_pressure(leontief=L, final_demand=y, intensity=intensity)
    
    assert abs(result.sum() - 1) < 1e-9
    assert result.idxmax() == "a"


def test_rank_foundational_sectors_limits():
    series = pd.Series([3, 2, 1], index=["x", "y", "z"])
    top = rank_foundational_sectors(series, top_n=2)
    assert list(top.index) == ["x", "y"]


def test_forward_linkages_row_sums():
    # Simple Ghosh matrix where row sums are clear
    sectors = ["a", "b"]
    G = pd.DataFrame([[0.5, 0.5], [0.2, 0.8]], index=sectors, columns=sectors)
    fl = forward_linkages(G)
    
    assert fl["a"] == 1.0
    assert fl["b"] == 1.0
    assert fl.name == "forward_linkages"


def test_demand_weighted_linkages_calculation():
    # G * y
    sectors = ["a", "b"]
    G = pd.DataFrame([[1.0, 0.0], [0.5, 0.5]], index=sectors, columns=sectors)
    y = pd.Series([10, 20], index=sectors)
    
    # a: 1*10 + 0*20 = 10
    # b: 0.5*10 + 0.5*20 = 5 + 10 = 15
    dwl = demand_weighted_linkages(ghosh=G, final_demand=y)
    
    assert dwl["a"] == 10.0
    assert dwl["b"] == 15.0
    assert dwl.name == "demand_weighted_linkages"

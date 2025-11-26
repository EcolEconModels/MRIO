import pandas as pd
import numpy as np

from mrio_pyramids.pyramids import (
    PyramidInputs,
    compute_leontief,
    embodied_pressure,
    rank_foundational_sectors,
    compute_ghosh,
)
from mrio_pyramids.graph import build_dependency_graph, detect_communities, symmetrize


def test_compute_leontief_identity():
    sectors = ["a", "b"]
    Z = pd.DataFrame([[0, 0], [0, 0]], index=sectors, columns=sectors)
    output = pd.Series([1, 1], index=sectors)
    L = compute_leontief(Z, output)
    assert L.equals(pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=sectors, columns=sectors))


def test_embodied_pressure_normalises():
    sectors = ["a", "b"]
    L = pd.DataFrame([[1, 0.5], [0, 1]], index=sectors, columns=sectors)
    y = pd.Series([1, 1], index=sectors)
    intensity = pd.Series([2, 1], index=sectors)
    inputs = PyramidInputs(leontief=L, final_demand=y, intensity=intensity)
    result = embodied_pressure(inputs)
    assert abs(result.sum() - 1) < 1e-9
    assert result.idxmax() == "a"


def test_rank_foundational_sectors_limits():
    series = pd.Series([3, 2, 1], index=["x", "y", "z"])
    top = rank_foundational_sectors(series, top_n=2)
    assert list(top.index) == ["x", "y"]


def test_compute_ghosh_identity_when_no_flows():
    sectors = ["a", "b"]
    Z = pd.DataFrame([[0, 0], [0, 0]], index=sectors, columns=sectors)
    output = pd.Series([1, 1], index=sectors)
    G = compute_ghosh(Z, output)
    assert G.equals(pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=sectors, columns=sectors))


def test_compute_ghosh_matches_manual_two_sector():
    sectors = ["a", "b"]
    Z = pd.DataFrame([[0, 1], [2, 0]], index=sectors, columns=sectors)
    output = pd.Series([3, 2], index=sectors)
    G = compute_ghosh(Z, output)
    expected = pd.DataFrame([[1.5, 0.5], [1.5, 1.5]], index=sectors, columns=sectors)
    pd.testing.assert_frame_equal(G, expected)


def test_build_dependency_graph_prunes_topk_threshold():
    mat = pd.DataFrame([[0.5, 0.1], [0.5, 0.9]], index=["a", "b"], columns=["a", "b"])
    G = build_dependency_graph(mat, top_k_per_downstream=1, threshold=0.2, normalize_by_column=False)
    assert set(G.edges()) == {("a", "a"), ("b", "b")}
    assert np.isclose(G["a"]["a"]["weight"], 0.5)


def test_detect_communities_on_small_graph():
    import networkx as nx

    G = nx.Graph()
    G.add_edge("a", "b", weight=1)
    G.add_edge("c", "d", weight=1)
    mapping = detect_communities(G, method="greedy")
    # Expect two communities
    assert len(set(mapping.values())) == 2

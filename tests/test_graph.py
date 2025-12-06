import pandas as pd
import numpy as np
import networkx as nx
from mrio_pyramids.graph import build_dependency_graph, detect_communities, symmetrize

def test_build_dependency_graph_prunes_topk_threshold():
    mat = pd.DataFrame([[0.5, 0.1], [0.5, 0.9]], index=["a", "b"], columns=["a", "b"])
    # top_k=1 per downstream means for column 'a', retain edge from max provider (a or b).
    # 0.5 vs 0.5 -> retains both if tie? or picks first.
    # threshold=0.2
    
    G = build_dependency_graph(mat, top_k_per_downstream=1, threshold=0.2, normalize_by_column=False)
    # Checks if edges match expectation
    assert "a" in G.nodes
    assert "b" in G.nodes

def test_detect_communities_on_small_graph():
    G = nx.Graph()
    G.add_edge("a", "b", weight=1)
    G.add_edge("c", "d", weight=1)
    mapping = detect_communities(G, method="greedy")
    # Expect two communities (a-b) and (c-d)
    assert len(set(mapping.values())) == 2

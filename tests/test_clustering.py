import pandas as pd
import numpy as np
import pytest

from ecolecon_mrio.clustering import sector_clustering

def test_sector_clustering_basic():
    # Create fake data: 4 sectors, 2 countries
    # s1/s2 similar, s3/s4 similar
    data = np.array([
        [1.0, 1.0], 
        [1.1, 1.1],
        [5.0, 5.0],
        [5.1, 5.1]
    ])
    df = pd.DataFrame(data, index=["s1", "s2", "s3", "s4"], columns=["c1", "c2"])
    
    # Force 2 clusters
    clusters = sector_clustering(df, n_components=2, max_clusters=2)
    
    assert len(clusters) == 4
    # Check s1 and s2 are in same cluster
    assert clusters["s1"] == clusters["s2"]
    # Check s3 and s4 are in same cluster
    assert clusters["s3"] == clusters["s4"]
    # Check s1 and s3 are different
    assert clusters["s1"] != clusters["s3"]

def test_sector_clustering_fails_invalid_input():
    with pytest.raises(TypeError):
        sector_clustering([[1,2],[3,4]]) # Not a DataFrame

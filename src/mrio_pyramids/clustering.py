from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA


def sector_clustering(
    matrix: pd.DataFrame,
    n_components: int = 3,
    distance_method: str = "ward",
    max_clusters: Optional[int] = None,
    t: float = 1.0,
) -> pd.Series:
    """
    Apply PCA followed by hierarchical clustering on sectoral scores across countries.

    Parameters
    ----------
    matrix : DataFrame
        Rows = sectors, columns = country scores (e.g., forward linkages).
    n_components : int
        Number of principal components to retain.
    distance_method : str
        Linkage method for hierarchical clustering (e.g., 'ward', 'average').
    max_clusters : int, optional
        If provided, cuts dendrogram at max_clusters clusters.
    t : float
        Threshold for fcluster if max_clusters is not specified.
    """
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame")
    # Center and reduce
    X = (matrix - matrix.mean()) / matrix.std(ddof=0)
    X = X.fillna(0.0).values

    pca = PCA(n_components=min(n_components, X.shape[1]))
    comps = pca.fit_transform(X)

    Z = linkage(comps, method=distance_method)
    if max_clusters:
        labels = fcluster(Z, max_clusters, criterion="maxclust")
    else:
        labels = fcluster(Z, t=t, criterion="distance")

    return pd.Series(labels, index=matrix.index, name="cluster")

# clustering/spectral.py

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseClusterer

class SpectralClusterer(BaseClusterer):
    """
    Spectral clustering for speaker embeddings.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters (speakers). If None, defaults to 2.
    affinity : {'rbf', 'nearest_neighbors', 'precomputed'}, default='rbf'
        How to construct the affinity matrix:
          - 'rbf': Gaussian RBF kernel on raw embeddings.
          - 'nearest_neighbors': k-NN graph (requires n_neighbors).
          - 'precomputed': you supply a cosine-similarity matrix (auto-computed here).
    gamma : float, optional
        Kernel coefficient for RBF kernel; ignored if affinity!='rbf'.
    n_neighbors : int, default=10
        Number of neighbors for k-NN graph; ignored if affinity!='nearest_neighbors'.
    assign_labels : {'kmeans', 'discretize'}, default='kmeans'
        Strategy to assign labels in the spectral embedding space.
    random_state : int, RandomState instance or None, default=0
        Seed for reproducibility.

    Example
    -------
    >>> from diarization.clustering.spectral import SpectralClusterer
    >>> clusterer = SpectralClusterer(n_clusters=3, affinity='precomputed')
    >>> labels = clusterer.fit_predict(embeddings)  # embeddings.shape == (N, D)
    """
    def __init__(self,
                 n_clusters: int = None,
                 affinity: str = 'rbf',
                 gamma: float = None,
                 n_neighbors: int = 10,
                 assign_labels: str = 'kmeans',
                 random_state: int = 0):
        self.n_clusters    = n_clusters
        self.affinity      = affinity
        self.gamma         = gamma
        self.n_neighbors   = n_neighbors
        self.assign_labels = assign_labels
        self.random_state  = random_state

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster the sequence of embeddings into speaker IDs.

        Parameters
        ----------
        embeddings : array-like, shape (N, D)
            Sequence of N d-dimensional embeddings.

        Returns
        -------
        labels : ndarray, shape (N,)
            Cluster label (speaker index) for each embedding.
        """
        # Determine number of clusters
        n_clusters = self.n_clusters or 2

        # Precomputed affinity?
        if self.affinity == 'precomputed':
            # Build cosine-similarity affinity matrix
            affinity_matrix = cosine_similarity(embeddings)
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels=self.assign_labels,
                random_state=self.random_state,
                n_jobs=-1
            )
            labels = model.fit_predict(affinity_matrix)

        else:
            # Use raw embeddings + built-in affinity
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity=self.affinity,
                gamma=self.gamma,
                n_neighbors=self.n_neighbors,
                assign_labels=self.assign_labels,
                random_state=self.random_state,
                n_jobs=-1
            )
            labels = model.fit_predict(embeddings)

        return labels

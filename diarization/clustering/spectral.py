# diarization/clustering/spectral.py

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseClusterer

class SpectralClusterer(BaseClusterer):
    """
    Spectral clustering for speaker embeddings.

    Parameters
    ----------
    n_clusters : int
        The number of speakers (clusters) to find.
    affinity : str, default="rbf"
        How to construct the affinity matrix:
          - "rbf": radial‐basis (Gaussian) kernel on embeddings
          - "nearest_neighbors": use k‐NN graph
          - "cosine": cosine‐similarity matrix
          - "precomputed": user must pass a precomputed affinity to fit_predict
    gamma : float, default=1.0
        Kernel bandwidth for rbf affinity.
    n_neighbors : int, default=10
        Number of neighbors for k-NN affinity.
    assign_labels : {"kmeans", "discretize"}, default="kmeans"
        How to assign labels in spectral embedding space.
    random_state : int or RandomState, default=None
        For reproducible results.
    """
    def __init__(
        self,
        n_clusters: int,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        assign_labels: str = "kmeans",
        random_state=None
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.random_state = random_state

    def fit_predict(self, embeddings: np.ndarray) -> list:
        """
        Fit the spectral clustering model on `embeddings` and return
        a label per embedding (speaker ID).
        
        Parameters
        ----------
        embeddings : np.ndarray, shape (n_segments, emb_dim)
            The per-segment embeddings.
        
        Returns
        -------
        labels : list[int], length n_segments
            Cluster assignments in {0, …, n_clusters-1}.
        """
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array, got shape {}".format(embeddings.shape))

        # Prepare input for SpectralClustering
        if self.affinity == "cosine":
            # build a cosine‐similarity matrix
            affinity_matrix = cosine_similarity(embeddings)
            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="precomputed",
                assign_labels=self.assign_labels,
                random_state=self.random_state
            )
            labels = sc.fit_predict(affinity_matrix)

        else:
            # affinity in {"rbf", "nearest_neighbors", "precomputed"}
            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity=self.affinity,
                gamma=self.gamma,
                n_neighbors=self.n_neighbors,
                assign_labels=self.assign_labels,
                random_state=self.random_state
            )
            labels = sc.fit_predict(embeddings)

        return labels.tolist()

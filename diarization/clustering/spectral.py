# diarization/clustering/spectral.py

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

def cluster_segments(segments,
                     n_clusters=2,
                     affinity='precomputed',
                     gamma=None,
                     n_neighbors=10,
                     assign_labels='kmeans',
                     random_state=0):
    """
    Spectral clustering on segment embeddings.

    :param segments: list of dicts with 'embedding'
    :param n_clusters: number of speakers
    :param affinity: 'rbf', 'nearest_neighbors', or 'precomputed'
    :param gamma: RBF kernel coefficient (if affinity='rbf')
    :param n_neighbors: for k-NN graph (if affinity='nearest_neighbors')
    :param assign_labels: 'kmeans' or 'discretize'
    :param random_state: for reproducibility
    :return: new list of dicts with added 'speaker' field
    """
    if not segments:
        return []

    X = np.vstack([np.array(s['embedding']) for s in segments])

    if affinity == 'precomputed':
        A = cosine_similarity(X)
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels=assign_labels,
            random_state=random_state,
            n_jobs=-1
        )
        labels = model.fit_predict(A)
    else:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            gamma=gamma,
            n_neighbors=n_neighbors,
            assign_labels=assign_labels,
            random_state=random_state,
            n_jobs=-1
        )
        labels = model.fit_predict(X)

    out = []
    for seg, label in zip(segments, labels):
        new_seg = seg.copy()
        new_seg['speaker'] = int(label)
        out.append(new_seg)
    return out

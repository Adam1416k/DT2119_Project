# diarization/clustering/spectral.py
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity


def cluster_segments(
    segments,
    n_clusters=2,
    affinity="precomputed",
    gamma=None,
    n_neighbors=10,
    assign_labels="kmeans",
    random_state=0,
):
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
    X = np.vstack([np.array(s["embedding"]) for s in segments])
    # catch any zero-norm rows
    norms = np.linalg.norm(X, axis=1)
    zero_idx = np.where(norms == 0)[0]
    if zero_idx.size:
        raise ValueError(f"Zero-norm embeddings at indices {zero_idx.tolist()}")
    if affinity == "precomputed":
        # build a cosine-similarity kernel
        A = cosine_similarity(X)
        # shift into [0,1], then scrub NaNs/Infs
        A = (A + 1.0) / 2.0
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        # remove self-loops
        np.fill_diagonal(A, 0)
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels=assign_labels,
            random_state=random_state,
            n_jobs=-1,
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
            n_jobs=-1,
        )
        labels = model.fit_predict(X)
    # annotate segments with speaker labels
    out = []
    for seg, label in zip(segments, labels):
        new_seg = seg.copy()
        new_seg["speaker"] = int(label)
        out.append(new_seg)
    return out

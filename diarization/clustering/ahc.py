import numpy as np
from scipy.spatial.distance import cdist
import sys
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import tqdm

# Import process_audio from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from segmenter import process_audio


def perform_clustering(embeddings):
    """
    Perform clustering on the embeddings by determining the best number of clusters.
    """
    best_score = -1
    best_n = 2  # Init to 2
    # Try clustering from 2 to min(n_emeddings, 9) speakers
    for n in range(2, min(len(embeddings), 10)):
        clustering = AgglomerativeClustering(
            n_clusters=n, metric="cosine", linkage="average", distance_threshold=None
        )
        labels = clustering.fit_predict(embeddings)

        # Compute silhouette score to evaluate clustering quality
        score = silhouette_score(embeddings, labels, metric="cosine")

        if score > best_score:
            best_score = score
            best_n = n

    # print(f"Best number of clusters (speakers): {best_n}")

    # Final clustering with the best number of clusters
    if len(embeddings) > 1:
        clustering = AgglomerativeClustering(
            n_clusters=best_n,
            metric="cosine",
            linkage="average",
            distance_threshold=None,
        )
        cluster_labels = clustering.fit_predict(embeddings)

    else:
        cluster_labels = [0]  # Only 1 segment/embedding -> Only 1 speaker

    return cluster_labels


def process_and_cluster_audio(all_segments):
    """
    Perform clustering on each segment, and return the results.
    """
    # Run segmentation and get all segments
    # all_segments = process_audio()

    clustered_results = []

    pbar = tqdm.tqdm(
        all_segments,
        total=len(all_segments),
        desc=f"Performing AHC",
    )

    for audio_data in pbar:
        filename = audio_data["filename"]
        segments = audio_data["segments"]

        # Collect embeddings for the current file
        embeddings = []
        for seg in segments:
            emb = np.array(seg["embedding"])
            if emb.ndim > 1:
                emb = emb.mean(axis=0)  # Make it 1D by averaging over time
            embeddings.append(emb)

        try:
            embeddings = np.vstack(embeddings)

        except ValueError:
            pass  # len(embedding) = 0, for some reason

        # Perform clustering (using the clustering logic)
        cluster_labels = perform_clustering(embeddings)

        # Assign cluster labels to segments
        for idx, segment in enumerate(segments):
            segment["cluster"] = int(
                cluster_labels[idx]
            )  # Ensure the cluster label is integer

        # Store the clustered result
        clustered_results.append({"filename": filename, "segments": segments})

    return clustered_results

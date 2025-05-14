import numpy as np
from scipy.spatial.distance import cdist
import sys
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Import process_audio from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmenter import process_audio

def perform_clustering(embeddings):
    """
    Perform clustering on the embeddings by determining the best number of clusters.
    """
    best_score = -1
    best_n = None
    
    for n in range(2, 10):  # Try clustering from 2 to 9 speakers
        clustering = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
        labels = clustering.fit_predict(embeddings)
        
        # Compute silhouette score to evaluate clustering quality
        score = silhouette_score(embeddings, labels, metric="cosine")
        
        if score > best_score:
            best_score = score
            best_n = n

    print(f"Best number of clusters (speakers): {best_n}")

    # Final clustering with the best number of clusters
    clustering = AgglomerativeClustering(n_clusters=best_n, metric="cosine", linkage="average")
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels


def process_and_cluster_audio():
    """
    Process audio files, perform clustering on each file, and return the results.
    """
    # Run segmentation and get all segments
    all_segments = process_audio()

    clustered_results = []
    
    for audio_data in all_segments:
        filename = audio_data["filename"]
        segments = audio_data["segments"]

        # Collect embeddings for the current file
        embeddings = []
        for seg in segments:
            emb = np.array(seg["embedding"])
            if emb.ndim > 1:
                emb = emb.mean(axis=0)  # Make it 1D by averaging over time
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)

        # Perform clustering (using the clustering logic)
        cluster_labels = perform_clustering(embeddings)

        # Assign cluster labels to segments
        for idx, segment in enumerate(segments):
            segment["cluster"] = int(cluster_labels[idx])  # Ensure the cluster label is integer

        # Store the clustered result
        clustered_results.append({
            "filename": filename,
            "segments": segments
        })

    return clustered_results

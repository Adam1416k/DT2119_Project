# diarization/clustering/bhmm.py

import numpy as np
from torch.distributions import MultivariateNormal
import torch
from sklearn.cluster import KMeans


class BayesianHMMClusterer:
    """
    Simple HMM-based clusterer:
     1) KMeans initializes hard labels.
     2) Estimates initial, transition & Gaussian emission parameters.
     3) Runs Viterbi to get smooth assignments.
    """

    def __init__(self, n_states: int = None):
        self.n_states = n_states or 2

    def fit(self, embeddings: np.ndarray):
        N, D = embeddings.shape
        K = self.n_states

        # 1) initialize with KMeans
        kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)
        labels = kmeans.labels_

        # 2) initial log‐probs
        counts = np.bincount(labels, minlength=K).astype(float) + 1e-6
        self.initial_log_probs_ = np.log(counts / counts.sum())

        # 3) transition log‐probs
        trans = np.ones((K, K), dtype=float) * 1e-6
        for t in range(N - 1):
            trans[labels[t], labels[t + 1]] += 1
        trans /= trans.sum(axis=1, keepdims=True)
        self.transition_log_probs_ = np.log(trans)

        # 4) Gaussian emissions
        self.means_ = np.zeros((K, D), float)
        self.covariances_ = np.zeros((K, D, D), float)
        for k in range(K):
            cluster_emb = embeddings[labels == k]
            if cluster_emb.shape[0] < 2:
                cluster_emb = embeddings
            self.means_[k] = cluster_emb.mean(axis=0)
            cov = np.cov(cluster_emb.T) + np.eye(D) * 1e-6
            self.covariances_[k] = cov

        return self

    def predict(self, embeddings: np.ndarray):
        N, D = embeddings.shape
        K = self.n_states

        # per‐state log likelihoods
        log_likes = np.zeros((N, K), float)
        for k in range(K):
            mvn = MultivariateNormal(
                loc=torch.tensor(self.means_[k]),
                covariance_matrix=torch.tensor(self.covariances_[k]),
            )
            log_likes[:, k] = mvn.log_prob(torch.tensor(embeddings)).numpy()

        # Viterbi
        dp = np.zeros((N, K), float)
        ptr = np.zeros((N, K), int)
        dp[0] = self.initial_log_probs_ + log_likes[0]
        for t in range(1, N):
            for j in range(K):
                scores = dp[t - 1] + self.transition_log_probs_[:, j]
                ptr[t, j] = np.argmax(scores)
                dp[t, j] = scores[ptr[t, j]] + log_likes[t, j]

        # backtrack
        states = np.zeros(N, int)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(N - 2, -1, -1):
            states[t] = ptr[t + 1, states[t + 1]]

        return states

    def fit_predict(self, embeddings: np.ndarray):
        return self.fit(embeddings).predict(embeddings)


def cluster_segments(segments, n_states=None):
    """
    Assigns a 'speaker' label to each segment.

    :param segments: list of dicts with keys 'start', 'end', 'embedding'
    :param n_states: number of speakers (int)
    :return: new list of dicts with added 'speaker' field
    """
    if not segments:
        return []

    # stack embeddings
    emb = np.vstack([np.array(s["embedding"]) for s in segments])

    # cluster
    clusterer = BayesianHMMClusterer(n_states=n_states)
    labels = clusterer.fit_predict(emb)

    # annotate
    out = []
    for seg, label in zip(segments, labels):
        new_seg = seg.copy()
        new_seg["speaker"] = int(label)
        out.append(new_seg)
    return out

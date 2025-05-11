# clustering/bhmm.py

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
from .base import BaseClusterer

tfd = tfp.distributions

class BayesianHMMClusterer(BaseClusterer):
    """
    A simple Bayesian HMM–based clusterer for speaker embeddings.
    1) Use KMeans to initialize speaker labels.
    2) Estimate initial state probs, transitions, Gaussian emissions.
    3) Run Viterbi to get final speaker sequence.
    """

    def __init__(self, n_states: int = None):
        """
        :param n_states: number of hidden states (speakers). If None, defaults to 2.
        """
        self.n_states = n_states
        # Will hold learned parameters after fit()
        self.initial_log_probs_ = None      # shape (K,)
        self.transition_log_probs_ = None   # shape (K, K)
        self.means_ = None                  # shape (K, D)
        self.covariances_ = None            # shape (K, D, D)

    def fit(self, embeddings: np.ndarray):
        """
        Estimate HMM parameters from embeddings via EM-like initialization.
        :param embeddings: array, shape (N, D)
        """
        N, D = embeddings.shape
        K = self.n_states or 2
        self.n_states = K

        # 1) KMeans for an initial hard assignment
        kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)
        labels = kmeans.labels_

        # 2) Initial state log-probs (with add-1 smoothing)
        counts = np.bincount(labels, minlength=K).astype(np.float64) + 1e-6
        self.initial_log_probs_ = np.log(counts / counts.sum())

        # 3) Transition log-probs
        trans_counts = np.ones((K, K), dtype=np.float64) * 1e-6
        for t in range(N-1):
            trans_counts[labels[t], labels[t+1]] += 1
        trans_norm = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        self.transition_log_probs_ = np.log(trans_norm)

        # 4) Emission Gaussians: means & covariances per state
        self.means_ = np.zeros((K, D), dtype=np.float64)
        self.covariances_ = np.zeros((K, D, D), dtype=np.float64)
        for k in range(K):
            cluster_emb = embeddings[labels == k]
            # if a cluster has only one point, fall back to global cov
            if len(cluster_emb) < 2:
                cluster_emb = embeddings
            self.means_[k] = cluster_emb.mean(axis=0)
            # empirical covariance + tiny regularizer
            cov = np.cov(cluster_emb.T) 
            cov += np.eye(D) * 1e-6
            self.covariances_[k] = cov

        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Run Viterbi decoding on the fitted HMM to get the most likely state per frame.
        :param embeddings: array, shape (N, D)
        :return: array of integer state labels, shape (N,)
        """
        N, D = embeddings.shape
        K = self.n_states

        # Precompute per-state log-likelihoods: shape (N, K)
        log_likes = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            mvn = tfd.MultivariateNormalFullCovariance(
                loc=self.means_[k],
                covariance_matrix=self.covariances_[k]
            )
            log_likes[:, k] = mvn.log_prob(embeddings).numpy()

        # Viterbi DP tables
        dp = np.zeros((N, K), dtype=np.float64)
        ptr = np.zeros((N, K), dtype=np.int32)

        # Initialization
        dp[0] = self.initial_log_probs_ + log_likes[0]

        # Recursion
        for t in range(1, N):
            for j in range(K):
                scores = dp[t-1] + self.transition_log_probs_[:, j]
                ptr[t, j] = np.argmax(scores)
                dp[t, j] = scores[ptr[t, j]] + log_likes[t, j]

        # Backtrack
        states = np.zeros(N, dtype=np.int32)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(N-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]

        return states

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Convenience method — estimate parameters, then decode.
        """
        return self.fit(embeddings).predict(embeddings)

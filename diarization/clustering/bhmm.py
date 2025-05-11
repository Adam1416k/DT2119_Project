# diarization/clustering/bhmm.py

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List
from .base import BaseClusterer

tfb = tfp.bijectors
tfd = tfp.distributions

class BayesianHMMClusterer(BaseClusterer):
    """
    HMM-based clustering using TensorFlow Probability.
    Fits a hidden Markov model with:
      - Categorical initial & transition distributions (learned via logits)
      - Multivariate Normal emissions (Gaussian per state)
    and then decodes the most likely hidden-state sequence (Viterbi).
    """

    def __init__(
        self,
        n_states: int,
        n_iter: int = 100,
        learning_rate: float = 0.05,
    ):
        """
        Args:
          n_states: number of hidden speakers/clusters
          n_iter: number of optimization steps (Baum–Welch via gradient ascent)
          learning_rate: for the Adam optimizer
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit_predict(self, embeddings: np.ndarray) -> List[int]:
        """
        Fit the HMM to the sequence of embeddings and return a list of
        length T (n_segments), with speaker‐cluster indices {0,...,n_states-1}.
        """
        # Ensure float32 and 2D: [T, D]
        embeddings = np.asarray(embeddings, dtype=np.float32)
        num_steps, obs_dim = embeddings.shape

        # --- 1) Define trainable parameters ---
        # Initial and transition logits for Categorical dists
        initial_logits = tf.Variable(
            tf.zeros([self.n_states]), name="initial_logits"
        )
        transition_logits = tf.Variable(
            tf.zeros([self.n_states, self.n_states]), name="transition_logits"
        )

        # Emission Gaussians: per-state mean & scale
        emission_loc = tf.Variable(
            tf.random.normal([self.n_states, obs_dim], stddev=0.5),
            name="emission_loc"
        )
        emission_scale = tfp.util.TransformedVariable(
            initial_value=0.5 * tf.ones([self.n_states, obs_dim]),
            bijector=tfb.Softplus(),
            name="emission_scale"
        )

        # --- 2) Target log-prob function ---
        def joint_log_prob():
            init_dist = tfd.Categorical(logits=initial_logits)
            trans_dist = tfd.Categorical(logits=transition_logits)
            obs_dist = tfd.MultivariateNormalDiag(
                loc=emission_loc,
                scale_diag=emission_scale
            )

            hmm = tfd.HiddenMarkovModel(
                initial_distribution=init_dist,
                transition_distribution=trans_dist,
                observation_distribution=obs_dist,
                num_steps=num_steps
            )
            # tfp expects observations shape [..., num_steps, obs_dim]
            # our embeddings is [num_steps, obs_dim], so we feed that directly.
            return tf.reduce_sum(hmm.log_prob(embeddings))

        # --- 3) Optimize parameters (approximate EM) ---
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for step in range(self.n_iter):
            with tf.GradientTape() as tape:
                loss = -joint_log_prob()
            grads = tape.gradient(
                loss,
                [initial_logits, transition_logits,
                 emission_loc, emission_scale.trainable_variables[0]]
            )
            optimizer.apply_gradients(zip(
                grads,
                [initial_logits, transition_logits,
                 emission_loc, emission_scale.trainable_variables[0]]
            ))

        # --- 4) Decode most likely state sequence ---
        trained_hmm = tfd.HiddenMarkovModel(
            initial_distribution=tfd.Categorical(logits=initial_logits),
            transition_distribution=tfd.Categorical(logits=transition_logits),
            observation_distribution=tfd.MultivariateNormalDiag(
                loc=emission_loc,
                scale_diag=emission_scale
            ),
            num_steps=num_steps
        )
        # posterior_mode runs Viterbi decoding
        most_likely_states = trained_hmm.posterior_mode(embeddings)

        # Convert to plain Python list of ints
        return most_likely_states.numpy().astype(int).tolist()

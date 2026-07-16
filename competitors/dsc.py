"""
Simple implementation of the DSC+ clustering algorithm.

GUIDELINES FOR ADDING NEW COMPETITORS:
=====================================
Function signature and parameter formats : (if needed, these are provided by the experiment framework)
    - Data : "X" (numpy.ndarray of shape (n_samples, n_features))
    - Affinity/adjacency matrix of a graph : "adjacency_matrix" (scipy.sparse.csr_matrix or numpy.ndarray, your code should handle both)
    - You can name your other hyperparameters as you wish.

Scikit-learn API Compatibility:
   The function should return an object compatible with sklearn clustering API:
   - Must have a fit() or fit_predict() method
   - Must have either a labels_ attribute (preferred) or predict() method

After adding a competitor here, you must also:
   - Add it as an option in the clusterer() function in utils/experiments_utils.py
   - Add parameter filtering in ExperimentConfig._filter_params_for_method()

"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from sklearn.cluster import KMeans  # type: ignore
from sklearn.neighbors import kneighbors_graph  # type: ignore
from sklearn.utils._param_validation import (
    _resolve_callable_param,  # type: ignore  # pyright: ignore[reportAttributeAccessIssue,reportPrivateImportUsage]
)


class DSC:
    """
    Directed Spectral Clustering (DSC+) implementation based on the Chung Laplacian, with teleportation-based smoothing to handle weakly-connected digraphs.
    """

    def __init__(
        self,
        n_clusters,
        n_neighbors,
        gamma,
        max_iter,
        tol,
        epsilon,
        affinity,
        random_state,
        n_init=1,
    ):

        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.affinity = affinity
        self.random_state = random_state
        self.n_init = n_init

    def _laplacian(self, X):
        context_kwargs = {"X": X}
        self.n_neighbors = _resolve_callable_param(self.n_neighbors, context_kwargs)
        if self.affinity == "precomputed":
            self.adjacency_matrix = X
        else:
            self.adjacency_matrix = kneighbors_graph(
                X, n_neighbors=self.n_neighbors, include_self=True
            )
        n = self.adjacency_matrix.shape[0]

        d_out = np.asarray(self.adjacency_matrix.sum(axis=1)).ravel()
        d_out[d_out <= 0] = self.epsilon  # prevent division by zero
        if sp.issparse(self.adjacency_matrix):
            P = sp.diags(1.0 / d_out) @ self.adjacency_matrix
        else:
            P = self.adjacency_matrix / d_out[:, None]

        P_teleport = np.ones((n, n)) / n
        P_smooth = self.gamma * P + (1 - self.gamma) * P_teleport
        pi = np.ones(n) / n

        for _ in range(self.max_iter):
            pi_next = pi @ P_smooth
            if np.allclose(pi_next, pi, atol=self.tol):
                break
            pi = pi_next
        pi = pi / np.sum(pi)
        pi = np.asarray(pi).flatten()
        pi[pi <= 0] = self.epsilon
        Pi_sqrt = np.diag(np.sqrt(pi))
        Pi_inv_sqrt = np.diag(1.0 / np.sqrt(pi))
        Theta = 0.5 * (
            Pi_sqrt @ P_smooth @ Pi_inv_sqrt + Pi_inv_sqrt @ P_smooth.T @ Pi_sqrt
        )

        L_dir = np.eye(n) - Theta

        return L_dir

    def _compute_embedding(self, X):
        L_dir = self._laplacian(X)
        _, eigenvectors = eigh(L_dir)
        embedding = eigenvectors[:, : self.n_clusters]
        return embedding

    def fit(self, X):
        embedding = self._compute_embedding(X)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        kmeans.fit(embedding)
        self.labels_ = kmeans.labels_

        return self.labels_

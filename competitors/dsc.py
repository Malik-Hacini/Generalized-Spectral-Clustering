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
from sklearn.cluster import KMeans # type: ignore
from sklearn.neighbors import kneighbors_graph # type: ignore
from scipy.linalg import eigh
from competitors.utils import _resolve_callable_param

class DSC:
      def __init__(self, n_clusters, n_neighbors, gamma, max_iter, tol, epsilon, random_state):

         self.n_clusters = n_clusters
         self.n_neighbors = n_neighbors
         self.gamma = gamma         
         self.max_iter = max_iter
         self.tol = tol
         self.epsilon = epsilon
         self.random_state = random_state

      def _laplacian(self, X):
         context_kwargs = {"X": X}
         self.n_neighbors = _resolve_callable_param(self.n_neighbors, context_kwargs)

         self.adjacency_matrix = kneighbors_graph(X, n_neighbors=self.n_neighbors, include_self=True)
         n = self.adjacency_matrix.shape[0]

         # Step 1: Normalize rows to get transition matrix P
         d_out = np.sum(self.adjacency_matrix, axis=1)
         d_out[d_out <= 0] = self.epsilon  # prevent division by zero
         P = self.adjacency_matrix / d_out[:, None]
         # Step 2: Teleportation-based smoothing (like in PageRank)
         P_teleport = np.ones((n, n)) / n
         P_smooth = self.gamma * P + (1 - self.gamma) * P_teleport
         # Step 3: Compute stationary distribution π with power iteration
         pi = np.ones(n) / n

         for _ in range(self.max_iter):
            pi_next = pi @ P_smooth
            if np.allclose(pi_next, pi, atol=self.tol):
                  break
            pi = pi_next
         pi = pi / np.sum(pi)
         pi = np.asarray(pi).flatten() 
         # Step 4: Symmetric operator Θ
         Pi_sqrt = np.diag(np.sqrt(pi))
         Pi_inv_sqrt = np.diag(1.0 / np.sqrt(pi))
         Theta = 0.5 * (Pi_sqrt @ P_smooth @ Pi_inv_sqrt +
                        Pi_inv_sqrt @ P_smooth.T @ Pi_sqrt)
         # Step 5: Chung Laplacian
         L_dir = np.eye(n) - Theta

         return L_dir

      def _compute_embedding(self, X):
         L_dir = self._laplacian(X)
         _, eigenvectors = eigh(L_dir)
         embedding = eigenvectors[:, :self.n_clusters]
         return embedding

      def fit(self, X):
         embedding = self._compute_embedding(X)
         kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
         kmeans.fit(embedding)
         self.labels_ = kmeans.labels_

         return self.labels_
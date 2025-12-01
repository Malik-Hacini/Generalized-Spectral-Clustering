"""
Simple implementation of the DI-SIM clustering algorithm.

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
from scipy.sparse.linalg import svds
from scipy.sparse import diags_array
from competitors.utils import _resolve_callable_param


def avg_deg_taus(adjacency_matrix, s):
   avg_deg = np.sum(adjacency_matrix) / adjacency_matrix.shape[0]
   return np.round(avg_deg * 10**s)

def safe_diags(vec, tau, epsilon):
    """
    Construct a safe diagonal matrix D^{-1/2} for DI-SIM.

    Parameters
    ----------
    vec : np.ndarray or list
        Degree vector.
    tau : float
        Regularization parameter to add to degrees.
    epsilon : float, optional
        Small constant to avoid divide-by-zero (default: 1e-8).

    Returns
    -------
    scipy.sparse.dia_matrix
        Diagonal matrix with entries 1 / sqrt(vec + tau + epsilon).
    """
    vec = np.asarray(vec)
    vec_safe = vec + tau
    vec_safe[vec_safe <= 0] = epsilon
    vec_safe = vec_safe.flatten()  
    return diags_array(1.0 / np.sqrt(vec_safe))


class DiSim:
   def __init__(self, n_clusters, n_neighbors, tau, embedding, epsilon, random_state):
      self.n_clusters = n_clusters
      self.n_neighbors = n_neighbors
      self.tau = tau
      self.embedding = embedding
      self.epsilon = epsilon
      self.random_state = random_state

   def _compute_embedding(self, X):
      self.n_neighbors = _resolve_callable_param(self.n_neighbors, {"X": X})
      self.adjacency_matrix = kneighbors_graph(X, n_neighbors=self.n_neighbors, include_self=True)
      self.tau = _resolve_callable_param(self.tau, {"adjacency_matrix": self.adjacency_matrix})
      d_out = np.sum(self.adjacency_matrix, axis=1)
      d_in = np.sum(self.adjacency_matrix, axis=0)


      D_out_inv_sqrt = safe_diags(d_out, self.tau, self.epsilon)
      D_in_inv_sqrt = safe_diags(d_in, self.tau, self.epsilon)

      A_hat = D_out_inv_sqrt @ self.adjacency_matrix @ D_in_inv_sqrt

      U, _, Vt = svds(A_hat, k=self.n_clusters)
      V = Vt.T

      if self.embedding == 'left':
         return U
      elif self.embedding == 'right':
         return V
      elif self.embedding == 'combined':
         return np.hstack([U, V])
      else:
         raise ValueError("Invalid embedding type specified.")

   def fit(self, X):
      self.embedding = self._compute_embedding(X)

      clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
      clusterer.fit(self.embedding)
      self.labels_ = clusterer.labels_

      return self.labels_


    

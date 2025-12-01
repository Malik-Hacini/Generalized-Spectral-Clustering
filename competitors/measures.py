"""
Different vertex measure strategies for generalized spectral clustering.

GUIDELINES FOR ADDING NEW MEASURE FUNCTIONS:
============================================

Function signature and parameter formats : (if needed, these are provided by the experiment framework)
    - Data : "X" (numpy.ndarray of shape (n_samples, n_features))
    - Affinity/adjacency matrix of a graph : "adjacency_matrix" (scipy.sparse.csr_matrix or numpy.ndarray, your code should handle both)
    - You can name your other hyperparameters as you wish.

Return value:
   - Must return a numpy.ndarray vector of shape (N,)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import matrix_power
from sklearn.neighbors import kneighbors_graph #type: ignore
from .neighbors import log_neighbors

def teleporting_undirected_measure(adjacency_matrix, alpha, t, epsilon=1e-8):
    """
    Builds the undirected vertex measure:
    v = (((P)^t)^T)**alpha


    """
    is_sparse = sp.issparse(adjacency_matrix)
    N = adjacency_matrix.shape[0]
    degree_vec = adjacency_matrix.sum(axis=1).A1 if is_sparse else adjacency_matrix.sum(axis=1)
    P = adjacency_matrix / degree_vec


    P_t = matrix_power(P.copy(), t)
 
    nu = (np.array(((1/N) * np.ones((1, N)) @ P_t)).T.flatten())**alpha
    nu[nu <= 0] = epsilon
    nu_normalized = nu / np.sum(nu)

    return nu_normalized





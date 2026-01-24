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
from functools import reduce
from sklearn.neighbors import kneighbors_graph #type: ignore
from .neighbors import log_neighbors

def teleporting_undirected_measure(adjacency_matrix, alpha, t, epsilon=1e-8):
    """
    Builds the undirected vertex measure:
    nu = ((1/N) * 1^T * P^t)^alpha
    
    Uses power iteration for O(t * nnz) complexity instead of O(N²) matrix power.
    Note: The t iterations are inherently sequential (each depends on previous).
    The sparse mat-vec operations use optimized scipy/BLAS routines.
    """
    is_sparse = sp.issparse(adjacency_matrix)
    N = adjacency_matrix.shape[0]
    
    # Build row-stochastic matrix P = D^{-1} A
    degree_vec = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
    degree_vec[degree_vec == 0] = 1  # Avoid division by zero
    
    if is_sparse:
        P = sp.diags(1.0 / degree_vec) @ adjacency_matrix
    else:
        P = adjacency_matrix / degree_vec[:, np.newaxis]
    
    # Power iteration: v = (1/N) * 1^T * P^t
    # Sequential iterations are unavoidable; each v @ P is O(nnz) optimized scipy
    v = reduce(lambda v, _: v @ P, range(t), np.ones(N) / N)
    
    nu = np.power(v, alpha)
    nu[nu <= 0] = epsilon
    nu /= nu.sum()

    return nu





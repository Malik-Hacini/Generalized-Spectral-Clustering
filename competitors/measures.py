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

    Uses power iteration for O(t * log(n)) complexity instead of O(N²) matrix power.
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
    v = reduce(lambda v, _: v @ P, range(t), np.ones(N) / N)

    nu = np.power(v, alpha)
    nu[nu <= 0] = epsilon
    nu /= nu.sum()

    return nu


def degree_measure(adjacency_matrix, gamma=0.5, epsilon=1e-8):
    """
    Builds the degree based vertex measure:
    nu = gamma degree_in + (1-gamma) degree_out

    For undirected graphs, degree_in = degree_out, and the stationary law is simply the normalized degree vector.

    For directed graphs, we compute the in-degree and out-degree separately and combine them.

    Arguments:
    - adjacency_matrix: The adjacency matrix of the graph
    - gamma: The weight for the in-degree
    - epsilon: A small value to avoid division by zero
    """
    is_sparse = sp.issparse(adjacency_matrix)
    N = adjacency_matrix.shape[0]

    # Compute in-degree and out-degree
    if is_sparse:
        in_degree = np.asarray(adjacency_matrix.sum(axis=0)).flatten()
        out_degree = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
    else:
        in_degree = adjacency_matrix.sum(axis=0)
        out_degree = adjacency_matrix.sum(axis=1)

    # Combine the degrees
    nu = gamma * in_degree + (1 - gamma) * out_degree

    # Normalize
    nu[nu <= 0] = epsilon
    nu /= nu.sum()

    return nu


def uniform_measure(adjacency_matrix):
    """
    Builds the uniform vertex measure:
    nu = 1/N

    Arguments:
    - adjacency_matrix: The adjacency matrix of the graph (not used in this measure)
    """
    N = adjacency_matrix.shape[0]
    nu = np.ones(N) / N
    return nu

def perron_vector_measure(adjacency_matrix, epsilon=1e-8):
    """
    Builds the Perron vector vertex measure:
    nu = leading left eigenvector of P

    For undirected graphs, this reduces to the degree measure.

    For directed graphs, this captures the stationary distribution of a random walk defined by P.

    Arguments:
    - adjacency_matrix: The adjacency matrix of the graph
    - epsilon: A small value to avoid division by zero
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

    # Compute leading left eigenvector (Perron vector)
    eigenvalues, left_eigenvectors = sp.linalg.eigs(P.T, k=1, which='LM')
    nu = np.real(left_eigenvectors[:, 0])

    nu[nu <= 0] = epsilon
    nu /= nu.sum()

    return nu
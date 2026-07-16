"""
Modularity computation for directed and undirected networks.

Based on:
- Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS.
- Leicht, E. A., & Newman, M. E. J. (2008). Community structure in directed networks.
  Physical Review Letters, 100(11), 118703. arXiv:0709.4500

The modularity Q measures the quality of a network partition by comparing
the actual edge density within clusters to the expected density under a null model.
"""

import numpy as np
import scipy.sparse as sp


def modularity(A, labels):
    """
    Compute modularity for a (possibly directed) network partition.

    For directed networks:
        Q = (1/m) * sum_{ij} [A_ij - (k_i^out * k_j^in) / m] * delta(c_i, c_j)

    For undirected networks (symmetric A):
        Q = (1/2m) * sum_{ij} [A_ij - (k_i * k_j) / 2m] * delta(c_i, c_j)

    Parameters
    ----------
    A : scipy.sparse matrix or numpy.ndarray
        Adjacency matrix of shape (n, n). Can be directed (asymmetric) or
        undirected (symmetric). Weights are supported.
    labels : array-like of shape (n,)
        Cluster assignment for each node.

    Returns
    -------
    float
        Modularity value in range [-0.5, 1]. Higher is better.
        Typical good partitions have Q in [0.3, 0.7].

    Notes
    -----
    - For sparse matrices, computation is O(m + n*k) where m is edges, k is clusters.
    - The function auto-detects directed vs undirected based on symmetry.
    - Self-loops are included in the computation.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> # Simple undirected graph: two cliques connected by one edge
    >>> A = csr_matrix([[0,1,1,0], [1,0,1,0], [1,1,0,1], [0,0,1,0]])
    >>> labels = np.array([0, 0, 0, 1])
    >>> Q = modularity(A, labels)
    """
    # Convert to sparse if needed
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = sp.csr_matrix(A)  # Ensure CSR format for efficient row operations

    n = A.shape[0]
    labels = np.asarray(labels)

    if len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) must match matrix size ({n})")

    # Total edge weight
    m = A.sum()
    if m == 0:
        return 0.0

    # Check if directed (asymmetric)
    is_directed = not _is_symmetric(A)

    # Compute degrees
    k_out = np.asarray(A.sum(axis=1)).flatten()  # Row sums (out-degree)
    k_in = np.asarray(A.sum(axis=0)).flatten()  # Column sums (in-degree)

    # Get unique clusters
    unique_labels = np.unique(labels)

    Q = 0.0

    for c in unique_labels:
        # Nodes in cluster c
        mask = labels == c
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        # Sum of edges within cluster c: sum_{i,j in c} A_ij
        A_c = A[np.ix_(idx, idx)]
        e_c = A_c.sum()

        # Expected edges under null model
        if is_directed:
            # Directed: (sum k_out in c) * (sum k_in in c) / m
            sum_k_out_c = k_out[mask].sum()
            sum_k_in_c = k_in[mask].sum()
            expected = sum_k_out_c * sum_k_in_c / m
        else:
            # Undirected: (sum k in c)^2 / (2m)
            sum_k_c = k_out[mask].sum()  # k_out == k_in for undirected
            expected = sum_k_c**2 / (2 * m)

        Q += e_c - expected

    # Normalize
    if is_directed:
        Q = Q / m
    else:
        Q = Q / (2 * m)

    return float(Q)


def _is_symmetric(A, tol=1e-10):
    """Check if sparse matrix is symmetric."""
    diff = A - A.T
    if sp.issparse(diff):
        return abs(diff).sum() < tol
    return np.allclose(diff, 0, atol=tol)

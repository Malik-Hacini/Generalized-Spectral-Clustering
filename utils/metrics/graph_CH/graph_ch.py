"""
Graph Calinski-Harabasz index based on random walk (diffusion) distance.

Computes a clustering quality score for graph partitions using a distance
derived from the transition matrix P = D^{-1}A. Supports polynomial filters
g(P) = sum_k a_k P^k for multi-scale distance computation.

The key idea (Coifman & Lafon, 2006) is that two nodes are "close" if
the random walk started from either reaches similar distributions after
some number of steps. This naturally captures cluster structure: nodes
within the same cluster have similar transition profiles.

The CH index (Calinski & Harabasz, 1974) — ratio of between-cluster to
within-cluster dispersion — is then computed using these graph distances
instead of Euclidean distances.

References
----------
Coifman, R. R. & Lafon, S. (2006). "Diffusion maps."
    Applied and Computational Harmonic Analysis, 21(1), 5-30.

Calinski, T. & Harabasz, J. (1974). "A dendrite method for cluster analysis."
    Communications in Statistics, 3(1), 1-27.

See derivation.md for the full mathematical derivation.
"""

import numpy as np
import scipy.sparse as sp


def graph_calinski_harabasz(A, labels, filter_coeffs=None, weighted=False, epsilon=1e-10):
    """
    Compute the graph Calinski-Harabasz index for a partition of a (directed) graph.

    Uses a diffusion distance derived from the transition matrix P = D^{-1}A,
    optionally smoothed by a polynomial filter g(P) = sum_k a_k P^k.

    Parameters
    ----------
    A : scipy.sparse matrix or numpy.ndarray
        Adjacency matrix of shape (n, n). Can be directed or undirected.
    labels : array-like of shape (n,)
        Cluster assignment for each node.
    filter_coeffs : dict, optional
        Polynomial filter coefficients as {power: coefficient} mapping.
        Examples:
            - {1: 1.0}           → use P directly (one-step transitions)
            - {3: 1.0}           → use P^3 (3-step diffusion)
            - {1: 0.5, 2: 0.5}  → average of P and P^2
            - {1: 0.33, 2: 0.33, 3: 0.34} → uniform average of P, P^2, P^3
        Default: {1: 1.0} (single-step transition matrix).
    weighted : bool, optional
        If True, weight the distance by the inverse stationary distribution
        (Coifman-Lafon weighting). If False, use uniform weighting.
        Default: False (uniform — safer for directed graphs).
    epsilon : float, optional
        Small constant for numerical stability. Default: 1e-10.

    Returns
    -------
    float
        Graph CH score. Higher values indicate better-separated clusters.
        Returns 0.0 for degenerate cases (single cluster, all-zero adjacency).

    Notes
    -----
    The distance between nodes i and j is:

        d(i,j)^2 = sum_z [Z(i,z) - Z(j,z)]^2 / phi(z)

    where Z = g(P) is the filtered transition matrix and phi is either
    uniform (phi=1) or the stationary distribution.

    The CH index is computed using the pairwise-distance identity to avoid
    the need for explicit graph centroids:

        WCSS = sum_c (1/2n_c) sum_{x,y in C_c} d(x,y)^2
        TSS  = (1/2n) sum_{x,y} d(x,y)^2
        BCSS = TSS - WCSS
        CH   = [BCSS/(k-1)] / [WCSS/(n-k)]

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> # Two-community directed graph
    >>> A = csr_matrix([[0,1,1,0], [1,0,1,0], [0,1,0,1], [0,0,1,0]])
    >>> labels = np.array([0, 0, 1, 1])
    >>> score = graph_calinski_harabasz(A, labels)
    >>> score > 0
    True
    >>> # Multi-scale filter
    >>> score_ms = graph_calinski_harabasz(A, labels, filter_coeffs={1: 0.5, 2: 0.5})
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = sp.csr_matrix(A)

    n = A.shape[0]
    labels = np.asarray(labels)

    if labels.shape[0] != n:
        raise ValueError(f"labels length ({labels.shape[0]}) must match matrix size ({n})")

    n_clusters = len(np.unique(labels))
    if n_clusters <= 1 or n_clusters >= n:
        return 0.0

    if A.nnz == 0:
        return 0.0

    if filter_coeffs is None:
        filter_coeffs = {1: 1.0}

    # Build transition matrix P = D^{-1} A
    P = _build_transition_matrix(A)

    # Compute filtered representation Z = g(P) = sum_k a_k P^k
    Z = _apply_polynomial_filter(P, filter_coeffs)

    # Optionally weight by inverse stationary distribution
    if weighted:
        phi = _estimate_stationary(P, epsilon)
        # Weight columns of Z by 1/sqrt(phi) so that ||Z_i - Z_j||^2 becomes
        # the weighted diffusion distance
        inv_sqrt_phi = 1.0 / np.sqrt(np.maximum(phi, epsilon))
        Z = Z.multiply(inv_sqrt_phi[np.newaxis, :]) if sp.issparse(Z) else Z * inv_sqrt_phi[np.newaxis, :]

    # Convert to dense for CH computation (Z rows are the node representations)
    if sp.issparse(Z):
        Z = Z.toarray()

    # Compute CH using efficient vectorised pairwise distance identity
    return _compute_ch_from_embeddings(Z, labels, n_clusters)


def _build_transition_matrix(A):
    """Build row-stochastic transition matrix P = D^{-1} A.

    Dangling nodes (zero out-degree) get uniform transition row.
    """
    n = A.shape[0]
    out_degree = np.asarray(A.sum(axis=1)).flatten()
    dangling = out_degree == 0

    # Avoid division by zero
    out_degree[dangling] = 1.0
    D_inv = sp.diags(1.0 / out_degree)
    P = D_inv @ A

    # Handle dangling nodes: uniform transition
    if np.any(dangling):
        dangling_idx = np.where(dangling)[0]
        P = P.tolil()
        for i in dangling_idx:
            P[i, :] = 1.0 / n
        P = sp.csr_matrix(P)

    return P


def _apply_polynomial_filter(P, filter_coeffs):
    """Compute g(P) = sum_k a_k P^k using iterative sparse matrix multiplication.

    Parameters
    ----------
    P : sparse matrix
        Row-stochastic transition matrix.
    filter_coeffs : dict
        {power: coefficient} mapping. E.g. {1: 0.5, 3: 0.5}.

    Returns
    -------
    Z : sparse or dense matrix
        The filtered matrix g(P).
    """
    n = P.shape[0]
    max_power = max(filter_coeffs.keys())

    # Include identity term if present
    if 0 in filter_coeffs:
        Z = filter_coeffs[0] * sp.eye(n, format='csr')
    else:
        Z = sp.csr_matrix((n, n))

    # Iteratively compute P^k and accumulate weighted terms
    P_power = sp.eye(n, format='csr')  # P^0 = I
    for k in range(1, max_power + 1):
        P_power = P_power @ P  # P^k

        if k in filter_coeffs:
            Z = Z + filter_coeffs[k] * P_power

    return Z


def _estimate_stationary(P, epsilon, max_iter=1000, tol=1e-10):
    """Estimate stationary distribution via power iteration.

    For directed graphs this may not converge; falls back to uniform.
    """
    n = P.shape[0]
    pi = np.ones(n) / n

    for _ in range(max_iter):
        pi_new = pi @ P
        pi_new = np.maximum(pi_new, 0)
        s = pi_new.sum()
        if s > 0:
            pi_new /= s

        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            return np.maximum(pi_new, epsilon)
        pi = pi_new

    return pi


def _compute_ch_from_embeddings(Z, labels, n_clusters):
    """Compute CH index from embedding matrix Z using the pairwise-distance identity.

    Uses the algebraic identity:
        sum_{x,y in C} ||Z_x - Z_y||^2 = 2|C| sum_{x in C} ||Z_x||^2 - 2||sum_{x in C} Z_x||^2

    This gives O(n * d) computation instead of O(n^2 * d) pairwise.

    Parameters
    ----------
    Z : ndarray of shape (n, d)
        Embedding matrix (rows are node representations).
    labels : ndarray of shape (n,)
        Cluster labels.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    float
        CH score.
    """
    n = Z.shape[0]

    # Precompute squared norms of each row
    row_sq_norms = np.sum(Z ** 2, axis=1)  # shape (n,)

    # Total sum of pairwise squared distances:
    # sum_{x,y} ||Z_x - Z_y||^2 = 2n * sum ||Z_x||^2 - 2 * ||sum Z_x||^2
    total_sq_norm_sum = row_sq_norms.sum()
    global_sum = Z.sum(axis=0)
    global_sum_sq_norm = np.sum(global_sum ** 2)
    tss_pairwise = 2 * n * total_sq_norm_sum - 2 * global_sum_sq_norm

    # TSS = (1/2n) * sum_{x,y} ||Z_x - Z_y||^2
    tss = tss_pairwise / (2 * n)

    # Within-cluster sum of pairwise squared distances
    wcss = 0.0
    unique_labels = np.unique(labels)
    for c in unique_labels:
        mask = labels == c
        n_c = mask.sum()
        if n_c <= 1:
            continue

        cluster_sq_norm_sum = row_sq_norms[mask].sum()
        cluster_sum = Z[mask].sum(axis=0)
        cluster_sum_sq_norm = np.sum(cluster_sum ** 2)

        # sum_{x,y in C_c} ||Z_x - Z_y||^2 = 2*n_c * sum||Z_x||^2 - 2*||sum Z_x||^2
        wcss_c_pairwise = 2 * n_c * cluster_sq_norm_sum - 2 * cluster_sum_sq_norm
        wcss += wcss_c_pairwise / (2 * n_c)

    bcss = tss - wcss

    # CH = [BCSS / (k-1)] / [WCSS / (n-k)]
    if wcss <= 0:
        return 0.0

    k = n_clusters
    ch = (bcss / (k - 1)) / (wcss / (n - k))
    return max(float(ch), 0.0)

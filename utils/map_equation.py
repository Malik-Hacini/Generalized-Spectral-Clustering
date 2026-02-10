"""
Map equation score for a fixed partition on a directed or undirected graph.

References
----------
Rosvall, M., & Bergstrom, C. T. (2008).
"Maps of random walks on complex networks reveal community structure."
PNAS 105(4), 1118-1123. arXiv:0707.0609.

The two-level map equation for a partition M is:

    L(M) = q^* H(Q) + sum_i p_i^* H(P_i)

where q^* is the probability of switching modules, H(Q) is the entropy of the
module-level codebook, and H(P_i) is the entropy of the within-module codebook
(including the exit code).

We use a random-walk with teleportation probability tau to ensure ergodicity in
directed graphs, consistent with the paper's random surfer model.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def map_equation(
    A: sp.spmatrix | np.ndarray,
    labels: np.ndarray,
    teleportation: float = 0.15,
    tol: float = 1e-12,
    max_iter: int = 10000,
    log_base: float = 2.0,
) -> float:
    """
    Compute the two-level map equation code length for a fixed partition.

    Parameters
    ----------
    A : scipy.sparse matrix or numpy.ndarray
        Adjacency matrix of shape (n, n). Directed or undirected. Weights allowed.
    labels : array-like of shape (n,)
        Cluster assignment for each node.
    teleportation : float, optional
        Teleportation probability tau in the random surfer model. Default 0.15.
    tol : float, optional
        Convergence tolerance for the stationary distribution. Default 1e-12.
    max_iter : int, optional
        Maximum number of power-iteration steps. Default 10000.
    log_base : float, optional
        Logarithm base for entropy (2.0 gives bits). Default 2.0.

    Returns
    -------
    float
        Map equation code length L(M). Lower is better.
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = sp.csr_matrix(A)

    n = A.shape[0]
    labels = np.asarray(labels)

    if labels.shape[0] != n:
        raise ValueError(f"labels length ({labels.shape[0]}) must match matrix size ({n})")
    if n == 0:
        return 0.0

    if teleportation < 0.0 or teleportation > 1.0:
        raise ValueError("teleportation must be in [0, 1]")

    # Outgoing weights and dangling nodes.
    out_weight = np.asarray(A.sum(axis=1)).flatten()
    dangling = out_weight == 0.0

    # Power iteration for stationary distribution of random surfer.
    p = np.full(n, 1.0 / n)
    uniform = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        prev = p
        with np.errstate(divide="ignore", invalid="ignore"):
            temp = np.zeros_like(prev)
            nz = out_weight > 0.0
            temp[nz] = prev[nz] / out_weight[nz]

        link_flow = A.T.dot(temp)
        dangling_mass = prev[dangling].sum()
        p = (1.0 - teleportation) * (link_flow + dangling_mass * uniform) + teleportation * uniform

        # L1 convergence check.
        if np.linalg.norm(p - prev, ord=1) < tol:
            break

    # Map labels to contiguous ids.
    unique_labels, label_ids = np.unique(labels, return_inverse=True)
    m = unique_labels.shape[0]
    module_sizes = np.bincount(label_ids, minlength=m)

    # Precompute node-wise in-module weight sums.
    in_module_weight = np.zeros(n, dtype=float)
    for i in range(n):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        if row_end == row_start:
            continue
        cols = A.indices[row_start:row_end]
        data = A.data[row_start:row_end]
        same = label_ids[cols] == label_ids[i]
        if np.any(same):
            in_module_weight[i] = data[same].sum()

    # Exit rates q_i and visit rates per module.
    q_i = np.zeros(m, dtype=float)
    p_module = np.bincount(label_ids, weights=p, minlength=m)

    for i in range(n):
        mod = label_ids[i]
        size = module_sizes[mod]
        if out_weight[i] > 0.0:
            in_prob = in_module_weight[i] / out_weight[i]
        else:
            # Dangling nodes follow uniform transitions.
            in_prob = size / n

        stay_prob = (1.0 - teleportation) * in_prob + teleportation * (size / n)
        exit_prob = 1.0 - stay_prob
        q_i[mod] += p[i] * exit_prob

    q_total = q_i.sum()

    # Entropy helper.
    def _entropy(probs: np.ndarray) -> float:
        probs = probs[probs > 0.0]
        if probs.size == 0:
            return 0.0
        return float(-(probs * (np.log(probs) / np.log(log_base))).sum())

    # Index codebook.
    H_Q = _entropy(q_i / q_total) if q_total > 0.0 else 0.0

    # Module codebooks.
    L = q_total * H_Q
    for mod in range(m):
        p_i = p_module[mod] + q_i[mod]
        if p_i <= 0.0:
            continue

        node_probs = p[label_ids == mod] / p_i
        exit_prob = q_i[mod] / p_i
        H_P = _entropy(np.concatenate(([exit_prob], node_probs)))
        L += p_i * H_P

    return float(L)

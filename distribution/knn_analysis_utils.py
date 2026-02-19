"""
KNN graph analysis utilities: in-degree distribution and C-L reciprocity.

Pure functions for analyzing directed KNN graphs built from point-cloud data.
All functions operate on sparse adjacency matrices and numpy arrays.

References
----------
- C-L Reciprocity: Eq. 14 in the GSC paper
- Standard Reciprocity: Eq. 12 in the GSC paper
- See distribution/derivation.md for optimized derivations
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


def build_knn_graph(X: np.ndarray, n_neighbors: int) -> sp.csr_matrix:
    """
    Build a directed KNN graph (binary connectivity, no symmetrization).

    This produces the raw directed adjacency matrix W where W[i,j] = 1
    iff j is among i's K nearest neighbors. Out-degree is exactly K
    for every node; in-degree varies.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Point-cloud feature matrix.
    n_neighbors : int
        Number of nearest neighbors K.

    Returns
    -------
    sp.csr_matrix of shape (N, N)
        Directed binary adjacency matrix with nnz = N * K.
    """
    return kneighbors_graph(
        X, n_neighbors=n_neighbors, mode='connectivity', include_self=False
    )


def compute_in_degrees(W: sp.spmatrix) -> np.ndarray:
    """
    Compute in-degree of each node in a directed graph.

    In-degree of node j = number of edges pointing to j = sum of column j.

    Parameters
    ----------
    W : sp.spmatrix of shape (N, N)
        Directed adjacency matrix.

    Returns
    -------
    np.ndarray of shape (N,)
        In-degree vector.
    """
    return np.asarray(W.sum(axis=0)).flatten()


def reciprocity(W: sp.spmatrix) -> float:
    """
    Standard reciprocity of a directed graph (Eq. 12).

    For binary W: fraction of directed edges that are reciprocated.
    reciprocity = |{(i,j): W_ij=1 AND W_ji=1}| / nnz(W)

    Computed via sparse Hadamard product in O(nnz).

    Parameters
    ----------
    W : sp.spmatrix of shape (N, N)
        Directed adjacency matrix.

    Returns
    -------
    float
        Reciprocity in [0, 1].
    """
    total_edges = W.sum()
    if total_edges == 0:
        return 0.0
    reciprocated = W.multiply(W.T).sum()
    return float(reciprocated / total_edges)


def inter_cluster_edge_matrix(W: sp.spmatrix, labels: np.ndarray) -> np.ndarray:
    """
    Compute the inter-cluster edge count matrix E.

    E[a, b] = number of directed edges from cluster a to cluster b.
    Computed as E = Y^T @ W @ Y where Y is one-hot cluster indicator.

    Parameters
    ----------
    W : sp.spmatrix of shape (N, N)
        Directed adjacency matrix.
    labels : np.ndarray of shape (N,)
        Cluster assignment for each node.

    Returns
    -------
    np.ndarray of shape (k, k)
        Inter-cluster edge count matrix, where k = number of unique labels.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    # Build one-hot indicator matrix Y (N x k), sparse for efficiency
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    row_indices = np.arange(len(labels))
    col_indices = np.array([label_to_idx[l] for l in labels])
    Y = sp.csr_matrix(
        (np.ones(len(labels)), (row_indices, col_indices)),
        shape=(len(labels), k)
    )

    # E = Y^T @ W @ Y — sparse chain, result is dense (k x k)
    E = (Y.T @ W @ Y).toarray()
    return E


def cl_reciprocity(W: sp.spmatrix, labels: np.ndarray) -> float:
    """
    Cluster-Level (C-L) reciprocity of a directed graph (Eq. 14).

    Measures how balanced inter-cluster edges are directionally.
    For each cluster pair (a, b), computes:
        r_ab = 2 * min(E(a,b), E(b,a)) / (E(a,b) + E(b,a))
    then averages over all k(k-1) ordered pairs.

    Parameters
    ----------
    W : sp.spmatrix of shape (N, N)
        Directed adjacency matrix.
    labels : np.ndarray of shape (N,)
        Cluster assignment for each node.

    Returns
    -------
    float
        C-L reciprocity in [0, 1]. Returns 0.0 if fewer than 2 clusters.
    """
    k = len(np.unique(labels))
    if k < 2:
        return 0.0

    E = inter_cluster_edge_matrix(W, labels)
    n_pairs = k * (k - 1)
    total = 0.0

    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            denom = E[a, b] + E[b, a]
            if denom > 0:
                total += 2.0 * min(E[a, b], E[b, a]) / denom

    return total / n_pairs


def analyze_pointcloud(X: np.ndarray, labels: np.ndarray, n_neighbors: int) -> dict:
    """
    Run full KNN graph analysis on a point-cloud dataset.

    Builds a directed KNN graph from X, then computes in-degree
    distribution, standard reciprocity, and C-L reciprocity.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Feature matrix.
    labels : np.ndarray of shape (N,)
        Ground-truth cluster labels.
    n_neighbors : int
        Number of nearest neighbors K.

    Returns
    -------
    dict
        Analysis results with keys: n_samples, n_features, n_clusters,
        n_neighbors, in_degrees, reciprocity, cl_reciprocity, dataset_type.
    """
    W = build_knn_graph(X, n_neighbors)
    in_degrees = compute_in_degrees(W)

    return {
        'dataset_type': 'pointcloud',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_clusters': len(np.unique(labels)),
        'n_neighbors': n_neighbors,
        'in_degrees': in_degrees,
        'reciprocity': reciprocity(W),
        'cl_reciprocity': cl_reciprocity(W, labels),
    }


def analyze_graph(W: sp.spmatrix, labels: np.ndarray) -> dict:
    """
    Run reciprocity analysis on a pre-built directed graph (network dataset).

    No KNN construction needed — the adjacency matrix is given directly.

    Parameters
    ----------
    W : sp.spmatrix of shape (N, N)
        Directed adjacency matrix (possibly weighted).
    labels : np.ndarray of shape (N,)
        Ground-truth node labels.

    Returns
    -------
    dict
        Analysis results with keys: n_samples, n_edges, n_clusters,
        mean_out_degree, in_degrees, reciprocity, cl_reciprocity, dataset_type.
    """
    N = W.shape[0]
    in_degrees = compute_in_degrees(W)
    out_degrees = np.asarray(W.sum(axis=1)).flatten()

    return {
        'dataset_type': 'graph',
        'n_samples': N,
        'n_edges': W.nnz,
        'n_clusters': len(np.unique(labels)),
        'mean_out_degree': float(out_degrees.mean()),
        'in_degrees': in_degrees,
        'reciprocity': reciprocity(W),
        'cl_reciprocity': cl_reciprocity(W, labels),
    }

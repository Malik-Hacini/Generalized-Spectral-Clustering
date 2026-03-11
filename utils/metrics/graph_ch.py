"""Graph Calinski-Harabasz score on a diffusion embedding."""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import calinski_harabasz_score  # type: ignore


def graph_calinski_harabasz(A, labels, t=1, filter_coeffs=None):
    """Compute Graph-CH from an adjacency matrix and cluster labels."""
    is_sparse = sp.issparse(A)
    coeffs = {t: 1.0} if filter_coeffs is None else filter_coeffs
    Z = _build_diffusion_embedding(A, coeffs, is_sparse)
    if is_sparse:
        Z = sp.csr_matrix(Z).toarray()  # type: ignore

    return calinski_harabasz_score(Z, labels)


def _build_diffusion_embedding(A, filter_coeffs, is_sparse):
    P = _build_transition_matrix(A, is_sparse)
    Z = _apply_polynomial_filter(P, filter_coeffs, is_sparse)

    return Z

def _build_transition_matrix(A, is_sparse):
    degree_vec = np.asarray(A.sum(axis=1)).flatten()
    degree_vec[degree_vec == 0] = 1  # Avoid division by zero

    if is_sparse:
        P = sp.diags(1.0 / degree_vec) @ A
    else:
        P = A / degree_vec[:, np.newaxis]

    return P


def _apply_polynomial_filter(P, filter_coeffs, is_sparse):
    n = P.shape[0]
    powers = np.array(sorted(filter_coeffs), dtype=int)
    weights = np.array([filter_coeffs[k] for k in powers], dtype=float)

    if is_sparse:
        weighted_powers = [
            weights[i] * sp.linalg.matrix_power(P, int(power)) for i, power in enumerate(powers)
        ]
        if len(weighted_powers) == 1:
            return weighted_powers[0]

        stacked = sp.vstack(weighted_powers, format="csr")
        reducer = sp.kron(np.ones((1, len(weighted_powers))), sp.eye(n, format="csr"), format="csr")

        return reducer @ stacked

    matrix_powers = np.stack([np.linalg.matrix_power(P, int(power)) for power in powers], axis=0)

    return np.tensordot(weights, matrix_powers, axes=(0, 0))

"""
Generate synthetic datasets of 3 Gaussian clusters.

One cluster is artificially injected with higher edge weights using a Gaussian injection.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances  # type: ignore
from sklearn.datasets import make_blobs  # type: ignore


def _gaussian_injection(
    X_blobs, n_neighbors, injection_center, sigma_injection=1, alpha=0.5, bandwidth=1.0
):
    """Inject higher affinities around a specified center using a Gaussian function.

    Parameters
    ----------
    X_blobs : ndarray of shape (n_samples, n_features)
        The original data points.
    n_neighbors : int
        Number of neighbors for kNN graph construction.
    injection_center : tuple of float
        2D point around which to inject higher affinities.
    sigma_injection : float, optional
        Standard deviation of the Gaussian injection (controls how localized the injection is).
    alpha : float, optional
        Blending weight between the natural Gaussian affinity and the injection-based affinity. Should be in [0, 1].
    bandwidth : float, optional
        Bandwidth parameter for the natural Gaussian affinity.

    Returns
    -------
    injected_graph : ndarray of shape (n_samples, n_samples)
        The injected kNN graph.
    """
    distances_to_center = pairwise_distances(X_blobs, injection_center).ravel()

    v = np.exp(-(distances_to_center**2) / (2 * sigma_injection**2))
    injection_weights = np.outer(v, v)

    # Natural Gaussian affinity between points
    distances = pairwise_distances(X_blobs)
    natural_affinity = np.exp(-(distances**2) / (2 * bandwidth**2))
    injected_affinity = alpha * injection_weights + (1 - alpha) * natural_affinity

    np.fill_diagonal(injected_affinity, 0.0)

    # kNN is computed after injection.
    n = injected_affinity.shape[0]
    k_nn_graph = np.zeros_like(injected_affinity)
    for i in range(n):
        nn_idx = np.argpartition(injected_affinity[i], -n_neighbors)[-n_neighbors:]
        k_nn_graph[i, nn_idx] = injected_affinity[i, nn_idx]

    # injected_graph = 0.5 * (k_nn_graph + k_nn_graph.T)
    injected_graph = k_nn_graph
    return injected_graph


def generate_gaussian_injection(
    n_samples=900,
    centers=((0.0, 0.0), (3.0, 0.2), (1.5, 3.0)),
    std=0.45,
    n_neighbors=10,
    injection_center=(0.0, 0.0),
    sigma_injection=1,
    alpha=0.5,
    bandwidth=1.0,
    seed=42,
):
    """Generate a synthetic dataset of 3 Gaussian clusters with an injected affinity boost.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples across all clusters.
    centers : tuple of tuple of float, optional
        2D centers of the three Gaussian components.
    std : float, optional
        Standard deviation of the Gaussian blobs.
    n_neighbors : int, optional
        Number of neighbors for kNN graph construction.
    injection_center : tuple of float, optional
        2D point around which to inject higher affinities.
    sigma_injection : float, optional
        Standard deviation of the Gaussian injection (controls how localized the injection is).
    alpha : float, optional
        Blending weight between the natural Gaussian affinity and the injection-based affinity. Should be in [0, 1].
    bandwidth : float, optional
        Bandwidth parameter for the natural Gaussian affinity.
    seed : int, optional
        Random seed.

    Returns
    -------
    injected_graph : ndarray of shape (n_samples, n_samples)
        The injected kNN graph.
    labels : ndarray of shape (n_samples,)
        Ground-truth cluster labels.
    """
    X_blobs, labels = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=std,
        random_state=seed,
        return_centers=False,
    )
    injected_graph = _gaussian_injection(
        X_blobs,
        n_neighbors,
        injection_center,
        sigma_injection,
        alpha,
        bandwidth,
    )
    return injected_graph, labels, X_blobs

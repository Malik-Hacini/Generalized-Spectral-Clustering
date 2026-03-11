import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph  # type: ignore


def grid_imbalance(grid_size, n_high, n_low, k_neighbors=None, seed = 42):
    """
    Run experiment with grid of alternating high/low density clusters

    Parameters:
    -----------
    grid_size : int
        Grid dimension (will create grid_size x grid_size clusters)
    n_high : int
        Number of points in high density clusters
    n_low : int
        Number of points in low density clusters
    k_neighbors : int, optional
        Number of nearest neighbors to consider for the k-NN graph (default: int(2*np.log(n)))
    seed : int, optional
        Random seed

    Returns:
    --------
    adjacency_matrix : scipy.sparse.csr_matrix
        Adjacency matrix of the k-NN graph built on the generated data
    labels : numpy.ndarray
        Ground truth cluster labels for each point
    X: numpy.ndarray
        The generated data points (shape: (n_samples, 2))
    """
    np.random.seed(seed)

    # Create grid of cluster centers
    spacing = 4.0  # distance between cluster centers
    cluster_data_list = []
    ground_truth_labels = []

    cluster_id = 0
    for row in range(grid_size):
        for col in range(grid_size):
            # Checkerboard pattern: high density for (row + col) even, low density for odd
            is_high_density = (row + col) % 2 == 0

            center_x = col * spacing
            center_y = row * spacing

            if is_high_density:
                n_points = n_high
                cov = [[1, 0], [0, 1]]
            else:
                n_points = n_low
                cov = [[1, 0], [0, 1]]
            cluster_data = np.random.multivariate_normal(
                mean=[center_x, center_y],
                cov=cov,
                size=n_points
            )
            cluster_data_list.append(cluster_data)
            ground_truth_labels.extend([cluster_id] * n_points)
            cluster_id += 1

    X = np.vstack(cluster_data_list)
    labels = np.array(ground_truth_labels)
    n = len(X)
    if k_neighbors is None:
        k_neighbors = int(2*np.log(n))

    adjacency_matrix = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
    return adjacency_matrix, labels, X

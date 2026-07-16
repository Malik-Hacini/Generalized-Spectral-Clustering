import numpy as np


def grid_imbalance(grid_size, n_high, n_low, seed=42):
    """
    Generate grid of alternating high/low density clusters.

    Parameters:
    -----------
    grid_size : int or tuple of int
        Grid dimensions. If int, creates grid_size x grid_size square grid.
        If tuple (rows, cols), creates rows x cols rectangular grid.
    n_high : int
        Number of points in high density clusters
    n_low : int
        Number of points in low density clusters
    seed : int, optional
        Random seed

    Returns:
    --------
    X : numpy.ndarray
        The generated data points (shape: (n_samples, 2))
    labels : numpy.ndarray
        Ground truth cluster labels for each point
    """
    np.random.seed(seed)

    # Parse grid_size
    if isinstance(grid_size, tuple):
        n_rows, n_cols = grid_size
    else:
        n_rows = n_cols = grid_size

    # Create grid of cluster centers
    spacing = 4.0  # distance between cluster centers
    cluster_data_list = []
    ground_truth_labels = []

    cluster_id = 0
    for row in range(n_rows):
        for col in range(n_cols):
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
                mean=[center_x, center_y], cov=cov, size=n_points
            )
            cluster_data_list.append(cluster_data)
            ground_truth_labels.extend([cluster_id] * n_points)
            cluster_id += 1

    X = np.vstack(cluster_data_list)
    labels = np.array(ground_truth_labels)

    return X, labels

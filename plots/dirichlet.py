"""Generate the introductory Dirichlet-energy figures used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import kneighbors_graph


SEED = 31
N_POINTS_PER_CLUSTER = 50
N_NEIGHBORS = 10
CLUSTER_COLORS = ["#072AC8", "#FFBF46", "#FF1F2E"]
STATIONARY_CMAP = LinearSegmentedColormap.from_list(
    "gray_to_red", ["#A3D9FF", "#BF1363"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the introductory Dirichlet/GDE figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/figures"),
        help="Directory where the figure PDFs are written.",
    )
    return parser.parse_args()


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    cluster_params = [
        {"mean": [0, 0], "cov": [[0.5, 0], [0, 0.5]]},
        {"mean": [3, 3], "cov": [[0.4, 0.15], [0.15, 0.4]]},
        {"mean": [2, -4], "cov": [[0.6, -0.2], [-0.2, 0.6]]},
    ]

    clusters = []
    labels = []
    for label, params in enumerate(cluster_params):
        cluster = rng.multivariate_normal(
            mean=params["mean"],
            cov=params["cov"],
            size=N_POINTS_PER_CLUSTER,
        )
        clusters.append(cluster)
        labels.extend([label] * N_POINTS_PER_CLUSTER)

    return np.vstack(clusters), np.asarray(labels, dtype=int)


def build_directed_knn_graph(data: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(
        kneighbors_graph(
            data,
            n_neighbors=N_NEIGHBORS,
            mode="connectivity",
            include_self=False,
        )
    )


def edge_coordinates(adjacency: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    adjacency_coo = adjacency.tocoo()
    return adjacency_coo.row, adjacency_coo.col


def compute_stationary_distribution(adjacency: sp.csr_matrix) -> np.ndarray:
    dense_adjacency = adjacency.toarray().astype(float)
    out_degrees = dense_adjacency.sum(axis=1)
    out_degrees[out_degrees == 0] = 1.0
    transition = dense_adjacency / out_degrees[:, np.newaxis]

    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, stationary_idx])

    if stationary.sum() < 0:
        stationary *= -1.0
    stationary = np.clip(stationary, 0.0, None)

    total = stationary.sum()
    if total == 0.0:
        raise ValueError("Failed to compute a valid stationary distribution.")
    return stationary / total


def plot_directed_edges(
    ax: plt.Axes,
    data: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    alpha: float = 0.3,
    linewidth: float = 0.8,
) -> None:
    for i, j in zip(rows, cols):
        ax.annotate(
            "",
            xy=(data[j, 0], data[j, 1]),
            xytext=(data[i, 0], data[i, 1]),
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                alpha=alpha,
                lw=linewidth,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=0,
        )


def finalize_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_stationary_distribution(
    data: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    stationary: np.ndarray,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=stationary,
        s=100,
        alpha=1.0,
        linewidth=0.5,
        cmap=STATIONARY_CMAP,
        vmin=float(np.min(stationary)),
        vmax=float(np.max(stationary)),
    )
    plot_directed_edges(ax, data, rows, cols)
    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.05, format="%.2f")
    colorbar.set_label("Stationary Distribution", labelpad=-10)
    colorbar.set_ticks([float(np.min(stationary)), float(np.max(stationary))])
    finalize_axis(ax)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def build_partition_functions(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cluster_values = {0: -1.0, 1: 0.0, 2: 1.0}
    true_partition = np.asarray([cluster_values[label] for label in labels], dtype=float)

    rng = np.random.default_rng(42)
    mixed_labels = labels.copy()
    mixed_mask = np.isin(labels, [0, 1])
    mixed_labels[mixed_mask] = rng.choice([0, 1], size=int(np.sum(mixed_mask)))
    mixed_partition = np.asarray(
        [cluster_values[label] for label in mixed_labels], dtype=float
    )
    return true_partition, mixed_partition


def plot_partition(
    data: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    output_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_directed_edges(ax, data, rows, cols)

    for color_idx, value in enumerate(np.sort(np.unique(values))):
        mask = np.isclose(values, value, atol=1e-10)
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            c=CLUSTER_COLORS[color_idx % len(CLUSTER_COLORS)],
            s=100,
            alpha=1.0,
            linewidth=0.5,
            zorder=1,
        )

    finalize_axis(ax)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, labels = generate_data()
    adjacency = build_directed_knn_graph(data)
    rows, cols = edge_coordinates(adjacency)
    stationary = compute_stationary_distribution(adjacency)
    true_partition, mixed_partition = build_partition_functions(labels)

    plot_stationary_distribution(
        data,
        rows,
        cols,
        stationary,
        args.output_dir / "clustering_ergodic.pdf",
    )
    plot_partition(
        data,
        rows,
        cols,
        true_partition,
        args.output_dir / "dirichlet_true_labels.pdf",
    )
    plot_partition(
        data,
        rows,
        cols,
        mixed_partition,
        args.output_dir / "dirichlet_mixed_labels.pdf",
    )


if __name__ == "__main__":
    main()

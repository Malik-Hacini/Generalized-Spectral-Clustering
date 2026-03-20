"""Generate the Dirichlet/GDE illustration figures used in the paper.

This is a pipeline-integrated version of the original `plots/dirichlet.py` and
intentionally preserves its visual output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import kneighbors_graph

from plots.common import resolve_kind_dir


COLORS = ["#072AC8", "#FFBF46", "#FF1F2E"]
STATIONARY_CMAP = LinearSegmentedColormap.from_list("gray_to_red", ["#A3D9FF", "#BF1363"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Dirichlet/GDE illustration figures.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to plots/figures/.")
    parser.add_argument("--seed", type=int, default=31, help="Random seed for the Gaussian clouds.")
    parser.add_argument("--k", type=int, default=10, help="k used for the directed k-NN graph.")
    return parser.parse_args()


def dirichlet_energy(f: np.ndarray, transition: np.ndarray, stationary_dist: np.ndarray) -> float:
    energy = 0.0
    for i in range(len(f)):
        for j in range(len(f)):
            if transition[i, j] > 0:
                energy += stationary_dist[i] * transition[i, j] * (f[i] - f[j]) ** 2
    return float(energy)


def generate_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n_points_per_cluster = 50
    cluster_params = [
        {"mean": [0, 0], "cov": [[0.5, 0], [0, 0.5]]},
        {"mean": [3, 3], "cov": [[0.4, 0.15], [0.15, 0.4]]},
        {"mean": [2, -4], "cov": [[0.6, -0.2], [-0.2, 0.6]]},
    ]

    clusters = []
    labels_true = []
    for idx, params in enumerate(cluster_params):
        cluster_data = np.random.multivariate_normal(params["mean"], params["cov"], n_points_per_cluster)
        clusters.append(cluster_data)
        labels_true.extend([idx] * n_points_per_cluster)

    return np.vstack(clusters), np.asarray(labels_true, dtype=int)


def build_graph(data: np.ndarray, k: int):
    adjacency = kneighbors_graph(data, n_neighbors=k, mode="connectivity", include_self=False)
    return sp.coo_matrix(adjacency)


def transition_and_stationary(adjacency_coo) -> tuple[np.ndarray, np.ndarray]:
    adjacency = sp.csr_matrix(adjacency_coo).toarray()
    out_degrees = adjacency.sum(axis=1)
    out_degrees[out_degrees == 0] = 1
    transition = adjacency / out_degrees[:, np.newaxis]

    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    stationary_idx = int(np.argmin(np.abs(eigenvalues - 1)))
    stationary = np.real(eigenvectors[:, stationary_idx])
    stationary = stationary / stationary.sum()
    return transition, stationary


def draw_directed_edges(ax, data: np.ndarray, adjacency_coo) -> None:
    for i, j in zip(adjacency_coo.row, adjacency_coo.col, strict=True):
        ax.annotate(
            "",
            xy=(data[j, 0], data[j, 1]),
            xytext=(data[i, 0], data[i, 1]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=0.8, shrinkA=5, shrinkB=5),
            zorder=0,
        )


def hide_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_stationary_distribution(data: np.ndarray, adjacency_coo, stationary: np.ndarray, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=stationary,
        s=100,
        alpha=1,
        linewidth=0.5,
        cmap=STATIONARY_CMAP,
        vmin=float(np.min(stationary)),
        vmax=float(np.max(stationary)),
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.05, format="%.2f")
    cbar.set_label("Stationary Distribution", labelpad=-10)
    cbar.set_ticks([float(np.min(stationary)), float(np.max(stationary))])

    draw_directed_edges(ax, data, adjacency_coo)
    hide_axes(ax)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_partition(data: np.ndarray, adjacency_coo, values: np.ndarray, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_directed_edges(ax, data, adjacency_coo)

    unique_vals = np.sort(np.unique(values))
    for cluster_idx, value in enumerate(unique_vals):
        mask = np.isclose(values, value, atol=0.01)
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            c=COLORS[cluster_idx % len(COLORS)],
            s=100,
            alpha=1,
            linewidth=0.5,
            zorder=1,
        )

    hide_axes(ax)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)


def build_dirichlet_functions(data: np.ndarray, transition: np.ndarray, stationary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = 3
    cluster_size = len(data) // n_clusters
    true_labels = np.repeat(np.arange(n_clusters), cluster_size)

    cluster_values = {0: -1.0, 1: 0.0, 2: 1.0}
    f_true = np.asarray([cluster_values[int(label)] for label in true_labels], dtype=float)

    np.random.seed(42)
    mixed_labels = true_labels.copy()
    mixed_indices = np.arange(2 * cluster_size)
    mixed_labels[mixed_indices] = np.random.choice([0, 1], size=len(mixed_indices))
    f_mixed = np.asarray([cluster_values[int(label)] for label in mixed_labels], dtype=float)

    energy_true = dirichlet_energy(f_true, transition, stationary)
    energy_mixed = dirichlet_energy(f_mixed, transition, stationary)
    print(f"Dirichlet energy for true labels: {energy_true:.6f}")
    print(f"Initial Dirichlet energy for mixed version: {energy_mixed:.6f}")
    print(f"Energy difference: {abs(energy_true - energy_mixed):.8f}")
    print(f"Relative energy difference: {abs(energy_true - energy_mixed) / energy_true * 100:.4f}%")

    return f_true, f_mixed


def main() -> None:
    args = parse_args()
    output_dir = resolve_kind_dir(args.output_dir, "figures")

    data, labels_true = generate_data(args.seed)
    adjacency_coo = build_graph(data, args.k)
    transition, stationary = transition_and_stationary(adjacency_coo)

    print(f"Total points: {len(data)}")
    print(f"Points per cluster: {len(data) // 3}")
    print(f"k-NN parameter: k={args.k}")
    print(f"Total edges in graph: {adjacency_coo.nnz}")
    print(f"Average in-degree: {adjacency_coo.nnz / adjacency_coo.shape[0]:.2f}")

    plot_stationary_distribution(data, adjacency_coo, stationary, output_dir / "clustering_ergodic.pdf")

    f_true, f_mixed = build_dirichlet_functions(data, transition, stationary)
    plot_partition(data, adjacency_coo, f_true, output_dir / "dirichlet_true_labels.pdf")
    plot_partition(data, adjacency_coo, f_mixed, output_dir / "dirichlet_mixed_labels.pdf")

    print(f"Saved {output_dir / 'clustering_ergodic.pdf'}")
    print(f"Saved {output_dir / 'dirichlet_true_labels.pdf'}")
    print(f"Saved {output_dir / 'dirichlet_mixed_labels.pdf'}")


if __name__ == "__main__":
    main()

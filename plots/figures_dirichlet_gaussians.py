"""Generate the introductory Dirichlet/GDE illustration figures for the paper."""

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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import kneighbors_graph

from plots.common import configure_paper_style, resolve_kind_dir

CLUSTER_COLORS = ["#072AC8", "#FFBF46", "#FF1F2E"]
STATIONARY_CMAP = LinearSegmentedColormap.from_list("dirichlet_stationary", ["#A3D9FF", "#BF1363"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the introductory Dirichlet/GDE illustration figures."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to plots/figures/.",
    )
    parser.add_argument("--seed", type=int, default=31, help="Random seed for the Gaussian clouds.")
    parser.add_argument("--k", type=int, default=10, help="k used for the directed k-NN graph.")
    return parser.parse_args()


def generate_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cluster_params = [
        {"mean": [0.0, 0.0], "cov": [[0.5, 0.0], [0.0, 0.5]]},
        {"mean": [3.0, 3.0], "cov": [[0.4, 0.15], [0.15, 0.4]]},
        {"mean": [2.0, -4.0], "cov": [[0.6, -0.2], [-0.2, 0.6]]},
    ]
    points_per_cluster = 50
    clusters = [rng.multivariate_normal(spec["mean"], spec["cov"], points_per_cluster) for spec in cluster_params]
    labels = np.concatenate([np.full(points_per_cluster, idx, dtype=int) for idx in range(len(clusters))])
    return np.vstack(clusters), labels


def build_graph(data: np.ndarray, k: int):
    return kneighbors_graph(data, n_neighbors=k, mode="connectivity", include_self=False).tocsr()


def stationary_distribution(adjacency_matrix) -> np.ndarray:
    transition = adjacency_matrix.toarray().astype(float)
    out_degree = transition.sum(axis=1)
    out_degree[out_degree == 0] = 1.0
    transition = transition / out_degree[:, None]
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    stationary_idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, stationary_idx])
    stationary = stationary / stationary.sum()
    return stationary


def mixed_partition(labels_true: np.ndarray, stationary: np.ndarray, data: np.ndarray) -> np.ndarray:
    cluster_masses = [float(stationary[labels_true == cluster].sum()) for cluster in np.unique(labels_true)]
    sink_cluster = int(np.argmax(cluster_masses))
    mixed = np.zeros_like(labels_true)
    sink_mask = labels_true == sink_cluster
    mixed[sink_mask] = 0

    remaining_mask = ~sink_mask
    threshold = float(np.median(data[remaining_mask, 0]))
    mixed[remaining_mask & (data[:, 0] <= threshold)] = 1
    mixed[remaining_mask & (data[:, 0] > threshold)] = 2
    return mixed


def _hide_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _draw_edges(ax, data: np.ndarray, adjacency_matrix) -> None:
    coo = adjacency_matrix.tocoo()
    for src, dst in zip(coo.row, coo.col, strict=True):
        ax.annotate(
            "",
            xy=(data[dst, 0], data[dst, 1]),
            xytext=(data[src, 0], data[src, 1]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.22, lw=0.7, shrinkA=5, shrinkB=5),
            zorder=0,
        )


def plot_partition(data: np.ndarray, labels: np.ndarray, adjacency_matrix, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0), layout="constrained")
    _draw_edges(ax, data, adjacency_matrix)
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            c=CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)],
            s=55,
            linewidth=0.35,
            edgecolors="white",
            zorder=2,
        )
    _hide_axes(ax)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_stationary_distribution(data: np.ndarray, stationary: np.ndarray, adjacency_matrix, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 4.0), layout="constrained")
    _draw_edges(ax, data, adjacency_matrix)
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=stationary,
        s=55,
        linewidth=0.35,
        edgecolors="white",
        cmap=STATIONARY_CMAP,
        zorder=2,
    )
    _hide_axes(ax)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02, format="%.2f")
    cbar.set_ticks([float(stationary.min()), float(stationary.max())])
    cbar.set_label("Stationary distribution", labelpad=-8)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_paper_style(plt)

    output_dir = resolve_kind_dir(args.output_dir, "figures")
    data, labels_true = generate_data(args.seed)
    adjacency_matrix = build_graph(data, args.k)
    stationary = stationary_distribution(adjacency_matrix)
    labels_mixed = mixed_partition(labels_true, stationary, data)

    plot_stationary_distribution(data, stationary, adjacency_matrix, output_dir / "clustering_ergodic.pdf")
    plot_partition(data, labels_true, adjacency_matrix, output_dir / "dirichlet_true_labels.pdf")
    plot_partition(data, labels_mixed, adjacency_matrix, output_dir / "dirichlet_mixed_labels.pdf")

    print(f"Saved {output_dir / 'clustering_ergodic.pdf'}")
    print(f"Saved {output_dir / 'dirichlet_true_labels.pdf'}")
    print(f"Saved {output_dir / 'dirichlet_mixed_labels.pdf'}")


if __name__ == "__main__":
    main()

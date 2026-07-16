"""Plot in-degree distributions of k-NN graphs built from datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.extend([str(root), str(root / "scikit-learn")])

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

from competitors.neighbors import log_neighbors
from plots.common import (
    configure_paper_style,
    project_path,
    resolve_kind_dir,
    validate_selection,
)
from utils.file_manager import load_dataset

BAR_COLOR = "#072AC8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot in-degree distributions of dataset k-NN graphs."
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True, help="Datasets to plot."
    )
    parser.add_argument(
        "--factor", type=float, default=1.0, help="Factor used in log_neighbors."
    )
    parser.add_argument(
        "--datasets-dir", default="datasets", help="Datasets directory."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to plots/figures/.",
    )
    return parser.parse_args()


def factor_label(factor: float) -> str:
    text = f"{factor:g}"
    return text.replace(".", "_")


def plot_indegree_distribution(indegree: np.ndarray, output_file: Path) -> None:
    counts = np.bincount(indegree)
    frequencies = counts / indegree.size

    fig, ax = plt.subplots(figsize=(3.2, 2.4), layout="constrained")
    ax.bar(
        np.arange(len(frequencies)),
        frequencies,
        width=0.8,
        color=BAR_COLOR,
        edgecolor=BAR_COLOR,
    )
    ax.set_xlim(-0.5, len(frequencies) - 0.5)
    ax.set_xlabel("In-degree")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_paper_style(plt)

    datasets_dir = project_path(args.datasets_dir)
    available_datasets = sorted(
        path.name for path in datasets_dir.iterdir() if path.is_dir()
    )
    selected_datasets = validate_selection(
        available_datasets, args.datasets, "datasets"
    )
    output_dir = resolve_kind_dir(args.output_dir, "figures")
    suffix = factor_label(args.factor)

    for dataset in selected_datasets:
        X, _ = load_dataset(str(datasets_dir), dataset)
        if sp.issparse(X):
            raise ValueError(
                f"Dataset '{dataset}' is a graph dataset; expected point-cloud data."
            )

        k = log_neighbors(X, factor=args.factor)
        graph = kneighbors_graph(X, n_neighbors=k, include_self=False)
        indegree = np.asarray(graph.sum(axis=0)).ravel().astype(int)
        output_file = output_dir / f"{dataset}_indegree_factor_{suffix}.pdf"
        plot_indegree_distribution(indegree, output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()

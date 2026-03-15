"""
Benchmark clustering methods on degree-imbalance DSBM datasets.

Evaluates SC-N, DSC+, and GSC-N on directed SBM graphs with varying
out-degree imbalance between high-degree and low-degree blocks.
"""

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from competitors.measures import teleporting_undirected_measure
from competitors.neighbors import log_neighbors
from synthetic_data_gw.generate_disbm_datasets import degree_imbalance_sbm
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment


def save_graph_dataset(adjacency_matrix, labels, path: str, name: str) -> None:
    """Save a sparse graph dataset in the project's graph.npz format."""
    adjacency_matrix = sp.csr_matrix(adjacency_matrix)
    labels = np.asarray(labels)

    dataset_dir = Path(path) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        dataset_dir / "graph.npz",
        adj_data=adjacency_matrix.data,
        adj_indices=adjacency_matrix.indices,
        adj_indptr=adjacency_matrix.indptr,
        adj_shape=np.asarray(adjacency_matrix.shape, dtype=np.int64),
        labels=labels,
    )


def _fmt_prob(value: float) -> str:
    """Format probabilities as path-safe strings for dataset names."""
    return f"{value:.4f}".replace(".", "p")


"""
Basic experiment config:
"""
save_path = "results"
experiment_name = "benchmark_degree_imbalance"
mode = "grid_search"  # Either "score", "grid_search" or "viz"
metrics = ("ami", "graph_ch")  # Valid metrics include "ami", "graph_ch"
n_jobs = -1  # Number of parallel jobs (-1 to use all available cores)
verbose = True

"""
Degree imbalance DSBM configuration:
"""
# Fixed graph-generation parameters
block_sizes = [500, 500, 500]  # First block is high-degree, second is low-degree
p_intra = 0.05
p_high = 0.2

# Varying parameters
p_low_values = [p_high / 15, p_high / 10, p_high / 5, p_high / 3, p_high / 2]
n_seeds = 50

# Dataset generation
datasets_path = "datasets/degree_imbalance"
Path(datasets_path).mkdir(parents=True, exist_ok=True)

# Generate datasets
print("Generating degree-imbalance DSBM datasets...")
dataset_names = []

for p_low in p_low_values:
    for seed in range(n_seeds):
        dataset_name = (
            f"disbm_degimbal_b{block_sizes[0]}-{block_sizes[1]}"
            f"_pintra{_fmt_prob(p_intra)}"
            f"_phigh{_fmt_prob(p_high)}"
            f"_plow{_fmt_prob(p_low)}"
            f"_seed{seed}"
        )
        dataset_dir = Path(datasets_path) / dataset_name
        graph_file = dataset_dir / "graph.npz"

        if not graph_file.exists():
            adjacency_matrix, labels = degree_imbalance_sbm(
                block_sizes=block_sizes,
                p_intra=p_intra,
                p_high=p_high,
                p_low=p_low,
                seed=seed,
            )
            save_graph_dataset(
                adjacency_matrix=adjacency_matrix,
                labels=labels,
                path=datasets_path,
                name=dataset_name,
            )
            print(f"  Created: {dataset_name}")

        dataset_names.append(dataset_name)

print(f"Total datasets: {len(dataset_names)}")

"""
Methods configuration:
"""
method_specs = [
    ("spectral", "SC-N"),
    ("dsc", "DSC+"),
    ("spectral", "GSC-N"),
]

"""
Parameters configuration:
"""
default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42,
    "affinity": "precomputed",
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0, 1.5, 0.1), "t": range(0, 25)},
    ),  # Grid search for GSC methods
    "metric_params": {
        "graph_ch": {
            "filter_coeffs": {2: 0.5, 3: 0.5},
        }
    },
}

method_params = [
    ("SC-UN", {"laplacian_method": "unnorm", "standard": True, "measure": None}),
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    (
        "DSC+",
        {
            "gamma": np.arange(0, 1, 0.05),
        },
    ),
    ("GSC-N", {"laplacian_method": "norm"}),
    ("GSC-UN", {"laplacian_method": "unnorm"}),
]

dataset_params = []
method_dataset_params = []

"""
Do not edit below unless you really want to!
"""
config = ExperimentConfig(
    default_params=default_params,
    dataset_params=dataset_params,
    method_params=method_params,
    method_dataset_params=method_dataset_params,
)

if __name__ == "__main__":
    start = time.time()
    _ = experiment(
        experiment_name=experiment_name,
        dataset_names=dataset_names,
        method_specs=method_specs,
        config=config,
        load_path=datasets_path,
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    end = time.time()
    print(f"Experiment completed in {end - start:.2f} seconds.")

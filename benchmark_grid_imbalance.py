"""
Benchmark clustering methods on grid-imbalance datasets.

Evaluates SC-UN, SC-N, DSC+, GSC-UN, GSC-N on checkerboard grids with
varying imbalance between high and low density clusters.
"""

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from competitors.disim import avg_deg_taus
from competitors.measures import teleporting_undirected_measure, degree_measure
from competitors.neighbors import log_neighbors
from synthetic_data_gw.generate_imbalance_checkers import grid_imbalance
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment


"""
Basic experiment config:
"""
save_path = "results"
experiment_name = "benchmark_grid_imbalance"
mode = "grid_search"  # Either "score", "grid_search" or "viz"
metrics = ("ami",)  # Valid metrics: "ami", "ari", "nmi", "ch"
n_jobs = -1  # Number of parallel jobs (-1 to use all available cores)
verbose = True

"""
Grid imbalance configuration:
"""
# Fixed parameters
grid_size = 2
n_high = 300

# Varying parameters
n_low_values = [n_high // 15, n_high // 10, n_high // 5, n_high // 3, n_high // 2]
n_seeds = 50

# Dataset generation
datasets_path = "datasets/grid_imbalance"
Path(datasets_path).mkdir(parents=True, exist_ok=True)

# Generate datasets
print("Generating grid-imbalance datasets...")
dataset_names = []

for n_low in n_low_values:
    for seed in range(n_seeds):
        dataset_name = f"grid_{grid_size}x{grid_size}_high{n_high}_low{n_low}_seed{seed}"
        dataset_path = Path(datasets_path) / dataset_name

        # Generate if doesn't exist
        if not (dataset_path / "graph.npz").exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
            adjacency_matrix, labels, X = grid_imbalance(
                grid_size=grid_size,
                n_high=n_high,
                n_low=n_low,
                seed=seed
            )

            # Save graph in .npz format
            graph_file = dataset_path / "graph.npz"
            np.savez(
                graph_file,
                adj_data=adjacency_matrix.data,
                adj_indices=adjacency_matrix.indices,
                adj_indptr=adjacency_matrix.indptr,
                adj_shape=adjacency_matrix.shape,
                labels=labels,
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
    "random_state": 42,
    "affinity": "precomputed",  # Using precomputed adjacency matrices from graph datasets
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0, 1.5, 0.1), "t": range(0, 25)},
    ),  # Grid search for GSC methods
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
    results_df = experiment(
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

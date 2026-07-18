"""Benchmark SC-N, GSC-N, and DSC+ on Gaussian-injected graphs over alpha/sigma grids."""

from pathlib import Path
import time

import numpy as np

if __package__ is None or __package__ == "":
    from common import project_path
else:
    from experiments.common import project_path

from competitors.measures import teleporting_undirected_measure
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment

from utils.synthetic_data_gen.gaussian_injection import generate_gaussian_injection
from utils.file_manager import save_graph_dataset


def _fmt_float(value: float) -> str:
    """Format float values as path-safe tokens (e.g. 0.35 -> 0p3500)."""
    return f"{value:.4f}".replace(".", "p")


"""
Basic experiment config:
"""
save_path = project_path("results")
experiment_name = "benchmark_gaussian_injection_alpha_sigma"
mode = "grid_search"  # Either "score", "grid_search" or "viz"
metrics = ("ami", "nmi", "ari", "graph_ch")  # Graph-safe supervised metrics
n_jobs = -1
verbose = True

"""
Gaussian injection benchmark configuration:
"""
n_samples = 900
centers = ((-2.0, 0.0), (0.0, 2.0), (2.0, 0.0))
std = 1
n_neighbors = 6
injection_center = ((-2.0, 0.0),)
bandwidth = 1.0  # if None, the bandwidth is set using the basic heuristic: bandwidth = mean(pairwise_distances(X_blobs))

fixed_alpha_list = [0.5]
fixed_sigma_injection_list = [0.8, 1]

# Sweep parameters
alpha_values = np.arange(0.0, 1.01, 0.1)
sigma_injection_values = (0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0)
seeds = range(10)

sigma_alpha_combos = set()

for sigma in sigma_injection_values:
    for alpha in fixed_alpha_list:
        sigma_alpha_combos.add((float(sigma), float(alpha)))

for alpha in alpha_values:
    for sigma in fixed_sigma_injection_list:
        sigma_alpha_combos.add((float(sigma), float(alpha)))

sigma_alpha_combos = sorted(sigma_alpha_combos)

# ----- Dataset gen. -------

load_path = project_path("datasets/gaussian_injection_alpha_sigma")
Path(load_path).mkdir(parents=True, exist_ok=True)

dataset_names = []

for sigma_injection, alpha in sigma_alpha_combos:
    for seed in seeds:
        dataset_name = (
            "gaussian_inj"
            f"_n{n_samples}"
            f"_k{n_neighbors}"
            f"_bw{_fmt_float(bandwidth)}"
            f"_sigma{_fmt_float(float(sigma_injection))}"
            f"_alpha{_fmt_float(float(alpha))}"
            f"_seed{seed}"
        )

        dataset_dir = Path(load_path) / dataset_name
        graph_file = dataset_dir / "graph.npz"

        if not graph_file.exists():
            injected_graph, labels, _ = generate_gaussian_injection(
                n_samples=n_samples,
                centers=centers,
                std=std,
                n_neighbors=n_neighbors,
                injection_center=injection_center,
                sigma_injection=float(sigma_injection),
                alpha=float(alpha),
                bandwidth=bandwidth,
                seed=int(seed),
            )

            save_graph_dataset(
                adjacency_matrix=injected_graph,
                labels=labels,
                path=load_path,
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
    ("dsc", "Chung"),
    ("spectral", "GSC-N"),
    ("di_sim", "DI-SIM-R"),
    ("di_sim", "DI-SIM-L"),
    ("di_sim", "DI-SIM-C"),
]

"""
Parameters configuration:
"""
default_params = {
    "n_neighbors": n_neighbors,
    "random_state": 42,
    "n_init": 100,
    "affinity": "precomputed",
    "n_it": 5,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0.0, 2.1, 0.1), "t": range(0, 26)},
    ),
    "metric_params": {
        "graph_ch": {
            "filter_coeffs": {1: 1},
        }
    },
}

method_params = [
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    (
        "DSC+",
        {
            "gamma": np.arange(0.0, 1.01, 0.05),
            "affinity": "precomputed",
            "max_iter": 300,
            "tol": 1e-10,
            "epsilon": 1e-12,
        },
    ),
    ("Chung", {"gamma": 1}),
    (
        "DI-SIM-R",
        {
            "embedding": "right",
        },
    ),
    (
        "DI-SIM-L",
        {
            "embedding": "left",
        },
    ),
    (
        "DI-SIM-C",
        {
            "embedding": "combined",
        },
    ),
    ("GSC-N", {"laplacian_method": "norm", "standard": False}),
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
        load_path=load_path,
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    end = time.time()
    print(f"Experiment completed in {end - start:.2f} seconds.")

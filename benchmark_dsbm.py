"""
Clustering benchmark for the asymmetry-controlled DSBM datasets.
"""

import os
import time

import numpy as np

from competitors.measures import teleporting_undirected_measure
from competitors.neighbors import log_neighbors
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment

"""
Basic experiment config:
"""
save_path = "results"
experiment_name = "benchmark_dsbm"
mode = "grid_search"
metrics = ("ami", "graph_ch", "modularity", "map_equation")
n_jobs = -1
verbose = True

"""
Datasets and methods configuration:
"""
load_path = "DSBM_datasets"
# We only want to test the newly generated datasets (filtering by dsbm_gamma)
dataset_names = [
    name
    for name in os.listdir("DSBM_datasets")
    if os.path.isdir(os.path.join("DSBM_datasets", name))
    and name.startswith("dsbm_gamma")
]

method_specs = [
    ("spectral", "SC-N"),
    ("spectral", "GSC-N"),
]

"""
Parameters
"""
default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42,
    "affinity": "precomputed",
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0.0, 2.0, 0.5), "t": range(0, 11)},
    ),
}

dataset_params = []

method_params = [
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    ("GSC-N", {"laplacian_method": "norm"}),
]

method_dataset_params = []

config = ExperimentConfig(
    default_params=default_params,
    dataset_params=dataset_params,
    method_params=method_params,
    method_dataset_params=method_dataset_params,
)

if __name__ == "__main__":
    start = time.time()
    results_df_parallel = experiment(
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
    print(f"Experiment completed in {end - start} seconds.")

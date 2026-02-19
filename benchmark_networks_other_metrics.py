"""Network benchmark: modularity/map-equation proxy metrics (SC-N vs GSC-N)."""

import time

import numpy as np

from competitors.measures import teleporting_undirected_measure
from competitors.neighbors import log_neighbors
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment


save_path = "results"
experiment_name = "benchmark_networks_other_metrics"
mode = "grid_search"
metrics = ("ami", "modularity", "map_equation")
n_jobs = -1
verbose = True

load_path = "datasets"
dataset_names = [
    "email_eu_core",
    "polblogs",
#    "wiki_vote",
]

method_specs = [
    ("spectral", "SC-N"),
    ("spectral", "GSC-N"),
]

default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42,
    "affinity": "precomputed",
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0, 1.5, 0.5), "t": range(0, 10)},
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
    print(f"Datasets: {dataset_names}")
    start = time.time()
    experiment(
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

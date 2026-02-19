"""Network benchmark: Graph-CH filter-profile sweep (SC-N vs GSC-N)."""

import time

import numpy as np

from competitors.measures import teleporting_undirected_measure
from competitors.neighbors import log_neighbors
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment


def _build_graph_ch_metric_params_grid() -> list[dict]:
    profiles: list[dict] = []

    delta_scales = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    for k in delta_scales:
        profiles.append(
            {
                "profile_id": f"delta_k{k:02d}",
                "profile_family": "delta_k",
                "profile_scale": k,
                "graph_ch": {
                    "filter_coeffs": {k: 1.0},
                    "weighted": False,
                    "epsilon": 1e-10,
                },
            }
        )

    prefix_scales = [2,  4,  6, 8, 10, 12]
    for k in prefix_scales:
        profiles.append(
            {
                "profile_id": f"prefix_k{k:02d}",
                "profile_family": "prefix_k",
                "profile_scale": k,
                "graph_ch": {
                    "filter_coeffs": {j: 1.0 for j in range(1, k + 1)},
                    "weighted": False,
                    "epsilon": 1e-10,
                },
            }
        )

    return profiles


save_path = "results"
experiment_name = "benchmark_networks_graphch_profiles"
mode = "grid_search"
metrics = ("ami", "graph_ch")
n_jobs = -1
verbose = True

load_path = "datasets"
dataset_names = [
    "email_eu_core",
    "polblogs",
    #"wiki_vote",
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
        {"alpha": np.arange(0, 3, 0.2), "t": range(0, 20)},
    ),
    "metric_params": _build_graph_ch_metric_params_grid(),
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
    print(f"Graph-CH profiles: {len(default_params['metric_params'])}")
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

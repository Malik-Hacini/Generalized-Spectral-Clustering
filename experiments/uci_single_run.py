"""Single-run clustering benchmark for computational performance baselines."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

save_path = project_path("results")
experiment_name = "benchmark_uci_single_run"
mode = "grid_search"
metrics = ("ami", "ch")
n_jobs = -1
verbose = True

load_path = project_path("datasets")
dataset_names = [
    "breast_tissue",
    "wine",
    "iris",
    "seeds",
    "segmentation",
    "wdbc",
    "olivetti_faces",
    "mnist64",
    "ph_recognition",
]
method_specs = [
    ("spectral", "SC-UN"),
    ("spectral", "SC-N"),
    ("dsc", "DSC+"),
    ("di_sim", "DI-SIM-R"),
    ("di_sim", "DI-SIM-L"),
    ("di_sim", "DI-SIM-C"),
    ("spectral", "GSC-N"),
    ("spectral", "GSC-UN"),
]

default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42,
    "affinity": "nearest_neighbors",
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (teleporting_undirected_measure, {"alpha": 0.2, "t": 5}),
    "tau": (avg_deg_taus, {"s": 0.5}),
}

dataset_params = []
method_params = [
    ("SC-UN", {"laplacian_method": "unnorm", "standard": True, "measure": None}),
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    ("DSC+", {"gamma": 0.5}),
    ("DI-SIM-R", {"embedding": "right"}),
    ("DI-SIM-L", {"embedding": "left"}),
    ("DI-SIM-C", {"embedding": "combined"}),
    ("GSC-N", {"laplacian_method": "norm"}),
    ("GSC-UN", {"laplacian_method": "unnorm"}),
]
method_dataset_params = []

config = ExperimentConfig(
    default_params=default_params,
    dataset_params=dataset_params,
    method_params=method_params,
    method_dataset_params=method_dataset_params,
)

if __name__ == "__main__":
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

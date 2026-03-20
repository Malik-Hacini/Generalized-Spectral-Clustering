"""Benchmark SC and GSC on the DIGRAC lead-lag dataset suite."""

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

"""
Basic experiment config:
"""
save_path = project_path("results")
experiment_name = "benchmark_lead_lag"
mode = "grid_search"
metrics = ("ami", "graph_ch")
n_jobs = -1
verbose = True

"""
Datasets and methods configuration:
"""
load_path = project_path("datasets/digrac_directed")
dataset_names = sorted(
    path.name
    for path in Path(load_path).iterdir()
    if path.is_dir() and path.name.startswith("digrac_lead_lag_")
)

if not dataset_names:
    raise ValueError(f"No lead-lag DIGRAC datasets found in {load_path}")

method_specs = [
    ("spectral", "SC-UN"),
    ("spectral", "SC-N"),
    ("spectral", "GSC-UN"),
    ("spectral", "GSC-N"),
]

"""
Parameter hierarchy (lowest to highest precedence):
1. default_params : Base parameters for all experiments
2. General dataset parameters : [(dataset_name, params_dict), ...]
3. General method parameters: [(method_name, params_dict), ...]
4. Specific parameters for a method/dataset combination : [(method_name, [(dataset, params), ...]), ...]
"""

default_params = {
    "random_state": 42,
    "affinity": "precomputed",
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0, 1.5, 0.1), "t": range(0, 25)},
    ),
    "metric_params": {
        "graph_ch": {
            "filter_coeffs": {3: 0.5, 4: 0.5},
        },
    },
}

dataset_params = []
method_params = [
    ("SC-UN", {"laplacian_method": "unnorm", "standard": True, "measure": None}),
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    ("GSC-UN", {"laplacian_method": "unnorm"}),
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

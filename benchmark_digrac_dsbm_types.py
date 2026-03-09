"""DIGRAC DSBM benchmarks split by meta-graph type.

Runs one grid-search experiment per DIGRAC DSBM family (complete/cyclic/...)
using the same SC-N vs GSC-N setup and Graph-CH profile sweep.
"""

import os
import time

import numpy as np

from competitors.measures import teleporting_undirected_measure
from competitors.neighbors import log_neighbors
from utils.config import ExperimentConfig
from utils.experiments_utils import experiment


def _build_graph_ch_metric_params_grid() -> list[dict]:
    profiles: list[dict] = []

    delta_scales = [1]
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

    prefix_scales = [2]
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


def _discover_digrac_dsbm_groups(load_path: str) -> dict[str, list[str]]:
    if not os.path.isdir(load_path):
        return {}

    datasets = [
        name
        for name in os.listdir(load_path)
        if os.path.isdir(os.path.join(load_path, name))
    ]

    groups: dict[str, list[str]] = {}
    for name in datasets:
        family = name.split("_", 1)[0]
        groups.setdefault(family, []).append(name)

    for family in groups:
        groups[family] = sorted(groups[family])

    return dict(sorted(groups.items()))


def _filter_groups_by_env(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    raw = os.getenv("DIGRAC_DSBM_TYPES", "").strip()
    if not raw:
        return groups

    selected = {token.strip() for token in raw.split(",") if token.strip()}
    return {family: names for family, names in groups.items() if family in selected}


save_path = "results"
load_path = "DSBM_datasets/digrac"
experiment_prefix = os.getenv("EXPERIMENT_PREFIX", "benchmark_digrac_dsbm")
mode = "grid_search"
metrics = ("ami", "graph_ch", "modularity", "map_equation")
n_jobs = -1
verbose = True

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
        {"alpha": np.arange(0.0, 2.0, 0.5), "t": range(0, 11)},
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
    groups = _discover_digrac_dsbm_groups(load_path)
    groups = _filter_groups_by_env(groups)

    if not groups:
        print("No DIGRAC DSBM datasets found. Expected folders under DSBM_datasets/digrac/")
        raise SystemExit(1)

    print(f"Graph-CH profiles: {len(default_params['metric_params'])}")
    print(f"DSBM families: {', '.join(groups.keys())}")

    global_start = time.time()
    for family, dataset_names in groups.items():
        experiment_name = f"{experiment_prefix}_{family}_graphch_profiles"
        print(f"\n[{family}] Running {len(dataset_names)} datasets")

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
        print(f"[{family}] Completed in {end - start:.2f} seconds")

    global_end = time.time()
    print(f"All DIGRAC DSBM family benchmarks completed in {global_end - global_start:.2f} seconds")

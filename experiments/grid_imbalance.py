"""Benchmark clustering methods on grid-imbalance datasets."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

from pathlib import Path

from synthetic_data_gen.generate_imbalance_checkers import grid_imbalance
from utils.file_manager import save_dataset

save_path = project_path("results")
experiment_name = "benchmark_grid_imbalance"
mode = "grid_search"
metrics = ("ami", "ch")
n_jobs = -1
verbose = True

grid_size = (2, 1)
n_high = 1000
n_low_values = [n_high // 15, n_high // 10, n_high // 5, n_high // 3, n_high // 2]
n_seeds = 50
datasets_path = Path(project_path("../datasets/grid_imbalance"))

method_specs = [
    ("spectral", "SC-N"),
    ("dsc", "DSC+"),
    ("spectral", "GSC-N"),
]

default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42,
    "affinity": "nearest_neighbors",
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0, 1.5, 0.1), "t": range(0, 25)},
    ),
}

method_params = [
    ("SC-UN", {"laplacian_method": "unnorm", "standard": True, "measure": None}),
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    ("DSC+", {"gamma": np.arange(0, 1, 0.05)}),
    ("GSC-N", {"laplacian_method": "norm"}),
    ("GSC-UN", {"laplacian_method": "unnorm"}),
]

dataset_params = []
method_dataset_params = []

config = ExperimentConfig(
    default_params=default_params,
    dataset_params=dataset_params,
    method_params=method_params,
    method_dataset_params=method_dataset_params,
)


def generate_datasets() -> list[str]:
    datasets_path.mkdir(parents=True, exist_ok=True)
    print("Generating grid-imbalance datasets...")
    dataset_names = []
    for n_low in n_low_values:
        for seed in range(n_seeds):
            dataset_name = (
                f"grid_{grid_size}x{grid_size}_high{n_high}_low{n_low}_seed{seed}"
            )
            dataset_path = datasets_path / dataset_name
            needs_generation = (
                not dataset_path.exists()
                or not (dataset_path / "train").exists()
                or (dataset_path / "graph.npz").exists()
            )
            if needs_generation:
                if (dataset_path / "graph.npz").exists():
                    (dataset_path / "graph.npz").unlink()
                dataset_path.mkdir(parents=True, exist_ok=True)
                X, labels = grid_imbalance(
                    grid_size=grid_size, n_high=n_high, n_low=n_low, seed=seed
                )
                save_dataset(
                    data=X,
                    labels=labels,
                    path=str(datasets_path),
                    name=dataset_name,
                    feature_cols=["x", "y"],
                    label_col="labels",
                )
                print(f"  Created: {dataset_name}")
            dataset_names.append(dataset_name)
    print(f"Total datasets: {len(dataset_names)}")
    return dataset_names


if __name__ == "__main__":
    experiment(
        experiment_name=experiment_name,
        dataset_names=generate_datasets(),
        method_specs=method_specs,
        config=config,
        load_path=str(datasets_path),
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
    )

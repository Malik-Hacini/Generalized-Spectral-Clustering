"""Benchmark clustering methods on degree-imbalanced Gaussian datasets."""

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

from synthetic_data_gw.generate_degree_imabalanced_gaussians import (
    degree_imbalanced_gaussians,
)
from utils.file_manager import save_dataset


def _fmt_float(value: float) -> str:
    """Format float values as path-safe tokens (e.g. 0.35 -> 0p3500)."""
    return f"{value:.4f}".replace(".", "p")


"""
Basic experiment config:
"""
save_path = "../results"
experiment_name = "benchmark_degree_imbalance_gaussians"
mode = "grid_search"  # Either "score", "grid_search" or "viz"
metrics = ("ami", "ch")  # Point-cloud metrics
n_jobs = -1
verbose = True

"""
Degree-imbalanced Gaussian configuration:
"""
# Fixed Gaussian parameters
cluster_sizes = (300, 300, 300)
centers = ((0.0, 0.0), (3.0, 0.0), (1.5, 3.5))
dense_std = 0.25

# Varying parameters
sparse_std_values = [0.60, 0.80, 1.00, 1.30, 1.60]
n_seeds = 50

# Dataset generation
datasets_path = project_path("../datasets/degree_imbalance_gaussians")
Path(datasets_path).mkdir(parents=True, exist_ok=True)

print("Generating degree-imbalanced Gaussian datasets...")
dataset_names = []

for sparse_std in sparse_std_values:
    ratio = sparse_std / dense_std

    for seed in range(n_seeds):
        dataset_name = (
            f"gauss_degimbal_cs{'-'.join(map(str, cluster_sizes))}"
            f"_dense{_fmt_float(dense_std)}"
            f"_sparse{_fmt_float(sparse_std)}"
            f"_ratio{_fmt_float(ratio)}"
            f"_seed{seed}"
        )
        dataset_dir = Path(datasets_path) / dataset_name
        train_dir = dataset_dir / "train"

        if not train_dir.exists():
            X, labels = degree_imbalanced_gaussians(
                cluster_sizes=cluster_sizes,
                centers=centers,
                dense_std=dense_std,
                sparse_std=sparse_std,
                seed=seed,
            )
            save_dataset(
                data=X,
                labels=labels,
                path=datasets_path,
                name=dataset_name,
                feature_cols=["x", "y"],
                label_col="labels",
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
    "affinity": "rbf_nearest_neighbors",
    "gamma": 1.0,
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

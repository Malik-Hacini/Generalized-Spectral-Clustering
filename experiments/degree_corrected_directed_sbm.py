"""Benchmark clustering methods on degree-corrected directed SBM datasets."""

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *


def save_graph_dataset(adjacency_matrix, labels, path: str, name: str) -> None:
    """Save a sparse graph dataset in the project's graph.npz format."""
    adjacency_matrix = sp.csr_matrix(adjacency_matrix)
    labels = np.asarray(labels)

    dataset_dir = Path(path) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        dataset_dir / "graph.npz",
        adj_data=adjacency_matrix.data,
        adj_indices=adjacency_matrix.indices,
        adj_indptr=adjacency_matrix.indptr,
        adj_shape=np.asarray(adjacency_matrix.shape, dtype=np.int64),
        labels=labels,
    )


def _fmt_float(value: float) -> str:
    """Format float values as path-safe tokens."""
    return f"{value:.4f}".replace(".", "p")


"""
Basic experiment config:
"""
save_path = "../results"
experiment_name = "benchmark_degree_corrected_directed_sbm"
mode = "grid_search"
metrics = ("ami", "graph_ch")
n_jobs = -1
verbose = True

"""
Degree-corrected directed SBM configuration:
"""
# Fixed graph-generation parameters
block_sizes = [500, 500, 500]
p_intra = 0.05
p_inter = 0.01
power_law_exponents = (1.8, 3.5, 3.5)
high_block_scale = 2.5

# Varying parameters: lower values make blocks 2 and 3 increasingly low-degree
low_block_scales = [1.2, 0.9, 0.7, 0.5, 0.35]
n_seeds = 50

# Dataset generation
datasets_path = project_path("../datasets/degree_corrected_directed_sbm")
Path(datasets_path).mkdir(parents=True, exist_ok=True)

print("Generating degree-corrected directed SBM datasets...")
dataset_names = []
block_sizes_token = "-".join(str(size) for size in block_sizes)
exponent_token = "-".join(_fmt_float(value) for value in power_law_exponents)

for low_scale in low_block_scales:
    block_degree_scales = (high_block_scale, low_scale, low_scale)
    scale_ratio = low_scale / high_block_scale
    scale_token = "-".join(_fmt_float(value) for value in block_degree_scales)

    for seed in range(n_seeds):
        dataset_name = (
            f"dcdisbm_b{block_sizes_token}"
            f"_pintra{_fmt_float(p_intra)}"
            f"_pinter{_fmt_float(p_inter)}"
            f"_alpha{exponent_token}"
            f"_scale{scale_token}"
            f"_ratio{_fmt_float(scale_ratio)}"
            f"_seed{seed}"
        )
        dataset_dir = Path(datasets_path) / dataset_name
        graph_file = dataset_dir / "graph.npz"

        if not graph_file.exists():
            adjacency_matrix, labels = degree_corrected_directed_sbm(
                block_sizes=block_sizes,
                p_intra=p_intra,
                p_inter=p_inter,
                power_law_exponents=power_law_exponents,
                block_degree_scales=block_degree_scales,
                seed=seed,
            )
            save_graph_dataset(
                adjacency_matrix=adjacency_matrix,
                labels=labels,
                path=datasets_path,
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
    ("spectral", "GSC-N"),
]

"""
Parameters configuration:
"""
default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
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
            "filter_coeffs": {2: 0.5, 3: 0.5},
        }
    },
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

"""
Benchmark clustering methods on degree-imbalance DSBM datasets.

Evaluates SC-N, DSC+, and GSC-N on directed SBM graphs with varying
out-degree imbalance between high-degree and low-degree blocks.
"""

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


def _fmt_prob(value: float) -> str:
    """Format probabilities as path-safe strings for dataset names."""
    return f"{value:.4f}".replace(".", "p")


"""
Basic experiment config:
"""
save_path = project_path("results")
experiment_name = "benchmark_degree_imbalance"
mode = "grid_search"  # Either "score", "grid_search" or "viz"
metrics = ("ami", "graph_ch")  # Valid metrics include "ami", "graph_ch"
n_jobs = -1  # Number of parallel jobs (-1 to use all available cores)
verbose = True

"""
Degree imbalance DiSBM configuration:
Study how method performance varies with degree inhomogeneity between clusters.
Block 0 has high degree (scale=2.5), blocks 1-2 have varying scales to create imbalance.
"""
# Fixed graph-generation parameters
block_sizes = [300, 300, 300]
p_intra = 0.05
p_inter = 0.01
power_law_exponents = (1.8, 3.5, 3.5)  # Block 0 heavy-tailed, blocks 1-2 lighter-tailed
high_block_scale = 2.5

# Varying parameters: inhomogeneity controlled by low-block scales
# Lower scale values increase degree imbalance between blocks
low_block_scales = [1.2, 0.9, 0.7, 0.5, 0.35]
n_seeds = 20

# Dataset generation
datasets_path = project_path("datasets/degree_imbalance")
Path(datasets_path).mkdir(parents=True, exist_ok=True)

# Generate datasets
print("Generating degree-imbalance DiSBM datasets...")
dataset_names = []
block_sizes_token = "-".join(str(size) for size in block_sizes)

for low_scale in low_block_scales:
    for seed in range(n_seeds):
        dataset_name = (
            f"dcdisbm_degimbal_b{block_sizes_token}"
            f"_pintra{_fmt_prob(p_intra)}"
            f"_pinter{_fmt_prob(p_inter)}"
            f"_highscale{_fmt_prob(high_block_scale)}"
            f"_lowscale{_fmt_prob(low_scale)}"
            f"_seed{seed}"
        )
        dataset_dir = Path(datasets_path) / dataset_name
        graph_file = dataset_dir / "graph.npz"

        if not graph_file.exists():
            try:
                adjacency_matrix, labels = degree_corrected_directed_sbm(
                    block_sizes=block_sizes,
                    p_intra=p_intra,
                    p_inter=p_inter,
                    power_law_exponents=power_law_exponents,
                    block_degree_scales=(high_block_scale, low_scale, low_scale),
                    seed=seed,
                )
                save_graph_dataset(
                    adjacency_matrix=adjacency_matrix,
                    labels=labels,
                    path=datasets_path,
                    name=dataset_name,
                )
                print(f"  Created: {dataset_name}")
            except Exception as exc:
                print(f"  Skipped (generation failed): {dataset_name} -> {exc}")

        if graph_file.exists():
            dataset_names.append(dataset_name)
        else:
            print(f"  Skipped (missing graph.npz): {dataset_name}")

if not dataset_names:
    raise RuntimeError("No valid datasets were generated/found under datasets/degree_imbalance")

print(f"Total datasets: {len(dataset_names)}")

"""
Methods configuration:
"""
method_specs = [
    ("spectral", "SC-N"),
    ("spectral", "SC-UN"),
    ("dsc", "DSC+"),
    ("spectral", "GSC-N"),
    ("spectral", "GSC-UN"),
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
    ),  # Grid search for GSC methods
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

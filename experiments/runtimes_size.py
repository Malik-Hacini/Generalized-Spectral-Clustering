"""Runtime benchmark vs dataset size for SC-UN and GSC-UN variants.

This script builds synthetic DSBM graph datasets with increasing size and
compares runtime of:
1) SC-UN (standard unnormalized spectral clustering)
2) GSC-UN (grid search over t, alpha)
3) GSC-UN without tuning (fixed t, alpha)
"""

# A small one-off warm-up dataset to absorb first-call overhead
# (imports, BLAS thread pool init, sklearn internals).
warmup_n_nodes = 450
warmup_seed = 2026

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

from scipy.sparse.csgraph import connected_components
from utils.file_manager import save_graph_dataset


"""
Basic experiment config:
"""
save_path = project_path("results")
experiment_name = "benchmark_runtimes_size"
mode = "grid_search"
metrics = ("ami",)
n_jobs = -1
verbose = True


"""
Synthetic DSBM datasets with increasing size.
Expected degree scales like degree_factor * log(n), so the graph size is
O(n log n) asymptotically.

We use a directed 3-block SBM with strong diagonal structure and mild cyclic
asymmetry. This keeps the planted 3-way partition identifiable, unlike the old
core-periphery setup where the two peripheral blocks were statistically
indistinguishable.
"""
dataset_sizes = [600, 1200, 2400, 3600, 4800, 6000, 7200, 9600]
n_seeds = 3
n_clusters = 3
degree_factor = 1.0
block_probability_weights = np.array(
    [
        [0.20, 0.03, 0.005],
        [0.005, 0.20, 0.03],
        [0.03, 0.005, 0.20],
    ],
    dtype=float,
)
connectivity_growth = 1.15
max_generation_tries = 12

datasets_path = project_path("datasets/runtimes_size_disbm")


def _fmt_float(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def _dataset_name(n_nodes: int, seed: int, warmup: bool = False) -> str:
    prefix = "disbm_runtime_warmup" if warmup else "disbm_runtime"
    return (
        f"{prefix}_n{n_nodes}"
        f"_k{n_clusters}"
        f"_cycle3"
        f"_degf{_fmt_float(degree_factor)}"
        f"_seed{seed}"
    )


def _block_sizes(n_nodes: int) -> list[int]:
    base = n_nodes // n_clusters
    sizes = [base] * n_clusters
    sizes[0] += n_nodes - sum(sizes)
    return sizes


def _scale_probability_matrix(block_sizes: list[int], target_degree: float) -> np.ndarray:
    sizes = np.asarray(block_sizes, dtype=float)
    unit_degree = 0.0
    for block, size in enumerate(sizes):
        row_sum = float(np.dot(block_probability_weights[block], sizes) - block_probability_weights[block, block])
        unit_degree += size * row_sum
    unit_degree /= float(np.sum(sizes))
    scale = 0.0 if unit_degree == 0.0 else target_degree / unit_degree
    return np.minimum(1.0, scale * block_probability_weights)


def _generate_runtime_disbm(n_nodes: int, seed: int):
    block_sizes = _block_sizes(n_nodes)
    target_degree = degree_factor * np.log(n_nodes)
    for attempt in range(max_generation_tries):
        probability_matrix = _scale_probability_matrix(block_sizes, target_degree)
        adjacency_matrix, labels = directed_sbm(block_sizes, probability_matrix, seed=seed + attempt)
        adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        undirected = (adjacency_matrix + adjacency_matrix.T).sign().tocsr()
        if connected_components(undirected, directed=False, return_labels=False) == 1:
            return adjacency_matrix, np.asarray(labels, dtype=int)
        target_degree *= connectivity_growth

    raise RuntimeError(
        "Failed to generate a connected runtime DSBM. "
        f"Tried {max_generation_tries} times up to target degree {target_degree:.4f}."
    )


def _ensure_dataset(n_nodes: int, seed: int, warmup: bool = False) -> str:
    dataset_name = _dataset_name(n_nodes, seed, warmup=warmup)
    graph_file = Path(datasets_path) / dataset_name / "graph.npz"
    if not graph_file.exists():
        adjacency_matrix, labels = _generate_runtime_disbm(n_nodes=n_nodes, seed=seed)
        save_graph_dataset(adjacency_matrix, labels, datasets_path, dataset_name)
        print(f"  Created: {dataset_name}")
    return dataset_name


def generate_datasets() -> list[str]:
    Path(datasets_path).mkdir(parents=True, exist_ok=True)
    print("Generating size-scaling DSBM datasets...")
    dataset_names = [
        _ensure_dataset(n_nodes, seed)
        for n_nodes in dataset_sizes
        for seed in range(n_seeds)
    ]
    print(f"Total datasets: {len(dataset_names)}")
    return dataset_names


def generate_warmup_dataset() -> str:
    Path(datasets_path).mkdir(parents=True, exist_ok=True)
    return _ensure_dataset(warmup_n_nodes, warmup_seed, warmup=True)


"""
Methods configuration:
"""
method_specs = [
    ("spectral", "SC-UN"),
    ("spectral", "GSC-UN"),
    ("spectral", "GSC-UN-NoTune"),
]


"""
Parameters configuration:
"""
default_params = {
    "random_state": 42,
    "n_init": 1,
    "affinity": "precomputed",
    "n_it": 1,
    "assign_labels": "kmeans",
}

method_params = [
    (
        "SC-UN",
        {
            "laplacian_method": "unnorm",
            "standard": True,
            "measure": None,
        },
    ),
    (
        "GSC-UN",
        {
            "laplacian_method": "unnorm",
            "measure": (
                teleporting_undirected_measure,
                {"alpha": np.arange(0.0, 1.5, 0.1), "t": range(0, 25, 4)},
            ),
        },
    ),
    (
        "GSC-UN-NoTune",
        {
            "laplacian_method": "unnorm",
            "measure": (
                teleporting_undirected_measure,
                {"alpha": 0.4, "t": 10},
            ),
        },
    ),
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
    warmup_dataset_name = generate_warmup_dataset()
    dataset_names = generate_datasets()

    print("Running one warm-up pass to reduce first-run timing bias...")
    experiment(
        experiment_name=f"{experiment_name}_warmup",
        dataset_names=[warmup_dataset_name],
        method_specs=method_specs,
        config=config,
        load_path=datasets_path,
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=1,
        verbose=False,
    )

    experiment(
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

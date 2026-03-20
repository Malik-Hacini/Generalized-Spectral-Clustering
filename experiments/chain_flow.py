"""
Benchmark clustering methods on DiSBM-Chain networks with varying flow strength.

Evaluates SC-N, DSC+, and GSC-N on directed chain-structured SBM graphs with varying
inter-block flow strength (forward and backward edge probabilities).
"""

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

from utils.file_manager import save_graph_dataset


def _fmt_prob(value: float) -> str:
    """Format probabilities as path-safe strings for dataset names."""
    return f"{value:.4f}".replace(".", "p")


"""
Basic experiment config:
"""
save_path = project_path("results")
experiment_name = "benchmark_chain_flow"
mode = "grid_search"  # Either "score", "grid_search" or "viz"
metrics = ("ami", "graph_ch")  # Valid metrics include "ami", "graph_ch"
n_jobs = -1  # Number of parallel jobs (-1 to use all available cores)
verbose = True

"""
Chain flow strength DiSBM configuration:
Study how method performance varies with inter-block flow strength in chain-structured networks.
Flow strength is controlled by varying p_forward and p_backward (edge probabilities between blocks).
"""
# Fixed graph-generation parameters
block_sizes = [500, 500, 500]
p_intra = 0.1
base_p_forward = 0.15
base_p_backward = 0.01

# Varying parameters: flow strength multiplier

flow_strengths = [0.01, 0.05, 0.1, 0.2, 0.5]
n_seeds = 20

# Dataset generation
load_path = project_path("datasets/chain_flow")
datasets_path = load_path
Path(datasets_path).mkdir(parents=True, exist_ok=True)

# Generate datasets
print("Generating chain-flow DiSBM datasets...")
dataset_names = []
block_sizes_token = "-".join(str(size) for size in block_sizes)

for flow_strength in flow_strengths:
    p_forward = flow_strength
    p_backward = 0.01

    for seed in range(n_seeds):
        dataset_name = (
            f"disbm_chainflow_b{block_sizes_token}"
            f"_pintra{_fmt_prob(p_intra)}"
            f"_pfwd{_fmt_prob(p_forward)}"
            f"_pbwd{_fmt_prob(p_backward)}"
            f"_seed{seed}"
        )
        dataset_dir = Path(datasets_path) / dataset_name
        graph_file = dataset_dir / "graph.npz"

        if not graph_file.exists():
            try:
                adjacency_matrix, labels = chain_sbm(
                    block_sizes=block_sizes,
                    p_intra=p_intra,
                    p_forward=p_forward,
                    p_backward=p_backward,
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
    raise RuntimeError("No valid datasets were generated/found under datasets/chain_flow")

print(f"Total datasets: {len(dataset_names)}")

"""
Methods configuration:
"""
method_specs = [
    ("spectral", "SC-N"),
    ("spectral", "SC-UN"),
    ("dsc", "DSC+"),
    ("di_sim", "DI-SIM-R"),
    ("di_sim", "DI-SIM-L"),
    ("di_sim", "DI-SIM-C"),
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
        {"alpha": np.arange(0, 2, 0.1), "t": range(0, 25)},
    ),  # Grid search for GSC methods
    "tau": (
        avg_deg_taus,
        {"s": np.arange(-1, 1, 0.5)},
    ),  # Grid search for DI-SIM methods
    "metric_params": {
        "graph_ch": {
            "filter_coeffs": {2: 0.5, 3: 0.5},
        }
    },
}

method_params = [
    ("SC-UN", {"laplacian_method": "unnorm", "standard": True, "measure": None}),
    ("SC-N", {"laplacian_method": "norm", "standard": True, "measure": None}),
    ("DI-SIM-R", {"embedding": "right"}),
    ("DI-SIM-L", {"embedding": "left"}),
    ("DI-SIM-C", {"embedding": "combined"}),
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

if __name__ == "__main__":
    experiment(
        experiment_name=experiment_name,
        config=config,
        dataset_names=dataset_names,
        method_specs=method_specs,
        load_path=load_path,
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
    )

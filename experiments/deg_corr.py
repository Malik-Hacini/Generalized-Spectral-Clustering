"""
Benchmark clustering methods on degree-corrected directed SBM datasets.

This experiment varies degree-correction parameters in the first block and tracks
how method performance changes:
- `scale` sweep: vary first-block degree scale (`s1`) at fixed Pareto exponent (`a1`)
- `alpha` sweep: vary first-block Pareto exponent (`a1`) at fixed degree scale (`s1`)
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

from pathlib import Path

from utils.file_manager import save_graph_dataset


def _fmt_token(value: float) -> str:
    """Format numeric values as path-safe strings (e.g., 1.2500 -> 1p2500)."""
    return f"{value:.4f}".replace(".", "p")


"""Basic experiment config."""
save_path = project_path("results")
experiment_name = "benchmark_deg_corr"
mode = "grid_search"
metrics = ("ami", "graph_ch")
n_jobs = -1
verbose = True

"""Degree-corrected DiSBM configuration."""
block_sizes = [500, 500, 500]
p_intra = 0.05
p_inter = 0.01
n_seeds = 20

# Baselines for non-varied blocks and the fixed value in each sweep.
base_alpha_high = 1.8
base_scale_high = 2.5
other_alphas = (3.5, 3.5)
other_scales = (0.7, 0.7)

# Parameter sweeps for the first block.
scale_high_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
alpha_high_values = [1.2, 1.5, 1.8, 2.2, 2.8, 3.5]

# Dataset generation
load_path = project_path("datasets/deg_corr")
datasets_path = load_path
Path(datasets_path).mkdir(parents=True, exist_ok=True)


"""Methods configuration."""
method_specs = [
    ("spectral", "SC-N"),
    ("dsc", "DSC+"),
    ("di_sim", "DI-SIM-R"),
    ("di_sim", "DI-SIM-L"),
    ("di_sim", "DI-SIM-C"),
    ("spectral", "GSC-N"),
]

"""Parameters configuration."""
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
    "tau": (
        avg_deg_taus,
        {"s": np.arange(-1, 1, 0.5)},
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


def _dataset_name(
    block_sizes_token: str,
    sweep_mode: str,
    alpha_high: float,
    scale_high: float,
    seed: int,
) -> str:
    return (
        f"dcdisbm_degcorr_b{block_sizes_token}"
        f"_pintra{_fmt_token(p_intra)}"
        f"_pinter{_fmt_token(p_inter)}"
        f"_mode{sweep_mode}"
        f"_a1{_fmt_token(alpha_high)}"
        f"_s1{_fmt_token(scale_high)}"
        f"_seed{seed}"
    )


def generate_datasets() -> list[str]:
    print("Generating degree-corrected DiSBM datasets...")
    dataset_names: list[str] = []
    block_sizes_token = "-".join(str(size) for size in block_sizes)

    sweep_specs: list[tuple[str, float, float]] = []
    sweep_specs.extend(("scale", base_alpha_high, scale_high) for scale_high in scale_high_values)
    sweep_specs.extend(("alpha", alpha_high, base_scale_high) for alpha_high in alpha_high_values)

    for sweep_mode, alpha_high, scale_high in sweep_specs:
        power_law_exponents = (alpha_high, *other_alphas)
        block_degree_scales = (scale_high, *other_scales)

        for seed in range(n_seeds):
            dataset_name = _dataset_name(
                block_sizes_token=block_sizes_token,
                sweep_mode=sweep_mode,
                alpha_high=alpha_high,
                scale_high=scale_high,
                seed=seed,
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
                except Exception as exc:
                    print(f"  Skipped (generation failed): {dataset_name} -> {exc}")

            if graph_file.exists():
                dataset_names.append(dataset_name)
            else:
                print(f"  Skipped (missing graph.npz): {dataset_name}")

    if not dataset_names:
        raise RuntimeError("No valid datasets were generated/found under datasets/deg_corr")

    print(f"Total datasets: {len(dataset_names)}")
    return dataset_names


if __name__ == "__main__":
    experiment(
        experiment_name=experiment_name,
        config=config,
        dataset_names=generate_datasets(),
        method_specs=method_specs,
        load_path=load_path,
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
    )

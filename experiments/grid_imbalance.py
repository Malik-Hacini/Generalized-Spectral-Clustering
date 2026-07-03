"""Benchmark clustering methods on grid-imbalance datasets."""

from __future__ import annotations

import argparse
import re

if __package__ is None or __package__ == "":
    from common import *
else:
    from experiments.common import *

from pathlib import Path

from utils.synthetic_data_gen.generate_imbalance_checkers import grid_imbalance
from utils.file_manager import save_dataset

save_path = project_path("results")
experiment_name = "benchmark_grid_imbalance"
mode = "grid_search"
metrics = ("ami", "ch")
n_jobs = -1
verbose = True



n_seeds = 50
datasets_path = Path(project_path("datasets/grid_imbalance"))

method_specs = [
    ("spectral", "SC-UN"),
    ("spectral", "SC-N"),
    ("dsc", "DSC+"),
    ("spectral", "GSC-N"),
    ("spectral", "GSC-UN"),
    ("di_sim", "DI-SIM-R"),
    ("di_sim", "DI-SIM-L"),
    ("di_sim", "DI-SIM-C"),
]

default_params = {
    "n_neighbors": (log_neighbors, {"factor": 1}),
    "random_state": 42,
    "n_init": 1,
    "affinity": "nearest_neighbors",
    "n_it": 1,
    "assign_labels": "kmeans",
    "measure": (
        teleporting_undirected_measure,
        {"alpha": np.arange(0, 1.51, 0.1), "t": range(0, 26)},
    ),
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


def parse_grid_size(value: str) -> tuple[int, int]:
    raw = value.strip().lower()

    # Accept simple rectangular format, e.g. "2x1".
    if "x" in raw and "(" not in raw:
        rows, cols = raw.split("x", maxsplit=1)
        return int(rows.strip()), int(cols.strip())

    # Accept square shorthand, e.g. "4".
    if raw.isdigit():
        size = int(raw)
        return size, size

    # Accept tuple-product form, e.g. "(2,1)x(2,1)".
    tuple_product_match = re.fullmatch(
        r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*x\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
        raw,
    )
    if tuple_product_match is not None:
        r1, c1, r2, c2 = (int(group) for group in tuple_product_match.groups())
        if (r1, c1) != (r2, c2):
            raise argparse.ArgumentTypeError(
                f"Unsupported tuple-product grid size '{value}'. Expected both tuples to match."
            )
        return r1, c1

    raise argparse.ArgumentTypeError(
        "Invalid --grid-size value. Use one of: '4', '2x1', '(2,1)x(2,1)'."
    )


def format_grid_size(grid_size: tuple[int, int]) -> str:
    return f"{grid_size[0]}x{grid_size[1]}"


def generate_datasets(n_high:int, grid_size: tuple[int, int]) -> list[str]:
    datasets_path.mkdir(parents=True, exist_ok=True)
    print(f"Generating grid-imbalance datasets for {format_grid_size(grid_size)}...")
    dataset_names = []
    n_low_values = [n_high // 15, n_high // 10, n_high // 5, n_high // 3, n_high // 2]
    for n_low in n_low_values:
        for seed in range(n_seeds):
            dataset_name = f"grid_{format_grid_size(grid_size)}_high{n_high}_low{n_low}_seed{seed}"
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
    parser = argparse.ArgumentParser(description="Run the grid-imbalance benchmark")
    parser.add_argument(
        "--grid-size",
        type=parse_grid_size,
        default=(4, 4),
        help="Grid size, e.g. 2, 3, 4, or rectangular 2x1.",
    )
    parser.add_argument(
        "--n-high",
        type=int,
        default=300,
        help="Number of nodes in the high-density blocks (default: 300)",
    )
    args = parser.parse_args()
    grid_size = args.grid_size
    n_high = args.n_high

    experiment(
        experiment_name=experiment_name,
        dataset_names=generate_datasets(n_high, grid_size),
        method_specs=method_specs,
        config=config,
        load_path=str(datasets_path),
        save_path=save_path,
        mode=mode,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
    )

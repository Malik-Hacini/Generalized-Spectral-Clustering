"""Preprocessing benchmark for the size-scaling runtime DSBM experiment.

This benchmark uses the exact same datasets as `experiments/runtimes_size.py` and
measures only the preprocessing steps of each method, i.e. everything up to the
construction of the Laplacian.
"""

from __future__ import annotations

import time
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.extend([str(ROOT), str(ROOT / "scikit-learn")])
    import runtimes_size as size_benchmark
    from common import *
else:
    from experiments import runtimes_size as size_benchmark
    from experiments.common import *

import pandas as pd
from sklearn.manifold._laplacian import Laplacian

from utils.file_manager import load_dataset


save_path = project_path("results")
experiment_name = "benchmark_runtimes_size_preprocessing"


def _measure_grid(method_name: str) -> dict[str, object]:
    for name, params in size_benchmark.method_params:
        if name == method_name:
            return params["measure"][1]
    raise KeyError(f"Unknown method: {method_name}")


GRID_MEASURE = _measure_grid("GSC-UN")
NO_TUNE_MEASURE = _measure_grid("GSC-UN-NoTune")
METHODS = ("SC-UN", "GSC-UN", "GSC-UN-NoTune")


def _sc_un_preprocessing(adjacency_matrix) -> None:
    symmetric_adjacency = (0.5 * (adjacency_matrix + adjacency_matrix.T)).tocsr()
    Laplacian(symmetric_adjacency, standard=True, measure=None).unnormalized()


def _gsc_un_preprocessing(adjacency_matrix, alpha: float, t: int) -> None:
    measure = teleporting_undirected_measure(adjacency_matrix, alpha=alpha, t=t)
    Laplacian(adjacency_matrix, standard=False, measure=measure).unnormalized()


def _run_method_preprocessing(adjacency_matrix, method_name: str) -> float:
    start = time.perf_counter()
    if method_name == "SC-UN":
        _sc_un_preprocessing(adjacency_matrix)
    elif method_name == "GSC-UN":
        for alpha in GRID_MEASURE["alpha"]:
            for t in GRID_MEASURE["t"]:
                _gsc_un_preprocessing(adjacency_matrix, float(alpha), int(t))
    elif method_name == "GSC-UN-NoTune":
        _gsc_un_preprocessing(
            adjacency_matrix,
            alpha=float(NO_TUNE_MEASURE["alpha"]),
            t=int(NO_TUNE_MEASURE["t"]),
        )
    else:
        raise ValueError(f"Unsupported method: {method_name}")
    return time.perf_counter() - start


def benchmark_preprocessing(dataset_names: list[str]) -> pd.DataFrame:
    rows = []
    for dataset_name in dataset_names:
        adjacency_matrix, _ = load_dataset(size_benchmark.datasets_path, dataset_name)
        row = {"dataset": dataset_name, "n": int(adjacency_matrix.shape[0])}
        for method_name in METHODS:
            row[method_name] = _run_method_preprocessing(adjacency_matrix, method_name)
        rows.append(row)
    return pd.DataFrame(rows)


def save_runtime_csv(runtime_df: pd.DataFrame) -> Path:
    output_dir = Path(save_path) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{experiment_name}_runtimes.csv"
    runtime_df.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    warmup_dataset_name = size_benchmark.generate_warmup_dataset()
    dataset_names = size_benchmark.generate_datasets()

    print("Running one warm-up preprocessing pass to reduce first-run timing bias...")
    warmup_adjacency, _ = load_dataset(
        size_benchmark.datasets_path, warmup_dataset_name
    )
    for method_name in METHODS:
        _run_method_preprocessing(warmup_adjacency, method_name)

    runtime_df = benchmark_preprocessing(dataset_names)
    output_file = save_runtime_csv(runtime_df)
    print(f"Saved preprocessing runtimes to: {output_file}")

"""Shared helpers for the complexity-analysis experiments."""
# pyright: reportGeneralTypeIssues=false, reportMissingImports=false

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VENDORED_SKLEARN = ROOT / "scikit-learn"

for path in (VENDORED_SKLEARN, ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh
import sklearn
from sklearn import cluster
from sklearn.datasets import make_blobs
from sklearn.manifold._spectral_embedding import _graph_is_connected, _set_diag
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils._arpack import _init_arpack_v0

from competitors.measures import teleporting_undirected_measure
from disbm_utils.generate_disbm_datasets import core_periphery_disbm, directed_sbm

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"

sklearn_file = getattr(sklearn, "__file__", None)
if sklearn_file is None:
    raise ImportError("Unable to locate sklearn.__file__ for vendored import check.")
SKLEARN_FILE = Path(sklearn_file).resolve()
if VENDORED_SKLEARN not in SKLEARN_FILE.parents:
    raise ImportError(
        "These experiments must use the vendored scikit-learn tree. "
        f"Imported sklearn from {SKLEARN_FILE}."
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> list[int]:
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def log_neighbors_from_n(n_samples: int, factor: float = 1.0) -> int:
    return max(1, int(math.ceil(factor * math.log(n_samples))))


def generate_gaussian_point_cloud(
    n_samples: int,
    n_features: int,
    n_clusters: int = 3,
    cluster_std: float = 2.5,
    seed: int = 42,
    connectivity_n_neighbors: int | None = None,
    center_box: float = 4.0,
    center_shrink: float = 0.8,
    max_tries: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    current_center_box = float(center_box)
    for attempt in range(max_tries):
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            center_box=(-current_center_box, current_center_box),
            random_state=seed + attempt,
        )
        X = X.astype(np.float64)
        y = y.astype(np.int64)
        if connectivity_n_neighbors is None:
            return X, y

        W, _ = build_knn_graph(
            X,
            n_neighbors=connectivity_n_neighbors,
            algorithm="brute",
            n_jobs=1,
        )
        if _graph_is_connected(W):
            return X, y
        current_center_box *= center_shrink

    raise RuntimeError(
        "Failed to generate a connected Gaussian point cloud. "
        f"Tried {max_tries} times with final center_box={current_center_box:.4f}."
    )


def generate_sparse_network(
    n_nodes: int,
    n_clusters: int = 3,
    degree_factor: float = 1.0,
    p_in: float = 0.8,
    p_out: float = 0.2,
    seed: int = 42,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_clusters), n_nodes // n_clusters)
    if labels.size < n_nodes:
        labels = np.concatenate(
            [labels, np.zeros(n_nodes - labels.size, dtype=np.int64)]
        )
    rng.shuffle(labels)

    degree = log_neighbors_from_n(n_nodes, degree_factor)
    rows: list[int] = []
    cols: list[int] = []

    for i in range(n_nodes):
        probs = np.where(labels == labels[i], p_in, p_out).astype(np.float64)
        probs[i] = 0.0
        probs /= probs.sum()
        neighbors = rng.choice(
            n_nodes, size=min(degree, n_nodes - 1), replace=False, p=probs
        )
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors.tolist())

    data = np.ones(len(rows), dtype=np.float64)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return adjacency, labels


def generate_core_periphery_disbm(
    n_nodes: int,
    n_clusters: int = 3,
    p_core: float = 0.10,
    p_periphery: float = 0.01,
    p_core_periphery: float = 0.05,
    p_periphery_core: float = 0.005,
    degree_factor: float | None = None,
    seed: int = 42,
    ensure_connected: bool = True,
    max_tries: int = 12,
    connectivity_growth: float = 1.2,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    base = n_nodes // n_clusters
    sizes = [base] * n_clusters
    sizes[0] += n_nodes - sum(sizes)
    weights = np.full((n_clusters, n_clusters), p_periphery, dtype=np.float64)
    weights[0, 0] = p_core
    weights[0, 1:] = p_core_periphery
    weights[1:, 0] = p_periphery_core
    for i in range(1, n_clusters):
        weights[i, i] = p_periphery

    if degree_factor is None:
        adjacency, labels = core_periphery_disbm(
            block_sizes=sizes,
            p_core=p_core,
            p_periphery=p_periphery,
            p_core_periphery=p_core_periphery,
            p_periphery_core=p_periphery_core,
            seed=seed,
        )
        return sparse.csr_matrix(adjacency), np.asarray(labels, dtype=np.int64)

    target_degree = float(degree_factor * math.log(n_nodes))
    sizes_arr = np.asarray(sizes, dtype=np.float64)

    def scaled_probability_matrix(current_target_degree: float) -> np.ndarray:
        unit_degree = 0.0
        for a, size_a in enumerate(sizes_arr):
            row_sum = float(np.dot(weights[a], sizes_arr) - weights[a, a])
            unit_degree += size_a * row_sum
        unit_degree /= float(n_nodes)
        scale = 0.0 if unit_degree == 0.0 else current_target_degree / unit_degree
        return np.minimum(1.0, scale * weights)

    current_target_degree = target_degree
    for attempt in range(max_tries):
        P = scaled_probability_matrix(current_target_degree)
        adjacency, labels = directed_sbm(sizes, P, seed=seed + attempt)
        adjacency_csr = sparse.csr_matrix(adjacency)
        if (not ensure_connected) or _graph_is_connected(adjacency_csr):
            return adjacency_csr, np.asarray(labels, dtype=np.int64)
        current_target_degree *= connectivity_growth

    raise RuntimeError(
        "Failed to generate a connected sparse DISBM graph. "
        f"Tried {max_tries} times with final target degree {current_target_degree:.4f}."
    )


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int,
    algorithm: str = "auto",
    mode: str = "connectivity",
    n_jobs: int = 1,
) -> tuple[sparse.csr_matrix, str]:
    estimator = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric="euclidean",
        n_jobs=n_jobs,
    )
    estimator.fit(X)
    graph = estimator.kneighbors_graph(None, mode=mode)
    backend = str(getattr(estimator, "_fit_method", algorithm))
    return graph.tocsr(), backend


def build_transition_matrix(W: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    degree_safe = degree.copy()
    degree_safe[degree_safe == 0.0] = 1.0
    return sparse.csr_matrix(sparse.diags(1.0 / degree_safe) @ W)


def compute_measure(
    W: sparse.csr_matrix,
    alpha: float,
    t: int,
    epsilon: float = 1e-8,
) -> np.ndarray:
    return teleporting_undirected_measure(W, alpha=alpha, t=t, epsilon=epsilon)


def compute_measure_from_transition(
    P: sparse.spmatrix,
    alpha: float,
    t: int,
    epsilon: float = 1e-8,
) -> np.ndarray:
    n_nodes = P.shape[0]
    v = np.ones(n_nodes, dtype=np.float64) / n_nodes
    for _ in range(t):
        v = np.asarray(v @ P).ravel()
    nu = np.power(v, alpha)
    nu[nu <= 0.0] = epsilon
    nu /= nu.sum()
    return nu


def compute_xi(P: sparse.csr_matrix, nu: np.ndarray) -> np.ndarray:
    return np.asarray(P.T @ nu).ravel()


def symmetrize_graph(W: sparse.csr_matrix) -> sparse.csr_matrix:
    return sparse.csr_matrix(0.5 * (W + W.T))


def build_standard_normalized_laplacian(W_sym: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(W_sym.sum(axis=1)).ravel().astype(np.float64)
    degree_safe = degree.copy()
    degree_safe[degree_safe == 0.0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(degree_safe)
    L = sparse.csr_matrix(sparse.diags(degree) - W_sym)
    scale = sparse.diags(inv_sqrt)
    return sparse.csr_matrix(scale @ L @ scale)


def build_generalized_normalized_laplacian(
    P: sparse.csr_matrix,
    nu: np.ndarray,
    xi: np.ndarray,
) -> sparse.csr_matrix:
    D_nu = sparse.diags(nu)
    D_sum = sparse.diags(nu + xi)
    B = sparse.csr_matrix(D_nu @ P)
    L = sparse.csr_matrix(D_sum - (B + B.T))
    denom = nu + xi
    denom_safe = denom.copy()
    denom_safe[denom_safe == 0.0] = 1.0
    scale = sparse.diags(1.0 / np.sqrt(denom_safe))
    return sparse.csr_matrix(scale @ L @ scale)


def run_shift_invert_eigensolver(
    laplacian: sparse.csr_matrix,
    n_components: int,
    random_state: int,
    symmetric: bool = True,
) -> None:
    solver_input = _set_diag(laplacian.copy(), 1.0, True)
    if solver_input is None:
        raise RuntimeError("_set_diag returned None.")
    solver_input = sparse.csr_matrix(solver_input)
    solver_input *= -1.0
    rng = check_random_state(random_state)
    v0 = _init_arpack_v0(solver_input.shape[0], rng)
    if symmetric:
        eigsh(solver_input, k=n_components, sigma=1.0, which="LM", tol=0, v0=v0)
    else:
        eigs(solver_input, k=n_components, sigma=1.0, which="LM", tol=0, v0=v0)


def benchmark_runtime(func: Any, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return times


def graph_theory_term(n_nodes: int, n_features: int, n_neighbors: int, backend: str) -> float:
    if backend == "brute":
        return float(n_features * (n_nodes**2))
    log_n = math.log(n_nodes)
    return float(n_nodes * log_n + n_nodes * (n_features * log_n + n_neighbors))


def loglog_slope(x: Any, y: Any) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(x_arr[mask]), np.log(y_arr[mask]), deg=1)
    return float(slope)


def aggregate_results(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    group_cols = [c for c in raw_df.columns if c not in {"repeat", "runtime_seconds"}]
    summary_df = (
        raw_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            runtime_mean=("runtime_seconds", "mean"),
            runtime_std=("runtime_seconds", "std"),
            runtime_median=("runtime_seconds", "median"),
            runtime_min=("runtime_seconds", "min"),
            runtime_max=("runtime_seconds", "max"),
            n_repeats=("runtime_seconds", "size"),
        )
        .sort_values(["benchmark", "series", "x_value"])  # type: ignore[call-overload]
        .reset_index(drop=True)
    )
    summary_df["runtime_std"] = summary_df["runtime_std"].fillna(0.0)
    if "theory_term" in summary_df.columns:
        theory = pd.to_numeric(summary_df["theory_term"], errors="coerce")
        theory_arr = np.asarray(theory, dtype=float)
        if np.isfinite(theory_arr).any() and np.nanmax(theory_arr) > 0.0:
            summary_df["time_over_theory"] = np.where(
                theory_arr > 0.0, summary_df["runtime_mean"] / theory_arr, np.nan
            )

    fits: list[dict[str, Any]] = []
    for (benchmark, series, x_name), group in summary_df.groupby(
        ["benchmark", "series", "x_name"], dropna=False, sort=False
    ):
        payload: dict[str, Any] = {
            "benchmark": benchmark,
            "series": series,
            "x_name": x_name,
            "loglog_runtime_slope": loglog_slope(group["x_value"], group["runtime_mean"]),
        }
        if "time_over_theory" in group.columns:
            ratio = np.asarray(group["time_over_theory"], dtype=float)
            finite_ratio = ratio[np.isfinite(ratio)]
            payload["loglog_ratio_slope"] = loglog_slope(
                group["x_value"], group["time_over_theory"]
            )
            payload["mean_time_over_theory"] = (
                float(np.mean(finite_ratio)) if finite_ratio.size else float("nan")
            )
        fits.append(payload)
    return summary_df, fits


def save_outputs(
    benchmark_name: str,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fit_summary: list[dict[str, Any]],
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    ensure_dir(out_dir)
    raw_path = out_dir / f"{benchmark_name}_raw.csv"
    summary_path = out_dir / f"{benchmark_name}_summary.csv"
    json_path = out_dir / f"{benchmark_name}_summary.json"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "benchmark": benchmark_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": json_ready(config),
                "fit_summary": json_ready(fit_summary),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {"raw": raw_path, "summary": summary_path, "json": json_path}


def pointcloud_file_name(n_samples: int, n_features: int, seed: int) -> str:
    return f"gaussian_N{n_samples}_d{n_features}_seed{seed}.npz"


def save_pointcloud_dataset(
    X: np.ndarray,
    y: np.ndarray,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, X=X, y=y, metadata=json.dumps(metadata, sort_keys=True))


def load_pointcloud_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata"]))
    return payload["X"], payload["y"], metadata


def benchmark_sc_fit(
    X: Any,
    n_clusters: int,
    n_neighbors: int | None,
    random_state: int,
) -> None:
    kwargs: dict[str, Any] = {
        "n_clusters": n_clusters,
        "laplacian_method": "norm",
        "standard": True,
        "measure": None,
        "assign_labels": "kmeans",
        "random_state": random_state,
    }
    if n_neighbors is None:
        kwargs["affinity"] = "precomputed"
    else:
        kwargs["affinity"] = "nearest_neighbors"
        kwargs["n_neighbors"] = n_neighbors
    cluster.SpectralClustering(**kwargs).fit(X)


def benchmark_gsc_fit(
    X: Any,
    n_clusters: int,
    n_neighbors: int | None,
    alpha: float,
    t: int,
    random_state: int,
) -> None:
    kwargs: dict[str, Any] = {
        "n_clusters": n_clusters,
        "laplacian_method": "norm",
        "standard": False,
        "measure": (teleporting_undirected_measure, {"alpha": alpha, "t": t}),
        "assign_labels": "kmeans",
        "random_state": random_state,
    }
    if n_neighbors is None:
        kwargs["affinity"] = "precomputed"
    else:
        kwargs["affinity"] = "nearest_neighbors"
        kwargs["n_neighbors"] = n_neighbors
    cluster.SpectralClustering(**kwargs).fit(X)

"""
Generate a LaTeX table of dataset statistics for the paper.

Statistics computed per dataset:
- N         : number of nodes
- |E|       : number of edges
- K         : number of ground-truth classes
- Gini      : Gini coefficient of the in-degree distribution
- Reciprocity      : edge reciprocity  r = |{(i,j): A_ij=1 and A_ji=1}| / |E|
- C-L Reciprocity  : cluster-level reciprocity  (1/(k(k-1))) * sum_{a≠b} 2*min(E_ab,E_ba)/(E_ab+E_ba)
- #WCC      : number of weakly connected components
- #SCC      : number of strongly connected components

Usage
-----
# All datasets under datasets/
python plots/tables_dataset_infos.py

# Explicit list (folder names or synthetic generator names)
python plots/tables_dataset_infos.py --datasets karate dolphins polblogs email_eu_core
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from sklearn.neighbors import kneighbors_graph  # type: ignore

from competitors.neighbors import log_neighbors
from plots.common import project_path, resolve_output_file
from utils.synthetic_data_gen.generate_disbm_datasets import (
    chain_sbm,
    degree_corrected_directed_sbm,
)
from utils.file_manager import load_dataset, save_graph_dataset

# ---------------------------------------------------------------------------
# Default dataset list — edit this to control which datasets are processed
# when running without --datasets.
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = [
    "polblogs",
    "football",
    "email_eu_core",
    "polbooks",
    "breast_tissue",
    "wine",
    "iris",
    "seeds",
    "segmentation",
    "wdbc",
    "olivetti_faces",
    "mnist64",
    "ph_recognition",
]

# Keep k-NN construction aligned with experiment defaults: log_neighbors(X, factor=1).
DEFAULT_KNN_FACTOR = 1

SYNTHETIC_DATASET_SPECS = {
    "DiSBM_Chain": {
        "builder": chain_sbm,
        "params": {
            "block_sizes": [500, 500, 500],
            "p_intra": 0.1,
            "p_forward": 0.15,
            "p_backward": 0.01,
            "seed": 42,
        },
    },
    "Deg-corr": {
        "builder": degree_corrected_directed_sbm,
        "params": {
            "block_sizes": [500, 500, 500],
            "p_intra": 0.05,
            "p_inter": 0.01,
            "power_law_exponents": (1.8, 3.5, 3.5),
            "block_degree_scales": (2.5, 0.7, 0.7),
            "seed": 42,
        },
    },
}

# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def _load_graph_npz(path: Path) -> tuple[sp.csr_matrix, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    A = sp.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    labels = data["labels"] if "labels" in data else None
    return A, labels


def load_dataset_graph(datasets_root: Path, name: str) -> tuple[sp.csr_matrix, np.ndarray | None]:
    """Load a dataset graph; build default k-NN graph for point-cloud datasets."""
    dataset_dir = datasets_root / name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    graph_file = dataset_dir / "graph.npz"
    if graph_file.exists():
        return _load_graph_npz(graph_file)

    data, labels = load_dataset(str(datasets_root), name, split="train", label_col="labels")

    if sp.issparse(data):
        A = sp.csr_matrix(data)
    else:
        X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError(
                f"Point-cloud dataset '{name}' must have 2D features; got shape {X.shape}."
            )
        n_neighbors = int(log_neighbors(X, factor=DEFAULT_KNN_FACTOR))
        print(
            f"    No graph.npz found for '{name}'; building directed k-NN graph "
            f"with k={n_neighbors} (log_neighbors, factor={DEFAULT_KNN_FACTOR})."
        )
        A = sp.csr_matrix(
            kneighbors_graph(
                X,
                n_neighbors=n_neighbors,
                include_self=False,
                mode="connectivity",
            )
        )

    labels_arr = None if labels is None else np.asarray(labels)
    return A, labels_arr


def ensure_dataset_exists(datasets_root: Path, name: str) -> None:
    """Create selected synthetic datasets on demand when missing from disk."""
    graph_file = datasets_root / name / "graph.npz"
    if graph_file.exists():
        return

    spec = SYNTHETIC_DATASET_SPECS.get(name)
    if spec is None:
        return

    adjacency_matrix, labels = spec["builder"](**spec["params"])
    save_graph_dataset(adjacency_matrix=adjacency_matrix, labels=labels, path=str(datasets_root), name=name)
    print(f"  Generated synthetic dataset: {name}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    v = np.sort(np.abs(values.astype(float)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def _cluster_level_reciprocity(W: sp.csr_matrix, y: np.ndarray | None) -> float:
    """Cluster-level reciprocity: (1/(k(k-1))) * sum_{a≠b} 2*min(E_ab,E_ba)/(E_ab+E_ba)."""
    if y is None or len(np.unique(y)) <= 1:
        return float("nan")

    W_dense = W.toarray().astype(float)
    clusters = np.unique(y)
    k = len(clusters)

    E = np.zeros((k, k))
    for idx_a, a in enumerate(clusters):
        for idx_b, b in enumerate(clusters):
            nodes_a = np.where(y == a)[0]
            nodes_b = np.where(y == b)[0]
            E[idx_a, idx_b] = W_dense[np.ix_(nodes_a, nodes_b)].sum()

    reciprocity_sum = 0.0
    for a in range(k):
        for b in range(k):
            if a != b:
                denom = E[a, b] + E[b, a]
                if denom > 0:
                    reciprocity_sum += (2 * min(E[a, b], E[b, a])) / denom

    return reciprocity_sum / (k * (k - 1))


def compute_stats(A: sp.csr_matrix, labels: np.ndarray | None) -> dict:
    A = sp.csr_matrix(A, dtype=float)
    A_bin = sp.csr_matrix(A > 0, dtype=float)

    shape = A_bin.shape
    n_nodes = int(shape[0] if shape is not None else 0)
    n_edges = int(A_bin.nnz)
    n_classes = int(len(np.unique(labels))) if labels is not None else float("nan")

    # In-degree Gini
    in_degrees = np.asarray(A_bin.sum(axis=0)).ravel()
    gini = _gini(in_degrees)

    # Reciprocity: fraction of edges (i→j) for which j→i also exists
    A_T = A_bin.T.tocsr()
    mutual = A_bin.multiply(A_T)
    n_mutual_directed = int(mutual.nnz)   # counts both (i,j) and (j,i) sides
    reciprocity = n_mutual_directed / n_edges if n_edges > 0 else 0.0

    # Cluster-level reciprocity
    cl_reciprocity = _cluster_level_reciprocity(A_bin, labels)

    # WCC and SCC
    n_wcc = csgraph.connected_components(A_bin, directed=True, connection="weak")[0]
    n_scc = csgraph.connected_components(A_bin, directed=True, connection="strong")[0]

    return {
        "N": n_nodes,
        "|E|": n_edges,
        "K": n_classes,
        "Gini": gini,
        "Reciprocity": reciprocity,
        "CL-Reciprocity": cl_reciprocity,
        "WCC": n_wcc,
        "SCC": n_scc,
    }


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

_DISPLAY_NAMES: dict[str, str] = {
    "karate": "Karate",
    "dolphins": "Dolphins",
    "football": "Football",
    "polbooks": "PolBooks",
    "polblogs": "PolBlogs",
    "email_eu_core": "Email-EUcore",
    "wiki_vote": "WikiVote",
    "wikics": "WikiCS",
    "wikics_lcc": "WikiCS (LCC)",
    "cora": "Cora",
    "cora_lcc": "Cora (LCC)",
    "cora_ml": "Cora-ML",
    "cora_ml_lcc": "Cora-ML (LCC)",
    "citeseer": "CiteSeer",
    "citeseer_lcc": "CiteSeer (LCC)",
    "cornell": "Cornell",
    "texas": "Texas",
    "wisconsin": "Wisconsin",
    "telegram": "Telegram",
}


def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _fmt(value, fmt: str = ".0f") -> str:
    if isinstance(value, float) and np.isnan(value):
        return "--"
    return format(value, fmt)


def generate_dataset_table(
    rows: list[dict],
    caption: str = r"\textbf{Dataset statistics.} "
                   r"$N$: nodes, $|E|$: edges, $K$: classes, "
                   r"Gini: in-degree Gini coefficient, "
                   r"$\rho$: reciprocity, $\hat\rho$: cluster-level reciprocity, "
                   r"\#WCC / \#SCC: weakly/strongly connected components.",
    label: str = "tab:dataset_stats",
) -> str:
    col_spec = r"l|rrr|rrrr|rr"
    header = (
        r"    \textbf{Dataset} & $N$ & $|E|$ & $K$ & "
        r"\textbf{Gini} & $\rho$ & $\hat{\rho}$ & "
        r"\#\textbf{WCC} & \#\textbf{SCC} \\"
    )

    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{" + caption + r"}",
        r"  \label{" + label + r"}",
        r"  \begin{adjustbox}{width=\textwidth}",
        r"  \begin{tabular}{" + col_spec + r"}",
        r"    \Xhline{2\arrayrulewidth}",
        header,
        r"    \Xhline{2\arrayrulewidth}",
    ]

    for row in rows:
        name = _display_name(row["name"])
        s = row["stats"]
        cells = [
            name,
            _fmt(s["N"], ",d"),
            _fmt(s["|E|"], ",d"),
            _fmt(s["K"], "d") if not isinstance(s["K"], float) else "--",
            _fmt(s["Gini"], ".3f"),
            _fmt(s["Reciprocity"], ".3f"),
            _fmt(s["CL-Reciprocity"], ".3f"),
            _fmt(s["WCC"], "d"),
            _fmt(s["SCC"], "d"),
        ]
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.extend([
        r"    \Xhline{2\arrayrulewidth}",
        r"  \end{tabular}",
        r"  \end{adjustbox}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX dataset statistics table")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "Dataset names (folder names under datasets/ directory). "
            "If omitted, uses DEFAULT_DATASETS defined at the top of this file."
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="datasets",
        help="Path to the datasets root directory (default: datasets/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to plots/tables/.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="dataset_stats.tex",
        help="Output filename (default: dataset_stats.tex)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Override table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:dataset_stats",
        help="LaTeX label for the table (default: tab:dataset_stats)",
    )
    args = parser.parse_args()

    datasets_root = project_path(args.datasets_dir)

    # Determine which datasets to process
    if args.datasets is not None:
        names = args.datasets
    elif DEFAULT_DATASETS:
        names = DEFAULT_DATASETS
    else:
        names = sorted(
            d.name for d in datasets_root.iterdir()
            if d.is_dir() and (d / "graph.npz").exists()
        )

    if not names:
        print("No graph datasets found.")
        return

    # Compute stats
    rows = []
    for name in names:
        print(f"  Processing {name}...", end=" ")
        try:
            ensure_dataset_exists(datasets_root, name)
            A, labels = load_dataset_graph(datasets_root, name)
            stats = compute_stats(A, labels)
            rows.append({"name": name, "stats": stats})
            print(f"N={stats['N']:,}  |E|={stats['|E|']:,}  K={stats['K']}")
        except Exception as exc:
            print(f"SKIPPED ({exc})")

    if not rows:
        print("No results to write.")
        return

    rows.sort(key=lambda row: _display_name(row["name"]).casefold())

    output_file = resolve_output_file(
        args.output_dir,
        args.output_name,
        "tables",
        args.datasets_dir,
        "dataset_stats.tex",
    )

    kwargs: dict = {"label": args.label}
    if args.caption:
        kwargs["caption"] = args.caption

    latex = generate_dataset_table(rows, **kwargs)
    output_file.write_text(latex)
    print(f"\nLaTeX table saved to: {output_file}")


if __name__ == "__main__":
    main()

"""Benchmark xi = P^T nu against its O(m) sparse-matvec model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from common import (
    RESULTS_DIR,
    aggregate_results,
    build_knn_graph,
    build_transition_matrix,
    compute_measure_from_transition,
    compute_xi,
    ensure_dir,
    generate_gaussian_point_cloud,
    log_neighbors_from_n,
    parse_int_list,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="500,1000,2000,5000")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--cluster-std", type=float, default=2.5)
    parser.add_argument(
        "--graph-algorithm",
        choices=("auto", "brute", "kd_tree", "ball_tree"),
        default="auto",
    )
    parser.add_argument("--neighbors-factor", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows: list[dict[str, object]] = []

    for n_samples in parse_int_list(args.n_values):
        n_neighbors = log_neighbors_from_n(n_samples, factor=args.neighbors_factor)
        X, _ = generate_gaussian_point_cloud(
            n_samples=n_samples,
            n_features=args.dimension,
            n_clusters=args.clusters,
            cluster_std=args.cluster_std,
            seed=args.seed,
            connectivity_n_neighbors=n_neighbors,
        )
        W, _ = build_knn_graph(
            X,
            n_neighbors=n_neighbors,
            algorithm=args.graph_algorithm,
            n_jobs=args.n_jobs,
        )
        P = build_transition_matrix(W)
        nu = compute_measure_from_transition(P, alpha=args.alpha, t=args.t)

        for repeat in range(args.repeats):
            start = time.perf_counter()
            xi = compute_xi(P, nu)
            elapsed = time.perf_counter() - start
            raw_rows.append(
                {
                    "benchmark": "xi_construction",
                    "series": "xi",
                    "x_name": "n",
                    "x_value": float(n_samples),
                    "N": n_samples,
                    "d": args.dimension,
                    "K": n_neighbors,
                    "t": args.t,
                    "alpha": args.alpha,
                    "nnz": int(P.nnz),
                    "xi_sum": float(xi.sum()),
                    "theory_term": float(P.nnz),
                    "theory_label": "m",
                    "repeat": repeat,
                    "runtime_seconds": elapsed,
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    summary_df, fit_summary = aggregate_results(raw_df)
    paths = save_outputs(
        benchmark_name="xi_construction",
        raw_df=raw_df,
        summary_df=summary_df,
        fit_summary=fit_summary,
        config=vars(args),
        out_dir=ensure_dir(args.out_dir),
    )
    print(f"Saved summary to {paths['summary']}")


if __name__ == "__main__":
    main()

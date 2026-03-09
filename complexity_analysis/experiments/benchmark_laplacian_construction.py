"""Benchmark sparse symmetrization and Laplacian assembly steps."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from common import (
    RESULTS_DIR,
    aggregate_results,
    build_generalized_normalized_laplacian,
    build_knn_graph,
    build_standard_normalized_laplacian,
    build_transition_matrix,
    compute_measure_from_transition,
    compute_xi,
    ensure_dir,
    generate_gaussian_point_cloud,
    log_neighbors_from_n,
    parse_int_list,
    save_outputs,
    symmetrize_graph,
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
        W_sym = symmetrize_graph(W)
        P = build_transition_matrix(W)
        nu = compute_measure_from_transition(P, alpha=args.alpha, t=args.t)
        xi = compute_xi(P, nu)

        tasks = [
            (
                "sc_symmetrization",
                lambda: symmetrize_graph(W),
                float(W.nnz),
                "m",
            ),
            (
                "sc_norm_laplacian",
                lambda: build_standard_normalized_laplacian(W_sym),
                float(W_sym.nnz + W_sym.shape[0]),
                "m + N",
            ),
            (
                "gsc_norm_laplacian",
                lambda: build_generalized_normalized_laplacian(P, nu, xi),
                float(P.nnz + P.shape[0]),
                "m + N",
            ),
        ]

        for series, func, theory_term, theory_label in tasks:
            for repeat in range(args.repeats):
                start = time.perf_counter()
                matrix = func()
                elapsed = time.perf_counter() - start
                raw_rows.append(
                    {
                        "benchmark": "laplacian_construction",
                        "series": series,
                        "x_name": "n",
                        "x_value": float(n_samples),
                        "N": n_samples,
                        "d": args.dimension,
                        "K": n_neighbors,
                        "t": args.t,
                        "alpha": args.alpha,
                        "nnz": int(matrix.nnz),
                        "theory_term": theory_term,
                        "theory_label": theory_label,
                        "repeat": repeat,
                        "runtime_seconds": elapsed,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    summary_df, fit_summary = aggregate_results(raw_df)
    paths = save_outputs(
        benchmark_name="laplacian_construction",
        raw_df=raw_df,
        summary_df=summary_df,
        fit_summary=fit_summary,
        config=vars(args),
        out_dir=ensure_dir(args.out_dir),
    )
    print(f"Saved summary to {paths['summary']}")


if __name__ == "__main__":
    main()

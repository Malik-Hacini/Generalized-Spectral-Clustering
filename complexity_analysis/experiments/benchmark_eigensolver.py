"""Benchmark the sparse spectral step against the standard sparse-Krylov baseline."""

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
    generate_sparse_network,
    log_neighbors_from_n,
    parse_int_list,
    run_shift_invert_eigensolver,
    save_outputs,
    symmetrize_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-type", choices=("pointcloud", "network"), default="pointcloud")
    parser.add_argument("--n-values", default="500,1000,2000")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--cluster-std", type=float, default=2.5)
    parser.add_argument("--degree-factor", type=float, default=1.0)
    parser.add_argument(
        "--graph-algorithm",
        choices=("auto", "brute", "kd_tree", "ball_tree"),
        default="auto",
    )
    parser.add_argument("--neighbors-factor", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows: list[dict[str, object]] = []

    for n_samples in parse_int_list(args.n_values):
        if args.input_type == "pointcloud":
            n_neighbors = log_neighbors_from_n(n_samples, factor=args.neighbors_factor)
            X, _ = generate_gaussian_point_cloud(
                n_samples=n_samples,
                n_features=args.dimension,
                n_clusters=args.clusters,
                cluster_std=args.cluster_std,
                seed=args.seed,
                connectivity_n_neighbors=n_neighbors,
            )
            W, backend = build_knn_graph(
                X,
                n_neighbors=n_neighbors,
                algorithm=args.graph_algorithm,
                n_jobs=args.n_jobs,
            )
        else:
            W, _ = generate_sparse_network(
                n_nodes=n_samples,
                n_clusters=args.clusters,
                degree_factor=args.degree_factor,
                seed=args.seed,
            )
            n_neighbors = float("nan")
            backend = "given_sparse_graph"

        W_sym = symmetrize_graph(W)
        P = build_transition_matrix(W)
        nu = compute_measure_from_transition(P, alpha=args.alpha, t=args.t)
        xi = compute_xi(P, nu)
        L_sc = build_standard_normalized_laplacian(W_sym)
        L_gsc = build_generalized_normalized_laplacian(P, nu, xi)

        tasks = [
            ("sc_eigensolver", L_sc),
            ("gsc_eigensolver", L_gsc),
        ]

        baseline = float(W.nnz + W.shape[0] * args.n_components)
        for series, laplacian in tasks:
            for repeat in range(args.repeats):
                start = time.perf_counter()
                run_shift_invert_eigensolver(
                    laplacian,
                    n_components=args.n_components,
                    random_state=args.seed,
                    symmetric=True,
                )
                elapsed = time.perf_counter() - start
                raw_rows.append(
                    {
                        "benchmark": f"eigensolver_{args.input_type}",
                        "series": series,
                        "x_name": "n",
                        "x_value": float(n_samples),
                        "N": n_samples,
                        "d": args.dimension if args.input_type == "pointcloud" else float("nan"),
                        "K": n_neighbors,
                        "t": args.t if series == "gsc_eigensolver" else float("nan"),
                        "alpha": args.alpha if series == "gsc_eigensolver" else float("nan"),
                        "n_components": args.n_components,
                        "nnz": int(W.nnz),
                        "backend": backend,
                        "theory_term": baseline,
                        "theory_label": "baseline: m + N r (up to q)",
                        "repeat": repeat,
                        "runtime_seconds": elapsed,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    summary_df, fit_summary = aggregate_results(raw_df)
    paths = save_outputs(
        benchmark_name=f"eigensolver_{args.input_type}",
        raw_df=raw_df,
        summary_df=summary_df,
        fit_summary=fit_summary,
        config=vars(args),
        out_dir=ensure_dir(args.out_dir),
    )
    print(f"Saved summary to {paths['summary']}")


if __name__ == "__main__":
    main()

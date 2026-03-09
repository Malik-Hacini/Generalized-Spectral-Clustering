"""Benchmark GSC measure construction against the O(m + t m + N) model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from common import (
    RESULTS_DIR,
    aggregate_results,
    build_knn_graph,
    compute_measure,
    ensure_dir,
    generate_gaussian_point_cloud,
    log_neighbors_from_n,
    parse_int_list,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=("n", "t"), default="n")
    parser.add_argument("--n-values", default="500,1000,2000,5000")
    parser.add_argument("--t-values", default="1,2,4,8,16")
    parser.add_argument("--n-samples", type=int, default=2000)
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
    sweep_values = parse_int_list(args.n_values if args.sweep == "n" else args.t_values)
    raw_rows: list[dict[str, object]] = []

    for value in sweep_values:
        n_samples = value if args.sweep == "n" else args.n_samples
        t_value = args.t if args.sweep == "n" else value
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
        theory_term = float((t_value + 1) * W.nnz + W.shape[0])

        for repeat in range(args.repeats):
            start = time.perf_counter()
            nu = compute_measure(W, alpha=args.alpha, t=t_value)
            elapsed = time.perf_counter() - start
            raw_rows.append(
                {
                    "benchmark": "measure_construction",
                    "series": "nu",
                    "x_name": args.sweep,
                    "x_value": float(value),
                    "N": n_samples,
                    "d": args.dimension,
                    "K": n_neighbors,
                    "t": t_value,
                    "alpha": args.alpha,
                    "nnz": int(W.nnz),
                    "measure_sum": float(nu.sum()),
                    "theory_term": theory_term,
                    "theory_label": "(t + 1) m + N",
                    "repeat": repeat,
                    "runtime_seconds": elapsed,
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    summary_df, fit_summary = aggregate_results(raw_df)
    paths = save_outputs(
        benchmark_name="measure_construction",
        raw_df=raw_df,
        summary_df=summary_df,
        fit_summary=fit_summary,
        config=vars(args),
        out_dir=ensure_dir(args.out_dir),
    )
    print(f"Saved summary to {paths['summary']}")


if __name__ == "__main__":
    main()

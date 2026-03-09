"""Benchmark single-run SC and GSC end-to-end wall-clock times."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    RESULTS_DIR,
    aggregate_results,
    benchmark_gsc_fit,
    benchmark_sc_fit,
    ensure_dir,
    generate_gaussian_point_cloud,
    generate_sparse_network,
    log_neighbors_from_n,
    parse_int_list,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-type", choices=("pointcloud", "network"), default="pointcloud")
    parser.add_argument("--n-values", default="500,1000,2000")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--cluster-std", type=float, default=2.5)
    parser.add_argument("--degree-factor", type=float, default=1.0)
    parser.add_argument("--neighbors-factor", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows: list[dict[str, object]] = []

    for n_samples in parse_int_list(args.n_values):
        if args.input_type == "pointcloud":
            n_neighbors = log_neighbors_from_n(n_samples, factor=args.neighbors_factor)
            data, _ = generate_gaussian_point_cloud(
                n_samples=n_samples,
                n_features=args.dimension,
                n_clusters=args.clusters,
                cluster_std=args.cluster_std,
                seed=args.seed,
                connectivity_n_neighbors=n_neighbors,
            )
        else:
            data, _ = generate_sparse_network(
                n_nodes=n_samples,
                n_clusters=args.clusters,
                degree_factor=args.degree_factor,
                seed=args.seed,
            )
            n_neighbors = None

        tasks = [
            ("SC", lambda: benchmark_sc_fit(data, args.clusters, n_neighbors, args.seed)),
            (
                "GSC",
                lambda: benchmark_gsc_fit(
                    data, args.clusters, n_neighbors, args.alpha, args.t, args.seed
                ),
            ),
        ]

        for series, func in tasks:
            for repeat in range(args.repeats):
                start = time.perf_counter()
                func()
                elapsed = time.perf_counter() - start
                raw_rows.append(
                    {
                        "benchmark": f"full_run_{args.input_type}",
                        "series": series,
                        "x_name": "n",
                        "x_value": float(n_samples),
                        "N": n_samples,
                        "d": args.dimension if args.input_type == "pointcloud" else np.nan,
                        "K": n_neighbors if n_neighbors is not None else np.nan,
                        "t": args.t if series == "GSC" else np.nan,
                        "alpha": args.alpha if series == "GSC" else np.nan,
                        "repeat": repeat,
                        "runtime_seconds": elapsed,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    summary_df, fit_summary = aggregate_results(raw_df)
    paths = save_outputs(
        benchmark_name=f"full_run_{args.input_type}",
        raw_df=raw_df,
        summary_df=summary_df,
        fit_summary=fit_summary,
        config=vars(args),
        out_dir=ensure_dir(args.out_dir),
    )
    print(f"Saved summary to {paths['summary']}")


if __name__ == "__main__":
    main()

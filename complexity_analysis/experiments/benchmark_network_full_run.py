"""Benchmark single-run SC and GSC on precomputed DISBM networks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from common import (
    RESULTS_DIR,
    aggregate_results,
    benchmark_gsc_fit,
    benchmark_sc_fit,
    ensure_dir,
    generate_core_periphery_disbm,
    parse_int_list,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="500,1000,2000,5000")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--degree-factor", type=float, default=1.0)
    parser.add_argument("--p-core", type=float, default=0.10)
    parser.add_argument("--p-periphery", type=float, default=0.01)
    parser.add_argument("--p-core-periphery", type=float, default=0.05)
    parser.add_argument("--p-periphery-core", type=float, default=0.005)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows: list[dict[str, object]] = []

    for n_nodes in parse_int_list(args.n_values):
        adjacency, _ = generate_core_periphery_disbm(
            n_nodes=n_nodes,
            n_clusters=args.clusters,
            p_core=args.p_core,
            p_periphery=args.p_periphery,
            p_core_periphery=args.p_core_periphery,
            p_periphery_core=args.p_periphery_core,
            degree_factor=args.degree_factor,
            seed=args.seed,
        )
        theory_term = float(adjacency.nnz)

        tasks = [
            ("SC", lambda: benchmark_sc_fit(adjacency, args.clusters, None, args.seed)),
            (
                "GSC",
                lambda: benchmark_gsc_fit(
                    adjacency, args.clusters, None, args.alpha, args.t, args.seed
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
                        "benchmark": "full_run_disbm_network",
                        "series": series,
                        "x_name": "n",
                        "x_value": float(n_nodes),
                        "N": n_nodes,
                        "nnz": int(adjacency.nnz),
                        "clusters": args.clusters,
                        "t": args.t if series == "GSC" else float("nan"),
                        "alpha": args.alpha if series == "GSC" else float("nan"),
                        "p_core": args.p_core,
                        "p_periphery": args.p_periphery,
                        "p_core_periphery": args.p_core_periphery,
                        "p_periphery_core": args.p_periphery_core,
                        "theory_term": theory_term,
                        "theory_label": "m",
                        "repeat": repeat,
                        "runtime_seconds": elapsed,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    summary_df, fit_summary = aggregate_results(raw_df)
    paths = save_outputs(
        benchmark_name="full_run_disbm_network",
        raw_df=raw_df,
        summary_df=summary_df,
        fit_summary=fit_summary,
        config=vars(args),
        out_dir=ensure_dir(args.out_dir),
    )
    print(f"Saved summary to {paths['summary']}")


if __name__ == "__main__":
    main()

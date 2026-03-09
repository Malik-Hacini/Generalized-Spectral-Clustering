"""Measure the GSC preprocessing overhead on precomputed DISBM networks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    RESULTS_DIR,
    benchmark_gsc_fit,
    benchmark_sc_fit,
    build_generalized_normalized_laplacian,
    build_standard_normalized_laplacian,
    build_transition_matrix,
    compute_measure_from_transition,
    compute_xi,
    ensure_dir,
    generate_core_periphery_disbm,
    json_ready,
    parse_int_list,
    save_outputs,
    symmetrize_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="500,1000,2000,5000,10000")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--degree-factor", type=float, default=1.0)
    parser.add_argument("--p-core", type=float, default=0.10)
    parser.add_argument("--p-periphery", type=float, default=0.01)
    parser.add_argument("--p-core-periphery", type=float, default=0.05)
    parser.add_argument("--p-periphery-core", type=float, default=0.005)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def timed(callable_obj):
    start = time.perf_counter()
    value = callable_obj()
    return time.perf_counter() - start, value


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    raw_rows: list[dict[str, object]] = []
    derived_rows: list[dict[str, object]] = []

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

        for repeat in range(args.repeats):
            t_sc_full, _ = timed(
                lambda: benchmark_sc_fit(adjacency, args.clusters, None, args.seed)
            )
            t_gsc_full, _ = timed(
                lambda: benchmark_gsc_fit(
                    adjacency, args.clusters, None, args.alpha, args.t, args.seed
                )
            )

            t_sym, W_sym = timed(lambda: symmetrize_graph(adjacency))
            t_sc_lap, L_sc = timed(lambda: build_standard_normalized_laplacian(W_sym))

            t_P, P = timed(lambda: build_transition_matrix(adjacency))
            t_measure, nu = timed(
                lambda: compute_measure_from_transition(P, alpha=args.alpha, t=args.t)
            )
            t_xi, xi = timed(lambda: compute_xi(P, nu))
            t_gsc_lap, L_gsc = timed(
                lambda: build_generalized_normalized_laplacian(P, nu, xi)
            )

            _ = L_sc, L_gsc

            rows = [
                ("full_sc", t_sc_full),
                ("full_gsc", t_gsc_full),
                ("sc_symmetrization", t_sym),
                ("sc_laplacian", t_sc_lap),
                ("transition", t_P),
                ("measure", t_measure),
                ("xi", t_xi),
                ("gsc_laplacian", t_gsc_lap),
            ]

            for series, runtime in rows:
                raw_rows.append(
                    {
                        "benchmark": "network_overhead_disbm",
                        "series": series,
                        "x_name": "m",
                        "x_value": float(adjacency.nnz),
                        "N": n_nodes,
                        "nnz": int(adjacency.nnz),
                        "clusters": args.clusters,
                        "t": args.t,
                        "alpha": args.alpha,
                        "repeat": repeat,
                        "runtime_seconds": runtime,
                    }
                )

            extra_overhead = (t_P + t_measure + t_xi + t_gsc_lap) - (t_sym + t_sc_lap)
            derived_rows.append(
                {
                    "benchmark": "network_overhead_disbm",
                    "x_name": "m",
                    "x_value": float(adjacency.nnz),
                    "N": n_nodes,
                    "nnz": int(adjacency.nnz),
                    "clusters": args.clusters,
                    "t": args.t,
                    "alpha": args.alpha,
                    "repeat": repeat,
                    "full_sc": t_sc_full,
                    "full_gsc": t_gsc_full,
                    "transition": t_P,
                    "measure": t_measure,
                    "xi": t_xi,
                    "gsc_laplacian": t_gsc_lap,
                    "sc_symmetrization": t_sym,
                    "sc_laplacian": t_sc_lap,
                    "gsc_preprocessing": t_P + t_measure + t_xi + t_gsc_lap,
                    "sc_preprocessing": t_sym + t_sc_lap,
                    "extra_overhead_seconds": extra_overhead,
                    "overhead_fraction_of_gsc": extra_overhead / t_gsc_full,
                    "overhead_fraction_of_sc": extra_overhead / t_sc_full,
                    "full_runtime_ratio": t_gsc_full / t_sc_full,
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    derived_df = pd.DataFrame(derived_rows)

    summary_df = (
        derived_df.groupby(["benchmark", "x_name", "x_value", "N", "nnz", "clusters", "t", "alpha"], as_index=False)
        .agg(
            full_sc_mean=("full_sc", "mean"),
            full_sc_std=("full_sc", "std"),
            full_gsc_mean=("full_gsc", "mean"),
            full_gsc_std=("full_gsc", "std"),
            gsc_preprocessing_mean=("gsc_preprocessing", "mean"),
            sc_preprocessing_mean=("sc_preprocessing", "mean"),
            extra_overhead_mean=("extra_overhead_seconds", "mean"),
            extra_overhead_std=("extra_overhead_seconds", "std"),
            overhead_fraction_of_gsc_mean=("overhead_fraction_of_gsc", "mean"),
            overhead_fraction_of_gsc_std=("overhead_fraction_of_gsc", "std"),
            overhead_fraction_of_sc_mean=("overhead_fraction_of_sc", "mean"),
            full_runtime_ratio_mean=("full_runtime_ratio", "mean"),
            n_repeats=("repeat", "size"),
        )
        .sort_values(["x_value"])
        .reset_index(drop=True)
    )
    summary_df = summary_df.fillna(0.0)
    summary_df["extra_overhead_over_m"] = summary_df["extra_overhead_mean"] / summary_df["nnz"]

    benchmark_name = "network_overhead_disbm_components"
    paths = save_outputs(
        benchmark_name=benchmark_name,
        raw_df=raw_df,
        summary_df=summary_df,
        fit_summary=[],
        config=vars(args),
        out_dir=out_dir,
    )

    derived_path = out_dir / "network_overhead_disbm_derived_raw.csv"
    derived_df.to_csv(derived_path, index=False)
    meta_path = out_dir / f"{benchmark_name}.json"
    meta_path.write_text(
        json.dumps(
            {
                "benchmark": benchmark_name,
                "config": json_ready(vars(args)),
                "derived_raw_csv": str(derived_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Saved summary to {paths['summary']}")
    print(f"Saved derived rows to {derived_path}")


if __name__ == "__main__":
    main()
